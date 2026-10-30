# Discipline-aware convergence and error control

Status: design proposal with an opt-in prototype on `codex/discipline-tolerances`.
See `docs/disciplines.md` for the implemented API, model coverage, and limitations.
The design below includes follow-up work; it is not a claim that every phase is
complete. The prototype uses explicitly declared nominal residual references,
not contribution-based relative references, and preserves scalar-mode defaults.

## Related issue

[Issue #67](https://github.com/gdsfactory/circulax/issues/67) also requests
physical GMIN homotopy, per-unknown limiting, and iteration baselines. This
prototype addresses its metadata and tolerance checks, while keeping those
algorithm and benchmarking changes as separate, measurable follow-ups. It does
not close the full issue or claim faster Newton convergence.

## Objective

Give each solved quantity and each equation an appropriate physical tolerance,
so small currents, voltages, optical fields, and other quantities receive useful
accuracy checks in the same simulation. Keep the physical DAE contract
`F(y) + dQ(y)/dt = 0` and the existing Physics / Topology / Analysis separation.

Verilog-AMS supplies the vocabulary: a **nature** describes a quantity and its
units and absolute tolerance; a **discipline** associates potential and flow
natures. This is a useful model for metadata, not a requirement to implement the
Verilog-AMS language or reproduce one simulator's convergence algorithm.

## Findings in the current implementation

- `Circuit` and DC/HB entry points use scalar `rtol=1e-6`, `atol=1e-6`.
- `CircuitLinearSolver._run_newton` in `solvers/linear.py` passes its damped
  Newton update to `optimistix.FixedPointIteration`. The installed implementation
  tests `max(abs(y_new-y_old)/(atol+rtol*abs(y_new))) < 1`.
  This checks the applied update, without a separate physical residual test.
- The transient solver family repeats that pattern and inherits separate
  `newton_rtol=1e-5`, `newton_atol=1e-5` defaults. `Circuit.transient` applies
  circuit tolerances to initialization, but does not propagate them to these
  per-step Newton settings.
- `ComponentGroup` already separates `var_indices` and `eq_indices`, although
  native compilation currently assigns them the same index arrays. Equal indices
  do not imply equal physical units.
- `Inductor.i_L` is a current unknown, while its equation is voltage minus the
  time derivative of flux. Nodal voltage unknowns instead have current-balance
  equations. Internal state equations must therefore declare their own nature.
- Mixed optical/electrical components already exist. Optical pseudo-variables
  and power-normalized waves need metadata matching their actual `z0` convention;
  assigning voltage/current tolerances to every port would be incorrect.
- Ground is imposed through a `1e9` penalty. Linear backends also add `g_leak`
  to Jacobian diagonals. Such numerical additions need explicit treatment when
  evaluating physical residuals, damping, and scaling.

Separate absolute tolerances are justified by units. A common dimensionless
relative tolerance can remain the default; per-nature relative overrides are
optional. Better convergence checks improve confidence in the solution but do
not alone establish total simulation accuracy or parity with another solver.

## Proposed metadata

Add immutable definitions in `circulax/disciplines.py`:

- `Nature`: stable identifier, units, default absolute tolerance, optional
  nominal magnitude, and optional derivative/integral nature identifiers.
- `Discipline`: potential nature and flow nature, with a potential-only option
  for signal-flow models. Do not assume every domain uses a conservative pair.
- Component metadata: per-port discipline, per-state unknown nature, and
  per-state equation nature. Permit explicit equation scaling/reference metadata
  for normalized or generated models.

Keep `ports` and `states` tuples and the `(f_dict, q_dict)` return contract.
Extend `@component` and `@source` with optional metadata mappings keyed by those
existing names. Physics functions continue to operate on ordinary JAX arrays.
Units are declarations, not a runtime dimensional-analysis framework.

| Location | Unknown nature | Equation nature | Q nature, when present |
|---|---|---|---|
| Electrical node | Voltage (V) | Current (A) | Charge (C) |
| Inductor branch state | Current (A) | Voltage (V) | Flux linkage (V s) |
| Thermal node | Temperature (K) | Power (W) | Energy (J) |
| Translational velocity/force formulation | Velocity (m/s) | Force (N) | Momentum (N s) |
| Optical wave state | Declared wave amplitude | Declared wave equation units | Model dependent |
| Normalized model state | Declared normalized quantity | Declared equation units | Model dependent |

Mechanical formulations may choose displacement, velocity, or another potential;
there is no universal rule that a force is always a flow. Annotate the chosen
formulation. Optical ports and wave states must be audited independently.

During compilation, build a global layout with separate unknown and equation
nature IDs and diagnostic names. Validate connected port disciplines before
merging nodes. Resolve tolerance overrides to dense arrays outside the Newton
loop. Preserve compiler return compatibility, carrying local metadata on groups
and assembling the layout in an analysis helper.

Ground is a boundary/reference node and may serve several domains. Do not give
the single existing ground index an arbitrary electrical nature or reject a
valid mixed-domain circuit because its reference is shared. Validate non-ground
connections normally; represent reference constraints explicitly in the layout.
Domain-specific nonzero references require their own model/boundary definition.

## Numerical convergence contract

Use two independent, dimensionless infinity-norm checks at the candidate solution.
For unknown `j`:

```text
x_scale[j] = x_atol[j] + x_rtol[j] * max(abs(y_new[j]), abs(y_old[j]))
E_update = max_j(abs(delta_raw[j]) / x_scale[j])
```

`delta_raw` is the undamped Newton correction used to produce the candidate.
Also report the applied correction. A tiny damping factor must not turn a large
correction into convergence. Frozen-Jacobian methods use their computed raw
correction; the independent residual check remains essential.

For equation `i`, evaluate the actual discretized residual at `y_new`:

```text
r_scale[i] = r_atol[i] + r_rtol[i] * reference[i]
E_residual = max_i(abs(R[i](y_new)) / r_scale[i])
converged = finite_values_and_solves & (E_update <= 1) & (E_residual <= 1)
```

The equation tolerance has the units of `R`, not the units of the unknown at the
same index. Never construct `reference` from `abs(R)` itself: cancellation makes
a small net residual a poor measure of the magnitude of balanced quantities.

Initial reference policy:

- Conservation rows: accumulate magnitudes of component contributions before
  global cancellation, using the same physical discretization as the residual.
  For transient, combine each component's F and discretized Q contribution
  before taking its magnitude; do not use huge `abs(Q)/dt` history terms as an
  artificial relative-error allowance. This is a proposed scale, to validate
  against cancellation-heavy examples, not a claim of SPICE equivalence.
- Constitutive rows: allow a model-supplied magnitude in equation units, based
  on the terms being equated. Where unavailable, use an explicitly declared
  nominal magnitude; otherwise use absolute-only residual checking.
- Any fallback and its provenance are visible in layout diagnostics. Do not
  silently infer a state's nature or nominal scale from its name or value.

Start electrical presets at `voltage_atol=1e-6 V`, `current_atol=1e-12 A`,
retaining Circulax's `rtol=1e-6`. These are proposed initial values, not an
accuracy guarantee. Other domains require explicit, documented presets after
model audit. Permit per-nature, per-unknown, and per-equation overrides.

Treat ground rows as reference constraints with potential tolerances, excluding
their artificial penalty residual from the physical flow norm. Check reference
values explicitly. Longer term, exact constraint rows or reference elimination
can replace the penalty; assess this separately before changing assembly.

Distinguish Jacobian-only regularization from physical leakage. A Jacobian-only
term changes the iteration, not the target residual. Physical leakage requires
consistent F and Jacobian stamps and must be restricted to compatible domain
rows. Do not relabel today's all-diagonal `g_leak` as a physical conductance.

## Public configuration and compatibility

Introduce a `TolerancePolicy` with separate nonlinear update, nonlinear residual,
and transient local-error settings. Resolve policy precedence once:

1. Explicit per-analysis/per-variable/per-equation overrides.
2. Circuit policy overrides.
3. Model/nature defaults.

Within a level, exact variable/equation overrides win over nature overrides.
Reject ambiguous simultaneous legacy scalar arguments and a policy rather than
silently choosing one. A scalar legacy configuration remains available with its
existing behavior during migration; opt into the new checks through the policy.
Unannotated custom models work in legacy mode. Discipline-aware mode requires
complete metadata or an explicit fallback declaration, and reports any fallback.

Route the policy through circuit construction, analysis variants, initialization,
direct solver factories, DC homotopy, transient, HB, and analysis-cache keys.
Keep labels and nature definitions static; tolerances and scales become JAX
arrays. Do not make tolerance weights trainable model parameters.

## Implementation sequence

### 1. Metadata and compilation

Implement definitions, decorator metadata, global layout validation, and policy
resolution. Annotate electrical primitives first, especially ideal sources and
inductors. Audit generated rational states and OSDI descriptors: identify exposed
unknown/equation metadata and normalization, including Schur-reduced systems.
Use explicit adapter declarations where the backend lacks enough information;
do not assume every OSDI internal node is an electrical voltage/KCL pair.

Acceptance: a mixed voltage/current layout resolves correct arrays; incompatible
non-ground connections fail clearly; shared references remain valid; legacy
components and compilation APIs continue to work.

### 2. Electrical DC and shared convergence machinery

Add `solvers/convergence.py` and assembly support for residual reference scales.
Implement a reusable Optimistix-compatible iteration/termination path with raw
correction, accepted residual, finiteness, and linear-solve status. Keep existing
linear backends and implicit differentiation support. Evaluate an initial root
explicitly so an already solved initial guess can return successfully.

Integrate DC, checked solves, source stepping, and existing regularization
continuation. Continuation success requires the final target problem to pass;
intermediate success is insufficient. Tighten failure reporting without changing
legacy return shapes. Add opt-in diagnostics containing worst unknown/equation,
physical and normalized residual, update norm, iterations, damping, and reason.

Acceptance: small damped steps cannot hide a large physical residual; low-current
examples satisfy current tolerances; Dense and KLU agree within declared bounds.

### 3. Transient nonlinear solves and local error control

Reuse convergence machinery across BE, BDF2, trapezoidal, and SDIRK3, including
full, frozen, and refactored Jacobian variants. Each method supplies its own
discretized R and reference contributions; SDIRK checks every stage. Preserve
factorization lifetime, delay-history behavior, and step rejection on failure.

Scale adaptive local-error estimates by **unknown** natures, separately from
Newton residual tolerances. Pass vector tolerances to supported controllers and
validate broadcasting/norm behavior against the installed Diffrax API. Keep
explicit controller settings authoritative; reject conflicting configuration.
Use the same resolved policy for transient initialization and per-step solves.
Constant-step runs still check nonlinear convergence at each step.

Audit the existing methods' error estimators: tighter Newton checks cannot repair
an invalid or poorly calibrated local-error estimate. Include timestep refinement
and global waveform-error measurements in acceptance, rather than treating a
converged Newton solve as sufficient evidence of transient accuracy.

### 4. Optical/multiphysics and harmonic balance

Annotate optical ports, wave states, electro-optic components, carrier states,
and rational factories according to their actual conventions. Support custom
thermal/mechanical disciplines with a small coupled example. Resolve scales from
current parameters when normalization depends on `z0` or another parameter.

For real/imaginary unrolling, duplicate metadata and use complex magnitudes for
paired entries so convergence is invariant under phase rotation. Preserve the
real-only path. In HB, tile the layout over collocation samples and check physical
time-domain residuals with the existing FFT normalization. Harmonic truncation
error requires a separate refinement study. AC remains a linear analysis; apply
the policy to its operating point and use residual diagnostics for linear solves.

### 5. Validation and rollout

Add focused regression cases with independently known answers or tightened
references:

- Voltage-source/resistor circuits spanning pA to A; LC/RLC circuits with both
  voltage and current unknowns; cancellation-heavy KCL equations.
- A stagnating/damped nonlinear solve, nonfinite solve, and continuation failure.
- Unit conversion (V/mV, A/uA) and explicit equation rescaling with transformed
  tolerances/reference scales; physical answers and decisions must agree.
- Normalized internal states, parameter-dependent optical normalization, complex
  phase rotation, and a coupled electrical/thermal or mechanical example.
- Every transient integration/backend family, rejected steps, and delayed models.
- DC/transient/HB gradients against finite differences at well-converged points;
  comparisons across tolerances avoid differentiating discrete accept/reject choices.
- OSDI full/reduced systems when available; missing metadata fails explicitly.

Track solution error, physical residuals, Newton iterations, step counts, runtime,
and gradient error. Compare selected electrical cases against a pinned ngspice or
Xyce version with matched models, methods, and tolerances. Do not promise identical
iterations or simulator equivalence. Exercise x64 and explicitly assess attainable
accuracy in lower precision; no silent tolerance relaxation.

Document the policy, migration, custom discipline declarations, and diagnostics.
Promote discipline-aware defaults only after native-model coverage and accuracy /
runtime benchmarks are satisfactory. The first useful release is phases 1–2:
electrical DC with separate voltage/current update and residual checks.

## References

- [Accellera Verilog-AMS LRM 2023](https://accellera.org/images/downloads/standards/v-ams/VAMS-LRM-2023.pdf):
  nature attributes and potential/flow disciplines.
- [Optimistix fixed-point API](https://docs.kidger.site/optimistix/api/fixed_point/):
  iterate-based termination; installed source was also inspected for this plan.
- [Xyce 7.7 Reference Guide](https://xyce.sandia.gov/files/xyce/Xyce_Reference_Guide_7.7.pdf):
  separate nonlinear update and residual convergence controls. Its algorithm is
  reference material, not the contract proposed here.


## Prototype review and validation

The requested review-and-simplify agent reviewed the prototype and identified
phase-dependent complex adaptive scaling and premature failure of HB multistart
searches. Both were corrected and independently rechecked by the reviewer with
four focused regressions. Metadata collection, connection validation, and policy
resolution were separated to simplify the compiler-to-solver boundary.

The feature suite passes 37 tests, including all 12 transient method/backend
families, native complex optics, custom mechanical and electrothermal models,
JIT/vmap, and DC/transient parameter gradients. Both demonstration notebooks
execute with analytic assertions and rendered plots. Targeted existing circuit,
transient, HB, compiler, component, delay, and sensitivity tests were also run.
Two sparse optical transient failures reproduce on the original checkout and
are not attributed to this proposal. New solver modules pass Ruff.
