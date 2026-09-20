# Handoff: delay-aware fitting through the unified solver API

Status: **implemented with opt-in conservative automatic inference**.
The sections below preserve the original implementation requirements and rationale.
For the current interface see [fitting API](fitting_api.md); for measured order,
state/history cost and timings see [validation results](delay_aware_fitting_results.md).
The default remains `delay_mode="none"`; known supplied delays are the reliable path.

## Objective and rationale

De-embed physically justified propagation delays, fit a smaller rational core,
and restore the delays using Circulax's existing solver-independent delay
elements. Keep scikit-rf as the default core fitter and AAA as an optional
initialization experiment. Delay separation and AAA are independent choices.

A pure delay has transfer factor `exp(-s*d)` and is not a finite rational
function. A rational-only fit may spend many poles approximating propagation
phase across a wide band. Explicit delays can reduce that burden, but neither
fewer core poles nor a good phase fit guarantees a faster or physically correct
simulation. Count line variables, history storage, and actual solver costs too.

Do not frame this as scikit-rf lacking circuit-quality fitting. Its
`VectorFitting` fits rational responses; an external delay extraction stage can
feed it the de-embedded core. It also supplies passivity testing and enforcement.

## 1. Verified starting point

| Area | Existing capability / implementation anchor |
| --- | --- |
| Shared delay contract | `Signals.at_delay(tau)` in `circulax/components/base_component.py` |
| Compiler discovery | `circulax/compiler.py` infers delayed reads and validates one unique duration per component instance |
| Exact electrical delay | `TransmissionLine` in `circulax/components/electronic.py`: bidirectional matched line, same equations for DC, AC, HB, and transient |
| Delay interpretation | `circulax/solvers/assembly.py`, `ac_sweep.py`, `harmonic_balance.py`, `transient.py` |
| Existing delay fitting | `extract_group_delay`, `deembed_delay`, `embed_delay`, `fit_with_delay` in `circulax/fitting/sparam.py` |
| Current simple API | `fit_model`, `ModelFitOptions`, `ModelCoefficients`, `component_from_coefficients` in `circulax/fitting/api.py`; currently delay-free |
| Old fitted-delay wrapper | `rational_delay_component` in `circulax/components/rational.py` is a frequency-domain oracle, not the shared delay implementation |
| Composition | `compile_circuit` in `circulax/circuit.py` embeds compiled `Circuit` objects via stored subcircuit netlists |
| Existing exact-cascade test | `TestRationalDelayComponent.test_exact_lines_plus_reduced_model_match_fdomain_embedding` in `tests/test_rational.py` (locate by method name if class naming changes) |
| Solver contract tests | `tests/test_delay_contract.py`, `tests/test_delay.py`, `tests/test_subcircuit.py` |

The statement “delayed fitted models cannot run in transient” was too broad.
The old frequency-domain wrapper cannot; a rational core plus the unified exact
line elements can. The missing work is automatic assembly and persistence in
the simple fitting API, not a new delay solver.

Before changing anything, inspect `git status` and the current files. This
workspace also contains uncommitted conventional-VF/API and tutorial changes;
do not replace them with older HEAD versions. Preserve the unrelated
`examples/electrical/time_delay.ipynb` edits and `build/`. SignalIntegrity is a
mathematical reference only: do not copy its source.

## 2. Define one physical convention

Use **one-way duration in seconds per port**, `d[i] >= 0`, in new artifacts:

$$
P(s)=\operatorname{diag}(e^{-s d_i}),\qquad
S_{\rm full}(s)=P(s)S_{\rm core}(s)P(s).
$$

Therefore

$$
(S_{\rm core})_{ij}(j\omega)
=e^{j\omega(d_i+d_j)}(S_{\rm full})_{ij}(j\omega).
$$

A through response acquires delay `d[i]+d[j]`; reflection at port `i` acquires
`2*d[i]`. The line connected to that port uses `TransmissionLine(tau=d[i])`.

**Factor-of-two trap:** existing `sparam.py` stores `tau_per_port` and uses
`exp(-s*tau_per_port[i]/2)`. Convert explicitly with `d=tau_per_port/2`.
Do not silently reinterpret old parameters. Annotate units and the convention
in the serialized schema, docstrings, plots, and tests.

For real frequencies, `P` is unitary. Multiplying on either side preserves S
singular values, so positive ideal delays preserve scattering passivity of a
passive core. This does **not** prove that advancing measured data by an inferred
delay leaves a causal core. Delay inference must have separate safeguards.

## 3. Intended API and compatibility

Keep the two-stage workflow: fit coefficients, then create a usable circuit
model from those coefficients. Proposed settings, **not currently available**:

```python
options = ModelFitOptions(
    method="vector_fitting",
    delay_mode="auto",          # "none", "auto", or "supplied"
    port_delays=None,            # one-way seconds; required for "supplied"
    normalized_rmse=1e-4,
    max_absolute_error=1e-3,
)
coefficients = fit_model(S, freqs, options=options, z0=50.0)
coefficients.save("network.npz")
model = component_from_coefficients("network.npz", name="MeasuredNetwork")
# Both a leaf class and a Circuit subcircuit are accepted by compile_circuit:
# models = {"measured_network": model}
```

Recommended return contract: retain the existing component class for zero delay;
return a composed `Circuit` for nonzero delay. Document the union explicitly,
and show `compile_circuit(..., models_map={...})` as the common integration route.
Do not suggest that every result can be instantiated with `Model()` or passed
directly to low-level `compile_netlist`, which does not perform the same Circuit
embedding. If this contract proves awkward, resolve it with the owner before
introducing another public factory.

Keep `delay_mode="none"` behavior reproducible. Implement explicit delays first;
make `auto` the preferred/default workflow only after candidate selection and
regression tests pass. Supplied pole sets must not silently trigger automatic
delay inference: require explicit delays or leave them disabled on that branch.

## 4. Milestones, in implementation order

### A. Coefficient schema and pure evaluation

1. Add a validated per-port delay array to `ModelCoefficients`, defaulting to
   zeros. Validate shape, finiteness, and nonnegative durations, including on load.
2. Store the rational **core** coefficients, not an approximation to re-embedded
   delay. Keep the common real positive reference impedance explicit.
3. Make `evaluate(freqs)` return the full terminal response; provide a clearly
   named core evaluation internally for fitting/enforcement.
4. Version the NPZ/JSON schema. Read version-1 files as zero-delay models; write
   the new delay convention explicitly. Continue to load with `allow_pickle=False`.
5. Test unequal port delays, reflection versus transmission delay, zero-delay
   parity, invalid arrays, old-file compatibility, and save/load round trips.

Deliverable: delayed coefficients evaluate correctly without any solver changes.

### B. Solver-independent model assembly

1. Reuse the existing rational core and one `TransmissionLine` per delayed port.
   Zero-delay ports can connect directly. Different ports may have different
   delays; do not force multiple distinct delayed snapshots into one component.
2. Connect each external port to a line's first port and the line's second port
   to the corresponding core port. Use the coefficient artifact's `z0`.
3. Preserve core stability/conversion checks. Report a failed core realization
   rather than changing poles silently or falling back to a frequency-only model.
4. Test integration as a subcircuit, two fitted models in one parent, repeated
   instances, and nested composition. Avoid leaf-model name collisions and
   verify that retained source netlists are self-contained when re-embedded.
5. Retain `rational_delay_component` as a frequency-domain oracle. Do not use it
   as the production all-solver implementation or modify general solver code
   unless a contract test identifies an actual missing capability.

**Required edge case:** an ideal through core can have singular `I+D`, so it
cannot be represented by an ordinary finite admittance matrix. De-embedding an
ideal delayed through can produce exactly this case. Support it with the
existing line primitive or a correct algebraic wave/connection representation;
do not add artificial loss merely to make S-to-Y inversion succeed. Test both
lossless and attenuating static cores, including zero rational poles. Arbitrary
singular multiport cores may require an explicit supported-scope error, but an
ideal two-port through must work for this feature's acceptance.

Deliverable: known delayed coefficients work in DC, AC, HB, and transient.

### C. Supplied-delay fitting

1. De-embed known one-way delays using vectorized broadcasting over frequency
   and port axes. Fit the core with the existing conventional-VF default; allow
   the same operation with the optional AAA backend.
2. Apply optional physical enforcement to the core and keep its status explicit.
3. Re-embed and check the original complex-S accuracy targets against the
   **original training data**, after all corrections. Never use holdout data to
   estimate delay, choose poles, or select a candidate.
4. Store original-domain errors, core order, per-port delays, supplied/estimated
   provenance, and enforcement/validation diagnostics separately.
5. Handle exact or numerically constant cores without inventing dynamic poles.
   Any numerical constant-detection tolerance must also satisfy the caller's
   final accuracy target; do not let a hidden threshold override it.

Deliverable: known-delay synthetic data produce an accurate smaller core and
can be loaded and simulated with the delays intact.

### D. Conservative automatic candidate selection

Start with reciprocal two-ports where a shared propagation delay is plausible.
Known per-port delays should remain usable for arbitrary port counts and active
or nonreciprocal devices; do not infer their distribution from one transmission.

1. Fit the undelayed baseline at the same error target and retain its outcome.
2. Estimate candidate delay from training phase, checking transmission nulls,
   phase-linearity residuals, positive duration, and sampling ambiguity. Inspect
   both transmission directions. Expose safeguards as documented options rather
   than hiding unvalidated constants in the implementation.
3. Transmission identifies a **sum** of port delays, not their unique allocation.
   Record any equal-split assumption. Reflections may help constrain allocation;
   if they contradict it, decline the automatic estimate or require supplied
   delays. Do not assume an arbitrary multiport phase pattern factors as `P S P`.
4. Unwrapped phase cannot establish that a sweep is unaliased. In particular, a
   check on differences *after* unwrapping does not detect missing phase turns.
   Include an explicit delay bound or sampling warning; synthetic alias tests
   must expose this limitation rather than claim a causality certificate.
5. Treat phase slope as a proposal, not propagation-delay proof: resonances and
   active dynamics can also produce group delay. Reject negative estimates;
   do not silently clamp them and describe the result as a physical extraction.
6. Fit a bounded set of candidates (including smaller delay fractions when
   appropriate). Check the full re-embedded response, core physical properties,
   and constructibility of the final realization.
7. Prefer a passing delayed model only when it improves the chosen complexity
   metric. If the baseline fails, a passing delayed candidate may be selected,
   but do not claim a measured reduction relative to a missing baseline.
8. Record why inference was declined, fitting/enforcement failed, or the
   undelayed baseline was preferred. Fail explicitly when neither qualifies.

Default policy should target fewer rational poles **and report** total circuit
unknowns and history cost. A reduction in core poles alone is not sufficient
evidence of a faster simulator. Stability and sampled passivity checks also do
not prove that the inferred de-embedded core is physically causal.

Deliverable: safe fallbacks, transparent selection, no guaranteed reduction claim.

### E. Analytic validation and performance measurements

Use deterministic synthetic cases in addition to measured examples:

| Case | Required observation |
| --- | --- |
| Lossless delayed through | Exact propagation, no invented rational poles, no singular-Y workaround by artificial damping |
| Matched attenuating delayed through | Static core, preserved attenuation and delay |
| Stable passive rational core with known delay | Recovered core response and a smaller rational order than fitting the full delayed response at the same accuracy |
| Unequal port delays with reflections | Correct `2*d[i]` reflection and `d[i]+d[j]` transmission phases |
| Resonant response without justified propagation | Auto mode declines extraction or retains a validated baseline; no unjustified causality claim |
| Active/nonreciprocal response | Known delays work; automatic assumptions are not imposed and intentional gain is not passivated |
| Nulls, noisy phase, sparse/aliased samples | Diagnostics and conservative behavior, not silent over-de-embedding |
| Over-de-embedded input / invalid core | Rejection or explicit failed-candidate result |

For solver checks, compare against independent analytic delayed responses, not
only two paths sharing the same S/Y conversion:

- DC: delays have identity action; correct loaded operating point.
- AC: amplitude and phase agree across frequencies, including unequal delays.
- HB: source-driven steady-state harmonics agree with expected delay factors.
- Transient: pulse onset, propagation time, attenuation, and reflections agree;
  no pre-arrival signal beyond stated numerical/history tolerance.
- Repeat transient checks with sub-step delays, representative adaptive/fixed
  stepping, and supported complex-state paths. Define the prehistory consistently
  with the DC operating point; do not confuse DC history with noncausal leakage.
- Where gradients are advertised, compare derivatives with respect to line
  delay against finite differences away from interpolation/topology boundaries.

Report fit time, enforcement time, conversion/compilation time, warm evaluation
and transient costs separately. Include rational pole count, core state count,
line algebraic unknowns, history memory, final error, and physical checks. The
existing realization often repeats poles per port; a pole count is not a state
count. Do not compare a cold JAX compile with a warmed NumPy fit or run competing
timed workloads concurrently.

### F. Tutorial and API documentation

Extend `examples/fitting/vector_fitting.ipynb` with a linked delay-aware section
using a known delayed passive model before attempting extraction on measurement
data. Show the same conventional core fitter with delay separation disabled and
enabled. Explain why the ring-slot experiment alone did not assess this benefit.

Correct the distinction between unified solver delay support and the old
frequency-domain wrapper. Document per-port units, current versus proposed API,
selection/fallback behavior, subcircuit use, and limitations. Keep scikit-rf as
the fitting-theory reference; describe Circulax's value as explicit propagation,
component integration, and application-specific validation.

## 5. Suggested verification commands

From the repository root, after implementing the relevant milestones:

```bash
pixi run pytest -q tests/fitting/test_api.py tests/test_rational.py
pixi run pytest -q tests/test_delay_contract.py tests/test_delay.py tests/test_subcircuit.py
pixi run ruff check circulax/fitting/api.py tests/fitting/test_api.py
git diff --check
```

Add focused tests in `tests/fitting/test_delay_model_api.py` and a reproducible
benchmark such as `benchmarks/fitting/bench_delay_separation.py`. Execute the
combined notebook in a fresh kernel with the repository on `PYTHONPATH`; Jupyter
may require sandbox approval for local sockets. Respect the repository's output
stripping policy after checking execution results.

## 6. Definition of done

- A saved artifact preserves explicit physical delays and evaluates the same
  full S response after reload; old zero-delay files still load correctly.
- Component generation uses the unified delay elements and works through all
  four solver families without choosing a frequency-only fallback.
- An analytically specified delayed fixture demonstrates lower rational order
  at the same error target; total simulation cost is reported honestly.
- Automatic inference is optional, bounded, diagnosable, and does not claim
  causality from phase fitting alone. Supplied delays remain the reliable path.
- A model failing final error or realization checks cannot silently be admitted.
- Zero delay, ideal through, asymmetric delays, active models, and subcircuit
  composition are covered by regression tests.
- The tutorial distinguishes verified behavior, numerical limitations, and
  unmeasured extrapolation. Do not promise a speedup before measuring it.

## Prompt for the next implementer

> Read this plan and inspect the current worktree before editing. Implement
> milestones A–C first, with tests proving the one-way-delay convention and
> all-solver composition, including the ideal-through case. Then implement D
> only with transparent safeguards and baseline comparison. Reuse the unified
> delay contract; do not write a new solver or route transient models through
> the legacy frequency-domain wrapper. Keep scikit-rf as the default rational
> fitter, preserve optional AAA and existing user changes, and report measured
> order/cost/accuracy results rather than assuming a benefit. If the public
> return-type contract needs to change beyond the proposal, ask before expanding
> the API. Complete E–F before describing automatic delay fitting as supported.
