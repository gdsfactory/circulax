# Delayed nonlinear optical power waves

This focused milestone of [issue #112](https://github.com/gdsfactory/circulax/issues/112)
extracts the directional TPA waveguide from
`examples/photonic/photonics_transient.ipynb` into
`circulax.components.NonlinearOpticalWaveguide`. The factory
`nonlinear_optical_waveguide(r0=...)` supplies another static finite positive
real reference. The issue remains open; the concrete remaining inventory is in
[notebook observations](notebook_power_wave_observations.md#remaining-path-inventory).

## Original law and observation-plane audit

The notebook-local component already retained two incident unknowns, `a1,a2`.
Its equations were `v=a+b`, `i=(a-b)/z0`. At its authored `z0=1` these were
power-normalized, but changing `z0` alone changed the coordinate/power relation:
its incident unknowns had potential units, not reference-invariant sqrt(W).
The new model uses `potential=sqrt(r0)*(a+b)`,
`flow=(a-b)/sqrt(r0)` consistently. This is a coordinate correction for nonunit
references, not a reinterpretation of the original `z0` setting as launched
power. The reference is a factory argument, not a traced instance parameter.

The exact original directional law is preserved:

```text
L_cm = length_um/10000
alpha = loss_dB_cm*ln(10)/10                              [1/cm]
L_eff = (1-exp(-alpha*L_cm))/alpha, if alpha > 1e-12      [cm]
      = L_cm, otherwise
phi = 2*pi*neff*length_um*1000/wavelength_nm
T = 10**(-loss_dB_cm*L_cm/20)*exp(-j*phi)
tau = length_um*n_group/2.99792458e14                     [s]
G(u) = T*u/sqrt(1+tpa_coeff*|u|²*L_eff)
b1(t) = G(a2(t-tau)); b2(t) = G(a1(t-tau))
```

`tpa_coeff` is in 1/(W cm), waves in sqrt(W), powers in W, wavelengths in nm.
This is the integrated scalar law `dP/dz=-alpha*P-beta*P²` applied independently
to each direction. It contains no cross-direction TPA, standing-wave absorption,
Kerr phase, carrier accumulation, electrical photodetection or nonlinear flight
time. Returning light is propagated and compressed by the same directional law,
without being folded into the other incident intensity. No such extra physics
is introduced by the migration.

Intensity is evaluated at the **input plane at the retarded time**, followed by
linear attenuation and carrier phase to the opposite output plane. Applying TPA
to the attenuated output intensity or to `|potential|²` would change this law.
Linear attenuation is included once; the effective length accounts for its
interaction with nonlinear loss. The negative carrier-phase sign is unchanged.
There is no linearized wavelength-index dispersion correction in this model.

`expm1` improves evaluation of the existing effective-length expression without
changing its small-alpha branch. A safe inactive denominator prevents NaN
Jacobians at exactly zero loss. That preserved branch has its original derivative
behavior at alpha <= 1e-12; it is not an analytic continuation of the alpha
quotient's loss derivative through zero. Length, loss, group index and TPA
coefficient now require finite nonnegative values; wavelength must be finite and
positive. These checks define the passive domain rather than clipping parameters
or adding loss. Zero length, zero loss, zero TPA and zero group delay are allowed.

The original source fixed a potential difference
`sqrt(source_power)*sigmoid((t-delay)/rise)`. The migrated one-terminal source
emits exactly that envelope in sqrt(W) and absorbs arbitrary returning light.
In this particular notebook the matched output monitors and reflectionless
forward splitter excitation return no light to the source; at reference 1 the
new and old forward envelopes coincide. This does not assert equivalence of
potential and launched-wave sources in general reflecting networks.

The original 1-ohm resistors were ideal matched optical monitors, with no
responsivity, electrical terminals or carrier state. They become explicit
`OpticalWaveTermination` instances. Input power is incident into `SPLIT,p1`;
output powers are incident into `LOAD_SHORT,p` and `LOAD_LONG,p`, after the
respective delays, linear losses and TPA. Plot fields are these total incident
envelopes; plotted/settled powers use `PortWaves.incident_power`. The splitter
ratio, two physical lengths, pulse, delay, loss, compression law, comparison
thresholds and solver tolerances are preserved. The notebook explicitly uses
zero leakage and requires no optical ground. Output/metadata/cell IDs are retained.

## Retained constraints, flow and history

Each port retains `wave_p1` or `wave_p2` and the shared constraint
`F_wave=a+b-potential/sqrt(r0)`, with zero Q. The complete instantaneous terminal
row is `(2*a-potential/sqrt(r0))/sqrt(r0)`. It equals physical signed flow plus
the retained constraint divided by sqrt(r0); hence the same physical roots
hold for arbitrary returning light. No incident state is eliminated and no
external matching is assumed. S-to-Y conversion, `(I+S)` inversion, loss/reference
perturbations and optical-ground penalties are absent.

The compiler's established wave-reference and algebraic-flow metadata allow
`Circuit.port_waves` to reconstruct solved DC, transient and HB **time samples**
from local signed terminal F, without access to private state names or a history
buffer. Incident, outgoing and net powers are `|a|²`, `|b|²`, `|a|²-|b|²`, with
net positive into the selected component. At DC, summing the two net powers gives
nonnegative total linear-plus-TPA dissipation. During a transient, same-time net
input also accounts for energy in flight; it need not be nonnegative at every
sample. Retarded input/output comparisons establish directional attenuation.
The model provides no independent stored-energy state or absorption-rate probe.

Both directions share the existing single delay query per instance; different
instances retain their own delays. DC uses constant history. Transient holds the
complete `y0` before t0 and uses accepted-history/current-trial interpolation.
Zero delay uses the current trial. HB uses periodic full-complex Fourier shifts,
including independently populated negative and positive envelope harmonics.
The existing history capacity, interpolation, delay-derivative and refinement
limits in [bidirectional delays](bidirectional_optical_delays.md) apply unchanged.
No FIR/generalized multiple delays or solver/default changes are included.

Recompile circuits and rebuild raw initial guesses/histories when moving the
local notebook class to the library: incident state names, constraint signs,
terminal residual rows, source topology and reference configuration change.
Off-solution residual values are not a compatibility contract.

## Analysis and derivative contract

`holomorphic=False` is explicit even when an instance sets `tpa_coeff=0`.
DC and transient use both optical coordinates. `Circuit.small_signal` supports
the complete real-block delayed Jacobian and independent complex RF phasors via
`QuadratureDrive`. `Circuit.port_power_response` differentiates total traveling
power about that operating point and preserves RF phase separately from optical
phase. The source's forcing-aware terminal-flow limitation remains; observe the
attached device. Legacy `ac_response` refuses this component for all values of
its `holomorphic` argument. AC perturbations and HB coefficients are rejected as
total fields by `port_waves`; generic unsupported dynamic terminal-flow guards
remain unchanged.

For `k=beta*L_eff`, `d=1+k*|u|²` and a real coordinate perturbation du,

```text
DG(u)[du] = T/sqrt(d) * (du - k*u*Re(conj(u)*du)/d)
DP_out[du] = |T|²/d² * 2*Re(conj(u)*du)
```

The x and y columns of these real Jacobians independently carry complex RF
phasors. Delayed derivatives multiply by `exp(-2j*pi*f*tau)`. This includes the
conjugate/intensity derivative; a complex-linear derivative would discard it.
HB evaluates nonlinear power on reconstructed total samples, preserving carrier
interference and generated envelope harmonics. Refine the harmonic grid for
nonlinear reflected round trips rather than squaring individual coefficients.

## Fresh validation

CPU / Python 3.12 / JAX 0.7.2 float64 validation on 2026-11-13:

- **39 new acceptance cases passed**: a completed 37-case focused run plus the
  two subsequently added dynamic-objective gradient cases. Earlier overlapping
  development reruns are excluded. Tests make S-to-Y raise in the native and
  adapter modules, use zero leakage and no optical ground, and exercise public
  DC/checked-DC, transient, small-signal, wave/power accessors and HB APIs.
- Coverage includes both propagation directions and quadratures, arbitrary
  independent reverse drive, matched/partial/full reflecting controls, carrier
  phases 0/pi/2/pi, phase-invariant intensity, references 0.01/1/50, zero loss,
  zero length, zero TPA and zero group delay, passive balance and potential
  interference. State/history Jacobians and physical length/loss/TPA/input-field
  derivatives agree with independent references or finite differences. JIT/vmap
  and dense/sparse/KLU paths pass. Independent complex RF drive coordinates and
  nonlinear reflected round-trip power derivatives retain the delay phase.
- Bidirectional/reflected transient cases refine fixed timesteps from 40 to 10
  to 2.5 ps with strictly decreasing envelope error; the finest errors are below
  2e-5 sqrt(W). Nonzero constant prehistory is checked. Reflected nonlinear HB
  refines from 2 to 4 to 8 harmonics with strictly decreasing round-trip error;
  the finest is below 1e-9 sqrt(W). Powers use total samples. DC, transient and
  HB TPA objective gradients agree with central finite differences; dynamic
  gradients also agree with the independent integrated-loss derivative.
- All **seven Python cells** in the changed notebook executed, with Agg plots
  and S-to-Y patched to raise. Maximum short/long envelope errors were
  **1.08e-7 / 9.66e-8 sqrt(W)**. Measured differential delay was **133.4 ps**;
  settled powers were **0.1918 / 0.0619 mW**, passing the authored independent
  delay/power assertions. No outputs or metadata were saved.
- Scoped repository hooks pass whitespace, EOF, conflict and notebook stripping
  checks; YAML/TOML checks skip as nonapplicable. All three changed Python files
  pass formatting. New tests and component exports pass Ruff. `photonic.py`
  retains the same three inherited diagnostics; notebook lint decreases from
  four to two inherited diagnostics after removing its local class. Cell IDs,
  outputs and metadata match HEAD. No new lint findings are introduced.

The fresh disjoint regression results are recorded below. Earlier guides retain
their dated milestone results; those counts are not substituted for fresh runs.

| Disjoint selection | Fresh result |
| --- | --- |
| New nonlinear waveguide acceptance cases | **39 passed, 0 skipped, 0 failed** |
| Notebook observations, detector and both phase-shifter suites | **204 passed**, 1 inherited warning |
| Native passives, SAX, boundaries, delays and history suites | **334 passed**, 1 inherited warning |
| Delay contract, sources, compiler/circuit, small-signal, AC, HB, DC, transient, external boundaries, wavelength and sensitivity suites | **281 passed**, 24 inherited warnings |
| Total unique tests | **858 passed, 0 skipped, 0 failed** |

The inherited warnings are KLU `free_numeric` and `Circuit.ac()` deprecations
and the existing SAX conjugation/holomorphic warning. Development harness
failures (comparing a terminal potential with just its outgoing perturbation,
and JAX list indexing) were corrected before the counted runs; no unresolved
failures remain. The two nonapplicable hook skips are separate from pytest,
which skipped no tests. These are scoped CPU checks, not the full repository
suite, GPU execution, a documentation-site build, or completion of #112.

Runner scripts, logs, collection and lint-baseline evidence are retained under
`/home/cdaunt/.codex/validation/issue-112-tpa-20261113/`. The changed notebook was
executed without writing output/metadata. This milestone is committed on
`codex/nonlinear-optical-power-waves` for a draft PR targeting
`codex/issue-112-power-waves`. Integration into the feature branch and closure
of #112 remain separate. No remote issue-body update is included. Rational
realization, FIR, generalized multiple-delay and PR #98 work are outside this
milestone.
