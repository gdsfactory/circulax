# Bidirectional optical delays

Current follow-up: [notebook observations and small-signal traveling-power derivatives](notebook_power_wave_observations.md). The scope and validation below describe this earlier milestone.

This is the delay milestone of [issue #112](https://github.com/gdsfactory/circulax/issues/112),
built on the [native passive](native_passive_power_waves.md) and
[optical boundary](optical_wave_boundaries.md) milestones. The issue remains open.

## Physical equations and phase conventions

`OpticalDelayLine` retains two complex incident-wave states and implements
reciprocal carrier-envelope propagation, including arbitrary returning light:

```text
tau = length_um*n_group/2.99792458e14                    # seconds
T = 10**(-loss_dB_cm*length_um/10000/20)*exp(-1j*phi)
phi = 2*pi*neff*length_um*1000/wavelength_nm
b_p1(t) = T*a_p2(t-tau)
b_p2(t) = T*a_p1(t-tau)
potential_p = sqrt(r0)*(a_p+b_p)
flow_p = (a_p-b_p)/sqrt(r0)                            # into the component
F_wave_p = a_p+b_p-potential_p/sqrt(r0); Q = 0
P_incident = |a_p|²; P_outgoing = |b_p|²; P_net = |a_p|²-|b_p|²
```

Lengths are um, wavelengths nm, losses dB/cm, envelopes sqrt(W), and powers W.
The existing negative carrier-phase sign and nondispersive carrier law are
preserved. Unlike `OpticalWaveguide`, this model does not add the waveguide's
linearized wavelength-dependent index correction. Length, loss and group index
must be finite and nonnegative; wavelength must be finite and positive.
Group index zero is allowed as an algebraic control, including nonzero carrier
phase. Exact zero loss and zero delay need no numerical perturbation.

`delay_line_fdomain` retains its distinct, **modulation-only** convention:

```text
transmission(f) = 10**(-loss_dB_cm*length_um/10000/20)*exp(-2j*pi*f*tau)
b = [[0, transmission], [transmission, 0]] @ a
```

It has no `neff` or `wavelength_nm` parameter and does not include optical
carrier phase. It now uses the shared frequency-dependent SAX power-wave
operator, with terminal flows and incident constraints. Do not add carrier
phase twice when combining the models. At modulation frequency zero its
transmission is the attenuation; `OpticalDelayLine` also includes its carrier
phase. For matching carrier conventions, their DC/AC/HB results agree.

Neither realization uses S-to-Y conversion, inverts `I+S`, divides by `1-t²`,
injects artificial loss, perturbs references, or assumes external matching.
General conversion utilities remain available for unrelated conversion uses.

## Terminal flows and observables

Both models reuse the shared retained-wave construction. The time-domain
model uses an equivalent terminal row:

```text
F_terminal = (2*a-potential/sqrt(r0))/sqrt(r0); Q_terminal = 0
```

This is the physical `(a-b)/sqrt(r0)` row **plus** the retained incident
constraint divided by `sqrt(r0)`. It has exactly the same roots, for all
incident amplitudes and connected loads. It does not eliminate an incident
unknown. Delay appears in the constraints; terminal flow depends only on
current state. Consequently `Circuit.port_waves` can reconstruct the complete
signed terminal flows at DC, transient saved samples, and HB time samples
without the accepted history buffer. It evaluates terminal F, rather than
reading auxiliary state names or treating delayed potential as a traveling wave.
As with the other observables, inputs must be solved fields with the same
parameter updates. Residuals away from a solution depend on the chosen row
representation. Linear saved-sample interpolation preserves the algebraic flow
identity, though a delayed constraint at an interpolated sample can have the
usual time-discretization error.

The accessor continues to reject generic delayed models without an explicit
history-independent terminal-flow contract. It also rejects dynamic charges,
legacy potential sources, AC perturbation labels and HB coefficient labels.
The frequency-domain realization supports `port_waves` at DC only. Its
transient and HB-time terminal flows cannot be reconstructed from a single
sample without spectral/history context and remain explicitly rejected.
Boundary flows can still be observed at HB time samples in such a network.

## History, initialization and differentiation

There is one fixed scalar delay per instance; both directions reuse one
`Signals.at_delay(tau)` snapshot. Instances in one component group may have
different lengths/delays. No FIR propagation or generalized multiple-delay
support is introduced.

- DC evaluates a constant history equal to the current state. It is a steady
  scattering solve, including carrier phase and arbitrary reflections.
- Transient starts from `y0`, or the time-zero DC solution if omitted. Before
  `t0`, the entire state is held at `y0`. The public transient API has no
  arbitrary callable or sampled prehistory input. A supplied `y0` must contain
  consistent incident states as well as potentials; it is not just an input
  field. For a different earlier trajectory, start the simulation earlier.
- Accepted steps are stored in the existing bounded ring buffer. Queries
  between accepted times use linear interpolation. Queries between the last
  accepted point and the current evaluation time interpolate against the
  current Newton trial. This includes the nonzero current-state Jacobian when
  a step exceeds tau. Zero delay reads the current trial and reduces to the
  corresponding algebraic carrier-wave model, in both directions.
- Output `saveat` samples interpolate the solved state. They are not the
  accepted history mesh and must not be substituted for it in a later solve.
- Fixed-step history capacity is inferred from delay/dt; supported adaptive
  `dtmin` controllers use delay/dtmin. Other adaptive/traced runs default to
  1024 samples. Insufficient capacity raises; pass `history_capacity` as needed.
  Adaptive stepping is not capped by the delay. Refinement is still needed to
  resolve waveform features and repeated reflected pulses.
- JAX differentiates accepted history values, interpolation weights, delay
  parameters and current trials. Derivatives are piecewise smooth between
  interpolation knots. At an exact accepted-time hit, a change of bracket,
  zero-delay boundary, adaptive step decision or history-capacity transition,
  do not interpret a one-sided derivative as a globally smooth sensitivity.
  Finite-difference checks must stay within one smooth region.
- HB uses the existing full complex FFT periodic shift. It represents periodic
  steady state, including independently populated positive and negative
  envelope harmonics, rather than the transient startup history. AC separates
  direct and delayed Jacobians and multiplies delayed derivatives by
  `exp(-2j*pi*f*tau)`; full quadrature and holomorphic paths are covered.

## Public API and raw-state compatibility

Existing `OpticalDelayLine(...)` and `delay_line_fdomain(...)` instance settings
and defaults are preserved. Both default to reference `r0=1`. New factories
`optical_delay_line(r0=...)` and `optical_delay_line_fdomain(r0=...)` build the
same classes at another finite positive real static reference. They are exported
from `circulax.components` and `circulax.components.photonic`.

Use a consistent reference, carrier frame, time convention and mode throughout
a directly connected wave network. Reference changes are coordinate changes,
not S-parameter renormalization. Connect two compatible optical terminals per
node; use explicit splitters for branching. No optical ground or numerical
leakage is required. These connection rules are not compiler-enforced yet.

`OpticalDelayLine` replaces its legacy unilateral `i_p2` constraint with
`wave_p1` and `wave_p2`. It now loads its input physically and propagates
reflections back to the source. `delay_line_fdomain` gains the same two wave
states; its local operator is 4x4 rather than the former 2x2 admittance.
Recompile circuits and rebuild saved raw guesses, histories and direct physics
calls. Neither raw state arrays nor old `i_p2` names are compatible. Existing
`OpticalSource`/`OpticalSourcePulse` retain prescribed-potential semantics;
use explicit launched-wave boundaries for prescribed optical power.

```python
import jax.numpy as jnp
from circulax import compile_circuit
from circulax.components import optical_delay_line, optical_wave_source, optical_wave_termination

r0 = 50.0
circuit = compile_circuit({
    "instances": {
        "S": {"component": "source", "settings": {"amplitude": 0.6+0.8j}},
        "D": {"component": "delay", "settings": {"loss_dB_cm": 0.0}},
        "L": {"component": "load"},
    },
    "connections": {"S,p": "D,p1", "D,p2": "L,p"},
}, {
    "source": optical_wave_source(r0=r0),
    "delay": optical_delay_line(r0=r0),
    "load": optical_wave_termination(r0=r0),
}, backend="dense", g_leak=0.0)
y = circuit.dc()
assert jnp.allclose(circuit.port_waves(y, "D,p2").outgoing_power, 1.0)
```

## Network validation and scope

The focused tests use independent scattering matrices, delayed pulse envelopes,
reflective round-trip formulas and an all-pass ring transfer
`(t-z)/(1-t*z)`, where t is the coupler bar amplitude and z the delayed
round-trip transmission. A local exactly lossless delay is valid even when
`I+S` is singular. A decoupled, exactly resonant, lossless loop with `t*z=1`
has a physical circulating nullspace. The tests distinguish this whole-network
singularity from local device validity, without optical-ground penalties or
leakage. No solver convergence defaults change.

The subsequent [phase-shifter milestone](mixed_electro_optic_power_waves.md)
reuses this terminal-flow construction while preserving midpoint modulation and
electrical F/Q. Detectors, other mixed electro-optic laws, rational-model conversion paths, FIR,
notebook migration and PR #98 are outside this milestone. PR #98 remains
separate and opt-in. The validation results below describe the checked CPU
paths; they do not establish GPU or arbitrary broadband/dispersive accuracy.

Validation on Python 3.12 / CPU JAX with float64 (2026-11-10):

| Check | Result |
| --- | --- |
| `tests/test_optical_delay_power_waves.py` | 55 passed (53 core tests, then 2 sparse/KLU backend tests) |
| `tests/test_native_power_waves.py` | 105 passed |
| `test_power_wave_sax.py`, `test_sax_component.py`, `test_optical_wave_boundaries.py` | 133 passed |
| `test_delay.py`, `test_delay_history.py` | 41 passed |
| DC/checked-DC, transient, waveform sources, delay contract, AC sweep/response, small signal, HB, compiler, circuit, external boundary, wavelength, sensitivity | 272 passed, 9 skipped |

These selections contain 606 passing tests and 9 optional sensitivity skips
because bosdi/OSDI is unavailable. The documented Python example also executes.
The new tests make `s_to_y` raise in both the native and adapter modules.
They cover lossless/near-lossless and attenuated devices, zero length/group
delay, carrier phases 0/pi/2/pi, coherent bidirectional fields, three references,
reflective pulses, interpolation and timestep refinement, AC modulation/carrier
phase separation, positive/negative HB harmonics, a delayed ring and its
physical nullspace, JIT/vmap, parameter updates, and state/history/parameter
Jacobians. Existing fixed/adaptive transient parameter-gradient tests also pass.

Scoped repository pre-commit hooks and `git diff --check` pass; YAML, TOML and
notebook hooks skip as nonapplicable. The shared wave helper and new tests pass
Ruff; all five Python files changed by this milestone pass formatting checks.
Unrelated starting-version Ruff findings remain in `circuit.py` (10),
`photonic.py` (3) and component exports (1); comparison with the starting
versions confirms no new lint findings. The inherited sparse correction and
native-passive tests are unchanged. Existing SAX conjugation and KLU/analysis
deprecation warnings remain. GPU execution, full notebooks, the complete
repository test suite and a documentation-site build were not run.
