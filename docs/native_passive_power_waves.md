# Native passive power-wave migration

This is the native algebraic passive milestone of [issue #112](https://github.com/gdsfactory/circulax/issues/112),
built on the SAX and [optical boundary/observable](optical_wave_boundaries.md) milestones.
It does **not** complete the issue or migrate the entire photonic library.

## Ports and physical laws

`OpticalWaveguide`, `Grating`, `Splitter`, `DirectionalCoupler`, and
`TunableBeamSplitter` now use the shared SAX adapter and `power_wave_stamp` /
`power_wave_operator`. Every port retains one complex incident-wave unknown,
even for structurally matched models. No surrounding matching is assumed.
For a common finite positive real `r0`:

```text
b = S(parameters) @ a
potential = sqrt(r0)*(a+b)
flow = (a-b)/sqrt(r0)                  # into this component
F_wave = a+b-potential/sqrt(r0); Q = 0
P_incident = |a|²; P_outgoing = |b|²; P_net = |a|²-|b|²
```

There is no S-to-Y realization, inverse of `I+S`, complex reference perturbation,
artificial attenuation, or division by `1-t²` in these five paths. General
conversion utilities remain available for conversion applications.

| Model | Ordered ports | Scattering law and compatibility |
| --- | --- | --- |
| `OpticalWaveguide` | p1, p2 | Reciprocal matched through: `t=10**(-loss_dB_cm*length_um/10000/20)*exp(-1j*phi)`. Index slope `(neff-n_group)/center_wavelength_nm` and `phi=2*pi*n_eff_disp*length_um*1000/wavelength_nm` are unchanged. Zero length/loss and all carrier phases are supported. |
| `Grating` | grating, waveguide | Reciprocal matched through: loss in dB is `peak_loss_dB+((wavelength_nm-center_wavelength_nm)/(bandwidth_1dB/2))**2`. The old 0.9999 cap is removed; peak zero-loss transmission is exactly one. |
| `Splitter` | p1, p2, p3 | Reciprocal passive three-port with the existing transmission law and an omitted matched loss channel, described below. The former blanket lossless/unitary claim was incorrect. |
| `DirectionalCoupler` | p1, p2, p3, p4 | Reciprocal lossless four-port. Bar amplitude `sqrt(1-coupling)` connects p1/p3 and p2/p4; cross `1j*sqrt(coupling)` connects p1/p4 and p2/p3. Endpoint clipping is removed: 0 and 1 are exact. |
| `TunableBeamSplitter` | p1, p2, p3, p4 | Same reciprocal four-port topology, with bar `cos(theta)` and cross `1j*sin(theta)`. Forward matched transmission is preserved. Reverse/reflective response replaces the old unilateral voltage constraints. |

Wavelengths are nm, lengths um, amplitudes sqrt(W), powers W. These are
instantaneous **carrier-envelope** laws. `OpticalWaveguide` retains its carrier
phase and attenuation, but adds no envelope propagation delay. Static algebraic
transient/HB support does not establish broadband propagation accuracy.

Passive domains require nonnegative length/loss, positive waveguide wavelengths,
positive grating bandwidth, and power split/coupling in [0,1]. Invalid lengths,
losses, bandwidths, wavelengths, and ratios are rejected where specified in the
model, including JAX execution, rather than clipped. References are validated
when building the class. Square-root ratios are not differentiable at both
endpoints: response finiteness is required there, but finite ratio derivatives
are not. The angle parameterization remains smooth at bar/cross endpoints.

## Independent splitter audit

Writing `r=sqrt(split_ratio)` and `u=sqrt(1-split_ratio)`, the retained law is

```text
S = [[0, r, 1j*u],
     [r, 0, 0],
     [1j*u, 0, 0]]
```

`S.T=S` and its singular values are `(1,1,0)` for every valid ratio. Thus it is
reciprocal and passive but **not unitary**. A reciprocal, matched, isolated-output
three-port cannot provide this forward splitting law and conserve power for
all incident vectors without an extra channel. Here that channel is omitted
and matched: light entering it never returns. For arbitrary simultaneous
complex illumination,

```text
sum(P_incident)-sum(P_outgoing) = |1j*u*a_p2+r*a_p3|² >= 0
```

Forward input at p1 conserves power. At 50/50 splitting the normalized reverse
input `(a_p2,a_p3)=(1,+1j)/sqrt(2)` is fully lost, while `(1,-1j)/sqrt(2)` combines
losslessly into p1. A unitary reciprocal completion is obtained by adding port
p4 with `S24=S42=1j*u`, `S34=S43=r`, and all remaining new entries zero; terminating
p4 gives the three-port above. This is a physical omitted channel, not added
numerical loss. The audit preserves the original three-port amplitudes and phases;
it corrects the documentation and exposes the true loss under reverse/coherent
illumination. Use `DirectionalCoupler` when the fourth port must remain accessible
or reflected rather than absorbed. A reflecting Y-junction requires a different,
explicit scattering law; this milestone does not silently substitute one.

## API and state migration

Existing capitalized public names and parameter defaults remain available from
`circulax.components.photonic` (also exported from `circulax.components`). The
corresponding lowercase factories build the same classes for another static
reference:

```python
from circulax import compile_circuit
from circulax.components import optical_waveguide, optical_wave_source, optical_wave_termination

r0 = 50.0
circuit = compile_circuit({
    "instances": {
        "S": {"component": "source", "settings": {"amplitude": 0.6+0.8j}},
        "W": {"component": "waveguide", "settings": {"length_um": 0.0}},
        "L": {"component": "load"},
    },
    "connections": {"S,p": "W,p1", "W,p2": "L,p"},
}, {
    "source": optical_wave_source(r0=r0),
    "waveguide": optical_waveguide(r0=r0),
    "load": optical_wave_termination(r0=r0),
}, backend="dense", g_leak=0.0)
y = circuit.dc()
waves = circuit.port_waves(y, "L,p")
assert abs(float(waves.incident_power)-1.0) < 1e-10
```

The other factories are `grating`, `splitter`, `directional_coupler`, and
`tunable_beam_splitter`. Default classes use `r0=1`; `r0` is a class construction
coordinate, not a swept instance parameter or an S-parameter renormalization.
Change it consistently across connected components. Carrier frame, phase/time
convention and mode must also agree. Connect two compatible terminals per node;
use explicit splitters for branching. The compiler does not yet enforce these
optical connection rules.

Raw state layouts change: waveguides/gratings gain two complex wave states,
splitters three, directional couplers four. Tunable beam splitters replace
`i_top/i_bot` with four incident states. Names are `wave_<port>`; rebuild raw
initial guesses, saved states, and direct physics calls supplying these states.
Use public port observables instead of indexing internal arrays. Existing
`OpticalSource` and `OpticalSourcePulse` retain their potential-difference
semantics and parameter settings, including under reflection. Their `power`
is not a prescribed launched power. Existing `Circuit.port` still returns
potential, whose squared magnitude is generally not traveling-wave power.

DC, transient samples, and HB time samples use `Circuit.port_waves` with the
same parameter updates as the solve. AC responses are envelope perturbations;
HB coefficients are harmonic amplitudes, neither total fields nor total powers.
Use `analysis="hb_time"` and reconstruct samples before computing powers.
AC traveling-power observables remain unsupported. Static SAX export uses the
same retained construction and preserves these scattering laws.

## Network singularities and validation

A local exact through/short phase is valid even when `I+S` is singular. The
retained construction handles it without altering the device. Whole networks
can still be physically singular: a closed, isolated, exactly lossless resonant
loop has an arbitrary circulating amplitude. For the tested all-pass ring the
feedback denominator is `1-t*z` (bar amplitude t, round-trip transmission z).
At `t*z=1` its circulating solution is nonunique, or incompatible with a
nonzero forcing term. This network property is distinct from a removable local
admittance representation failure. Neither leakage nor optical-ground penalties
are used in the analytic tests; no convergence defaults are changed.

Retained rows also exposed a BiCGStab Jacobi-preconditioner issue in the legacy
optical transient regressions: a zero physical diagonal was scaled by numerical
leakage alone (about 1e9 at the default leakage). The sparse preconditioner now
uses the row's absolute coefficient sum when the physical diagonal is zero or
below its existing 1e-12 threshold. This changes only preconditioning, not the
linear operator, leakage, ground stiffness, Newton/linear tolerances, iteration
limits, or convergence defaults. Both failing legacy-source transient cases
pass with this correction; the other solver backends are unaffected.

`tests/test_native_power_waves.py` compares public scattering solves and wave
observables with independent analytic matrices and network formulas. It covers
full bidirectional matrices, coherent illumination, splitter bright/dark modes,
passivity and promised losslessness, grating spectral/peak response, zero/near-zero
loss, carrier phases, dispersion updates, endpoints, references 0.01/1/50,
reflective loads, MZI and ring feedback, supported DC/AC/HB/transient,
unknown/parameter Jacobians, JIT/vmap and smooth angle derivatives. It makes
`s_to_y` raise in both native and adapter modules. Regression results and
environment limitations are recorded below.

Validation on Python 3.12 / CPU JAX (float64):

| Final check | Result |
| --- | --- |
| `tests/test_native_power_waves.py` | 105 passed |
| `test_power_wave_sax.py`, `test_sax_component.py`, `test_optical_wave_boundaries.py` | 133 passed |
| Full `test_transient.py`, `test_dc.py`, `test_waveform_sources.py` after the sparse fix | 92 passed |
| Affected sparse AC/small-signal/HB/compiler/circuit/external-boundary/wavelength selection after the fix | 8 passed |

The broader AC, small-signal, HB, compiler, circuit, boundary, wavelength,
sensitivity, transient and waveform batch initially had 224 passes, 9 optional
sensitivity skips (unavailable bosdi/OSDI), and the two sparse transient failures
described above. Both failures passed against the original passive implementation,
then passed against the migrated implementation with the preconditioner correction;
the complete transient/DC/waveform and affected sparse-analysis selections were
rerun successfully. Counts above overlap and should not be added as unique tests.

The migration example executes successfully. Scoped repository hooks and
`git diff --check` pass; YAML/TOML/notebook hooks skip as nonapplicable.
Native component/export/test files pass Ruff and formatting checks. The existing
`solvers/linear.py` retains six unrelated Ruff findings, verified against its
starting version; the six-line preconditioner addition introduces none. Existing
SAX holomorphic/conjugation and analysis/KLU deprecation warnings remain.
GPU execution, full notebooks, and the full documentation-site build were not run.

## Remaining paths and milestones

The subsequent [bidirectional delay milestone](bidirectional_optical_delays.md)
migrates `OpticalDelayLine` and `delay_line_fdomain`, including history,
carrier/modulation phase and supported public observables. Rational
frequency-domain models still retain a conversion path. The subsequent
[phase-shifter milestone](mixed_electro_optic_power_waves.md) migrates `phase_shifter_factory`. Detectors and other mixed electro-optic models
have not been migrated here. Their compatibility must not be inferred from
the models above.
Detector semantics, remaining conversions, and wider notebook migrations
remain subsequent work. No FIR propagation or generalized delays are
implemented. PR #98 remains separate and opt-in; convergence defaults are untouched.
