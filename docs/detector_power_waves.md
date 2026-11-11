# Detector power waves

This detector milestone of [issue #112](https://github.com/gdsfactory/circulax/issues/112)
builds on the uncommitted [phase-shifter milestone](mixed_electro_optic_power_waves.md)
at `ffb35317f483b7389c2a539c0f0220b27bce15df`. It does not complete #112.

## Components and optical laws

`circulax.components` now exports `IdealPhotodetector`, `Photodiode` and
`CarrierPhotodiode`. Their factories `ideal_photodetector(r0=...)`,
`photodiode(r0=...)` and `carrier_photodiode(r0=...)` choose a static finite
positive real reference. Default classes use `r0=1`. Compile with
`is_complex=True` and enable JAX float64. Runtime settings, defaults, port
order `(optical_in, anode, cathode)`, and units come from
`examples/photonic/photodetectors.ipynb`, which now imports these implementations.

The one optical terminal uses shared retained-wave construction:

```text
a = wave_optical_in; b = reflection*a                # sqrt(W)
potential = sqrt(r0)*(a+b)
flow = (a-b)/sqrt(r0)                               # into detector
F_wave_optical_in = a+b-potential/sqrt(r0); Q_optical = 0
P_incident = |a|²; P_reflected = |b|²
P_absorbed = |a|²-|b|²
I_generated = responsivity_absorbed * P_absorbed
```

There is no division by `1+reflection`, S-to-Y conversion, loss perturbation,
optical-ground penalty, leakage, reference perturbation or changed convergence
default. The incident constraint remains even at zero reflection. Passive
reflectivity endpoints 0 and 1, including reflection phase pi (`reflection=-1`
at the center wavelength), are regular local models. A circuit with external
feedback can still be physically singular; the detector does not infer matching.

The retained wavelength laws are, with `d = wavelength_nm-center_wavelength_nm`:

```text
responsivity_absorbed = responsivity*(wavelength_nm/center_wavelength_nm)
                       * exp(-0.5*(d/responsivity_bandwidth_nm)²)
reflection = sqrt(reflectivity)*exp(-0.25*(d/reflection_bandwidth_nm)²)
             * exp(1j*(reflection_phase_rad+reflection_phase_slope_rad_nm*d))
```

Wavelengths/bandwidths are nm; phase is radians. `reflectivity` is incident
power reflectance at the center wavelength. Supply 0 <= reflectivity <= 1,
positive bandwidths and physical material settings; no new clipping changes
the original parameter laws. `responsivity` is A/W **of absorbed power**.
External incident-power responsivity is `responsivity_absorbed*(1-|reflection|²)`.
For a measured external responsivity, divide by the absorption fraction once
before setting the absorbed-power responsivity. Full reflection generates zero
current for finite responsivity. External responsivity cannot determine an
absorbed-power efficiency when absorption is zero.

## Electrical and carrier laws

`IdealPhotodetector` stamps `F_anode=-I_generated`, `F_cathode=I_generated`
and no electrical charge. `Photodiode` adds the unchanged dark I-V and charge:

```text
v = V_anode-V_cathode
I_dark = i_s*expm1(v/(ideality*v_t)) + i_leak*tanh(v/v_leak) + v/r_sh
F_anode = I_dark-I_generated; F_cathode = -F_anode
Q_anode = junction_charge(v); Q_cathode = -Q_anode
```

Electrical current is A, voltage V, charge C, capacitance F. Below
`v_join=fc*v_bi`, charge is `2*c0*v_bi*(1-sqrt(1-v/v_bi))`. Above the join,
it continues with its value, capacitance and capacitance slope at the join:
`Q_join+C_join*(v-v_join)+0.5*dC_join*(v-v_join)²`. The branch is selected by
`Re(v)`; the selected voltage remains complex in the electrical equations.
This continuation preserves the original electrical imaginary-block equations;
it is different from the phase-shifter junction's real-voltage convention.

For `CarrierPhotodiode`, define `q0=QE*volume_m3*density_scale_m3` in C.
`x_e`, `x_h` are normalized excess populations, with physical densities
`density_scale_m3*x` in m^-3. Its unchanged laws are:

```text
G = I_generated/q0                                  # normalized pairs/s
population = (max(Re(x_e),0)+max(Re(x_h),0))/2
E0 = (v_bi+max(-Re(v),0))/1e-6                       # V/m
E = E0/(1+screening*population/x_screen)
E_reference = (v_bi+v_ref)/1e-6
tau_e = tau_e0*E_reference/E; tau_h = tau_h0*E_reference/E
R = x_e*x_h/tau_rec
F_xe = tau_ref*(x_e/tau_e+R-G); Q_xe = tau_ref*x_e
F_xh = tau_ref*(x_h/tau_h+R-G); Q_xh = tau_ref*x_h
I_collected = q0*(electron_weight*x_e/tau_e+(1-electron_weight)*x_h/tau_h)
F_anode = I_dark-I_collected; F_cathode = -F_anode
```

Junction Q is the same as `Photodiode`. Recombination and population coordinates
remain complex in F/Q; screening uses real nonnegative populations and reverse
voltage. Lifetimes and `tau_ref` are seconds, volume is m³. `tau_ref` scales the
carrier equations and is not a physical collection lifetime. It cancels from
the differential equations. With fixed collection weight `w=electron_weight`:

```text
stored_collection_charge = q0*(w*x_e+(1-w)*x_h)
d stored_collection_charge/dt = I_generated-I_collected-q0*R
```

This weighted charge differs from the two separate population charges `q0*x_e`,
`q0*x_h` and from junction depletion charge. Carrier Q rows are normalized
state equations, not terminal charges. Optical absorption (W), generated pairs,
collected current (A), and stored charge (C) describe different quantities.
Unequal sweep times produce different populations and a weighted two-pole
response, even without recombination or screening. Recombination lowers collected
charge; screening slows collection and can change recombination loss. No new
terminal charge law, carrier floor, noise, avalanche or thermal law is introduced.

## Boundaries and observations

Use [explicit optical boundaries](optical_wave_boundaries.md), with the same
reference, carrier frame, mode and orientation on connected terminals. Connect
one source to one detector; use a physical splitter for branching. Returning
light is absorbed by `OpticalWaveSource`, independent of emission. There is no
optical ground. Electrical ground remains appropriate.

```python
import jax
from circulax import compile_circuit
from circulax.components import ideal_photodetector, optical_wave_source
from circulax.components.electronic import VoltageSource

jax.config.update("jax_enable_x64", True)
r0 = 50.0
circuit = compile_circuit({
    "instances": {
        "PD": {"component": "detector", "settings": {"reflectivity": 1.0,
               "reflection_phase_rad": 3.141592653589793}},
        "S": {"component": "source", "settings": {"amplitude": 0.003+0.004j}},
        "V": {"component": "bias", "settings": {"V": -3.0}},
        "GND": {"component": "ground"},
    },
    "connections": {"S,p": "PD,optical_in", "V,p1": "PD,anode",
                    "GND,p1": ("V,p2", "PD,cathode")},
}, {"detector": ideal_photodetector(r0=r0),
    "source": optical_wave_source(r0=r0), "bias": VoltageSource},
    is_complex=True, backend="dense", g_leak=0.0)
y = circuit.dc()
waves = circuit.port_waves(y, "PD,optical_in")
assert abs(float(waves.incident_power)-25e-6) < 1e-12
assert abs(float(waves.net_power)) < 1e-12
```

`Circuit.port_waves` supports optical total DC, transient samples
(`analysis="tran", t=sol.ts`) and HB time samples (`analysis="hb_time"`).
Pass the same scalar runtime settings as the solve; vmap observations when
settings vary across a sweep. Only `optical_in` declares wave metadata.
Anode/cathode and all internal states reject wave reconstruction. Optical Q is
zero and terminal F is the complete instantaneous signed optical flow even when
electrical/carrier Q is nonzero. Unsupported dynamic-flow reconstructions and
analysis labels retain their explicit rejection. `Circuit.port` still returns
potential; its squared magnitude is generally not traveling power.

The notebook replaces its local two-terminal `OpticalCW` and
`OpticalGaussianPulse` stamps with `optical_wave_source(waveform=...)`. Numeric
default keyword arguments of `waveform(t, ...)` now become runtime settings,
so power/phase/width/modulation sweeps keep their original units. `amplitude`
remains a separate multiplier defaulting to one. Independent small-signal source
metadata drives emitted sqrt(W) perturbations and does not differentiate or
multiply by the waveform. Power-parameter derivatives through solves still
use JAX. The notebook attaches electrical p1/p2 fixtures through
`attach_testbench` and connects optical `p` boundaries directly. Its electrical
SP postprocessing remains a measurement conversion, not a detector realization.

## Analyses, differentiation and compatibility

| Analysis | Contract |
| --- | --- |
| DC | All three models, complex reflection and wavelength profiles, original dark/carrier equilibria. |
| Transient | All three models, junction dQ/dt and carrier populations; optical flow needs no history. Check timestep refinement. |
| `small_signal` | Complete real-block optical intensity, complex electrical I/Q, carrier screening/recombination and parameters. RF phasor phase stays separate from optical x/y coordinates. |
| Legacy `ac_response` | Explicitly rejected for these non-holomorphic mixed models; use `small_signal`. |
| SP | Original notebook electrical S11 and reference-phase amplitude-modulation measurement remain supported with `holomorphic=False`. These do not establish arbitrary complex optical power reconstruction. |
| HB | Full nonlinear F/Q at time samples, including intensity and carrier dynamics; check harmonic convergence. Powers use time samples, not Fourier coefficients. |

`small_signal` quadratures and voltage remain available. AC traveling-power
reconstruction remains unsupported. The normalized-field `response.power`
helper rejects raw wave-component terminal potentials and declared non-wave
carrier states, even when the caller supplies `field_unit="sqrt(W)"`. Do not treat arbitrary
potential aliases as normalized traveling fields. JIT/vmap, runtime parameters
and gradients preserve both optical quadratures, electrical and carrier blocks.
Derivatives at clipping/branch boundaries are piecewise conventions. The original
`sqrt(reflectivity)` amplitude law is not differentiable with respect to power
reflectivity at zero; zero-reflection solves and state derivatives are supported,
while reflectivity gradients should be taken inside its physical domain.

Each detector gains one complex algebraic `wave_optical_in` unknown. Existing
carrier states retain their names and order, with the incident state appended.
Recompile circuits and rebuild raw initial guesses. At r0=1, old physical states
are compatible when initialized with the correct incident wave; the old
potential-based optical conversion is replaced. Public helper
`optical_conversion(incident, ...)` now returns `(reflected_envelope,
generated_current)`, not `(terminal_flow, generated_current)` from potential.
The helpers and classes now reside in `circulax.components.detectors`.
Electrical and carrier settings, defaults and laws are preserved.

Matched-resistor detector proxies in `mzm.ipynb` and
`examples/inverse_design/03_photonic_cmz_demux.ipynb`, and postprocessed
responsivity/EO power in `traveling_wave_mzm.ipynb`, remain separate follow-up
work. Establish observation planes and matching before replacing their powers;
a resistor proxy does not supply carrier dynamics. Rational-model conversions,
FIR, generalized multiple delays and PR #98 remain outside this milestone.
PR #98 stays separate and opt-in. There is no remote issue update or publication.

## Validation

Fresh CPU / Python 3.12 / JAX float64 validation on 2026-11-11:

| Check | Fresh result |
| --- | --- |
| `tests/test_detector_power_waves.py` | **97 passed** |
| Final carrier-observable guard rerun: detector suite + existing `test_small_signal.py` | **123 passed** (97 detector + 26 existing; overlaps the totals). |
| All 13 Python cells of `examples/photonic/photodetectors.ipynb` | Executed successfully in order with `g_leak=0.0` in every fixture, including every assertion; plots used the Agg backend and no outputs were written back. |
| Migration guide Python example | Executed successfully; relative guide links resolve. |
| Scoped pre-commit hooks | Passed whitespace, EOF, merge-conflict and notebook stripping checks; YAML/TOML skipped as nonapplicable. |
| New/changed detector, boundary, export and test Python | Ruff lint and formatting pass. |

The detector tests use public compiled circuits and class F/Q calls, independent
scattering, electrical derivatives, two-lifetime transfer functions, pulse
convolutions, recombination equilibria and charge balance. They check references
0.01/1/50, matched/partial/full reflection and phases 0/pi/2/pi, optical x/y
quadratures and independent RF phasors, optical pulses, unequal collection times,
screening/recombination, runtime parameters, JIT/vmap, solved parameter gradients,
full real-block state/parameter finite differences, timestep and harmonic
refinement, observable/analysis rejections, and conversion functions patched to
raise. Analytic wave networks use zero leakage and electrical ground only.
The notebook additionally checks loaded junction charge, nonlinear screening,
large-signal HB convergence, electrical SP, hierarchy and package parasitics.

An intermediate test run had nine JAX list-indexing errors in a final test
assertion and one incorrect test call supplying a DC time keyword. These test
errors were corrected; the final 97-test run above has no failures. No production
physics change was needed for those errors. Earlier debugging/rerun counts are
not added to the final counts.

Full touched-Python lint retains 12 inherited findings: ten in `circuit.py` and
two in `compiler.py`, with codes/messages verified identical to the base commit.
The compiler also retains its original multiline `configured` comprehension,
which the installed formatter would collapse. The new milestone introduces no
Python lint/format findings. The notebook retains unrelated existing lint
findings (18, all inherited; the base had 32); only its new import block was sorted. A complete repository test run,
GPU execution and a documentation-site build are outside this scoped validation.
The preceding milestone's 676 passes / 9 skips are historical evidence only;
the broader regression selection for this milestone is recorded separately below.

The regression selection includes `test_phase_shifter_power_waves.py`,
`test_phase_shifter.py`, `test_native_power_waves.py`, `test_power_wave_sax.py`,
`test_sax_component.py`, `test_optical_wave_boundaries.py`,
`test_optical_delay_power_waves.py`, `test_delay.py`, `test_delay_history.py`,
`test_delay_contract.py`, `test_waveform_sources.py`, `test_compiler.py`,
`test_circuit.py`, `test_small_signal.py`, `test_ac_response.py`,
`test_ac_sweep.py`, `test_harmonic_balance.py`, `test_dc.py`,
`test_dc_checked.py`, `test_transient.py`, `test_external_port_boundary.py`,
`test_sp_wavelength.py` and `test_sensitivity.py`.

Fresh regression result: **676 passed, 9 skipped**, with no test failures
(2026-11-11, 18 min 14 s). These are newly executed results for this detector
milestone, despite matching the preceding milestone's historical count.
The nine optional sensitivity tests require unavailable bosdi/OSDI support.
The 26 warnings are inherited KLU `free_numeric` deprecations, the existing SAX
holomorphic/conjugation warning, and the legacy `Circuit.ac()` deprecation.

The disjoint regression and detector selections total **773 passed, 9 skipped**.
The 123-pass detector/small-signal rerun after the final carrier-state guard is
an overlapping verification, not an additional 123 unique tests. It verifies the
final guard and all detector cases, including the 26 existing small-signal tests.
The complete changed notebook was then executed again with zero leakage in all
fixtures and passed every assertion. All changes, including the prerequisite
phase-shifter files, remain uncommitted on `codex/detector-power-waves`; the
source worktree is preserved. No remote update was made, and #112 remains open.
