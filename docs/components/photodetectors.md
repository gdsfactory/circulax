# Photodetectors

`circulax.components.photodetectors` promotes the current
[`photodetectors.ipynb`](../../examples/photonic/photodetectors.ipynb) equations and
package topology. The notebook now imports these definitions and retains its
pulse, spectral, AC, parasitic-package and HB demonstrations.

| API | Origin | Meaning |
| --- | --- | --- |
| `IdealPhotodetector` | notebook cell 3 | Absorbing one-port and instantaneous electrical photocurrent |
| `Photodiode` | notebook cell 5 | Adds dark conduction and integrated depletion charge |
| `CarrierPhotodiode` | notebook cell 7 | Adds electron/hole extraction, recombination and field screening |
| `OpticalCW`, `OpticalGaussianPulse` | notebook cells 5/7 | Matched incident-power sources |
| `make_photodetector_subcircuit` | notebook cell 23 | Intrinsic detector, true series resistor, pad resistor and optional pad capacitor |

All three detectors use ports `optical_in`, `anode`, `cathode`, with currents
positive into the component. Photocurrent leaves the anode, so a load from anode
to ground develops positive voltage under reverse cathode bias. The package adds
`pad_ground`, which must be explicitly grounded or connected. The intrinsic anode
is behind `R_series`; the pad R and C connect the external anode to `pad_ground`.
`with_parasitics=False` removes all three parasitic branches and `pad_ground`.
The factory creates its own model mapping and has no notebook runtime dependency.

## Normalization and equations

At reference impedance 1, incident amplitude has units sqrt(W). With complex
reflection coefficient `r`, `v=(1+r)*a`, `i=(1-r)*a`. Absorbed power is
`(1-|r|²)*|v/(1+r)|²`, and photocurrent is absorbed power times responsivity in A/W.
This is an absorbing optical boundary, with no transmitted port. It is neither an
ideal non-invasive monitor nor a semiconductor energy-balance model.
`OpticalCW.power` and `OpticalGaussianPulse.power` specify incident power in W;
the CW modulation is fractional **power** modulation. Both source parameters and
optical reflection may have arbitrary phase.

The retained illustrative spectral profile is
`R_abs=responsivity*(wavelength_nm/center_wavelength_nm)*exp(-detuning²/(2*bandwidth²))`.
The reflected field has Gaussian magnitude and linear phase versus wavelength.
Wavelengths/bandwidths are in nm and phase slopes in rad/nm. This is a parameterized
spectral curve, not material dispersion or experimentally calibrated quantum
efficiency. Users must choose nonnegative responsivity below the quantum limit
where that physical interpretation is desired. Use `0 <= reflectivity < 1` and
positive spectral widths/wavelengths; perfect reflection `r=-1` is outside this
one-port admittance formulation. The supported optical normalization is z0=1.

`Photodiode` adds Shockley current, smooth tanh leakage and shunt resistance.
Junction charge integrates `C0/sqrt(1-V/Vbi)` below `fc*Vbi`; above that join its
quadratic continuation has continuous differential capacitance. It is not
`C(V)*V`. Units are A, V, F, ohm and seconds. Use positive thermal voltage,
ideality, leakage scale, shunt resistance and built-in potential, and `0<=fc<1`.
No avalanche, diffusion-charge diode recovery, stochastic noise or temperature
calibration is implied. Very large forward bias can overflow the retained
unlimited Shockley exponential; this is not a power-diode model.

`CarrierPhotodiode` retains the notebook's two scaled excess populations
`x_e`, `x_h`, with charge scale `q_e*volume_m3*density_scale_m3`. Extraction times
increase with screened population and decrease with reverse bias; recombination
is `x_e*x_h/tau_rec`. Electron/hole extraction currents are weighted separately.
Use positive geometry, density scale, lifetimes, screening scale and reference
scaling time, nonnegative screening and `0<=electron_weight<=1`. This is a compact
phenomenological transit model, not a Poisson/drift-diffusion solution. Population
states initialize to zero in darkness; DC supplies illuminated equilibrium.
The electrical RC network and these intrinsic transit states are separate poles.

The one necessary promotion correction makes carrier rates and photocurrent use
real carrier populations and adds restoring imaginary residuals to carrier states.
Their storage retains full complex state, giving imaginary decay time `tau_ref`.
This preserves real-state notebook behavior and prevents imaginary populations
from producing complex physical photocurrent. Electrical dark conduction and
junction storage retain their original complex continuation, constraining the
imaginary electrical block near physical operating points. Models are explicitly
non-holomorphic and require real-block optical linearization.

## Usage

```python
from circulax.components.photodetectors import (
    CarrierPhotodiode, OpticalCW, make_photodetector_subcircuit,
)

models = {"detector": CarrierPhotodiode, "light": OpticalCW}
# A reusable Circuit object can be used in an outer compile_circuit model map.
models["packaged_detector"] = make_photodetector_subcircuit(
    model_kind="carrier", series_resistance=20.0,
    pad_resistance=1e6, pad_capacitance=100e-15,
)
```

See the source notebook for complete loaded circuits. No external VA, OSDI,
proprietary simulator or reference checkout is needed to import or evaluate these
native models. Existing notebook equations are the implementation provenance;
no GPL photonic source was copied or translated.

## Verification and limits

`tests/test_photodetectors.py` includes frozen constitutive outputs from the
original notebook at complex optical field and real electrical/carrier states;
loaded compiled circuits with reflected CW light and electrical loads; phase
invariance; signed responsivity; optical absorption; dark I/V; integrated junction
capacitance; package topology, series drop and pad leakage division; independent
RC and two-extraction-pole small-signal transfer; diode/carrier Gaussian pulse integrated charge
with two timestep sizes; small-modulation carrier HB/SP agreement and three/five
harmonic convergence; real-block forward/reverse AD, JIT/vmap and dynamic
parameter updates. Both optical quadratures are included in the full real-block
Jacobian test. Small-signal optical transfer uses the public non-holomorphic SP
path at zero source phase, with independent network conversion after the solve.

The original notebook's broader spectral/power sweeps, carrier screening plots
and HB demonstrations are retained, but this promotion does not by itself claim
that all notebook cells were rerun. No external cross-simulator comparison,
material calibration, measured high-speed link, BER or experimental verification
is claimed. Finite-power nonlinear transit pulse/HB convergence and both-sideband
EO/OE measurements beyond the checked small-signal case remain future work.

A fresh CPU process, float64/complex128, with responsivity, extraction lifetimes
and screening supplied dynamically measured the five-complex-variable carrier
core (ten real inputs): F/Q lowering 0.028 s, XLA compilation 0.064 s, warm
execution 3.25 µs; full real-block Jacobian lowering 0.054 s, compilation 0.107 s,
warm execution 16.3 µs (100 calls). These are representative local timings, not
portable performance guarantees or constant-folded configurations.

A loaded packaged carrier-detector circuit has eight complex unknowns (sixteen
real equations). Its first dense DC solve took 0.926 s in a fresh process; ten
successive power-parameter update/solve calls averaged 0.822 s each, including
Python dispatch, parameter-update work and any retracing. This circuit figure
is deliberately separate from the already-compiled constitutive timings.

The absorbing admittance formulation requires `0 <= reflectivity < 1`; invalid
values are rejected at evaluation, including under JIT. An ideal reflecting
termination can make `1+r=0`; use the wave-constraint `OpticalInterface` for
that endpoint rather than this absorbing detector. No epsilon loss is added.
