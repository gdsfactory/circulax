# Photonic component conventions

This contract covers the generic photonic library expansion. Existing examples
are the implementation baseline; model-specific documentation records retained
legacy conventions and any demonstrated corrections.

## Fields, ports and power

Optical circuits use complex slowly varying envelopes with a carrier convention
`Re(E(t) exp(+i omega_ref t))`. Forward propagation contributes
`exp(-i beta L)` and an envelope delay `E(t-tau)`. A component's positive control
phase may deliberately be `exp(+i phase)`; its sign is documented explicitly.
Keep carrier frequency/wavelength and envelope modulation frequency distinct.
Do not add carrier propagation phase twice when adding envelope delay.

Currents are positive into each device. At real positive reference impedance
`z0`, incident and outgoing power waves are
`a=(v+z0*i)/(2*sqrt(z0))` and `b=(v-z0*i)/(2*sqrt(z0))`.
Thus `v=sqrt(z0)*(a+b)`, `i=(a-b)/sqrt(z0)`, and net inward power is
`|a|^2-|b|^2` watts. The optical default is **z0=1**, so the numerical wave
amplitude is in sqrt(W). No extra factor of two is applied to optical power.
Legacy example states named `a` sometimes store incident pseudo-voltage rather
than power wave: these coincide at z0=1. Preserve example behavior there and
explicitly document or correct scaling before supporting other impedances.

A total node voltage is not an incident field when there is reflection.
Use matched source/load boundaries and the wave equations for passive devices.
Legacy buffered/source-clamped models use explicitly directed field amplitudes;
retain them as documented directional behavioral models, with no reciprocal
scattering claim. The existing OpticalSource prescribes a field voltage and is
not a calibrated matched incident-power source under arbitrary reflections.
The photodetectors notebook's OpticalCW is a matched incident-power source.

Optical external ports of passive models are genuine scattering interfaces.
Do not mask singular ideal S-to-Y endpoints with loss, clipping or leakage.
Constraint/wave states are acceptable and compose with ordinary KCL topology.
Do not claim a splitter is unitary without checking its entire S matrix.

## Electrical, thermal and cavity quantities

Electrical ports use volts and currents into the device in amperes; electron
photocurrent in the promoted detectors leaves the anode and enters the cathode.
Electrical/thermal quantities are physical real values. In a complex mixed
circuit, retain a restoring imaginary equation (e.g. use full complex voltage
in electrical charge/resistance, and real voltage for optical control). Taking
real() everywhere must not leave imaginary circuit unknowns unconstrained.

Thermal ports represent temperature rise in K, with heat flow in W and storage
in J. An exposed free thermal/auxiliary node must appear in connections or in
the declared netlist ports; document its allocation and reference connection.
Use explicit ambient temperature and positive Rth/Cth where a dynamic thermal
model is claimed. Account for Joule heating and modeled absorption separately.

Cavity envelope amplitudes use sqrt(J), so |a_cavity|^2 is stored energy.
State whether a lifetime is an amplitude-decay or energy-decay lifetime:
`da/dt=-gamma*a` implies energy decay rate `2*gamma`.

## Analysis and implementation

Use native F+dQ/dt=0 and supported signals.at_delay facilities, with float64 /
complex128 enabled for verification. Mark intensity/conjugation/real-control
models non-holomorphic. Check full real/imaginary Jacobians and both sidebands
for mixed EO/OE response; a wavelength sweep does not verify envelope dynamics.
Fixed carrier settings may parameterize a time-domain model; they do not make
that model fdomain-only. No stochastic samples belong in deterministic residuals.

Keep the coupled core compact, promote existing equations before deriving new
ones, and use subcircuits for arms, parasitics, electrode and thermal networks.
Hierarchy is flattened, not separately JIT compiled. Benchmark dynamic parameters
and real-block Jacobians, not constant-folded single configurations.

## Example-to-library starting inventory

| Existing source | Definitions / coverage | Action |
|---|---|---|
| `examples/photonic/photodetectors.ipynb`, cells 3/5/7/23 | IdealPhotodetector, Photodiode, CarrierPhotodiode, spectral/charge/carrier helpers, matched OpticalCW, package factory | Promote equations and factory; import them from the notebook |
| `examples/photonic/ring_modulator.ipynb`, cells 4/6/18 | RingModulator, RingModSetup/setup, RingModulatorEO and pulse/CW helpers | Promote legacy directed cores; add clearly distinct passive reciprocal CMT cores as needed |
| `examples/photonic/photonics_transient.ipynb`, cell 3 | NonlinearOpticalWaveguide, bidirectional delay and TPA | Promote and regression-check; preserve notebook narrative |
| `examples/photonic/traveling_wave_mzm.ipynb`, cell 7 | PNJunctionPhaseShifter, QuadratureCombiner | Promote exact definitions; retain aliases and units |
| `examples/photonic/traveling_wave_mzm_reference.ipynb`, cells 3/8 | PhaseBias, PNPhaseShifter, optical_netlist | Promote with targeted edits preserving existing work |
| `examples/photonic/traveling_wave_mzm_tutorial.py` | electrode_netlist, loaded_group_index; waveform/eye helpers | Promote circuit builders, retain analysis utilities in examples |
| `examples/photonic/mzm.ipynb` and inverse-design CMZ notebook | Existing coupler/waveguide MZI composition | Reuse topology; avoid duplicating device physics |
| `circulax/components/photonic.py` | Static waveguide/grating, couplers, sources, directed delay/tunable splitter | Preserve APIs; audit claims and singular ideal endpoints |

This inventory records code-cell indices at the start of the implementation;
subsequent targeted notebook edits can move them. Original extracted code is
available during implementation in `/tmp/circulax-photonic-example-inventory/`.
It is not a packaged/runtime dependency. Final model pages record actual source
provenance, analytical checks and remaining validity limits.
