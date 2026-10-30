# Photonic component library

For a practical introduction, start with [Devices: choosing and connecting components](devices.md).

The library uses native complex envelopes and physical electrical/thermal loading.
Start with the [normalization and port contract](photonic_conventions.md): a total
optical node voltage is not incident power under reflection. All additions below
have time-domain F/Q realizations; fixed-delay devices also use native history.
None requires a Verilog-A compiler, reference checkout or proprietary simulator.

## Catalog

| Family / public API | Implementation and composition | Time domain / frequency-only | Verification and limits |
|---|---|---|---|
| `OpticalAttenuator`, `OpticalPhaseShifter`, `OpticalTermination`, `OpticalInterface` | [Reciprocal matched wave constraints](optical_passive.md), exact ideal endpoints | Algebraic time domain; no propagation dynamics | Full multi-input scattering, passivity, reciprocity, reflection phase; positive real z0 |
| Existing `DirectionalCoupler`, `Splitter`, `OpticalWaveguide`, `Grating` | Reused with exact wave constraints; coupling clipping removed | Algebraic time domain; wavelength settings are carrier parameters | Full matrix checks; the three-port splitter is contractive with singular values (1,1,0), not unitary |
| `NonlinearOpticalWaveguide` | Promoted delayed waveguide with directional TPA | Native fixed delay | Original notebook, independent shifted pulse and DC attenuation; no cross-TPA/Kerr/carrier heating |
| `ElectricalPhaseShifter`, `PNJunctionPhaseShifter`, `PNPhaseShifter`, `loaded_phase_shifter` | [Ideal control and promoted delayed PN cores](phase_shifters.md); finite series R/shunt C factory | Native F/Q, PN fixed delay | Integrated charge, real-block AD, loaded RC and coherent EO response; generic parameters need calibration |
| `JouleHeater`, `ThermalPhaseShifter`, `thermo_optic_phase_shifter` | [Heater + explicit thermal R/C + optical core](thermo_optic.md) | Native thermal/electrical F/Q | Joule power, thermal rise/relaxation and optical phase; explicitly terminate thermal_reference; no absorption heating |
| `mach_zehnder`, `lumped_mzm` | [Real coupler/arm subcircuits](modulators.md), imbalance, bias, loss, single-arm/push-pull loading | Native F/Q | Independent coherent matrix and RC references, finite-drive link; no material calibration |
| `AllPassRing`, `AddDropRing`, `AllPassRingEO`, `loaded_ring` | [Reciprocal single-resonance CMT](resonators.md), counterpropagating modes, loaded electrical tuning | Native cavity F/Q | Through/drop transfer, energy balance and amplitude/energy lifetimes; narrowband around one resonance, no backscattering or multi-FSR claim |
| `RingModulator`, `RingModulatorEO`, `RingModSetup` | Promoted notebook baseline, explicitly legacy directed behavior | Native F/Q; illustrative static setup | Preserved example equations and separate independent checks; retained legacy detuning convention |
| `IdealPhotodetector`, `Photodiode`, `CarrierPhotodiode`, `make_photodetector_subcircuit` | [Promoted absorbing detector cores and R/RC package](photodetectors.md) | Native charge/carrier F/Q | Actual loaded I/V, absorption/reflection, phase invariance, RC/transit response, pulse charge and small-signal HB; carrier transport is phenomenological |
| `OpticalCW`, `OpticalGaussianPulse` | Promoted matched incident-power sources | Time-domain envelope sources | Power in W at z0=1; these are prescribed sources, not laser dynamics |
| `OpticalFilter` | [Reciprocal causal one-pole envelope filter](optical_filters.md) | Native F/Q | Independent transfer and step response; one Lorentzian pole, no broadband measured-device claim |
| `traveling_wave_mzm`, `electrode_netlist`, `optical_netlist`, `loaded_group_index` | [Promoted distributed electrode/PN/delay composition](modulators.md) using existing RLGC/Foster work | Native rational line F/Q + optical delay | Loaded electrode ABCD, mismatch and spatial convergence; line fits require a declared fit band/error |
| Existing `OpticalDelayLine`, `TunableBeamSplitter`, `QuadratureCombiner` | Retained directed/buffered behavioral blocks | Native time-domain/delay | Not advertised as reciprocal passive scattering devices |
| Existing `delay_line_fdomain`, `fdomain_component`, fitted frequency data | Retained frequency-response path | **Frequency-only where no causal realization is supplied** | Does not become a transient model merely by specifying a wavelength |

See each model page for exact ports, units, valid domains, initialization, measurements
and supported analysis paths. Composition is flattened before compilation; a subcircuit
is not a separate JIT boundary.

## Example-to-library promotion

| Source inspected (actual code cells/helpers) | Promoted library destination | Example update |
|---|---|---|
| `examples/photonic/photodetectors.ipynb` | `photodetectors.py`: all three cores, helpers, matched sources, package factory | Local model definitions replaced by imports; analysis and plots retained |
| `examples/photonic/photonics_transient.ipynb` | `optical_passive.py`: `NonlinearOpticalWaveguide` | Local definition replaced by import; notebook executed successfully |
| `examples/photonic/ring_modulator.ipynb` | `resonators.py`: legacy cores/setup | Targeted definition/import changes; example source/analysis remains local |
| `examples/photonic/traveling_wave_mzm.ipynb` | `phase_shifters.py`, `modulators.py`: original PN law and combiner | Targeted definition/import changes |
| `examples/photonic/traveling_wave_mzm_reference.ipynb` | PNPhaseShifter, PhaseBias alias, optical topology | Targeted source edits preserving pre-existing work |
| `examples/photonic/traveling_wave_mzm_tutorial.py` | Distributed electrode and loaded-group-index builders | Compatibility imports expose the promoted builders; waveform/eye analyses remain in examples |
| `examples/photonic/mzm.ipynb`, inverse-design CMZ example | Existing coupler/waveguide composition reused | No new duplicate physics |

Regression checks reproduce representative original equations alongside independent
physics checks. The ring notebook baseline through cell 18 also executed against its analytical
steady transmission (both 0.925). Only the delayed-waveguide notebook has been run end-to-end during
this change; the larger notebooks' saved plots/outputs do not constitute a fresh run.

## Running examples and small-signal response

From an installed environment or repository root:

```bash
python examples/photonic/library_link.py
python examples/photonic/library_traveling_wave_mzm.py
python examples/photonic/library_components.py
```

The first example composes a matched CW source, a finite-load push-pull MZM and
an absorbing photodiode with a 50-ohm load. It checks DC coherent recombination
and EO/OE conversion against independent formulas. The focused tests additionally
check its finite-drive charging transient at two timesteps.

`Circuit.ac_response` deliberately refuses complex nonholomorphic circuits.
Use `circulax.quadrature_response` for native mixed-domain models:

```python
from circulax import quadrature_response

response = quadrature_response(
    circuit, freqs=frequencies, y_dc=bias,
    sources={"DRIVE": 1.0},  # peak electrical volts, I quadrature
)
idx = circuit.port_map["output"]
voltage_phasor = response["in_phase"][:, idx]
```

Source values can be `(I, Q)` pairs of complex **modulation phasors**. For a
matched optical source without an excitation stamp, use
`parameter_drives={"LIGHT.power": delta_power}` in W. This differentiates F and Q
at fixed bias and does not change the DC operating point.
For optical fields `delta E(t)=Re(I exp(i Omega t))+i Re(Q exp(i Omega t))`.
The upper sideband is `(I+iQ)/2`, the lower is `conj(I-iQ)/2`; both are returned.
Do not call `Circuit.port` on the quadrature arrays, which are already phasors:
index directly with `port_map`. All physical loads must be in the netlist.
The utility supports native fixed delays and dynamic scalar parameter updates,
but rejects OSDI and frequency-only circuits until their two-sideband contracts
are implemented. It does not claim those paths are validated by native-model checks.

## Evidence, provenance and cost

The implementation baseline is this repository's existing examples. New compact
cores use independently stated Ohm/thermal balance, wave constraints and linear
coupled-mode/state-space equations. No GPL Verilog-A implementation was copied or
translated. The user-supplied [external repository](https://github.com/keikawa/Verilog-A-photonic-model-library)
was inspected at `8b10116b0dcd537e5935cfbab2369672f7c353d4` as an unverified
GPL-3.0 comparison candidate; its optical-bus conventions are not assumed identical.
The [blog](https://www.simpletechtrend.com/en/post/verilog-a-eo-cosimulation-siph-transceiver-64gbps-en)
is discovery context. The linked [primary-paper DOI](https://doi.org/10.1109/JMW.2025.3576358)
and the old ring notebook's UGent PDF could not be accessed during this run.
No external VA/OSDI comparison, measured-data calibration, experimental validation,
BER/noise validation or calibrated high-speed link claim is made.

`benchmarks/sample_components.py --complex-state` retains both input quadratures
and benchmarks the full real-block F/Q and Jacobian, with continuous parameters
passed dynamically. `--imag-bias` supplies imaginary physical-state values.
For example:

```bash
python -m benchmarks.sample_components circulax.components.optical_passive OpticalPhaseShifter \
    --complex-state --bias '[1,0,1,0]' --imag-bias '[0.2,0.1,0.2,0.1]'
```

On CPU, Python 3.12.14/JAX 0.7.2, that fresh-process 4-complex-unknown core
measured F/Q trace **12.8 ms**, XLA compile **35.8 ms**, warm **38.5 us**;
Jacobian trace **24.1 ms**, compile **48.9 ms**, warm **33.8 us** (100 calls).
The composed link has **30 complex unknowns**; construction plus first DC solve
measured **1.91 s**. These are local measurements, not universal performance promises.
These final core measurements include runtime domain guards and shared CPU load.
Per-family pages record additional sizes/costs and exact comparison coverage.
See the [verification record](photonic_verification.md) for exact test commands and run boundaries.

Full laser/SOA/EDFA material dynamics, stochastic ASE/RIN/linewidth, large AWGs,
multimode/polarization and foundry-specific PCW physics remain deferred. Balanced
receivers, optical hybrids and gain blocks are follow-on work. A higher-order
filter is unnecessary for the focused first-order implementation.
