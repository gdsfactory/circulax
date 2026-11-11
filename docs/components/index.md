# Components

Use this catalog to choose a component, then follow its link for ports, parameters, equations, and source code. Register component classes or compiled subcircuits in the model mapping passed to `compile_circuit`; use instance `settings` to configure their parameters. See [Writing Components](../writing_components.md) for the component interface.

## Electrical components

### Passive elements and interconnects

| Component | Purpose |
| --- | --- |
| [`Resistor`](../references/components/electronic.md#circulax.components.electronic.Resistor) | Two-terminal resistor with resistance in ohms. |
| [`Capacitor`](../references/components/electronic.md#circulax.components.electronic.Capacitor) | Two-terminal capacitor with capacitance in farads. |
| [`Inductor`](../references/components/electronic.md#circulax.components.electronic.Inductor) | Two-terminal inductor with inductance in henries. |
| [`CoupledInductors`](../references/components/electronic.md#circulax.components.electronic.CoupledInductors) | Two magnetically coupled inductors with `M = k * sqrt(L1 * L2)`. |
| [`IdealTransformer`](../references/components/electronic.md#circulax.components.electronic.IdealTransformer) | Ideal transformer with turns ratio `n = N1 / N2` (`n > 0`). |
| [`DifferentialAdapter`](../references/components/electronic.md#circulax.components.electronic.DifferentialAdapter) | Lossless bidirectional differential/common to conductor adapter. |
| [`TransmissionLine`](../references/components/electronic.md#circulax.components.electronic.TransmissionLine) | Matched bidirectional transmission line with exact fixed delay. |

### Sources and controlled elements

| Component | Purpose |
| --- | --- |
| [`VoltageSource`](../references/components/electronic.md#circulax.components.electronic.VoltageSource) | Step voltage source. |
| [`CurrentSource`](../references/components/electronic.md#circulax.components.electronic.CurrentSource) | Constant current source. |
| [`VoltageSourceAC`](../references/components/electronic.md#circulax.components.electronic.VoltageSourceAC) | Sinusoidal voltage source. |
| [`SmoothPulse`](../references/components/electronic.md#circulax.components.electronic.SmoothPulse) | Voltage source with a sigmoid-smoothed pulse. |
| [`PulseVoltageSource`](../references/components/electronic.md#circulax.components.electronic.PulseVoltageSource) | SPICE-compatible PULSE voltage source: PULSE(v1 v2 td tr tf pw per). |
| [`WaveformVoltageSource`](../references/components/electronic.md#circulax.components.electronic.WaveformVoltageSource) | SPICE voltage source with optional DC override and SIN/PULSE/PWL waveform. |
| [`WaveformCurrentSource`](../references/components/electronic.md#circulax.components.electronic.WaveformCurrentSource) | Independent current source with a DC bias and a SIN, PULSE or PWL transient waveform. |
| [`VCVS`](../references/components/electronic.md#circulax.components.electronic.VCVS) | Voltage Controlled Voltage Source. |
| [`VCCS`](../references/components/electronic.md#circulax.components.electronic.VCCS) | Voltage Controlled Current Source. |
| [`CCVS`](../references/components/electronic.md#circulax.components.electronic.CCVS) | Current Controlled Voltage Source: V_out = R * i_ctrl, V_in = 0 (short). |
| [`CCCS`](../references/components/electronic.md#circulax.components.electronic.CCCS) | Current Controlled Current Source: I_out = alpha * i_ctrl, V_in = 0 (short). |
| [`IdealOpAmp`](../references/components/electronic.md#circulax.components.electronic.IdealOpAmp) | Ideal voltage amplifier with configurable open-loop gain. |
| [`VoltageControlledSwitch`](../references/components/electronic.md#circulax.components.electronic.VoltageControlledSwitch) | Voltage Controlled Switch. |

For waveform definitions and examples, see [Waveform Sources](../waveform_sources.md).

### Semiconductor devices

| Component | Purpose |
| --- | --- |
| [`Diode`](../references/components/electronic.md#circulax.components.electronic.Diode) | Ideal diode using the Shockley equation `I = Is * (exp(Vd / n*Vt) - 1)`. |
| [`ZenerDiode`](../references/components/electronic.md#circulax.components.electronic.ZenerDiode) | Zener diode with forward Shockley conduction and reverse breakdown. |
| [`NMOS`](../references/components/electronic.md#circulax.components.electronic.NMOS) | N-channel MOSFET with square-law DC model and channel-length modulation. |
| [`PMOS`](../references/components/electronic.md#circulax.components.electronic.PMOS) | P-channel MOSFET with square-law DC model, formulated in terms of `Vsg` and `Vsd`. |
| [`NMOSDynamic`](../references/components/electronic.md#circulax.components.electronic.NMOSDynamic) | NMOS with square-law DC model and Meyer gate capacitance model. |
| [`BJT_NPN`](../references/components/electronic.md#circulax.components.electronic.BJT_NPN) | NPN BJT using the transport form of the Ebers-Moll DC model. |
| [`BJT_NPN_Dynamic`](../references/components/electronic.md#circulax.components.electronic.BJT_NPN_Dynamic) | NPN BJT with the Ebers-Moll DC model and first-order junction charge dynamics. |

The static transistor models describe DC conduction; the dynamic variants also include charge storage. Choose the model according to the dynamics required by your circuit.

### Filters

| Component | Purpose |
| --- | --- |
| [`LowPass1`](../references/components/filters.md#circulax.components.filters.LowPass1) | Unity-DC-gain first-order low-pass; `f3db` is the -3 dB frequency. |
| [`Biquad`](../references/components/filters.md#circulax.components.filters.Biquad) | Unity-DC-gain second-order low-pass with -3 dB frequency and quality factor. |
| [`Bessel3`](../references/components/filters.md#circulax.components.filters.Bessel3) | Unity-DC-gain third-order Bessel low-pass, normalized at -3 dB. |

## Photonic components

| Component | Purpose |
| --- | --- |
| [`OpticalWaveguide`](../references/components/photonic.md#circulax.components.photonic.OpticalWaveguide) | Waveguide with optical phase, propagation loss, and first-order dispersion. |
| [`OpticalDelayLine`](../references/components/photonic.md#circulax.components.photonic.OpticalDelayLine) | Reciprocal retained-wave carrier-envelope delay, including reflected light. |
| [`Grating`](../references/components/photonic.md#circulax.components.photonic.Grating) | Grating coupler with Gaussian wavelength-dependent insertion loss. |
| [`Splitter`](../references/components/photonic.md#circulax.components.photonic.Splitter) | Reciprocal passive three-port splitter with configurable power split and an omitted matched loss channel. |
| [`DirectionalCoupler`](../references/components/photonic.md#circulax.components.photonic.DirectionalCoupler) | 4-port directional coupler (2×2 beamsplitter) with configurable power split ratio. |
| [`TunableBeamSplitter`](../references/components/photonic.md#circulax.components.photonic.TunableBeamSplitter) | Ideal lossless 2×2 directional coupler with a variable coupling angle. |
| [`OpticalSource`](../references/components/photonic.md#circulax.components.photonic.OpticalSource) | Ideal CW optical source for DC and small-signal AC analysis. |
| [`OpticalSourcePulse`](../references/components/photonic.md#circulax.components.photonic.OpticalSourcePulse) | Time-dependent optical source with a sigmoid turn-on profile. |
| [`phase_shifter_factory`](../references/components/phase_shifter.md#circulax.components.phase_shifter.phase_shifter_factory) | Electro-optic phase shifter from voltage-dependent index, loss and optional junction charge. |
| [`pn_diode`](../references/components/phase_shifter.md#circulax.components.phase_shifter.pn_diode) | Reusable electrical junction with depletion charge and diode current. |
| [`junction_model`](../references/components/phase_shifter.md#circulax.components.phase_shifter.junction_model) | Electrical junction from supplied current, charge or capacitance functions. |
| [`delay_line_fdomain`](../references/components/photonic.md#circulax.components.photonic.delay_line_fdomain) | Retained-wave modulation-only delay for DC, AC and harmonic balance. |

The five algebraic native passives use [retained power-wave ports](../native_passive_power_waves.md),
including exact lossless through and coupling endpoints. Use explicit launched-wave
sources/terminations and `Circuit.port_waves` for traveling envelopes and powers.
`Splitter` is forward-lossless but absorbs one coherent reverse-input mode;
`TunableBeamSplitter` now propagates in both directions. The
[bidirectional delays](../bidirectional_optical_delays.md) and
[mixed phase shifter](../mixed_electro_optic_power_waves.md) share this optical
contract with their documented timing and analysis limits. Legacy sources,
detectors and other mixed models have separate semantics.

Optical circuits use complex field envelopes. Enable float64 and compile with `is_complex=True`. Use `OpticalDelayLine` when explicit propagation delay is needed in a transient simulation; `delay_line_fdomain` supplies a frequency-domain representation for AC and harmonic balance.

For complete circuits, see the [Mach–Zehnder modulator](../examples/mzm.md), [traveling-wave MZM](../examples/traveling_wave_mzm.md), [photonic transient](../examples/photonics_transient.md), and [ring modulator](../examples/ring_modulator.md) tutorials. The traveling-wave tutorial defines its PN phase-shifter component and composes the MZM from electrode and optical sections.

### Photodetectors

`IdealPhotodetector`, `Photodiode` and `CarrierPhotodiode` are reusable absorbed-power
models with one optical terminal and electrical anode/cathode. Their factories
`ideal_photodetector`, `photodiode` and `carrier_photodiode` accept a static optical
`r0`. See [Detector Power Waves](../detector_power_waves.md) for responsivity,
carrier/charge balance, parameter units, supported analyses and migration.

## Fitted transmission lines

| Component | Purpose |
| --- | --- |
| [`rlgc_transmission_line`](../references/components/transmission_line.md#circulax.components.transmission_line.rlgc_transmission_line) | Fit frequency-dependent RLGC per metre and return a line factory. |
| [`mode_transmission_line`](../references/components/transmission_line.md#circulax.components.transmission_line.mode_transmission_line) | Generate a passive RLC line factory from uniform-mode propagation data. |
| [`reduced_rlgc_transmission_line`](../references/components/transmission_line.md#circulax.components.transmission_line.reduced_rlgc_transmission_line) | Create a compact, transient-capable uniform line from RLGC versus Hz. |
| [`characteristic_rlgc_transmission_line`](../references/components/characteristic_line.md#circulax.components.characteristic_line.characteristic_rlgc_transmission_line) | Compact uniform line with fitted characteristic admittance and explicit propagation delay. |
| [`RLGCTransmissionLine`](../references/components/transmission_line.md#circulax.components.transmission_line.RLGCTransmissionLine) | Reusable fitted line; call with a length in metres to create a Circuit. |

These factories construct circuit models from sampled line properties. The Foster ladder uses passive RLC elements; the reduced and characteristic models use rational responses checked for sampled passivity. Start with the [reduced RLGC transmission-line example](../examples/rlgc_transmission_line.md), or use the [traveling-wave MZM tutorial](../examples/traveling_wave_mzm.md) to connect modal lines to distributed junction loads. The [Fitting API](../fitting_api.md) explains the input data and fit controls.

## Rational models

| Component | Purpose |
| --- | --- |
| [`rational_component`](../references/components/rational.md#circulax.components.rational.rational_component) | Create a time-domain component from a rational-fitting SSModel. |
| [`rational_fdomain_component`](../references/components/rational.md#circulax.components.rational.rational_fdomain_component) | Create a frequency-domain oracle component from a rational-fitting SSModel. |
| [`rational_delay_component`](../references/components/rational.md#circulax.components.rational.rational_delay_component) | Create an fdomain component from SSModel + per-port delay. |

`rational_component` provides a time-domain realization. The frequency-domain factories support frequency-response analysis. See [Rational Model Enforcement](../rational_model_enforcement.md) and the [vector-fitting example](../examples/vector_fitting.md) for fitting and validation.

See [Bidirectional Optical Delays](../bidirectional_optical_delays.md) for phase conventions, history semantics, observables and raw-state migration.
