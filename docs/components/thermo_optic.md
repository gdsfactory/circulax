# Thermo-optic phase shifter

`circulax.components.thermo_optic` provides `JouleHeater`,
`ThermalPhaseShifter`, and the reusable `thermo_optic_phase_shifter` Circuit
factory. This is new library coverage: there was no corresponding thermo-optic
model to promote from the audited photonic notebooks. Its provenance is an
independent implementation of Ohm's law, lumped thermal energy conservation and
linear thermo-optic tuning. No external VA implementation was copied, translated
or used as a validation oracle, and no measured calibration is claimed.

## Physical contract

The heater has electrical ports `hp`, `hn` (V, inward A) and thermal port `dt`
(temperature rise above ambient in K, inward W). With
`R(T)=resistance*(1+alpha*dt)`, `I=(Vhp-Vhn)/R(T)` and physical heating
`P=real(Vhp-Vhn)²/R(T)`. `alpha` has units 1/K; resistance is specified at
`ambient_temperature_K`. Use positive resistance throughout the operating range.
Ambient temperature is explicit; this first model only uses its rise for
resistance and phase, and includes no other absolute-temperature dependence.

The thermal network is an ordinary resistor in K/W and capacitor in J/K:

`Cth d(dt)/dt + dt/Rth = P`.

For constant resistance and a heater-voltage step, the independently derived
solution is `dt(t)=Rth*V²/R*(1-exp(-t/(Rth*Cth)))`; cooling is exponential with
that same time constant. Temperature-dependent resistance changes this response
and equilibrium; the constant-R exponential is not claimed when `alpha != 0`.

The optical core has ports `o1`, `o2`, `dt`, and incident pseudo-voltage states
`a1`, `a2`. It is matched and reciprocal, with
`S21=S12=10**(-loss_dB/20)*exp(+i*(phase+efficiency*real(dt)))`, `S11=S22=0`.
Efficiency is rad/K; phase is rad and loss is power attenuation in dB.
At reference impedance `z0`, each incident power is `|a|²/z0` W.
Wave constraint states handle zero phase and ideal through endpoints without
singular S-to-Y conversion. Both directions share one local temperature.
No envelope propagation delay or broadband material dispersion is modeled.

Electrical conduction retains the full complex electrical voltage. Heating and
optical tuning use real physical voltage and temperature. A positive imaginary
thermal conductance constrains the auxiliary imaginary temperature in the mixed
complex representation without injecting physical heat. The physical thermal
network also retains full complex thermal voltage in its R/C laws. These models
are non-holomorphic and need full real-block linearization.

The factory uses genuine heater/core/Rth/Cth circuit composition, flattened by
Circulax. It exposes `hp`, `hn`, `o1`, `o2`, `dt`, and `thermal_reference`; the `dt` port explicitly
allocates the otherwise auxiliary thermal node and permits observation or
additional thermal coupling. Connect `thermal_reference` to ground explicitly; this defines zero temperature
rise at ambient for Rth and Cth. An internal ground in a flattened child is not
assumed to establish this parent boundary. Do not clamp `dt` unless an isothermal boundary is
intended. Multiple devices have independent temperatures unless their exposed
thermal nodes are explicitly connected; shared nodes sum heat and storage.

## Runnable example

```python
from circulax import attach_testbench, compile_circuit
from circulax.components.electronic import Resistor, VoltageSource
from circulax.components.photodetectors import OpticalCW
from circulax.components.thermo_optic import thermo_optic_phase_shifter

package = thermo_optic_phase_shifter(
    resistance=100.0, thermal_resistance=1000.0,
    thermal_capacitance=1e-6, efficiency=0.02, phase=0.3,
)
device = {
    "instances": {"TO": {"component": "heater_phase"}},
    "ports": {p: f"TO,{p}" for p in ("hp", "hn", "o1", "o2", "dt", "thermal_reference")},
}
net = attach_testbench(
    device,
    sources={
        "hp": {"name": "DRIVE", "component": "voltage", "settings": {"V": 1.0}},
        "o1": {"name": "LIGHT", "component": "light", "settings": {"power": 1e-3}},
    },
    loads={"o2": {"component": "load", "settings": {"R": 1.0}}},
    gnd=["hn", "thermal_reference"],
)
circuit = compile_circuit(net, {
    "heater_phase": package, "voltage": VoltageSource, "light": OpticalCW,
    "load": Resistor, "ground": lambda: 0,
}, is_complex=True, backend="dense")
op = circuit.dc()
print(circuit.port(op, "dt"))  # 10 K: 1 V² / 100 ohm * 1000 K/W
```

## Verification and limitations

Focused tests independently verify electrical loading, `V²/R` heating,
`P*Rth` equilibrium temperature, loaded coherent output phase, heating/cooling
exponentials and two transient timesteps, both optical propagation directions,
full real-block forward/reverse Jacobians, JIT/vmap and a tuning-parameter
reverse derivative. Inputs with invalid static R, Rth, Cth, z0, ambient
or passive loss are rejected by the factory. Lower-level native components,
like other native constitutive primitives, assume documented parameter domains.

This is a single thermal mode, linear phase tuning and lumped ohmic heater.
Optical insertion loss is not converted to local heat; its absorbed optical
energy exits to an unmodeled bath. No absorption-heating, distributed thermal
coupling, temperature-dependent optical loss, mechanical stress, stochastic noise
or device calibration is implied. Core parameters can be updated after circuit
construction through flattened instance parameters; topology is fixed.
DC and native transient behavior are checked. Full thermo-optic modulation
sidebands and HB thermal feedback are not independently verified here.

A fresh CPU process with continuous resistance/temperature coefficient and
phase-efficiency/bias parameters supplied dynamically measured:

| Core | Lowering F/Q / Jacobian | XLA compile F/Q / Jacobian | Warm F/Q / Jacobian |
| --- | --- | --- | --- |
| JouleHeater (3 complex variables) | 0.009 / 0.026 s | 0.024 / 0.056 s | 3.95 / 5.30 µs |
| ThermalPhaseShifter (5 complex variables) | 0.011 / 0.020 s | 0.029 / 0.045 s | 2.83 / 2.80 µs |

Jacobians are full real-block derivatives; warm figures average 100 synchronized
calls. These local timings exclude circuit assembly and are not portable
performance guarantees.

The loaded example contains eight complex unknowns (sixteen real equations); a
fresh dense DC solve took 0.586 s on the same CPU. Heating/cooling step checks
explicitly select `BDF2VectorizedTransientSolver`, with 10 µs and 5 µs steps for
a 1 ms thermal time constant. An instantaneous change in an ideal voltage source
makes the previous algebraic initial state inconsistent; the repository's default
trapezoidal integrator can then ring between source voltages and corrupt squared
Joule power. BDF2 provides a damped algebraic startup. A resolved finite-rise drive
is another appropriate approach; no model adjustment hides that numerical effect.
