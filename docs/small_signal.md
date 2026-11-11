# Small-signal analysis

`Circuit.small_signal()` linearizes the physical circuit at a fixed DC operating
point and returns a reusable, callable `SmallSignalModel`. It uses the circuit's
actual sources and loads, without adding scattering terminations.

```python
ss = circuit.small_signal(params={"Vdrive.dc": 0.5})
response = ss(freqs=freqs, sources={"Vdrive": 1j})
# Equivalent:
response = ss.response(freqs=freqs, sources={"Vdrive": 1j})
```

Parameters, including optical carrier wavelength, are fixed when the model is
constructed. Frequencies and independent-source amplitudes are inputs to each
response calculation. A supplied `y_dc` skips the DC solve and must correspond
to the same parameter values, using the usual DC coordinate layout. Construct
a new model after changing component parameters or bias. As with `dc()`, callers
are responsible for establishing a valid operating point; `dc_checked()` can be
used to obtain and inspect the DC convergence flag before supplying `y_dc`.

## Electrical RC example

```python
import jax.numpy as jnp
from circulax import compile_circuit
from circulax.components.electronic import (
    Capacitor, Resistor, WaveformVoltageSource,
)

circuit = compile_circuit(
    {
        "instances": {
            "V": {"component": "voltage", "settings": {"dc": 2.0}},
            "R": {"component": "resistor", "settings": {"R": 1000.0}},
            "C": {"component": "capacitor", "settings": {"C": 1e-6}},
            "GND": {"component": "ground"},
        },
        "connections": {
            "V,p1": "R,p1",
            "R,p2": "C,p1",
            "GND,p1": ("V,p2", "C,p2"),
        },
        "ports": {"out": "C,p1"},
    },
    {"voltage": WaveformVoltageSource, "resistor": Resistor, "capacitor": Capacitor},
)

ss = circuit.small_signal()
freqs = jnp.array([0.0, 100.0, 1000.0])
response = ss(freqs=freqs, sources={"V": 1j})
vout = response.voltage("out")  # 1j / (1 + 2j*pi*freqs*1e-3)
vr = response.voltage("R,p1", "R,p2")
```

Source keys identify independent-source **instances**, not node names. Omitted
sources have zero AC perturbation while retaining their DC settings. Complex
source amplitudes are peak phasors, in volts for voltage sources and amperes for
current sources. Current-source signs follow their existing component stamps.
A drive of `1j` is a 90-degree RF time phase, not an imaginary electrical voltage
coordinate. Multiple sources superpose within the same fixed linearization.

## Optical power modulation

An optical envelope is `E = x + j*y`. Under an RF perturbation, both physical
coordinates have their own complex peak phasors:

```text
x(t) = x_dc + Re(dx * exp(+j*omega*t))
y(t) = y_dc + Re(dy * exp(+j*omega*t))
```

The solver keeps both phasors. They cannot be recovered in general from a single
`dx + j*dy` number. For a power-normalized field in sqrt(W), the physical power
is `P = |E|**2`, and the first-order power response is:

```text
dP = 2 * (x_dc*dx + y_dc*dy)
```

For example, define a low-pass electro-optic probe:

```python
from circulax.components.base_component import component
from circulax.components.photonic import OpticalSource

@component(ports=("ein", "out", "voltage"), states=("u", "iout"), holomorphic=False)
def electro_optic_probe(signals, tau=1e-10):
    target = signals.ein * ((1 + 1j) + (2 + 3j)*jnp.real(signals.u))
    return {
        "ein": 0j, "voltage": 0j, "out": signals.iout,
        "iout": signals.out - target,
        "u": signals.u - signals.voltage,
    }, {"u": tau*signals.u}

optical_circuit = compile_circuit(
    {
        "instances": {
            "laser": {"component": "laser", "settings": {"power": 1e-9}},
            "V": {"component": "voltage"},
            "probe": {"component": "probe"},
            "load": {"component": "resistor", "settings": {"R": 1.0}},
            "GND": {"component": "ground"},
        },
        "connections": {
            "laser,p1": "probe,ein", "V,p1": "probe,voltage",
            "probe,out": "load,p1",
            "GND,p1": ("laser,p2", "V,p2", "load,p2"),
        },
        "ports": {"out": "probe,out"},
    },
    {"laser": OpticalSource, "voltage": WaveformVoltageSource,
     "probe": electro_optic_probe, "resistor": Resistor},
    is_complex=True,
)

ss = optical_circuit.small_signal()
response = ss(freqs=freqs, sources={"V": 1.0})
dP = response.power("out", field_unit="sqrt(W)")
# dP = 1e-8 / (1 + 2j*pi*freqs*1e-10), in watts
magnitude = jnp.abs(dP)
phase = jnp.angle(dP)
```

`power()` returns complex power **modulation**, not the DC power or total power.
Its magnitude is the peak sinusoidal modulation; RMS modulation is that magnitude
divided by sqrt(2). The DC formula and zero-frequency small-signal derivative
have no sinusoidal RMS conversion.

`field_unit="sqrt(W)"` explicitly declares the selected unknown to be the intended
power-normalized optical field. The compiler currently has no general discipline
metadata and does not infer units from names. In particular, the squared total
field at a mismatched boundary is not an incident or transmitted power wave.
Electrical absorbed-power extraction is not currently implemented.

For advanced use, `response.quadratures("out")` returns `(dx, dy)`, each a complex
RF phasor. `response.sidebands("out")` returns the actual envelope coefficients:

```text
E(t) = E_dc + upper*exp(+j*omega*t) + lower*exp(-j*omega*t)
upper = (dx + j*dy)/2
lower = (conj(dx) + j*conj(dy))/2
```

These are positive/negative **envelope-frequency** coefficients. Calling them the
optical upper/lower sidebands additionally requires a carrier time convention:
for the analytic field `E(t)*exp(+j*omega_c*t)`, they lie at `omega_c +/- omega`.
At zero modulation frequency the two envelope terms coincide.

## Optical excitation

A scalar source phasor always drives the source's x coordinate. To excite both
optical field coordinates independently, use `QuadratureDrive`:

```python
from circulax import QuadratureDrive

response = ss(
    freqs=freqs,
    sources={"laser": QuadratureDrive(x=0.1j, y=0.2)},
)
```

The x and y entries are each complex RF phasors in field units. This explicit
convention differs from the historical holomorphic optical excitation accepted
by `ac_response()`; that existing API retains its behavior and safety checks.

## JAX and supported models

The model and response are JAX pytrees. Arrays have final axes `(frequency,
unknown)` and frequency is in Hz. For complex-mode models, raw `solution` retains
the `[x coordinates, y coordinates]` block layout. `port_map` preserves node and
internal-state aliases. Real models have one coordinate block; their second
quadrature is zero. Additional leading axes come from external `jax.vmap`.

```python
import jax

run = jax.jit(lambda amplitude: ss(freqs=freqs, sources={"V": amplitude}))
responses = jax.vmap(run)(jnp.array([1.0, 1j]))
power = responses.power("out", field_unit="sqrt(W)")
```

Parameter arrays are not implicitly batched by `small_signal()`. Use external
`vmap` around model construction and response evaluation. Native analysis
registrations can be prepared before tracing with
`circuit.prepare_analysis("small_signal")`; this does not solve DC.

Supported physics includes ordinary real and complex F/Q components, fixed
delays, frequency-domain admittances, and real native models. Native conductance
is differentiated using DC registrations while charge derivatives use AC
registrations. Complex-mode native/combined real-only Jacobians are explicitly
unsupported because they do not supply a complete doubled-coordinate system.

In complex mode, frequency-domain admittances are evaluated at both `+f` and
`-f` to distinguish RF phase from optical field rotation. Define the response
on signed frequencies: real electrical models obey `Y(-f) = conj(Y(f))`.
A frequency-independent complex optical admittance instead retains its fixed
field rotation. Both conventions can include an additional temporal response.

The new API preserves non-holomorphic derivatives automatically. There is no
`holomorphic=True` escape hatch. `sp()` remains the scattering analysis with
reference impedances and matching terminations; `small_signal()` retains the
physical source/load circuit and its full internal response. Pole/zero and noise
analysis are not part of this API yet.

## Retained optical wave terminals

`power()` differentiates a caller-declared normalized field intensity. A wave
component's terminal potential includes both incident and returning light; its
squared magnitude is not traveling power. Declared wave-component terminal
names now reject this helper, including electrical terminals of mixed models.
Other aliases still require the caller's explicit normalization/matching
assertion; an alias does not make a reflected potential a traveling field.
Use `Circuit.port_waves` for supported total DC, transient and HB time samples.
AC traveling-power reconstruction remains deferred; field quadratures and
sidebands remain available. The [phase-shifter guide](mixed_electro_optic_power_waves.md)
describes full optical transit, midpoint voltage modulation and electrical Q.
