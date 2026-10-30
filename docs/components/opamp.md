# Generic nonideal operational amplifier

`circulax.components.opamp.OpAmp` has ports `out_p`, `out_m`, `in_p`,
`in_m`, matching `IdealOpAmp`. It adds one internal state, `v_drive`,
representing the differential Thevenin driver voltage behind `Rout`.

| Parameter | Meaning | Default |
|---|---|---|
| `A` | DC open-loop gain | 100000 |
| `GBW` | Gain-bandwidth product, Hz | 1 MHz |
| `Rout` | Output resistance, ohms | 50 |
| `Vmin`, `Vmax` | Driver limits relative to `out_m`, V | -15, +15 |
| `SR` | Maximum internal slew rate, V/s | 1e6 |
| `RailTau` | Response time of dynamic rail constraint, s | 1e-7 |

Use positive `A`, `GBW`, `Rout`, `SR`, `RailTau`, with `Vmin < Vmax`.
The inputs have infinite impedance. The output is floating: the current
entering `out_p` is `((Vout_p - Vout_m) - v_drive)/Rout`, with equal
opposite current entering `out_m`.

The dominant time constant is `tau = A/(2*pi*GBW)`. Away from limits,
the rate is `SR*tanh((A*(Vin_p - Vin_m) - v_drive)/(tau*SR))`.
Its linearization at zero signal gives the standard open-loop transfer
`A/(1+s*tau)`. Large errors limit the derivative to `SR`.

A dynamic rail constraint multiplies the requested rate by a smooth
headroom taper, `tanh(headroom/(max(abs(rate),1)*RailTau))`, where headroom
is measured toward the rail in the requested direction. Outside the rail,
a separate linear restoring term returns the state toward the nearest rail
while retaining any inward requested motion. This also rules out spurious
out-of-range DC equilibria when the input error happens to be zero.
The taper approaches one away from the rails and zero at a rail. Its
transition width adapts to the requested rate, preserving small-signal
bandwidth even when the slew-rate ceiling is very high.
The driver itself is the integrated state, so prolonged saturation does
not accumulate a hidden voltage that delays recovery. The rail constraint
is piecewise differentiable; JAX derivatives remain finite at its transitions.

Rails are parameter settings; supply currents, input common-mode limits,
input bias currents, extra poles and detailed overload recovery are not
modeled. Reactive loads can briefly pull the external terminal voltage
beyond the internal driver rails through `Rout`. This abstraction provides
bandwidth, loading, slew and saturation effects with one dynamic state.

A unity-gain follower with a 1 kohm load:

```python
from circulax import compile_circuit
from circulax.components.electronic import Resistor, VoltageSource
from circulax.components.opamp import OpAmp

net = {
    "instances": {
        "A": {"component": "amp", "settings": {"GBW": 1e6}},
        "V": {"component": "voltage", "settings": {"V": 0.1}},
        "R": {"component": "resistor", "settings": {"R": 1000.}},
    },
    "connections": {
        "V,p1": "A,in_p", "V,p2": "GND,p1",
        "A,in_m": "A,out_p", "A,out_p": "R,p1",
        "A,out_m": "GND,p1", "R,p2": "GND,p1",
    },
    "ports": {"output": "A,out_p"},
}
circuit = compile_circuit(net, {
    "amp": OpAmp, "voltage": VoltageSource, "resistor": Resistor})
y = circuit.dc()
output = circuit.port(y, "output")
```

For load attenuation `alpha = Rload/(Rload+Rout)`, this follower has
transfer `alpha*A/(1+alpha*A+s*tau)` in its linear regime. Tests check
that DC/AC reference, the analytical closed-loop step, slew-limited
transient slope, rail saturation and recovery in an actual follower,
JIT/vmap and parameter/voltage derivatives. No direct VA/OSDI counterpart
is claimed. DC, AC, transient and HB share the native current/charge DAE;
no runtime compiler or extra dependencies are required.

On this CPU with dynamic parameters (JAX 0.7.2), F/Q trace/compile takes
0.010/0.027 s, Jacobian trace/compile 0.027/0.042 s, and synchronized warm
evaluations are about 12–13 us. Timings include dispatch overhead.
