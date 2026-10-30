# Dynamic comparator with hysteresis

`circulax.components.comparator.Comparator` uses differential ports
`out_p`, `out_m`, `in_p`, `in_m` and one normalized internal state `x`.
It models finite output levels, output resistance, finite response time
and optional regenerative hysteresis, using native JAX current/charge equations.

| Parameter | Meaning | Default |
|---|---|---|
| `Vlow`, `Vhigh` | Driver levels relative to `out_m`, V | 0, 5 |
| `Vth` | Differential-input threshold center, V | 0 |
| `Vh` | Positive feedback voltage, V | 0 |
| `Vsmooth` | Input transition smoothing width, V | 1 mV |
| `Tau` | Response time constant, s | 1 ns |
| `Rout` | Thevenin output resistance, ohms | 50 |

Use `Tau > 0`, `Vsmooth > 0`, `Rout > 0`, `Vh >= 0`, `Vhigh > Vlow`.
The governing equation is

`Tau*dx/dt = tanh((Vin_p - Vin_m - Vth + Vh*x)/Vsmooth) - x`.

The output driver is `(Vhigh+Vlow)/2 + (Vhigh-Vlow)*x/2`.
For initial `x` in [-1,1], the continuous dynamics preserve those bounds.
An out-of-range initial state moves inward. Inputs have infinite impedance;
the output is floating with equal opposite terminal currents. Supply
currents, common-mode limits, transistor-level saturation recovery and noise
are outside this behavioral model. Loads attenuate the terminal levels by
`Rload/(Rload+Rout)`. Capacitive loads can temporarily move terminal voltages
outside the driver limits.

Without hysteresis (`Vh=0`), the DC driver follows the input tanh and has a
single-pole response with time constant `Tau`. Around bias `v`, the unloaded
small-signal gain is

`((Vhigh-Vlow)/(2*Vsmooth))*sech((v-Vth)/Vsmooth)^2/(1+s*Tau)`.

True bistability appears when `Vh > Vsmooth`. The quasi-static switching
thresholds are `Vth +/- h`, where

`z=sqrt(1-Vsmooth/Vh)` and `h=Vh*z-Vsmooth*atanh(z)`.

Thus `Vh` approaches the half hysteresis width for small smoothing. It is
not exactly the threshold offset at finite `Vsmooth`. Between the thresholds,
the state remembers its branch. Finite ramp speed adds dynamic switching
delay, especially near a threshold. If `Vh <= Vsmooth`, the transfer remains
single-valued and does not provide useful static memory.

A complete setup with explicit initial-branch selection:

```python
import jax.numpy as jnp
from circulax import compile_circuit
from circulax.components.comparator import Comparator
from circulax.components.electronic import Resistor, VoltageSource

net = {
    "instances": {
        "C": {"component": "comparator", "settings": {
            "Vh": .1, "Vsmooth": .002, "Vlow": -1., "Vhigh": 1.}},
        "V": {"component": "voltage", "settings": {"V": 0.}},
        "R": {"component": "resistor", "settings": {"R": 1000.}},
    },
    "connections": {
        "V,p1": "C,in_p", "V,p2": "GND,p1", "C,in_m": "GND,p1",
        "C,out_p": "R,p1", "C,out_m": "GND,p1", "R,p2": "GND,p1",
    },
    "ports": {"output": "C,out_p"},
}
circuit = compile_circuit(net, {
    "comparator": Comparator, "voltage": VoltageSource, "resistor": Resistor})
guess = jnp.zeros(circuit.sys_size).at[circuit.port_map["C,x"]].set(1.)
y_high = circuit.dc(guess)
# Set x=-1 in the guess for the low branch. For transients pass the selected
# operating point as y0, or begin at an input outside the hysteresis window.
```

There is no hidden Python memory. The internal DAE state carries history,
so JIT, batching, differentiation and independent circuit instances remain
compatible. At a perfectly symmetric input/initial state, a DC solve can
return the unstable midpoint: explicitly choose the branch as above or
use prior bias continuation. Derivatives of the constitutive equations
remain finite; sensitivity of an operating point can diverge at a
bifurcation, as expected for regenerative switching.

Tests cover loaded analytical DC/AC at three biases, a finite-response
step, rising/falling thresholds on an actual slow-ramp circuit, different
outputs at the same input during that ramp, explicit DC branch selection,
output limits, JIT/vmap and parameter derivatives. No direct OSDI catalog
counterpart is claimed. DC, AC, transient and HB use the same native F/Q core.

For hysteretic switching, resolve the regenerative timescale
`Tau/(Vh/Vsmooth-1)` near the midpoint as well as the input ramp. The circuit
threshold test uses `Vh=.1 V`, `Vsmooth=.01 V`, `Tau=1 us` and a 20 ns step;
coarse fixed steps can fail a nonlinear solve or skip the switching dynamics.

On this CPU with dynamic parameters (JAX 0.7.2), F/Q trace/compile takes
0.013/0.020 s, Jacobian trace/compile 0.017/0.026 s, and synchronized warm
evaluations are about 13–15 us. Timings include dispatch overhead.
