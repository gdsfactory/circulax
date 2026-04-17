# Circulax

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo-white.svg">
  <img src="docs/images/logo.svg" alt="logo" width="300">
</picture>

**A differentiable circuit simulator built on JAX.**
Define netlists, run transient / DC / AC / harmonic-balance analysis, and differentiate through the solver for gradient-based optimization and inverse design. Circulax aims to be flexible multi-diciplined circuit simulator and offering a similar interface to the linear simulator [SAX](https://github.com/flaport/sax).

**[Read the Documentation here](https://cdaunt.github.io/circulax/)**

## Installation
```sh
pip install circulax
```

## Quickstart

Simulate an underdamped LCR circuit in the time domain:

![LCR transient animation](docs/images/lcr_animation.gif)

```python
import diffrax, jax, jax.numpy as jnp
from circulax import compile_circuit
from circulax.components.electronic import Capacitor, Inductor, Resistor, VoltageSource
from circulax.solvers import setup_transient

jax.config.update("jax_enable_x64", True)

net_dict = {
    "instances": {
        "GND": {"component": "ground"},
        "V1":  {"component": "source_voltage", "settings": {"V": 1.0, "delay": 0.25e-9}},
        "R1":  {"component": "resistor",        "settings": {"R": 10.0}},
        "C1":  {"component": "capacitor",       "settings": {"C": 1e-11}},
        "L1":  {"component": "inductor",        "settings": {"L": 5e-9}},
    },
    "connections": {
        "GND,p1": ("V1,p2", "C1,p2"),
        "V1,p1": "R1,p1",  "R1,p2": "L1,p1",  "L1,p2": "C1,p1",
    },
}

models = {
    "resistor": Resistor, "capacitor": Capacitor,
    "inductor": Inductor, "source_voltage": VoltageSource, "ground": lambda: 0,
}

circuit = compile_circuit(net_dict, models)
y_op    = circuit()
sim     = setup_transient(groups=circuit.groups, linear_strategy=circuit.solver)

sol = sim(
    t0=0.0, t1=3e-9, dt0=3e-12, y0=y_op,
    saveat=diffrax.SaveAt(ts=jnp.linspace(0, 3e-9, 500)),
    max_steps=100_000,
)

v_cap = circuit.get_port_field(sol.ys, "C1,p1")  # capacitor voltage over time
```

## Defining Components

Components are plain Python functions — no boilerplate, no subclassing:

```python
from circulax.components.base_component import component, Signals, States

@component(ports=("p1", "p2"))
def Resistor(signals: Signals, s: States, R: float = 1e3):
    i = (signals.p1 - signals.p2) / R
    return {"p1": i, "p2": -i}, {}          # (currents, charges)

@component(ports=("p1", "p2"))
def Capacitor(signals: Signals, s: States, C: float = 1e-12):
    q = C * (signals.p1 - signals.p2)
    return {}, {"p1": q, "p2": -q}          # dq/dt becomes current automatically
```

Non-linear opto-electronic components are just as simple — the Jacobian is computed automatically via [Automatic Differentiation](https://docs.jax.dev/en/latest/automatic-differentiation.html):

```python
@component(ports=("optical_in", "anode", "cathode"))
def Photodetector(signals: Signals, s: States,
                  responsivity: float = 0.8, dark_current: float = 1e-9):
    optical_power = jnp.abs(signals.optical_in) ** 2           # non-linear
    i_photo = responsivity * optical_power + dark_current
    i_reflect = -0.01 * signals.optical_in                     # small back-reflection
    return {"optical_in": i_reflect, "anode": i_photo, "cathode": -i_photo}, {}
```

Existing [SAX](https://flaport.github.io/sax/) models plug in directly — reuse your photonic PDK as-is:

```python
import sax
from circulax.s_transforms import sax_component

Straight = sax_component(sax.models.straight)   # that's it — ready to simulate
```

## Features

- **Transient** — implicit ODE stepping via [Diffrax](https://docs.kidger.site/diffrax/); handles stiff circuits.
- **DC operating point** — Newton-Raphson root-finding via [Optimistix](https://github.com/patrick-kidger/optimistix).
- **Harmonic Balance** — periodic steady state directly in the frequency domain.
- **AC sweep** — linearise at DC op-point, sweep frequency, return S-parameters.
- **Automatic differentiation** — differentiate through the solver for gradient-based inverse design.
- **Hardware-agnostic** — CPU, GPU, or TPU with no code changes.
- **Mixed-domain** — electronic and photonic circuits in a single netlist.

## Comparison to SPICE

Circulax is a SPICE-like simulator but built with modern tooling so users can easily create their own models in a language they know.

| | SPICE | circulax |
|---|---|---|
| Model definition | Verilog-A / hardcoded C++ | Python functions |
| Derivatives | Hardcoded or compiler-generated | Automatic differentiation |
| Solver | Fixed/heuristic stepping | Adaptive ODE (Diffrax) |
| Hardware | CPU-only | CPU / GPU / TPU |

## Inverse Design via Back-propagation

Because the entire solver is written in JAX, gradients flow end-to-end from a loss function back through the simulation and into component parameters. Use `jax.grad` and standard optimizers to automatically tune circuit designs — the cost is one forward + one backward pass regardless of parameter count.

See the [Inverse Design guide](docs/inverse_design.md) for a comparison with finite differences and worked examples.

---

Copyright © 2026 Chris Daunt — [Apache-2.0](https://github.com/cdaunt/circulax/blob/master/LICENSE)
