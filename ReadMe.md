# Circulax

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo-white.svg">
  <img src="docs/images/logo.svg" alt="logo" width="300">
</picture>

**A differentiable circuit simulator built on JAX.**
Define netlists, run transient / DC / AC / harmonic-balance analysis, and differentiate through the solver for gradient-based optimization and inverse design.

**[Documentation](https://cdaunt.github.io/circulax/)**

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

## Features

- **Transient** — implicit ODE stepping via [Diffrax](https://docs.kidger.site/diffrax/); handles stiff circuits.
- **DC operating point** — Newton-Raphson root-finding via [Optimistix](https://github.com/patrick-kidger/optimistix).
- **Harmonic Balance** — periodic steady state directly in the frequency domain.
- **AC sweep** — linearise at DC op-point, sweep frequency, return S-parameters.
- **Automatic differentiation** — differentiate through the solver for gradient-based inverse design.
- **Hardware-agnostic** — CPU, GPU, or TPU with no code changes.
- **Mixed-domain** — electronic and photonic circuits in a single netlist.

## vs SPICE

| | SPICE | circulax |
|---|---|---|
| Model definition | Verilog-A / hardcoded C++ | Python functions |
| Derivatives | Hardcoded or compiler-generated | Automatic differentiation |
| Solver | Fixed/heuristic stepping | Adaptive ODE (Diffrax) |
| Hardware | CPU-only | CPU / GPU / TPU |

---

Copyright © 2026 Chris Daunt — [Apache-2.0](https://github.com/cdaunt/circulax/blob/master/LICENSE)
