## OSDI Components (Verilog-A models)

OSDI (Open Source Device Interface) lets you drop compiled Verilog-A device models directly into a circulax circuit, without rewriting the physics in Python. This is useful when an accurate vendor model already exists in Verilog-A form — BJTs, MOSFETs, diodes, ferroelectrics, and so on.

!!! note "Optional dependency"
    OSDI support requires the **bosdi** package. If it is not installed, circulax loads normally but raises an `ImportError` when you try to use an OSDI component. See [Installation](#installation) below.

---

## How it works

OSDI models are `.osdi` binary files produced by compiling a Verilog-A source (`.va`) with **OpenVAF**. At runtime, circulax loads the binary, calls into it to evaluate currents and Jacobians, and slots the result into the standard Newton-Raphson assembly loop — the same path used by Python-defined components.

```
Verilog-A (.va)  →  OpenVAF compiler  →  .osdi binary  →  osdi_component()  →  compile_netlist()
```

---

## Installation

### OpenVAF (compiler)

Pre-compiled OpenVAF binaries for Linux, macOS, and Windows are available at:

> **<https://fides.fe.uni-lj.si/openvaf/download/>**

The source and build instructions are at:

> **<https://github.com/OpenVAF/OpenVAF-Reloaded>**

Compile a `.va` file to `.osdi` with:

```bash
openvaf resistor.va          # produces resistor.osdi
```

### bosdi

`bosdi` is the Python bridge between circulax and OSDI binaries. Install it (or add its `src/` directory to `PYTHONPATH`) before using OSDI components. bosdi is not available on all platforms (e.g. Windows).

---

## Usage

### 1. Create a descriptor with `osdi_component()`

```python
from circulax import osdi_component

OsdiResistor = osdi_component(
    osdi_path="/path/to/resistor.osdi",
    ports=("A", "B"),
    param_names=("R", "m"),
    default_params={"R": 1000.0, "m": 1.0},
)
```

| Argument | Description |
|---|---|
| `osdi_path` | Absolute path to the compiled `.osdi` binary |
| `ports` | Port names in the same order as the Verilog-A terminals |
| `param_names` | Parameter names in the order the OSDI model expects |
| `default_params` | Default parameter values (used when a netlist instance omits a setting) |

The returned object (`OsdiResistor` above) acts as a component class in `compile_netlist` — pass it in the `models` map alongside ordinary Python components.

### 2. Build a netlist

```python
netlist = {
    "instances": {
        "R1": {"component": "osdi_res", "settings": {"R": 470.0, "m": 1.0}},
        "V1": {"component": "vsrc",     "settings": {"dc": 1.0}},
    },
    "connections": {
        "R1,A":  "V1,p",
        "R1,B":  "GND,p1",
        "V1,n":  "GND,p1",
    },
    "ports": {"in": "V1,p"},
}
```

### 3. Compile and solve

```python
from circulax import compile_circuit
from circulax.components.electronic import VoltageSource

models = {
    "osdi_res": OsdiResistor,
    "vsrc":     VoltageSource,
}

circuit = compile_circuit(netlist, models)
dc = circuit.solve_dc()
```

OSDI components participate in transient simulation the same way:

```python
from circulax.solvers import setup_transient
import diffrax, jax.numpy as jnp

term   = setup_transient(circuit)
sol    = diffrax.diffeqsolve(term, ...)
```

---

## Limitations

- **Stateful models** (`num_states > 0`) are not yet supported. Trying to load one raises `NotImplementedError`.
- **Windows** is not supported by bosdi.
- OSDI components do not yet participate in `jax.grad` through the model evaluation itself (Jacobians come from the OSDI binary analytically, not via JAX autodiff).
