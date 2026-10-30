# MOSVAR 1.3

`circulax.components.mosvar.MOSVAR` implements the deterministic MOSVAR 1.3
MOS-varactor equations as ordinary JAX F/Q contributions. It includes the finite
salicide/poly/substrate/end resistances, voltage-dependent access conductance,
fringe and terminal charges, recombination/generation state and gate tunneling
branches. It requires neither OSDI nor bosdi at runtime.

MOSVAR was developed by **Arizona State University, Freescale Semiconductor and
the Silicon Integration Initiative**. The original notices and redistribution
terms are preserved in the generated component and vendored source. The reference
is [VA-Models commit 6099b4e12e6f66ff530f59b8d01fb909094be85a](https://github.com/dwarning/VA-Models/tree/6099b4e12e6f66ff530f59b8d01fb909094be85a/code/MOSVAR/vacode).

```python
from circulax.components.mosvar import MOSVAR

model = MOSVAR(W=3e-6, L=2e-6, TOXO=2e-9, TAU=.1)
```

Ports, in source order, are `g` (gate), `bi` (internal substrate tap exposed by the
reference) and `b` (outer substrate). Connect `bi` and `b` together for an ordinary
two-terminal device, while retaining the internal substrate branch to `ci`.
Finite gate/substrate resistance is retained with **SWRES fixed at 1**; the
constructor does not accept `SWRES`. LEVEL is fixed at 1000. These topology
choices avoid applying conditional node-collapse hints to finite resistances.

The local DAE contains six auxiliary variables:

| State | Meaning |
|---|---|
| `v_gii`, `v_gi` | Gate voltages after salicide and poly resistance |
| `v_ci` | Intrinsic substrate voltage |
| `v_n` | Inversion-charge relaxation variable; stored quantity is `TAU*v_n` |
| `i_b_rgs`, `i_s_rgs` | Algebraic branch-current observation variables retained from the reference DAE |

For direct `solver_call` evaluation, the vector order is
`g,bi,b,v_gii,v_gi,v_ci,v_n,i_b_rgs,i_s_rgs`. The auxiliary electrical voltages
participate in KCL, whereas `v_n` and the branch-current rows are internal
constraints. Circulax allocates and solves these states automatically.

The source's ordinary parameter names/defaults are retained, including `TYPE`
and `TYPEP` (±1), `QMC`, `SWIGATE`, `W,L,TOXO` (m), `NSUBO,NPO` (m⁻³),
`TAU` (s), `DTA` (K), `TR` (°C), `RPV`, `RSHG`, `RSHS`, `REND`, fringe-charge
and leakage parameters. See the vendored `mosvar.va` for all parameter units
and reference bounds. Ambient temperature is `_temperature` (K, default 300.15)
and parallel multiplicity is `_mfactor` (default 1). Use `DTA` for the source's
`TRISE` alias. Physically meaningful parameter domains should satisfy the source
bounds; simulator warning messages/operating-point metadata are not implemented.

The `QMC` switch changes the charge equations and `SWIGATE` enables actual
tunneling contributions when leakage coefficients are nonzero. Both are tested
against independent OSDI evaluations. Noise-only `RACNOISE` does not affect
deterministic F/Q. Noise analysis and the SWRES=0 collapsed topology are outside
this implementation.

All continuous parameters are JAX inputs, including parameters used during
initialization. A width derivative of intrinsic charge is checked against finite
differences. Discrete mode switches are selectable but are not differentiable
choices. The emitter bounds inactive exponential/log/square-root/division
expressions to keep JAX evaluation finite; numerical parity is verified over the
tested physical operating range rather than claimed for unphysical extremes.

## Circuit use

```python
import jax.numpy as jnp
from circulax import compile_circuit
from circulax.components.electronic import VoltageSource
from circulax.components.mosvar import MOSVAR

net = {
    "instances": {
        "M": {"component": "MOSVAR"},
        "V": {"component": "V", "settings": {"V": .5}},
        "GND": {"component": "ground"},
    },
    "connections": {
        "V,p1": "M,g",
        "GND,p1": ("V,p2", "M,bi", "M,b"),
    },
    "ports": {"gate": "M,g"},
}
circuit = compile_circuit(net, {"MOSVAR": MOSVAR, "V": VoltageSource})
y_dc = circuit.dc()
ac = circuit.ac_response(freqs=jnp.array([1e3, 1e6]), sources={"V": 1.})
```

The MOSVAR core retains its own coupled network to preserve consistent dynamic
charge calculations. It can itself be included as a leaf inside larger recursive
subcircuits. Its internal auxiliary voltage states must not be substituted by
external terminal voltages, since doing so removes the intended parasitic poles.

## Verification, regeneration and compile cost

Run `.venv/bin/python -m pytest tests/test_mosvar.py -q`. The 20 tests include
finite eager/JIT evaluation and differentiation, electrical KCL/charge
conservation, active QMC and leakage modes, width gradients, time-varying
`F + dQ/dy * dy/dt`, actual circuit DC/AC solves, and independent OSDI circuit
input-admittance comparison.

With `openvaf-r`, `osdi_loader` and `osdi_jax`, optional reference tests compile
the vendored sources in a temporary directory. They compare F/Q and full G/C
matrices across four internal-state bias vectors and seven parameter sets,
including both substrate/poly polarities, quantum correction disabled,
nonuniform doping, geometry, temperature, multiplicity and gate leakage enabled.
Dedicated leakage tests use a bias with no parasitic resistor current so tiny
tunnel currents remain measurable. Optional reference cases skip when the
required tools are absent. Full solved transient-trajectory/HB parity has not
been measured.

The reproducible generation script is
`tests/data/va_models/mosvar/regenerate.py`; it records its validated bosdi source
revision. It compiles unoptimized MIR, retains all nodes, specializes topology,
normalizes MIR integer/boolean casts, compacts initialization-cache slots, then
emits ordinary Circulax components. Run it with that bosdi checkout on
`PYTHONPATH`, followed by `ruff check --fix circulax/components/mosvar.py` and
`ruff format circulax/components/mosvar.py`.

On the development CPU with JAX 0.7.2 and Python 3.12.14, the default nine-variable
model took **0.95 s to trace and 0.67 s to compile F/Q**. Its voltage/state Jacobian
took **3.02 s to trace and 3.97 s to compile**. Warm calls including dispatch and
synchronization averaged 142 µs for F/Q and 474 µs for the Jacobian (100 repeats).
These are component measurements, not complete circuit solve timings.

```sh
.venv/bin/python -m benchmarks.sample_components circulax.components.mosvar MOSVAR \
  --bias '[0.5,0,0,0.49,0.48,0.02,-0.005,0.02,0]'
```
