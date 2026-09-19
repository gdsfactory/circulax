# S-Parameters and SAX Dispatch

Circulax and [SAX](https://github.com/flaport/sax) solve overlapping problems from
opposite ends. SAX composes S-matrices: fast, but only for circuits that *are* an
S-matrix — linear, memoryless, frequency-domain. Circulax solves a DAE system, which
covers nonlinear devices, sources and time-domain analysis, and covers linear photonic
circuits too, by stamping each S-matrix as an admittance and letting the global sparse
solve do the composition.

That overlap used to mean choosing a tool up front, and switching if the circuit grew a
diode. `Circuit.sdict()` removes the choice: it always returns S-parameters, and picks
the backend for you.

```python
from sax.models import coupler, straight

from circulax import compile_circuit

circuit = compile_circuit(mzi_netlist, {"coupler": coupler, "straight": straight}, is_complex=True)

s = circuit.sdict(wl=1.55)  # SAX composes this one
transmission = abs(s[("out0", "in0")]) ** 2
```

## When SAX is used

Dispatch happens when all three hold:

1. **Every instantiated model is a SAX model** — a callable with all-defaulted
   parameters returning an S-dict (the same check that lets you pass raw SAX models to
   `compile_circuit`). One nonlinear or stateful component anywhere makes the circuit a
   DAE system with no S-parameter representation, so the whole netlist falls back.
   An unused entry in the models mapping is not a blocker; only what is instantiated
   counts.
2. **No `ground` / `GND` instance** — a netlist with one is a driven test bench rather
   than a scattering object.
3. **External `ports` are declared** — SAX composes an S-matrix *between ports*, so
   without them there is nothing to return.

`Circuit.sax_dispatch` reports the verdict, and says what cost you the fast path:

```python
>>> circuit.sax_dispatch
SaxDispatch(dispatchable=False, reasons=("Instance 'D1' uses model 'diode', which is not a SAX model.",))
```

## Both paths give the same matrix

The fallback computes S-parameters from the nodal system via the small-signal sweep at
`z0=1` — the normalisation `sax_component` stamps admittances with — evaluated at
`f = 0`, since a scattering network has no electrical storage and its S-matrix therefore
does not depend on the small-signal frequency. Optical dispersion enters through model
parameters (`wl`) in both paths, which is why they agree to solver tolerance;
`tests/test_sax_dispatch.py` asserts it across the band.

So `backend` is a performance and debugging knob, not a semantic one:

```python
circuit.sdict(wl=1.55)                    # auto: SAX when eligible (default)
circuit.sdict(wl=1.55, backend="sax")     # force SAX; raises if ineligible
circuit.sdict(wl=1.55, backend="nodal")   # force the nodal solve — use to cross-check
```

## What it buys

On a cascade of MZIs built from `sax.models`, once both paths are JIT-compiled, per
wavelength point:

| Circuit | SAX | Nodal solve |
|---|---|---|
| 1 MZI (4 instances) | ~2 µs | ~210 µs |
| 4 MZIs (13 instances) | ~4 µs | ~150 µs |
| 16 MZIs (49 instances) | ~12 µs | ~1.4 ms |

Roughly two orders of magnitude, and the SAX path also broadcasts over a wavelength
array in a single call where the nodal path needs one solve per point, and skips the
~1 s XLA compile the nodal path pays per call shape.

Both paths stay differentiable — `jax.grad` through `sdict()` works either way, so
inverse-design code does not need to know which ran.

## Dense form

`Circuit.smatrix()` takes the same arguments and returns `(s_matrix, port_order)`, with
the matrix indexed `[..., out, in]`.
