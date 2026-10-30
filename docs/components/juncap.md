# JUNCAP200

`circulax.components.juncap.JUNCAP200` implements the JUNCAP200 200.6.2
(2021) junction current and terminal charge equations as native JAX. It has
anode `A` and cathode `K`, with no internal unknowns:

```python
from circulax.components.juncap import JUNCAP200

models = {"junction": JUNCAP200}
# Netlist instance settings retain the reference VA parameter names.
settings = {"AB": 1e-12, "LS": 1e-6, "LG": 1e-6, "TYPE": 1}
```

The bottom, STI-edge and gate-edge contributions retain their individual
geometry, saturation-current, depletion-charge, generation/recombination,
tunneling and breakdown parameters. Both the full model (`SWJUNEXP=0`,
default) and express model (`SWJUNEXP=1`) are included. `TYPE=-1` reverses
polarity, `MULT` scales parallel devices, and `TRISE` and `DTA` offset the
ambient `_temperature` in kelvin (default 300.15 K). The reference `LEVEL`
is fixed at 200. Use `TRISE`, rather than the VA alias `DTEMP`. `_mfactor`
corresponds to the simulator multiplier and defaults to 1.

This is a nonlinear time-domain component: `F` contains current and `Q`
contains terminal charge. DC, linearized AC, transient and harmonic-balance
analyses can use those same equations. Shot-noise sources and operating-point
metadata are excluded. No OSDI library, bosdi installation, or compiler is
needed to run this component. The coupled junction equations remain together
because introducing separate subcircuit unknowns does not benefit this
two-terminal model.

## Reference and regeneration

The retained VA source and includes under `tests/data/va_models/juncap/` come
from [VA-Models](https://github.com/dwarning/VA-Models/tree/6099b4e12e6f66ff530f59b8d01fb909094be85a/code/psp103/vacode),
commit `6099b4e12e6f66ff530f59b8d01fb909094be85a`. The source copyright notices
are preserved. `regenerate.py` lowers OpenVAF's **unoptimized** MIR using
bosdi, fixes only `LEVEL=200`, collapses reference nodes, and emits an ordinary
Circulax `@component`; the optional bosdi analytical-Jacobian wrapper is
removed so JAX differentiates the emitted equations. The generator compacts
init-cache references before emitting setup and evaluation to avoid an index
mismatch in older ordinary-component emitters. Generation was checked
with bosdi source commit `159907ddf29a0225614cf1b7999214401b0b5a2d`.

The unoptimized route matters: older optimized-MIR lowering mishandled nested
initialization branches. There is no runtime lowering step. The checked-in
Python file is the artifact used by simulations.

## Verification

`tests/test_juncap.py` verifies terminal-current and charge conservation,
nonzero default behavior, polarity reversal, parallel scaling, a geometry
parameter derivative, and netlist compilation. Optional tests compile the
pinned source using `openvaf-r`, then independently compare **both ports** of
`F`, `Q`, `dF/dV`, and `dQ/dV` against `osdi_jax.osdi_eval`. Cases cover
full/express branches, both polarities, altered temperature, geometry and
multiplicity, over -2 to 0.7 V. A prescribed sinusoidal voltage also compares
the total instantaneous current `F + (dQ/dV) dV/dt` against OSDI. These are
constitutive-equation checks; full circuit transient integration and parameter
sweeps beyond these cases are not claimed verified.

The shared benchmark with parameters traced as arguments reported `F,Q`:
1.02 s tracing, 2.00 s compilation, 133 microseconds warm; Jacobian: 1.57 s
tracing, 4.57 s compilation, 206 microseconds warm.
The initial CPU measurement with parameters closed over (float64) gave about
1.68 s cold JIT including tracing/compilation and 20 microseconds per warmed
single-device `F,Q` call with closed-over default parameters. This is a local
measurement, not a performance guarantee. Differentiating with respect to
`AB` is supported through initialization; a finite-difference test checks it.
Use `benchmarks/sample_components.py` for reproducible `F,Q` and Jacobian
performance measurements.
