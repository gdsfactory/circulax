# Generic dynamic PMOS

`circulax.components.mosfet.PMOSDynamic` complements the existing
`NMOSDynamic` with the same square-law current and piecewise Meyer terminal
charge equations. It has drain, gate and source ports `d,g,s` and returns
conservative currents and charges using
`PMOS(v) = -NMOS(-v)` with equal positive physical parameter magnitudes.

```python
from circulax.components.mosfet import PMOSDynamic

pmos = PMOSDynamic(Vth=-1., Kp=1e-5, W=20e-6, L=1e-6,
                   Cox=1e-3, Cgd_ov=1e-15, Cgs_ov=1e-15)
f, q = pmos(d=0, g=0, s=3)
```

Default parameters match the existing static PMOS: `Kp=1e-5 A/V²`,
`W=20 µm`, `L=1 µm`, `Vth=-1 V` and `lam=0 V⁻¹`. Vth is converted to its
absolute value for the NMOS equations. `Cox` is F/m² and the two overlap
coefficients are F. Nonnegative overlap coefficients give positive linear
overlap capacitances, including when the channel is off. Parameters remain
JAX-traceable and differentiable away from piecewise boundaries.

This is a generic three-terminal teaching/sample model. It inherits the existing
NMOSDynamic charge partition and charge discontinuities at region boundaries.
Its intended operating range is forward PMOS source/drain operation (`Vs >= Vd`)
with positive Kp/W/L and nonnegative capacitance coefficients. It has no bulk
terminal, body junctions, subthreshold conduction, self-heating or advanced
charge-recovery model. It has no independent VA/OSDI equivalent; verification
uses exact polarity mirroring of the existing NMOS physics and analytical
overlap-capacitance checks. EKV26 is a separate four-terminal reference model
with more detailed physics and its own upstream PMOS overlap convention.

Run `.venv/bin/python -m pytest tests/test_pmos_dynamic.py -q`. Seven tests cover
cutoff/linear/saturation mirroring, KCL and charge conservation, finite Jacobians,
positive overlap capacitance, width gradients, JIT compilation and a compiled
CMOS inverter DC/AC solve at both logic levels.

A single-component CPU benchmark with JAX 0.7.2/Python 3.12.14 measured 10 ms
tracing and 27 ms compilation for F/Q, and 27 ms tracing and 38 ms compilation
for its voltage Jacobian. Warm dispatch/synchronized calls averaged 16 µs and
13 µs respectively (100 repeats). Reproduce with:

```sh
.venv/bin/python -m benchmarks.sample_components circulax.components.mosfet PMOSDynamic \
  --bias '[0,0,3]'
```
