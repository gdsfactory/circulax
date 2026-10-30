# PNP bipolar transistors

`circulax.components.bipolar` provides `BJT_PNP` and `BJT_PNP_Dynamic`,
with ports `c`, `b`, `e`. Both are the polarity complements of the existing
NPN models: `F_PNP(v) = -F_NPN(-v)` and `Q_PNP(v) = -Q_NPN(-v)`.
Currents are positive entering each terminal. Use positive parameter magnitudes.

```python
from circulax.components.bipolar import BJT_PNP_Dynamic

pnp = BJT_PNP_Dynamic(Is=1e-14, BetaF=100., Cje=2e-12, Tf=1e-9)
current, charge = pnp(c=-2., b=-0.6, e=0.)
```

`BJT_PNP` accepts `Is`, `BetaF`, `BetaR`, `Vt`, matching `BJT_NPN`.
The dynamic model additionally accepts `Cje`, `Cjc`, `Vje`, `Vjc`, `Mje`,
`Mjc`, `Tf`, `Tr`, matching `BJT_NPN_Dynamic`. Units are amperes, volts,
farads and seconds; gains and grading coefficients are dimensionless.

Register either class in the model mapping passed to `compile_circuit` to
use it in a netlist. The dynamic variant supports DC, linearized AC, transient
and HB through the ordinary current/charge interface. No extra internal state
is required. Terminal currents and charges sum to zero. JAX transformations
can differentiate voltages and parameters and batch the solver interface.

These generic Ebers–Moll models inherit the NPN numerical clipping and
junction-charge extrapolation. They do not add Early effect, self-heating,
noise or advanced bipolar physics. Verification uses NPN polarity symmetry,
terminal conservation, AC Jacobians, finite-difference charge derivatives,
JIT/vmap, parameter gradients and a biased common-emitter circuit.
The shared junction-charge continuation also keeps reverse-mode voltage
derivatives finite above the junction potential. A separate regression covers
that forward-bias region for both NPN and PNP models.
