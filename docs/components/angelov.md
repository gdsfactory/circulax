# Angelov

`circulax.components.angelov.Angelov` implements the deterministic standard
`angelov.va` transistor model from the pinned VA-Models catalog. It retains
channel and gate currents, charge/capacitance modes, series RLC branches,
input/output dispersion, the rational channel-delay network, and optional
self-heating. It is separate from `AngelovGaN`.

```python
from circulax.components.angelov import Angelov

models = {"fet": Angelov}
settings = {
    "Idsmod": 3, "Capmod": 2, "Selft": 1,
    "Cgs0": 2e-12, "Cgd0": 1e-12,
    "Ld": 1e-9, "Ls": 2e-9, "Lg": 5e-10,
    "Tau": 1e-10, "Crf": 1e-12, "Crfin": 1e-12, "Rth": 100.,
}
```

Ports are `d`, `g`, `s`. The component has 19 unknowns: three ports, eleven
internal voltages, and five branch currents. The coupled channel, delay and
thermal equations remain in one component. This avoids splitting their
shared nonlinear calculations or duplicating internal model unknowns.
Two noise-only coordinates from the reference are omitted.

The internal order, useful for standalone evaluations and benchmarks, is:

```text
d, g, s,
v_di, v_gi, v_si, v_sii, v_gdi, v_gsi, v_bi, v_rf, v_t, v_xt1, v_xt2,
i_un_xt1_xt2_16, i_un_g_gi_17, i_un_si_sii_18, i_un_sii_s_19, i_un_di_d_20
```

All four `Idsmod` choices, both `Igmod` choices, and all three `Capmod`
choices are implemented. `Capmod=0` uses linear capacitances, `Capmod=1`
retains the reference voltage-dependent `C(V) V` storage, and `Capmod=2`
uses the reference charge-based equations. `Tau` uses the reference finite
network, rather than an ideal time delay. Thus its equations work in DC,
linearized AC, transient and HB analyses. Zero inductances, zero `Tau`, and
zero dispersion capacitors are supported exactly.

## Defaults and supported topology

`Angelov()` uses unchanged source defaults. The reference already defaults
`Rg`, `Rd`, `Rs`, `Ri`, and `Rgd` to 0.05 ohm, `Rc` to 10 kohm, and `Rcin`
to 100 kohm. This implementation supports that finite-resistance topology;
all seven resistances must remain strictly positive. The constructor rejects
zero values explicitly, including concrete JAX-array values. It does not
replace zero resistances with small finite resistors. Zero `Rc`/`Rcin`
short-circuit topology variants are not included in this first implementation.

`Selft=1` includes the thermal RC equation and the source's temperature
feedback. `v_t` is the internal temperature rise. With `Selft=0`, its small
reference shunt is retained. `Temp` and `Tnom` use degrees Celsius;
`_temperature` is ambient kelvin (default 300.15 K), and `Trise` is a kelvin
increment. The constructor preserves `$param_given` behavior: omitted
`Temp` uses ambient plus `Trise`; explicitly supplied `Temp` selects the
specified device temperature. Omitted `Tnom` uses the reference 27 C.
Supplying `Ne` while omitting `Pg` derives the gate coefficient from `Ne`
and thermal voltage. Explicit `Pg` takes precedence. Optional fields use NaN as the "not given" sentinel. Presence is inferred
inside physics on each call, so fitting or circuit parameter updates from
omitted to finite `Temp`, `Tnom`, `Pg`, or `Ne` values take effect naturally.

Noise is fixed off; noise-only parameters do not add stochastic behavior.
The native model has no bosdi, OpenVAF or OSDI runtime dependency.

## Source and regeneration

The source and standard constants/disciplines are retained under
`tests/data/va_models/angelov/`, from
[VA-Models commit 6099b4e12e6f66ff530f59b8d01fb909094be85a](https://github.com/dwarning/VA-Models/tree/6099b4e12e6f66ff530f59b8d01fb909094be85a/code/angelov/vacode).
The model's MIT copyright/license is preserved in the generated Python.
The broad legacy `compact.vams` is replaced by a small compatibility header
containing only the two numerical definitions actually used by this model:
`DEFAULT_TNOM=27` and `SPICE_GMIN=1e-12`.

`regenerate.py` records the generation procedure using OpenVAF-reloaded
`20260616-4-g5ed9e63a` and bosdi source
`159907ddf29a0225614cf1b7999214401b0b5a2d`. It restores unnamed branch-current
identities in unoptimized MIR, preserves parameter presence, normalizes
integer-to-boolean casts, compacts the init cache, disables stale phi-batching
metadata, and uses logical inversion consistently. Four pinned SSA repairs
restore the exact multi-way channel/capacitance switch equations that the
lowerer otherwise reduced to default branches. Their short source-equivalent
formulas live in `channel_helpers.txt`; the constructor boundary is retained
in `constructor.txt`. All physical equations are checked into the Python
module; no generation happens during import or simulation.

## Verification

Sixteen independently compiled OSDI fixture cases cover the full cross product
of four current models and three capacitance models, both gate models,
nonzero polynomial coefficients, thermal feedback, RLC parasitics, delay and
dispersion, both bias polarities, and explicit versus omitted temperature and
gate coefficients. Tests compare every deterministic internal and terminal
row of `F`, `Q`, `dF/dV`, and `dQ/dV`. Additional checks cover eager/JIT
agreement, an `Ipk0` gradient against finite differences, instantaneous
`F + (dQ/dV) dV/dt`, and actual biased-amplifier DC and frequency-domain
circuit solutions. An optional fresh OSDI compilation checks the stored
fixtures and circuit responses. The tested 1 MHz–1 GHz amplifier response
agrees with OSDI to approximately 3e-15 absolute S-parameter error.

Full large-signal switching or pulse-circuit transient integration has not
yet been independently verified. Parameter-domain behavior beyond these
cases remains subject to the reference equations' numerical limitations.

## Performance

A local CPU measurement with the active settings above reported `F,Q`:
0.086 s tracing, 0.129 s compilation, 124 microseconds warm; Jacobian:
0.206 s tracing, 0.307 s compilation, 137 microseconds warm. Reproduce it with:

```bash
.venv/bin/python -m benchmarks.sample_components circulax.components.angelov Angelov \
  --bias '[2,-0.3,0,1.99,-0.31,0.01,0.005,-0.315,-0.32,0.015,0.02,10,0.04,0.04,0.04,-1e-8,0.04,0.04,-0.04]' \
  --settings '{"Selft":1,"Idsmod":3,"Cgs0":2e-12,"Cgd0":1e-12,"Ld":1e-9,"Ls":2e-9,"Lg":5e-10,"Tau":1e-10,"Crf":1e-12,"Crfin":1e-12,"Rth":100}'
```

These development-machine timings are not performance guarantees.
