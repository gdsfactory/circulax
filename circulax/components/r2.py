"""Native deterministic R2_CMC 1.0.1 resistor equations.

Adapted from VA-Models commit 6099b4e12e6f66ff530f59b8d01fb909094be85a.
Copyright (c) 2020 Silicon Integration Initiative, Inc.
Licensed under the Educational Community License, Version 2.0:
https://opensource.org/licenses/ECL-2.0
Distributed AS IS, without warranties or conditions of any kind.
See tests/data/va_models/r2_cmc/{LICENSE,NOTICE}.txt for upstream notices.

Geometry uses metres; offsets/field coefficients use upstream micrometre units.
Unspecified ``w``, ``l`` and ``r`` use NaN to preserve $param_given semantics.
The supported domain is strictly positive resistance and effective geometry.
Noise and simulator diagnostic/operating-point metadata are not implemented.
"""

import jax
import jax.numpy as jnp

from circulax.components.base_component import PhysicsReturn, Signals, component

__all__ = ["R2CMC", "R2ETCMC", "r2_thermal_subcircuit"]


def _clip_temperature(value: float, lower: float, upper: float) -> jax.Array:
    # Upstream CLIPB1p0. Bound inactive exponentials to keep AD finite.
    low = lower + jnp.exp(jnp.minimum(value - lower - 1.0, 0.0))
    high = upper - jnp.exp(jnp.minimum(upper - value - 1.0, 0.0))
    return jnp.where(value < lower + 1.0, low, jnp.where(value > upper - 1.0, high, value))


def _geometry(
    *, w: float, l: float, r: float, rsh: float, scale: float, shrink: float, xw: float, xl: float, c1: float, c2: float
) -> tuple[jax.Array, ...]:
    given_w, given_l, given_r = ~jnp.isnan(w), ~jnp.isnan(l), ~jnp.isnan(r)
    w = jnp.where(given_w, w, 1e-6)
    l = jnp.where(given_l, l, 1e-6)
    r = jnp.where(given_r, r, 100.0)
    contact = (c1 + c2) / 2.0
    offset = xl * contact
    factor = scale * (1.0 - 0.01 * shrink) * 1e6
    length, width = l * factor, w * factor
    effective_l, effective_w = length + offset, width + xw
    # l+r (no w): infer width. r (no l): infer length, using specified/default w.
    infer_w = given_l & given_r & ~given_w
    infer_l = given_r & ~given_l
    effective_w = jnp.where(infer_w, rsh * effective_l / r, effective_w)
    width = jnp.where(infer_w, effective_w - xw, width)
    effective_l = jnp.where(infer_l, r * effective_w / rsh, effective_l)
    length = jnp.where(infer_l, effective_l - offset, length)
    resistance = jnp.where(infer_w | infer_l, r, rsh * effective_l / effective_w)
    return length, width, effective_l, effective_w, resistance, contact


def _resistance(
    voltage: float,
    rise: float,
    *,
    w: float,
    l: float,
    r: float,
    rsh: float,
    c1: float,
    c2: float,
    scale: float,
    shrink: float,
    xw: float,
    xl: float,
    dxle: float,
    sw_efgeo: float,
    p2: float,
    q2: float,
    p3: float,
    q3: float,
    temperature: float,
    trise: float,
    tnom: float,
    tminclip: float,
    tmaxclip: float,
    tc1: float,
    tc2: float,
    tc1l: float,
    tc2l: float,
    tc1w: float,
    tc2w: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    length, width, effective_l, effective_w, r0, contact = _geometry(
        w=w, l=l, r=r, rsh=rsh, scale=scale, shrink=shrink, xw=xw, xl=xl, c1=c1, c2=c2
    )
    tc1e = tc1 + contact * tc1l / effective_l + tc1w / effective_w
    tc2e = tc2 + contact * tc2l / effective_l + tc2w / effective_w
    delta = _clip_temperature(temperature + trise + rise - 273.15, tminclip, tmaxclip) - tnom
    tcr = 1.0 + delta * (tc1e + delta * tc2e)
    # Upstream CLIPL0p1(tcr,tcr,0.01).
    tcr = jnp.where(tcr < 0.11, 0.01 + 0.1 * jnp.exp(jnp.minimum(10.0 * (tcr - 0.01) - 1.0, 0.0)), tcr)
    field = voltage / (jnp.where(sw_efgeo != 0, effective_l, length) + dxle)
    q2e, q3e = q2 * field, q3 * jnp.abs(field)
    nonlinear = 1.0 - p2 - p3 + p2 * jnp.sqrt(1.0 + q2e**2) + p3 * (1.0 + q3e**3) ** (1.0 / 3.0)
    return r0 * tcr * nonlinear, length, width


@component(ports=("n1", "n2"))
def R2CMC(
    signals: Signals,
    *,
    w: float = float("nan"),
    l: float = float("nan"),
    r: float = float("nan"),
    rsh: float = 100.0,
    c1: float = 1.0,
    c2: float = 1.0,
    scale: float = 1.0,
    shrink: float = 0.0,
    xw: float = 0.0,
    xl: float = 0.0,
    dxle: float = 0.0,
    sw_efgeo: float = 0.0,
    p2: float = 0.0,
    q2: float = 0.0,
    p3: float = 0.0,
    q3: float = 0.0,
    temperature: float = 300.15,
    trise: float = 0.0,
    tnom: float = 27.0,
    tminclip: float = -100.0,
    tmaxclip: float = 500.0,
    tc1: float = 0.0,
    tc2: float = 0.0,
    tc1l: float = 0.0,
    tc2l: float = 0.0,
    tc1w: float = 0.0,
    tc2w: float = 0.0,
    m: float = 1.0,
) -> PhysicsReturn:
    """Isothermal R2_CMC resistor with geometry, field and temperature dependence.

    ``temperature`` is ambient kelvin; ``tnom`` and clip limits are Celsius.
    ``r`` is per segment and ``m`` is the number of parallel segments.

    Nonlinear resistor and explicit electrothermal composition.

    **Implementation source:** Native JAX adaptation of the Verilog-A model,
    version 1.0.1, from VA-Models commit `6099b4e12e6f66ff530f59b8d01fb909094be85a`.
    [Pinned upstream `r2_cmc.va`](https://github.com/dwarning/VA-Models/blob/6099b4e12e6f66ff530f59b8d01fb909094be85a/code/r2_cmc/vacode/r2_cmc.va).
    Attribution: Silicon Integration Initiative (Educational Community License 2.0).
    The shared equations are in [r2_cmc_body.include](https://github.com/dwarning/VA-Models/blob/6099b4e12e6f66ff530f59b8d01fb909094be85a/code/r2_cmc/vacode/r2_cmc_body.include).
    The retained source and includes are under `tests/data/va_models/r2_cmc/`.

    **Validation reference:** Verilog-A/OSDI comparisons use the same pinned
    [upstream source](https://github.com/dwarning/VA-Models/blob/6099b4e12e6f66ff530f59b8d01fb909094be85a/code/r2_cmc/vacode/r2_cmc.va),
    with comparison cases in `tests/test_r2.py`.
    These are implementation/parity checks, not measured-device calibration.
    Live comparisons require optional compiler/OSDI tools; stored fixtures are
    used where provided. Test coverage and adaptations are described in the model reference.

    Device group:
        Electronic / Compact devices

    Device reference:
        r2.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.r2 import R2CMC

        jax.config.update("jax_enable_x64", True)
        model = R2CMC()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.float64)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(R2CMC.__name__, "ports:", model.ports)
        ```

    """
    voltage = signals.n1 - signals.n2
    rise = 0.0
    resistance, _length, _width = _resistance(
        voltage,
        rise,
        w=w,
        l=l,
        r=r,
        rsh=rsh,
        c1=c1,
        c2=c2,
        scale=scale,
        shrink=shrink,
        xw=xw,
        xl=xl,
        dxle=dxle,
        sw_efgeo=sw_efgeo,
        p2=p2,
        q2=q2,
        p3=p3,
        q3=q3,
        temperature=temperature,
        trise=trise,
        tnom=tnom,
        tminclip=tminclip,
        tmaxclip=tmaxclip,
        tc1=tc1,
        tc2=tc2,
        tc1l=tc1l,
        tc2l=tc2l,
        tc1w=tc1w,
        tc2w=tc2w,
    )
    current = m * voltage / resistance
    return {"n1": current, "n2": -current}, {}


@component(ports=("n1", "n2", "dt"))
def R2ETCMC(
    signals: Signals,
    *,
    w: float = float("nan"),
    l: float = float("nan"),
    r: float = float("nan"),
    rsh: float = 100.0,
    c1: float = 1.0,
    c2: float = 1.0,
    scale: float = 1.0,
    shrink: float = 0.0,
    xw: float = 0.0,
    xl: float = 0.0,
    dxle: float = 0.0,
    sw_efgeo: float = 0.0,
    p2: float = 0.0,
    q2: float = 0.0,
    p3: float = 0.0,
    q3: float = 0.0,
    temperature: float = 300.15,
    trise: float = 0.0,
    tnom: float = 27.0,
    tminclip: float = -100.0,
    tmaxclip: float = 500.0,
    tc1: float = 0.0,
    tc2: float = 0.0,
    tc1l: float = 0.0,
    tc2l: float = 0.0,
    tc1w: float = 0.0,
    tc2w: float = 0.0,
    m: float = 1.0,
    sw_et: float = 1.0,
    gth0: float = 1e6,
    gthp: float = 0.0,
    gtha: float = 0.0,
    cth0: float = 0.0,
    cthp: float = 0.0,
    ctha: float = 0.0,
) -> PhysicsReturn:
    """Electrothermal resistor; ``dt`` is temperature rise in K, flow is W.

    Leave ``dt`` as a free circuit node for self-heating, or connect to
    external thermal networks. ``m`` scales both electrical/thermal flows.
    ``sw_et=0`` suppresses thermal feedback; clamp ``dt`` to zero in that mode.

    Nonlinear resistor and explicit electrothermal composition.

    **Implementation source:** Native JAX adaptation of the Verilog-A model,
    version 1.0.1, from VA-Models commit `6099b4e12e6f66ff530f59b8d01fb909094be85a`.
    [Pinned upstream `r2_et_cmc.va`](https://github.com/dwarning/VA-Models/blob/6099b4e12e6f66ff530f59b8d01fb909094be85a/code/r2_cmc/vacode/r2_et_cmc.va).
    Attribution: Silicon Integration Initiative (Educational Community License 2.0).
    The shared equations are in [r2_cmc_body.include](https://github.com/dwarning/VA-Models/blob/6099b4e12e6f66ff530f59b8d01fb909094be85a/code/r2_cmc/vacode/r2_cmc_body.include).
    The retained source and includes are under `tests/data/va_models/r2_cmc/`.

    **Validation reference:** Verilog-A/OSDI comparisons use the same pinned
    [upstream source](https://github.com/dwarning/VA-Models/blob/6099b4e12e6f66ff530f59b8d01fb909094be85a/code/r2_cmc/vacode/r2_et_cmc.va),
    with comparison cases in `tests/test_r2.py`.
    These are implementation/parity checks, not measured-device calibration.
    Live comparisons require optional compiler/OSDI tools; stored fixtures are
    used where provided. Test coverage and adaptations are described in the model reference.

    Device group:
        Electronic / Compact devices

    Device reference:
        r2.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.r2 import R2ETCMC

        jax.config.update("jax_enable_x64", True)
        model = R2ETCMC()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.float64)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(R2ETCMC.__name__, "ports:", model.ports)
        ```

    """
    voltage = signals.n1 - signals.n2
    rise = sw_et * signals.dt
    resistance, length, width = _resistance(
        voltage,
        rise,
        w=w,
        l=l,
        r=r,
        rsh=rsh,
        c1=c1,
        c2=c2,
        scale=scale,
        shrink=shrink,
        xw=xw,
        xl=xl,
        dxle=dxle,
        sw_efgeo=sw_efgeo,
        p2=p2,
        q2=q2,
        p3=p3,
        q3=q3,
        temperature=temperature,
        trise=trise,
        tnom=tnom,
        tminclip=tminclip,
        tmaxclip=tmaxclip,
        tc1=tc1,
        tc2=tc2,
        tc1l=tc1l,
        tc2l=tc2l,
        tc1w=tc1w,
        tc2w=tc2w,
    )
    current = m * voltage / resistance
    perimeter = 2.0 * length + (c1 + c2) * width
    area = length * width
    conductance = gth0 + gthp * perimeter + gtha * area
    capacitance = cth0 + cthp * perimeter + ctha * area
    return {"n1": current, "n2": -current, "dt": m * rise * conductance - voltage * current}, {"dt": m * rise * capacitance}


def r2_thermal_subcircuit(
    *,
    resistor_settings: dict[str, float] | None = None,
    thermal_resistance: float = 1e3,
    thermal_capacitance: float = 1e-3,
) -> dict:
    """Return a recursive-netlist body with an explicit thermal R/C network.

    Register model names ``R2ETCMC``, ``Resistor`` and ``Capacitor`` when
    compiling. Ports are ``n1``, ``n2`` and ``dt`` (temperature rise in K).
    Thermal resistance is K/W and capacitance is J/K. The built-in thermal
    network is replaced, so geometry-dependent thermal parameters supplied
    in ``resistor_settings`` are overridden by the explicit R/C values.

    Nonlinear resistor and explicit electrothermal composition.

    Device group:
        Electronic / Compact devices

    Device reference:
        r2.md

    Examples:
        ```python
        from circulax.components.r2 import r2_thermal_subcircuit

        netlist = r2_thermal_subcircuit(thermal_resistance=1e3, thermal_capacitance=1e-6)
        assert {"core", "Rth", "Cth"} <= netlist["instances"].keys()
        # Register R2ETCMC, Resistor and Capacitor in the parent model mapping.
        print("Thermal subcircuit ports:", tuple(netlist["ports"]))
        ```

    """
    if thermal_resistance <= 0 or thermal_capacitance < 0:
        message = "thermal_resistance must be positive and thermal_capacitance nonnegative"
        raise ValueError(message)
    settings = dict(resistor_settings or {})
    settings.update(gth0=0.0, gthp=0.0, gtha=0.0, cth0=0.0, cthp=0.0, ctha=0.0, sw_et=1.0)
    # m scales the reference's thermal network together with electrical power.
    multiplicity = settings.get("m", 1.0)
    return {
        "instances": {
            "core": {"component": "R2ETCMC", "settings": settings},
            "Rth": {"component": "Resistor", "settings": {"R": thermal_resistance / multiplicity}},
            "Cth": {"component": "Capacitor", "settings": {"C": thermal_capacitance * multiplicity}},
            "GND": {"component": "ground"},
        },
        "connections": {"core,dt": ("Rth,p1", "Cth,p1"), "GND,p1": ("Rth,p2", "Cth,p2")},
        "ports": {"n1": "core,n1", "n2": "core,n2", "dt": "core,dt"},
    }
