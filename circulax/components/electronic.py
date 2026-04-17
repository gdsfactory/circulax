"""Electronic components."""

import jax.nn as jnn
import jax.numpy as jnp

from circulax.components.base_component import (
    PhysicsReturn,
    Signals,
    States,
    component,
    source,
)


@component(ports=("p1", "p2"))
def Resistor(signals: Signals, s: States, R: float = 1e3) -> PhysicsReturn:
    """Ohm's Law: I = V/R."""
    i = (signals.p1 - signals.p2) / (R + 1e-12)
    return {"p1": i, "p2": -i}, {}


@component(ports=("p1", "p2"))
def Capacitor(signals: Signals, s: States, C: float = 1e-12) -> PhysicsReturn:
    """Q = C * V.
    Returns Charge (q) so the solver computes I = dq/dt.
    """  # noqa: D205
    v_drop = signals.p1 - signals.p2
    q_val = C * v_drop
    return {}, {"p1": q_val, "p2": -q_val}


@component(ports=("p1", "p2"), states=("i_L",))
def Inductor(signals: Signals, s: States, L: float = 1e-9) -> PhysicsReturn:
    """V = L * di/dt formulated via flux: f['i_L'] = V, q['i_L'] = -L*i_L."""
    v_drop = signals.p1 - signals.p2
    return ({"p1": s.i_L, "p2": -s.i_L, "i_L": v_drop}, {"i_L": -L * s.i_L})


# ===========================================================================
# Sources (Time-Dependent)
# ===========================================================================


@source(ports=("p1", "p2"), states=("i_src",), amplitude_param="V")
def VoltageSource(signals: Signals, s: States, t: float, V: float = 0.0, delay: float = 0.0) -> PhysicsReturn:
    """Step voltage source."""
    v_val = jnp.where(t >= delay, V, 0.0)
    constraint = (signals.p1 - signals.p2) - v_val
    return {"p1": s.i_src, "p2": -s.i_src, "i_src": constraint}, {}


@source(ports=("p1", "p2"), states=("i_src",), amplitude_param="V")
def SmoothPulse(
    signals: Signals,
    s: States,
    t: float,
    V: float = 1.0,
    delay: float = 1e-9,
    tr: float = 1e-10,
) -> PhysicsReturn:
    """Sigmoid-smoothed pulse."""
    k = 10.0 / tr
    v_val = V * jnn.sigmoid(k * (t - delay))
    constraint = (signals.p1 - signals.p2) - v_val
    return {"p1": s.i_src, "p2": -s.i_src, "i_src": constraint}, {}


@source(ports=("p1", "p2"), states=("i_src",), amplitude_param="V")
def VoltageSourceAC(
    signals: Signals,
    s: States,
    t: float,
    V: float = 0.0,
    freq: float = 1e6,
    phase: float = 0.0,
    delay: float = 0.0,
) -> PhysicsReturn:
    """Sinusoidal voltage source."""
    omega = 2.0 * jnp.pi * freq
    v_ac = V * jnp.sin(omega * t + phase)
    v_val = jnp.where(t >= delay, v_ac, 0.0)
    constraint = (signals.p1 - signals.p2) - v_val
    return {"p1": s.i_src, "p2": -s.i_src, "i_src": constraint}, {}


@source(ports=("p1", "p2"), states=("i_src",), amplitude_param="v2")
def PulseVoltageSource(
    signals: Signals,
    s: States,
    t: float,
    v1: float = 0.0,
    v2: float = 1.0,
    td: float = 0.0,
    tr: float = 1e-9,
    tf: float = 1e-9,
    pw: float = 1e-3,
    per: float = 2e-3,
) -> PhysicsReturn:
    """SPICE-compatible PULSE voltage source: PULSE(v1 v2 td tr tf pw per).

    Parameters
    ----------
    v1  : initial (low) voltage
    v2  : pulsed (high) voltage
    td  : delay time before first edge
    tr  : rise time (v1 → v2)
    tf  : fall time (v2 → v1)
    pw  : pulse width (time at v2)
    per : period (must satisfy per >= td + tr + pw + tf)

    """
    t_shifted = jnp.where(t >= td, t - td, 0.0)
    t_in_period = jnp.mod(t_shifted, per)
    v_rising = v1 + (v2 - v1) * t_in_period / (tr + 1e-30)
    v_falling = v2 - (v2 - v1) * (t_in_period - tr - pw) / (tf + 1e-30)
    v_periodic = jnp.where(
        t_in_period < tr,
        v_rising,
        jnp.where(t_in_period < tr + pw, v2, jnp.where(t_in_period < tr + pw + tf, v_falling, v1)),
    )
    v_val = jnp.where(t < td, v1, v_periodic)
    constraint = (signals.p1 - signals.p2) - v_val
    return {"p1": s.i_src, "p2": -s.i_src, "i_src": constraint}, {}


@component(ports=("p1", "p2"), amplitude_param="I")
def CurrentSource(signals: Signals, s: States, I: float = 0.0) -> PhysicsReturn:
    """Constant current source."""
    return {"p1": I, "p2": -I}, {}


# ===========================================================================
# Diodes
# ===========================================================================


@component(ports=("p1", "p2"))
def Diode(signals: Signals, s: States, Is: float = 1e-12, n: float = 1.0, Vt: float = 25.85e-3) -> PhysicsReturn:
    """Ideal diode using the Shockley equation ``I = Is * (exp(Vd / n*Vt) - 1)``.

    Junction voltage is clipped to ``[-5, 5]`` V for numerical stability.

    Args:
        signals: Port voltages at anode (``p1``) and cathode (``p2``).
        s: Unused.
        Is: Saturation current in amperes. Defaults to ``1e-12``.
        n: Ideality factor. Defaults to ``1.0``.
        Vt: Thermal voltage in volts. Defaults to ``25.85e-3``.

    """
    vd = signals.p1 - signals.p2
    # Clip for numerical stability
    vd_safe = jnp.clip(vd, -5.0, 5.0)
    i = Is * (jnp.exp(vd_safe / (n * Vt)) - 1.0)
    return {"p1": i, "p2": -i}, {}


@component(ports=("p1", "p2"))
def ZenerDiode(
    signals: Signals,
    s: States,
    Vz: float = 5.0,
    Is: float = 1e-12,
    n: float = 1.0,
    Vt: float = 25.85e-3,
) -> PhysicsReturn:
    """Zener diode with forward Shockley conduction and reverse breakdown.

    Breakdown is modelled as a reverse exponential that activates when
    ``Vd < -Vz``.

    Args:
        signals: Port voltages at anode (``p1``) and cathode (``p2``).
        s: Unused.
        Vz: Zener breakdown voltage in volts. Defaults to ``5.0``.
        Is: Saturation current in amperes. Defaults to ``1e-12``.
        n: Ideality factor. Defaults to ``1.0``.
        Vt: Thermal voltage in volts. Defaults to ``25.85e-3``.

    """
    vd = signals.p1 - signals.p2
    i_fwd = Is * (jnp.exp(vd / (n * Vt)) - 1.0)
    # Zener breakdown modeled as reverse exponential
    i_rev = -Is * (jnp.exp(-(vd + Vz) / (n * Vt)) - 1.0)
    i_total = i_fwd + jnp.where(vd < -Vz, i_rev, 0.0)
    return {"p1": i_total, "p2": -i_total}, {}


# ===========================================================================
# Transistors (MOSFETs)
# ===========================================================================


def _nmos_current(v_d: float, v_g: float, v_s: float, Kp: float, W: float, L: float, Vth: float, lam: float) -> float:
    """Compute NMOS drain current for cutoff, linear, and saturation regions.

    Args:
        v_d: Drain voltage.
        v_g: Gate voltage.
        v_s: Source voltage.
        Kp: Process transconductance parameter in A/V².
        W: Gate width in metres.
        L: Gate length in metres.
        Vth: Threshold voltage in volts.
        lam: Channel-length modulation coefficient in V⁻¹.

    Returns:
        Drain current ``i_ds`` in amperes.

    """
    vgs = v_g - v_s
    vds = v_d - v_s

    beta = Kp * (W / L)
    v_over = vgs - Vth

    linear_current = beta * (v_over * vds - 0.5 * vds**2) * (1 + lam * vds)
    sat_current = (beta / 2.0) * (v_over**2) * (1 + lam * vds)

    return jnp.where(vgs <= Vth, 0.0, jnp.where(vds < v_over, linear_current, sat_current))


@component(ports=("d", "g", "s"))
def NMOS(
    signals: Signals,
    s: States,
    Kp: float = 2e-5,
    W: float = 10e-6,
    L: float = 1e-6,
    Vth: float = 1.0,
    lam: float = 0.0,
) -> PhysicsReturn:
    """N-channel MOSFET with square-law DC model and channel-length modulation.

    Gate current is zero (infinite input impedance).

    Args:
        signals: Port voltages at drain (``d``), gate (``g``), and source (``s``).
        s: Unused.
        Kp: Process transconductance parameter in A/V². Defaults to ``2e-5``.
        W: Gate width in metres. Defaults to ``10e-6``.
        L: Gate length in metres. Defaults to ``1e-6``.
        Vth: Threshold voltage in volts. Defaults to ``1.0``.
        lam: Channel-length modulation coefficient in V⁻¹. Defaults to ``0``.

    """
    i_ds = _nmos_current(signals.d, signals.g, signals.s, Kp, W, L, Vth, lam)
    return {"d": i_ds, "g": 0.0, "s": -i_ds}, {}


@component(ports=("d", "g", "s"))
def PMOS(
    signals: Signals,
    s: States,
    Kp: float = 1e-5,
    W: float = 20e-6,
    L: float = 1e-6,
    Vth: float = -1.0,
    lam: float = 0.0,
) -> PhysicsReturn:
    """P-channel MOSFET with square-law DC model, formulated in terms of ``Vsg`` and ``Vsd``.

    Gate current is zero (infinite input impedance).

    Args:
        signals: Port voltages at drain (``d``), gate (``g``), and source (``s``).
        s: Unused.
        Kp: Process transconductance parameter in A/V². Defaults to ``1e-5``.
        W: Gate width in metres. Defaults to ``20e-6``.
        L: Gate length in metres. Defaults to ``1e-6``.
        Vth: Threshold voltage in volts (negative for PMOS). Defaults to ``-1.0``.
        lam: Channel-length modulation coefficient in V⁻¹. Defaults to ``0``.

    """
    vsg = signals.s - signals.g
    vsd = signals.s - signals.d

    beta = Kp * (W / L)
    vth_abs = jnp.abs(Vth)
    v_over = vsg - vth_abs

    linear_current = beta * (v_over * vsd - 0.5 * vsd**2) * (1 + lam * vsd)
    sat_current = (beta / 2.0) * (v_over**2) * (1 + lam * vsd)

    i_sd = jnp.where(vsg <= vth_abs, 0.0, jnp.where(vsd < v_over, linear_current, sat_current))

    return {"d": -i_sd, "g": 0.0, "s": i_sd}, {}


@component(ports=("d", "g", "s"))
def NMOSDynamic(
    signals: Signals,
    s: States,
    Kp: float = 2e-5,
    W: float = 10e-6,
    L: float = 1e-6,
    Vth: float = 1.0,
    lam: float = 0.0,
    Cox: float = 1e-3,
    Cgd_ov: float = 1e-15,
    Cgs_ov: float = 1e-15,
) -> PhysicsReturn:
    """NMOS with square-law DC model and Meyer gate capacitance model.

    Gate charge is split into bias-dependent intrinsic charge (Meyer) and
    linear overlap contributions at the drain and source.

    Args:
        signals: Port voltages at drain (``d``), gate (``g``), and source (``s``).
        s: Unused.
        Kp: Process transconductance parameter in A/V². Defaults to ``2e-5``.
        W: Gate width in metres. Defaults to ``10e-6``.
        L: Gate length in metres. Defaults to ``1e-6``.
        Vth: Threshold voltage in volts. Defaults to ``1.0``.
        lam: Channel-length modulation coefficient in V⁻¹. Defaults to ``0``.
        Cox: Gate oxide capacitance per unit area in F/m². Defaults to ``1e-3``.
        Cgd_ov: Gate-drain overlap capacitance in farads. Defaults to ``1e-15``.
        Cgs_ov: Gate-source overlap capacitance in farads. Defaults to ``1e-15``.

    """
    i_ds = _nmos_current(signals.d, signals.g, signals.s, Kp, W, L, Vth, lam)
    f_dict = {"d": i_ds, "g": 0.0, "s": -i_ds}

    vgs = signals.g - signals.s
    vds = signals.d - signals.s
    vgd = signals.g - signals.d

    WL = W * L
    Cox_total = Cox * WL
    v_over = vgs - Vth

    cutoff = vgs <= Vth
    saturation = vds >= v_over

    # Meyer Capacitance Logic
    Qg_cut = 0.0
    Qg_sat = (2.0 / 3.0) * Cox_total * v_over
    Qg_lin = 0.5 * Cox_total * v_over

    Qg = jnp.where(cutoff, Qg_cut, jnp.where(saturation, Qg_sat, Qg_lin))
    Qd = jnp.where(cutoff, 0.0, jnp.where(saturation, 0.0, -0.5 * Qg))
    Qs = -Qg - Qd

    Q_gate = Qg + Cgd_ov * vgd + Cgs_ov * vgs
    Q_drain = Qd - Cgd_ov * vgd
    Q_source = Qs - Cgs_ov * vgs

    return f_dict, {"d": Q_drain, "g": Q_gate, "s": Q_source}


# ===========================================================================
# BJTs
# ===========================================================================


def _junction_charge(v, Cj0, Vj, m) -> float:
    """Integrate the SPICE depletion capacitance model to obtain junction charge.

    Returns ``Q = integral from 0 to v of Cj(v') dv'``, so ``dQ/dv = +Cj(v) > 0``.
    Uses linear extrapolation beyond ``fc * Vj`` (``fc = 0.5``) to avoid the
    power-law singularity as ``v → Vj``.  The anchor at the threshold is a
    precomputed constant so the diverging ``dq_normal/dv`` cannot leak through
    ``jnp.where`` into the extrapolation region.

    Args:
        v: Junction voltage in volts.
        Cj0: Zero-bias junction capacitance in farads.
        Vj: Built-in junction potential in volts.
        m: Junction grading coefficient.

    Returns:
        Junction charge in coulombs.

    """
    fc = 0.5
    v_thresh = fc * Vj
    # Standard SPICE depletion charge: Cj0*Vj/(1-m) * [1 - (1 - v/Vj)^(1-m)]
    q_normal = Cj0 * Vj / (1.0 - m) * (1.0 - jnp.power(jnp.maximum(0.0, 1.0 - v / Vj), 1.0 - m))

    # Linear extrapolation beyond threshold: fixed anchor + constant slope C_linear
    C_linear = Cj0 / jnp.power(1.0 - fc, m)  # Cj(v_thresh)
    q_thresh = Cj0 * Vj / (1.0 - m) * (1.0 - jnp.power(1.0 - fc, 1.0 - m))  # q_normal(v_thresh)
    q_high = q_thresh + C_linear * (v - v_thresh)

    return jnp.where(v < v_thresh, q_normal, q_high)


@component(ports=("c", "b", "e"))
def BJT_NPN(
    signals: Signals,
    s: States,
    Is: float = 1e-12,
    BetaF: float = 100.0,
    BetaR: float = 1.0,
    Vt: float = 25.85e-3,
) -> PhysicsReturn:
    """NPN BJT using the transport form of the Ebers-Moll DC model.

    Junction voltages are clipped to ``[-5, 2]`` V before exponentiation.
    For transient simulations with junction charge dynamics use
    :func:`BJT_NPN_Dynamic` instead.

    Args:
        signals: Port voltages at collector (``c``), base (``b``), and emitter (``e``).
        s: Unused.
        Is: Saturation current in amperes. Defaults to ``1e-12``.
        BetaF: Forward common-emitter current gain. Defaults to ``100``.
        BetaR: Reverse common-emitter current gain. Defaults to ``1``.
        Vt: Thermal voltage in volts. Defaults to ``25.85e-3``.

    """
    vbe = signals.b - signals.e
    vbc = signals.b - signals.c

    alpha_f = BetaF / (1.0 + BetaF)
    alpha_r = BetaR / (1.0 + BetaR)

    vbe_safe = jnp.clip(vbe, -5.0, 2.0)
    vbc_safe = jnp.clip(vbc, -5.0, 2.0)

    i_f = Is * (jnp.exp(vbe_safe / Vt) - 1.0)
    i_r = Is * (jnp.exp(vbc_safe / Vt) - 1.0)

    i_c = alpha_f * i_f - i_r
    i_e = -i_f + alpha_r * i_r
    i_b = -(i_c + i_e)

    return {"c": i_c, "b": i_b, "e": i_e}, {}


@component(ports=("c", "b", "e"))
def BJT_NPN_Dynamic(
    signals: Signals,
    s: States,
    Is: float = 1e-12,
    BetaF: float = 100.0,
    BetaR: float = 1.0,
    Vt: float = 25.85e-3,
    Cje: float = 1e-12,
    Cjc: float = 1e-12,
    Vje: float = 0.75,
    Vjc: float = 0.75,
    Mje: float = 0.33,
    Mjc: float = 0.33,
    Tf: float = 0.0,
    Tr: float = 0.0,
) -> PhysicsReturn:
    """NPN BJT with the Ebers-Moll DC model and first-order junction charge dynamics.

    DC currents follow the transport form of the Ebers-Moll equations. Junction
    voltages are clipped to ``[-5, 2]`` V before exponentiation to prevent
    overflow during transient solver iterations.

    Charge dynamics combine two contributions at each junction:

    - **Depletion charge** — modelled as a nonlinear junction capacitance via
      :func:`_junction_charge`, using the standard abrupt/graded junction
      formula with ideality parameters ``Vj`` and ``Mj``.
    - **Diffusion charge** — proportional to the junction current scaled by the
      transit time (``Tf`` for BE, ``Tr`` for BC``), representing minority
      carrier storage in the base.

    Args:
        signals: Port voltages at collector (``c``), base (``b``), and
            emitter (``e``).
        s: Unused; present to satisfy the component protocol.
        Is: Saturation current in amperes. Defaults to ``1e-12``.
        BetaF: Forward common-emitter current gain. Defaults to ``100``.
        BetaR: Reverse common-emitter current gain. Defaults to ``1``.
        Vt: Thermal voltage in volts. Defaults to ``25.85e-3`` (room temperature).
        Cje: Zero-bias BE junction capacitance in farads. Defaults to ``1e-12``.
        Cjc: Zero-bias BC junction capacitance in farads. Defaults to ``1e-12``.
        Vje: BE built-in junction potential in volts. Defaults to ``0.75``.
        Vjc: BC built-in junction potential in volts. Defaults to ``0.75``.
        Mje: BE junction grading coefficient. Defaults to ``0.33``.
        Mjc: BC junction grading coefficient. Defaults to ``0.33``.
        Tf: Forward transit time in seconds. Defaults to ``0`` (no BE diffusion charge).
        Tr: Reverse transit time in seconds. Defaults to ``0`` (no BC diffusion charge).

    Returns:
        A two-tuple ``(f, q)`` where:

        - **f** — DC current dict ``{"c": i_c, "b": i_b, "e": i_e}``.
        - **q** — Junction charge dict ``{"c": Q_collector, "b": Q_base, "e": Q_emitter}``,
            where ``Q_base = Q_be_total + Q_bc_total``, ``Q_collector = -Q_bc_total``,
            and ``Q_emitter = -Q_be_total``.

    """
    vbe = signals.b - signals.e
    vbc = signals.b - signals.c

    vbe_safe = jnp.clip(vbe, -5.0, 2.0)
    vbc_safe = jnp.clip(vbc, -5.0, 2.0)

    alpha_f = BetaF / (1.0 + BetaF)
    alpha_r = BetaR / (1.0 + BetaR)

    i_f = Is * (jnp.exp(vbe_safe / Vt) - 1.0)
    i_r = Is * (jnp.exp(vbc_safe / Vt) - 1.0)

    i_c = alpha_f * i_f - i_r
    i_e = -i_f + alpha_r * i_r
    i_b = -(i_c + i_e)

    f_dict = {"c": i_c, "b": i_b, "e": i_e}

    Qje_depl = _junction_charge(vbe, Cje, Vje, Mje)
    Qjc_depl = _junction_charge(vbc, Cjc, Vjc, Mjc)

    Qbe_diff = Tf * i_f
    Qbc_diff = Tr * i_r

    Q_be_total = Qje_depl + Qbe_diff
    Q_bc_total = Qjc_depl + Qbc_diff

    Q_base = Q_be_total + Q_bc_total
    Q_collector = -Q_bc_total
    Q_emitter = -Q_be_total

    return f_dict, {"c": Q_collector, "b": Q_base, "e": Q_emitter}


# ===========================================================================
# Controlled Sources & OpAmps
# ===========================================================================


@component(ports=("out_p", "out_m", "ctrl_p", "ctrl_m"), states=("i_src",))
def VCVS(signals: Signals, s: States, A: float = 1.0) -> PhysicsReturn:
    """Voltage Controlled Voltage Source."""
    constraint = (signals.out_p - signals.out_m) - A * (signals.ctrl_p - signals.ctrl_m)
    return {
        "out_p": s.i_src,
        "out_m": -s.i_src,
        "ctrl_p": 0.0,
        "ctrl_m": 0.0,
        "i_src": constraint,
    }, {}


@component(ports=("out_p", "out_m", "ctrl_p", "ctrl_m"))
def VCCS(signals: Signals, s: States, G: float = 0.0) -> PhysicsReturn:
    """Voltage Controlled Current Source."""
    i = G * (signals.ctrl_p - signals.ctrl_m)
    return {"out_p": i, "out_m": -i, "ctrl_p": 0.0, "ctrl_m": 0.0}, {}


@component(ports=("out_p", "out_m", "in_p", "in_m"), states=("i_src",))
def IdealOpAmp(signals: Signals, s: States, A: float = 1e6) -> PhysicsReturn:
    """Ideal Op Amp."""
    constraint = (signals.out_p - signals.out_m) - A * (signals.in_p - signals.in_m)
    return {
        "out_p": s.i_src,
        "out_m": -s.i_src,
        "in_p": 0.0,
        "in_m": 0.0,
        "i_src": constraint,
    }, {}


@component(ports=("p1", "p2", "cp", "cm"))
def VoltageControlledSwitch(signals: Signals, s: States, Ron: float = 1.0, Roff: float = 1e6, Vt: float = 0.0) -> PhysicsReturn:
    """Voltage Controlled Switch."""
    v_ctrl = signals.cp - signals.cm
    k = 10.0
    sig = jnn.sigmoid(k * (v_ctrl - Vt))

    g_on = 1.0 / Ron
    g_off = 1.0 / Roff
    g_eff = g_off + (g_on - g_off) * sig

    i = (signals.p1 - signals.p2) * g_eff
    return {"p1": i, "p2": -i, "cp": 0.0, "cm": 0.0}, {}


@component(ports=("out_p", "out_m", "in_p", "in_m"), states=("i_src", "i_ctrl"))
def CCVS(signals: Signals, s: States, R: float = 1.0) -> PhysicsReturn:
    """Current Controlled Voltage Source: V_out = R * i_ctrl, V_in = 0 (short)."""
    eq_in = signals.in_p - signals.in_m
    eq_out = (signals.out_p - signals.out_m) - (R * s.i_ctrl)
    return {
        "out_p": s.i_src,
        "out_m": -s.i_src,
        "in_p": s.i_ctrl,
        "in_m": -s.i_ctrl,
        "i_src": eq_out,
        "i_ctrl": eq_in,
    }, {}


@component(ports=("out_p", "out_m", "in_p", "in_m"), states=("i_ctrl",))
def CCCS(signals: Signals, s: States, alpha: float = 1.0) -> PhysicsReturn:
    """Current Controlled Current Source: I_out = alpha * i_ctrl, V_in = 0 (short)."""
    eq_in = signals.in_p - signals.in_m
    i_out = alpha * s.i_ctrl
    return {
        "out_p": i_out,
        "out_m": -i_out,
        "in_p": s.i_ctrl,
        "in_m": -s.i_ctrl,
        "i_ctrl": eq_in,
    }, {}
