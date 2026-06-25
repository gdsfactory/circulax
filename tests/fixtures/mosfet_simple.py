"""Simplified JAX-native MOSFET sufficient to run a ring oscillator.

Target: a pure-JAX `@component` equivalent of PSP103 that's just
functional enough to demonstrate the ring oscillator running without
the bosdi FFI path.  Used for the "is the OSDI interface the
bottleneck?" experiment — compare wall time vs the OSDI path on the
same ring.

Physics, deliberately simplified:

- Drain current: smoothed level-1 (square-law + channel-length-mod)
  with a softplus-based Vgs_eff so sub-threshold isn't a hard zero.
- Gate capacitance: single-piece Meyer-like Q_g = Cox·W·L·V̄ where
  V̄ = Vg - (Vs+Vd)/2, split evenly to source and drain.  No
  bulk-charge, no bulk diodes, no overlap capacitance.
- ``type`` = +1 for NMOS, −1 for PMOS (matches PSP103 convention).

This is NOT accurate to PSP103 at real operating points — it's
quantitatively off by O(20 %) on Id and qualitatively wrong on many
things PSP103 handles correctly (short-channel effects, velocity
saturation, junction parasitics).  It exists only to light up a
ring oscillator through pure JAX so we can compare wall times.
"""

from __future__ import annotations

import jax.nn as jnn
import jax.numpy as jnp

from circulax.components.base_component import (
    PhysicsReturn,
    Signals,
    States,
    component,
)


@component(ports=("D", "G", "S", "B"))
def MosfetSimple(
    signals: Signals,
    s: States,
    # Sign convention: +1 for NMOS, −1 for PMOS (PSP103 TYPE parameter).
    type: float = 1.0,
    # Geometry.
    W: float = 10e-6,
    L: float = 1e-6,
    # Threshold + transconductance (fitted to PSP103 via
    # scripts/fit_mosfet_params.py at Vds=1.0/Vgs=0.7, Vds=0.6/Vgs=1.2,
    # Vds=0.6/Vgs=0.3 biases; within 13 % of PSP103 on all three).
    Vt: float = 0.171,
    KP: float = 530e-6,       # A/V²  NMOS fit; PMOS ≈ 590e-6
    LAMBDA: float = 0.0,      # 1/V channel-length modulation (fit picked 0)
    THETA: float = 0.0,       # 1/V mobility degradation (fit picked 0)
    # Sub-threshold smoothing (soft step near Vt).
    N_SMOOTH: float = 0.04,   # V — softplus scale; smaller → sharper transition
    # Gate capacitance (Meyer-like, single-piece).  8.5e-3 F/m² ≈ 4 nm oxide;
    # empirically tuned so the 9-stage ring oscillates near PSP103's 289 MHz
    # with the calibrated Id above.  Real PSP103 Cox (TOXO=1.5 nm) is higher
    # but Meyer underestimates because it skips charge redistribution in the
    # channel and the bulk-junction depletion caps; this COX is an effective
    # lump that compensates.
    COX: float = 8.5e-3,
) -> PhysicsReturn:
    """Simplified smoothed-square-law MOSFET with Meyer gate capacitance.

    Ports: D, G, S, B (bulk is a dummy — currents are zero).
    States: none (all terminals, no internal nodes like PSP103's di/si).
    """
    V_D = signals.D
    V_G = signals.G
    V_S = signals.S
    # V_B currently unused — bulk isn't modelled.  Kept so the port exists.
    _ = signals.B

    # Flip sign for PMOS so the same equations work (matches PSP103 TYPE).
    Vgs = type * (V_G - V_S)
    Vds = type * (V_D - V_S)

    # --- Drain current ------------------------------------------------------
    beta = KP * W / L

    # Softplus-smoothed Vgs_eff = max(Vgs - Vt, 0) with a scale-controlled
    # transition.  Sub-threshold (Vgs < Vt) decays exponentially; well above
    # Vt, Vgs_eff ≈ Vgs - Vt.  This gives a smooth Id vs Vgs with no kinks.
    Vgs_eff = N_SMOOTH * jnn.softplus((Vgs - Vt) / N_SMOOTH)

    # tanh-smoothed saturation (0 → 1 as Vds grows past Vdsat ≈ Vgs_eff).
    Vdsat_plus = Vgs_eff + 1e-3                   # avoid /0 near cutoff
    sat_ramp = jnp.tanh(Vds / Vdsat_plus)         # smooth linear→sat

    # Canonical strong-inversion sat current with channel-length modulation
    # and mobility-degradation factor 1/(1+THETA·Vgs_eff).
    mobility_factor = 1.0 / (1.0 + THETA * Vgs_eff)
    Id_strong = 0.5 * beta * Vgs_eff**2 * mobility_factor * (1.0 + LAMBDA * jnp.abs(Vds))

    # Assemble.  Multiply by sat_ramp to get smooth linear→saturation.
    # `type` flips the sign for PMOS so drain current direction follows
    # PSP103's convention.
    I_ds = type * Id_strong * sat_ramp

    # --- Gate capacitance ---------------------------------------------------
    # Simple Meyer: Qg = Cox·W·L · (Vg − (Vs+Vd)/2), split evenly to D and S.
    Cox_tot = COX * W * L
    Vmid = 0.5 * (V_S + V_D)
    Qg = Cox_tot * (V_G - Vmid)

    # Distribute to each port.  The complement of Qg is split between S and D.
    Q_G = Qg
    Q_D = -0.5 * Qg
    Q_S = -0.5 * Qg
    Q_B = jnp.zeros_like(Qg)

    # KCL currents at each port: drain sinks I_ds, source sources it, gate/bulk zero DC.
    f = {
        "D":  I_ds,
        "G":  jnp.zeros_like(I_ds),
        "S": -I_ds,
        "B":  jnp.zeros_like(I_ds),
    }
    q = {
        "D": Q_D,
        "G": Q_G,
        "S": Q_S,
        "B": Q_B,
    }
    return f, q
