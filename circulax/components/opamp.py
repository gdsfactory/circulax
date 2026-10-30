# Copyright (c) 2026 Circulax contributors
"""Generic finite-gain operational amplifier with output-stage dynamics."""

import jax.numpy as jnp

from circulax.components.base_component import PhysicsReturn, Signals, component


@component(ports=("out_p", "out_m", "in_p", "in_m"), states=("v_drive",))
def OpAmp(  # noqa: PLR0917
    signals: Signals,
    A: float = 1e5,
    GBW: float = 1e6,
    Rout: float = 50.0,
    Vmin: float = -15.0,
    Vmax: float = 15.0,
    SR: float = 1e6,
    RailTau: float = 1e-7,
) -> PhysicsReturn:
    """Dominant-pole op-amp with finite output resistance, rails and slew rate.

    ``A`` is dimensionless DC open-loop gain, ``GBW`` the gain-bandwidth
    product in Hz, ``Rout`` output resistance in ohms, ``SR`` slew rate in
    V/s, and ``RailTau`` output limiting response in seconds. ``Vmin/Vmax``
    bound the differential internal driver voltage relative to ``out_m``.
    Require positive A/GBW/Rout/SR/RailTau and Vmin < Vmax.

    A dynamic rail constraint limits outward motion of the driver state,
    avoiding saturation windup. Inputs have infinite impedance; the output
    is a floating Thevenin driver. Explicit rail settings do not model supply
    loading or common-mode limits. Reactive loads can briefly pull terminal
    voltage beyond the internal driver rails through Rout.

    Finite gain/bandwidth, output resistance, slew and rail limits.

    Device group:
        Electronic / Compact devices

    Device reference:
        opamp.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.opamp import OpAmp

        jax.config.update("jax_enable_x64", True)
        model = OpAmp()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.float64)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(OpAmp.__name__, "ports:", model.ports)
        ```

    """
    tau = A / (2.0 * jnp.pi * GBW)
    error = A * (signals.in_p - signals.in_m) - signals.v_drive
    rate = SR * jnp.tanh(error / (tau * SR))
    headroom = jnp.where(rate >= 0.0, Vmax - signals.v_drive, signals.v_drive - Vmin)
    normalized = headroom / (jnp.maximum(jnp.abs(rate), 1.0) * RailTau)
    # Smooth slowing inside the rails; linear restoring force outside them.
    taper = jnp.where(normalized >= 0.0, jnp.tanh(normalized), normalized)
    rate = jnp.where(
        signals.v_drive > Vmax,
        jnp.minimum(rate, 0.0) - (signals.v_drive - Vmax) / RailTau,
        jnp.where(
            signals.v_drive < Vmin,
            jnp.maximum(rate, 0.0) + (Vmin - signals.v_drive) / RailTau,
            rate * taper,
        ),
    )
    current = ((signals.out_p - signals.out_m) - signals.v_drive) / Rout
    return {
        "out_p": current,
        "out_m": -current,
        "in_p": 0.0,
        "in_m": 0.0,
        "v_drive": -tau * rate,
    }, {"v_drive": tau * signals.v_drive}
