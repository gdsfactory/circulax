# Copyright (c) 2026 Circulax contributors
"""Smooth dynamic comparator with optional regenerative hysteresis."""

import jax.numpy as jnp

from circulax.components.base_component import PhysicsReturn, Signals, component


@component(ports=("out_p", "out_m", "in_p", "in_m"), states=("x",))
def Comparator(  # noqa: PLR0917
    signals: Signals,
    Vlow: float = 0.0,
    Vhigh: float = 5.0,
    Vth: float = 0.0,
    Vh: float = 0.0,
    Vsmooth: float = 1e-3,
    Tau: float = 1e-9,
    Rout: float = 50.0,
) -> PhysicsReturn:
    """Finite-response comparator with state-based positive-feedback hysteresis.

    ``Tau * dx/dt = tanh((Vin_p - Vin_m - Vth + Vh*x)/Vsmooth) - x``.
    The bounded normalized state x drives a floating Thevenin output between
    Vlow and Vhigh relative to out_m. Tau is seconds, Rout is ohms, other
    parameters are volts. Require Tau/Rout/Vsmooth > 0, Vh >= 0 and Vhigh > Vlow.

    Hysteresis is bistable when Vh > Vsmooth. The quasi-static thresholds are
    Vth +/- (Vh*z - Vsmooth*atanh(z)), z=sqrt(1-Vsmooth/Vh). Vh therefore
    approaches the half hysteresis width as Vsmooth becomes small. Select a
    stable initial branch through x or a prior unambiguous input bias; there
    is no hidden Python state. Inputs have infinite impedance. Supply loading,
    common-mode limits and stochastic switching are not modeled.

    Finite response and dynamic hysteresis.

    Device group:
        Electronic / Compact devices

    Device reference:
        comparator.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.comparator import Comparator

        jax.config.update("jax_enable_x64", True)
        model = Comparator()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.float64)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(Comparator.__name__, "ports:", model.ports)
        ```

    """
    differential = signals.in_p - signals.in_m - Vth
    target = jnp.tanh((differential + Vh * signals.x) / Vsmooth)
    drive = 0.5 * (Vhigh + Vlow) + 0.5 * (Vhigh - Vlow) * signals.x
    current = ((signals.out_p - signals.out_m) - drive) / Rout
    return {
        "out_p": current,
        "out_m": -current,
        "in_p": 0.0,
        "in_m": 0.0,
        "x": signals.x - target,
    }, {"x": Tau * signals.x}
