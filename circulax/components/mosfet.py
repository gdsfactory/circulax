# Copyright (c) 2026 Circulax contributors.
"""Complementary generic MOSFET components sharing the existing NMOS physics."""

import jax.numpy as jnp

from circulax.components.base_component import PhysicsReturn, Signals, component
from circulax.components.electronic import NMOSDynamic


@component(ports=("d", "g", "s"))
def PMOSDynamic(  # noqa: PLR0917
    signals: Signals,
    Kp: float = 1e-5,
    W: float = 20e-6,
    L: float = 1e-6,
    Vth: float = -1.0,
    lam: float = 0.0,
    Cox: float = 1e-3,
    Cgd_ov: float = 1e-15,
    Cgs_ov: float = 1e-15,
) -> PhysicsReturn:
    """PMOS polarity complement of the generic square-law ``NMOSDynamic``.

    ``Vth`` defaults to -1 V; its magnitude is passed to the NMOS model.
    ``Kp`` is A/V², W/L are m, ``lam`` is 1/V, ``Cox`` is F/m², and the
    two overlap coefficients are F. Currents are positive into terminals.
    Both currents and stored charge obey PMOS(v) = -NMOS(-v). This gives
    positive overlap capacitances for nonnegative overlap coefficients.

    The same piecewise Meyer charge model and its region-boundary
    discontinuities as ``NMOSDynamic`` apply. No bulk terminal, junctions,
    subthreshold conduction or charge-recovery states are included.

    Generic PMOS conduction with intrinsic/overlap charge.

    Device group:
        Electronic / Compact devices

    Device reference:
        pmos_dynamic.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.mosfet import PMOSDynamic

        jax.config.update("jax_enable_x64", True)
        model = PMOSDynamic()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.float64)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(PMOSDynamic.__name__, "ports:", model.ports)
        ```

    """
    nmos = NMOSDynamic(Kp=Kp, W=W, L=L, Vth=jnp.abs(Vth), lam=lam, Cox=Cox, Cgd_ov=Cgd_ov, Cgs_ov=Cgs_ov)
    f, q = nmos(d=-signals.d, g=-signals.g, s=-signals.s)
    return {name: -value for name, value in f.items()}, {name: -value for name, value in q.items()}
