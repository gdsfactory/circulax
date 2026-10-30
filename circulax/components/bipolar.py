# Copyright (c) 2026 Circulax contributors
"""Complementary bipolar components sharing the existing NPN physics."""

from circulax.components.base_component import CircuitComponent, PhysicsReturn, Signals, component
from circulax.components.electronic import BJT_NPN, BJT_NPN_Dynamic


def _mirror_npn(model: CircuitComponent, signals: Signals) -> PhysicsReturn:
    """Reverse voltage, current and charge signs without changing parameters."""
    f, q = model(c=-signals.c, b=-signals.b, e=-signals.e)
    return {port: -value for port, value in f.items()}, {port: -value for port, value in q.items()}


@component(ports=("c", "b", "e"))
def BJT_PNP(
    signals: Signals,
    Is: float = 1e-12,
    BetaF: float = 100.0,
    BetaR: float = 1.0,
    Vt: float = 25.85e-3,
) -> PhysicsReturn:
    """PNP Ebers–Moll transistor, the polarity complement of ``BJT_NPN``.

    Currents are positive entering the named terminals. Parameters retain the
    positive magnitudes and units of the NPN model. Use ``BJT_PNP_Dynamic`` for
    depletion and diffusion charge. The same junction-voltage clipping as the
    NPN reference applies.

    PNP conduction; dynamic variant adds terminal storage.

    Device group:
        Electronic / Compact devices

    Device reference:
        pnp.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.bipolar import BJT_PNP

        jax.config.update("jax_enable_x64", True)
        model = BJT_PNP()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.float64)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(BJT_PNP.__name__, "ports:", model.ports)
        ```

    """
    return _mirror_npn(BJT_NPN(Is=Is, BetaF=BetaF, BetaR=BetaR, Vt=Vt), signals)


@component(ports=("c", "b", "e"))
def BJT_PNP_Dynamic(  # noqa: PLR0917
    signals: Signals,
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
    """PNP Ebers–Moll transistor with conservative terminal charges.

    Parameters match ``BJT_NPN_Dynamic``: ``Is`` in A, ``Vt/Vje/Vjc`` in V,
    ``Cje/Cjc`` in F, ``Tf/Tr`` in s, and dimensionless gains and grading
    coefficients. Both currents and charges obey ``PNP(v) = -NPN(-v)``.
    Consequently DC, AC and transient behavior use a single set of NPN physics
    equations; all inherited numerical limits also apply.

    PNP conduction; dynamic variant adds terminal storage.

    Device group:
        Electronic / Compact devices

    Device reference:
        pnp.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.bipolar import BJT_PNP_Dynamic

        jax.config.update("jax_enable_x64", True)
        model = BJT_PNP_Dynamic()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.float64)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(BJT_PNP_Dynamic.__name__, "ports:", model.ports)
        ```

    """
    return _mirror_npn(
        BJT_NPN_Dynamic(
            Is=Is,
            BetaF=BetaF,
            BetaR=BetaR,
            Vt=Vt,
            Cje=Cje,
            Cjc=Cjc,
            Vje=Vje,
            Vjc=Vjc,
            Mje=Mje,
            Mjc=Mjc,
            Tf=Tf,
            Tr=Tr,
        ),
        signals,
    )
