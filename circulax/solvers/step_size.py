# Copyright (c) 2026 Circulax contributors.
"""Step-size control for Circulax's flat real/imaginary state layout."""

from typing import Any

import diffrax
import jax
import jax.numpy as jnp


def complex_max_norm(value: jax.Array) -> jax.Array:
    """Maximum complex magnitude of an unrolled flat vector."""
    width = value.shape[-1] // 2
    return jnp.max(jnp.hypot(value[..., :width], value[..., width:]))


class ComplexPIDController(diffrax.PIDController):
    """Give each complex pair one physical relative-error denominator.

    Diffrax scales coordinates before applying its norm. Repeating each
    complex magnitude as the state supplied to that scaling gives both
    quadratures the same denominator, while preserving the error itself and
    all parent PID logic. Intended for Circulax runs with an explicit dt0.
    """

    def adapt_step_size(  # noqa: PLR0917 -- Diffrax controller interface
        self,
        t0: Any,
        t1: Any,
        y0: jax.Array,
        y1_candidate: jax.Array,
        args: Any,
        y_error: jax.Array | None,
        error_order: Any,
        controller_state: Any,
    ) -> tuple:
        """Delegate PID control using a shared magnitude for each complex pair."""
        width = y0.shape[-1] // 2

        def magnitudes(value: jax.Array) -> jax.Array:
            absolute = jnp.hypot(value[..., :width], value[..., width:])
            return jnp.concatenate((absolute, absolute), axis=-1)

        return super().adapt_step_size(
            t0, t1, magnitudes(y0), magnitudes(y1_candidate), args, y_error, error_order, controller_state
        )
