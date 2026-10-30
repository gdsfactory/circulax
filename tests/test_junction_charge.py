# Copyright (c) 2026 Circulax contributors
"""Forward-bias continuation must remain differentiable in reverse mode."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax.components.bipolar import BJT_PNP_Dynamic
from circulax.components.electronic import BJT_NPN_Dynamic, _junction_charge


@pytest.mark.parametrize("voltage", [-1.0, 0.2, 0.375, 0.6, 0.9, 1.2])
def test_junction_charge_continuation_reverse_gradient(voltage: float) -> None:
    capacitance, potential, grading = 2e-12, 0.75, 0.33

    def charge(v: jax.Array) -> jax.Array:
        return _junction_charge(v, capacitance, potential, grading)

    reverse = jax.jit(jax.grad(charge))(jnp.asarray(voltage))
    expected = capacitance / (1 - min(voltage, 0.5 * potential) / potential) ** grading
    assert np.isfinite(reverse)
    assert reverse == pytest.approx(expected, rel=1e-12)
    np.testing.assert_allclose(reverse, jax.jacfwd(charge)(jnp.asarray(voltage)), rtol=1e-12)
    if voltage >= 0.5 * potential:
        anchor = capacitance * potential / (1 - grading) * (1 - 0.5 ** (1 - grading))
        assert charge(jnp.asarray(voltage)) == pytest.approx(anchor + expected * (voltage - 0.5 * potential), abs=1e-25)


@pytest.mark.parametrize("polarity", [1.0, -1.0])
def test_dynamic_bjt_reverse_charge_gradient_above_junction_potential(polarity: float) -> None:
    cls = BJT_NPN_Dynamic if polarity > 0 else BJT_PNP_Dynamic
    model = cls(Cje=2e-12, Cjc=3e-12)
    bias = polarity * jnp.array([2.0, 0.9, 0.0])

    def charge(v: jax.Array) -> jax.Array:
        return cls.solver_call(0.0, v, model)[1][1]

    reverse = jax.jit(jax.grad(charge))(bias)
    forward = jax.jacfwd(charge)(bias)
    assert np.isfinite(reverse).all()
    np.testing.assert_allclose(reverse, forward, rtol=1e-12, atol=1e-25)
    step = 1e-5
    direction = jnp.array([0.2, -0.1, 0.3])
    central = (charge(bias + step * direction) - charge(bias - step * direction)) / (2 * step)
    np.testing.assert_allclose(reverse @ direction, central, rtol=1e-8, atol=1e-24)


def test_continuation_parameter_gradients() -> None:
    def charge(parameters: jax.Array) -> jax.Array:
        return _junction_charge(0.9, *parameters)

    parameters = jnp.array([2e-12, 0.75, 0.33])
    gradient = jax.jit(jax.grad(charge))(parameters)
    assert np.isfinite(gradient).all()
    step = 1e-4
    for index in range(3):
        delta = jnp.zeros(3).at[index].set(step * parameters[index])
        central = (charge(parameters + delta) - charge(parameters - delta)) / (2 * delta[index])
        np.testing.assert_allclose(gradient[index], central, rtol=1e-7, atol=1e-24)
