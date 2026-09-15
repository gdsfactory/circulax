"""Compare barycentric responses, not phase-ambiguous SVD weights."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax.fitting.aaa import aaa_scalar
from circulax.fitting.aaa_jax import aaa_scalar_jax


def evaluate(w, nodes, values, z):
    basis = 1 / (z[:, None] - nodes)
    return (basis @ (w * values)) / (basis @ w)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_exact_rational_holdout(backend):
    z = 1j * np.linspace(0.1, 10, 100)
    holdout = z + 0.013j

    def response(s):
        return 0.2 + (1 + 0.3j) / (s + 0.5 + 2j) + (1 - 0.3j) / (s + 0.5 - 2j)

    result = aaa_scalar(response(z), z, mmax=8, backend=backend)
    np.testing.assert_allclose(evaluate(*result, holdout), response(holdout), rtol=1e-8)
    assert len(result[0]) == 3


@pytest.mark.parametrize("value", [0., 2. + 1j])
def test_jax_constant_stops_after_one_support(value):
    z = jnp.arange(10, dtype=float).astype(complex)
    result = aaa_scalar_jax(jnp.full(10, value, dtype=complex), z, mmax=8)
    assert int(result.count) == 1
    np.testing.assert_allclose(result.weights, [1, 0, 0, 0, 0, 0, 0, 0])


def test_jax_compiles_and_batches_without_host_compaction():
    z = 1j * jnp.linspace(0.1, 10, 50)
    responses = jnp.stack([1 / (z + 1), 2 / (z + 3)])
    run = jax.jit(jax.vmap(lambda f: aaa_scalar_jax(f, z, mmax=4)))
    result = run(responses)
    assert result.weights.shape == (2, 4)
    np.testing.assert_array_equal(result.count, [2, 2])


def test_ring_slot_backend_parity():
    skrf = pytest.importorskip("skrf")
    from circulax.fitting import evaluate_sparameter_model, fit_with_delay

    network = skrf.data.ring_slot
    train = np.arange(len(network.f)) % 5 != 0
    predictions = []
    for backend in ("numpy", "jax"):
        ss, tau, metadata = fit_with_delay(
            network.s[train], network.f[train], fit_domain="s", delay_mode="none",
            aaa_backend=backend, tol=1e-8, mmax=12, max_poles=4,
            enforce_passive=False, causality="ignore", verbose=False,
        )
        assert metadata["aaa_backend"] == backend
        prediction = evaluate_sparameter_model(ss, network.f[~train], tau)
        assert np.linalg.norm(prediction - network.s[~train]) / np.linalg.norm(network.s[~train]) < 1e-4
        predictions.append(prediction)
    np.testing.assert_allclose(*predictions, atol=1e-7)
