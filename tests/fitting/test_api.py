"""Public coefficient fitting, persistence and component boundary."""

import numpy as np
import pytest

from circulax.fitting import ModelCoefficients, ModelFitOptions, component_from_coefficients, fit_model


def test_fit_save_load_component(tmp_path):
    freqs = np.linspace(0, 3, 60)
    source = ModelCoefficients(np.array([-2.0]), np.array([[[0.3]]]), np.array([[0.2]]))
    fitted = fit_model(
        source.evaluate(freqs), freqs, initial_poles=source.poles, options=ModelFitOptions(iterations=0, normalized_rmse=1e-10)
    )
    path = tmp_path / "coefficients.npz"
    fitted.save(path)
    restored = ModelCoefficients.load(path)
    np.testing.assert_allclose(restored.evaluate(freqs), source.evaluate(freqs), atol=1e-12)
    assert restored.metadata == fitted.metadata
    assert restored.z0 == 50.0
    component = component_from_coefficients(path, name="TestFit")
    assert component.__name__ == "TestFit"
    instance = component()
    actual = np.asarray(instance.ss_D) + np.einsum(
        "ik,fk,kj->fij",
        instance.ss_C,
        1 / (2j * np.pi * freqs[:, None] - np.asarray(instance.ss_A)),
        instance.ss_B,
    )
    expected_s = source.evaluate(freqs)
    np.testing.assert_allclose(actual, (1 - expected_s) / (1 + expected_s) / 50.0, atol=1e-12)


def test_constant_auto_fit_component():
    freqs = np.linspace(0, 3, 40)
    fitted = fit_model(np.full((len(freqs), 1, 1), 0.2), freqs)
    assert len(fitted.poles) == 0
    component = component_from_coefficients(fitted)()
    assert len(component.ss_A) == 0
    np.testing.assert_allclose(component.ss_D, [[0.8 / 1.2 / 50.0]])


def test_ring_slot_public_workflow(tmp_path):
    skrf = pytest.importorskip("skrf")
    network = skrf.data.ring_slot
    mask = np.arange(len(network.f)) % 5 != 0
    coefficients = fit_model(
        network.s[mask],
        network.f[mask],
        options=ModelFitOptions(vector_fit_order=(4, 0), normalized_rmse=1e-4, max_absolute_error=1e-3),
    )
    assert coefficients.metadata["fitter"] == "scikit-rf"
    path = tmp_path / "ringslot.npz"
    coefficients.save(path)
    instance = component_from_coefficients(path)()
    assert np.asarray(instance.ss_A).real.max() < 0
    assert np.linalg.norm(coefficients.evaluate(network.f[~mask]) - network.s[~mask]) / np.linalg.norm(network.s[~mask]) < 1e-4


def test_unstable_y_rejected():
    source = ModelCoefficients(np.array([-1.0]), np.array([[[-2.0]]]), np.array([[0.0]]))
    with pytest.raises(ValueError, match="unstable Y"):
        component_from_coefficients(source)


def test_post_enforcement_accuracy_is_checked():
    freqs = np.linspace(0, 3, 40)
    with pytest.raises(ValueError, match="Final fit exceeds"):
        fit_model(np.full((len(freqs), 1, 1), 1.2), freqs, options=ModelFitOptions(enforce_passivity=True, normalized_rmse=0.01))
    corrected = fit_model(
        np.full((len(freqs), 1, 1), 1.2),
        freqs,
        options=ModelFitOptions(enforce_passivity=True, normalized_rmse=0.3, max_absolute_error=0.3),
    )
    assert corrected.metadata["enforcement"]["converged"]
    assert not corrected.metadata["global_passivity_certified"]


def test_invalid_coefficients_and_options():
    with pytest.raises(ValueError, match="symmetry"):
        ModelCoefficients(np.array([-1.0]), np.array([[[1j]]]), np.array([[0.0]]))
    with pytest.raises(ValueError, match="backend"):
        ModelFitOptions(aaa_backend="unknown")
    with pytest.raises(ValueError, match="tolerances"):
        ModelFitOptions(tol=np.nan)
    with pytest.raises(ValueError, match="method"):
        ModelFitOptions(method="unknown")
    with pytest.raises(ValueError, match="vector_fit_order"):
        ModelFitOptions(vector_fit_order=(0, 0))
    with pytest.raises(ValueError, match="only"):
        ModelFitOptions(method="aaa", vector_fit_order=(3, 0))


def test_modified_coefficients_revalidated():
    source = ModelCoefficients(np.array([-1.0]), np.array([[[0.1]]]), np.array([[0.0]]))
    source.poles[:] = 1
    with pytest.raises(ValueError, match="unstable S"):
        component_from_coefficients(source)


def test_jax_discovery_adapter():
    freqs = np.linspace(0.01, 3, 40)
    source = ModelCoefficients(np.array([-2.0]), np.array([[[0.3]]]), np.array([[0.2]]))
    fitted = fit_model(source.evaluate(freqs), freqs, options=ModelFitOptions(method="aaa", aaa_backend="jax", mmax=5))
    np.testing.assert_allclose(fitted.evaluate(freqs), source.evaluate(freqs), atol=1e-8)


def test_default_matches_skrf_auto_fit():
    skrf = pytest.importorskip("skrf")
    from skrf.vectorFitting import VectorFitting

    network = skrf.data.ring_slot
    mask = np.arange(len(network.f)) % 5 != 0
    fitted = fit_model(network.s[mask], network.f[mask])
    reference = VectorFitting(network[mask])
    reference.auto_fit()
    expected = np.stack([reference.get_model_response(r, c, network.f) for r in range(2) for c in range(2)], axis=-1).reshape(
        -1, 2, 2
    )
    np.testing.assert_allclose(fitted.evaluate(network.f), expected, atol=1e-12)
    assert fitted.metadata["order_selection"] == "automatic"
    with pytest.raises(ValueError, match="unstable Y"):
        component_from_coefficients(fitted)


def test_optional_numpy_aaa_and_conflicting_orders():
    freqs = np.linspace(0.01, 3, 40)
    source = ModelCoefficients(np.array([-2.0]), np.array([[[0.3]]]), np.array([[0.2]]))
    fitted = fit_model(source.evaluate(freqs), freqs, options=ModelFitOptions(method="aaa", mmax=5))
    np.testing.assert_allclose(fitted.evaluate(freqs), source.evaluate(freqs), atol=1e-8)
    with pytest.raises(ValueError, match="not both"):
        fit_model(source.evaluate(freqs), freqs, initial_poles=source.poles, options=ModelFitOptions(vector_fit_order=(1, 0)))


def test_vector_fitting_preserves_directionality():
    freqs = np.linspace(0.01, 3, 60)
    source = ModelCoefficients(np.array([-2.0]), np.array([[[0.3], [0.1]], [[0.2], [0.4]]]), np.eye(2) * 0.1)
    fitted = fit_model(
        source.evaluate(freqs), freqs, options=ModelFitOptions(reciprocal=False, vector_fit_order=(1, 0), normalized_rmse=1e-8)
    )
    np.testing.assert_allclose(fitted.evaluate(freqs), source.evaluate(freqs), atol=1e-9)
