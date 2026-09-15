"""Rational coefficient correction, distinct from sample conditioning."""

import numpy as np
import pytest

from circulax.fitting.enforcement_numpy import enforce_s_passivity_numpy
from circulax.fitting.reduction_numpy import evaluate_numpy
from circulax.fitting.types import VFModel


def test_constant_and_input_unchanged():
    model = VFModel(np.array([], complex), np.zeros((2, 2, 0), complex), np.diag([1.2, 0.5]), np.zeros((2, 2)))
    result, report = enforce_s_passivity_numpy(model, np.linspace(0, 10, 10))
    assert report["converged"]
    assert not report["global_passivity_certified"]
    np.testing.assert_allclose(result.D, np.diag([0.999999, 0.5]), atol=1e-8)
    assert model.D[0, 0] == 1.2


def test_passive_model_unchanged():
    model = VFModel(np.array([-2.0 + 0j]), np.array([[[0.3]]]), np.array([[0.2]]), np.zeros((1, 1)))
    result, report = enforce_s_passivity_numpy(model, np.linspace(0, 10, 20))
    assert report["converged"]
    np.testing.assert_array_equal(result.poles, model.poles)
    np.testing.assert_allclose(result.residues, model.residues, atol=1e-12)


def test_frequency_dependent_violation_and_failure_reporting():
    model = VFModel(np.array([-2.0 + 0j]), np.array([[[2.0]]]), np.array([[0.5]]), np.zeros((1, 1)))
    freqs = np.linspace(0, 10, 40)
    _, failed = enforce_s_passivity_numpy(model, freqs, max_iterations=1)
    assert not failed["converged"]
    result, report = enforce_s_passivity_numpy(model, freqs)
    assert report["converged"]
    assert np.abs(evaluate_numpy(result, freqs)).max() <= 1
    np.testing.assert_array_equal(result.poles, model.poles)


@pytest.mark.parametrize(("pole", "e"), [(2.0, 0.0), (-2.0, 1.0)])
def test_reject_unstable_or_improper(pole, e):
    model = VFModel(np.array([pole + 0j]), np.ones((1, 1, 1)), np.zeros((1, 1)), np.array([[e]]))
    with pytest.raises(ValueError, match="stable and proper"):
        enforce_s_passivity_numpy(model, np.arange(10))


def test_ring_slot_rational_enforcement():
    skrf = pytest.importorskip("skrf")
    from benchmarks.fitting.bench_rational_enforcement import oracle
    from circulax.fitting import scattering_state_space_to_admittance
    from circulax.fitting.reduction_numpy import errors_numpy, fit_s_numpy
    from circulax.fitting.types import vfmodel_to_ss

    network = skrf.data.ring_slot
    mask = np.arange(len(network.f)) % 5 != 0
    with pytest.warns(RuntimeWarning, match="reflected"):
        model, _ = fit_s_numpy(
            network.s[mask], network.f[mask], reduction_stage="refined", normalized_rmse=8e-7, max_absolute_error=1e-5
        )
    assert len(oracle(model, network)) > 0
    corrected, report = enforce_s_passivity_numpy(
        model, network.f[mask], enforcement_freqs=np.r_[network.f[mask], np.geomspace(1e5, 1e15, 4000)]
    )
    assert report["converged"]
    assert len(oracle(corrected, network)) == 0
    np.testing.assert_array_equal(model.poles, corrected.poles)
    np.testing.assert_allclose(corrected.residues, corrected.residues.swapaxes(0, 1))
    assert errors_numpy(corrected, network.s[~mask], network.f[~mask])[0] < 3e-6
    ss, _ = scattering_state_space_to_admittance(vfmodel_to_ss(corrected, 2))
    assert np.asarray(ss.A).real.max() < 0
    assert np.linalg.eigvalsh(ss.D).min() > 0
