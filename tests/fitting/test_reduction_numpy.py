"""Numerical checks for compact CPU pole screening and VF refinement."""

import numpy as np
import pytest

from circulax.fitting.reduction_numpy import (
    candidate_subsets,
    errors_numpy,
    evaluate_numpy,
    fit_s_numpy,
    pole_groups,
    refine_numpy,
    refit_numpy,
    screen_masked_numpy,
)
from circulax.fitting.types import VFModel


@pytest.mark.parametrize("reciprocal", [True, False])
def test_numpy_refinement_recovers_complex_rational_network(reciprocal):
    f = np.linspace(0.01, 3, 160)
    poles = np.array([-1.0, -0.4 - 4j, -0.4 + 4j])
    residue = np.array([[0.1, -0.04], [-0.04 if reciprocal else -0.08, 0.2]])
    residues = np.stack([residue, residue * (1 + 0.3j), residue * (1 - 0.3j)], axis=-1)
    exact = VFModel(poles, residues, np.eye(2) * 0.1, np.zeros((2, 2)))
    S = evaluate_numpy(exact, f)
    fitted = refine_numpy(S, f, poles * 1.15, reciprocal=reciprocal)
    assert errors_numpy(fitted, S, f)[0] < 1e-8
    test_f = f + 0.003
    np.testing.assert_allclose(evaluate_numpy(fitted, test_f), evaluate_numpy(exact, test_f), atol=1e-9)
    for subset in candidate_subsets(fitted, f):
        pole_groups(fitted.poles[subset])
        refit = refit_numpy(S, 2j * np.pi * f, fitted.poles[subset])
        assert np.all(np.isfinite(evaluate_numpy(refit, test_f)))
    subsets = candidate_subsets(fitted, f)
    for subset, masked in zip(subsets, screen_masked_numpy(S, f, fitted, subsets), strict=True):
        compact = refit_numpy(S, 2j * np.pi * f, fitted.poles[subset])
        np.testing.assert_allclose(evaluate_numpy(masked, test_f), evaluate_numpy(compact, test_f), atol=1e-10)


def test_broken_conjugate_pairs_rejected():
    with pytest.raises(ValueError, match="conjugate"):
        pole_groups(np.array([-1 + 2j]))


def test_constant_network_needs_no_poles():
    f = np.linspace(0.0, 10.0, 40)
    S = np.broadcast_to(np.array([[0.1, 0.2], [0.2, 0.1]]), (len(f), 2, 2))
    model, metadata = fit_s_numpy(S, f)
    assert metadata["pole_count"] == 0
    np.testing.assert_allclose(evaluate_numpy(model, f), S, atol=1e-14)


@pytest.mark.parametrize("screening", ["compact", "masked"])
def test_ring_slot_selects_compact_accurate_model(screening):
    skrf = pytest.importorskip("skrf")
    network = skrf.data.ring_slot
    train = np.arange(len(network.f)) % 5 != 0
    with pytest.warns(RuntimeWarning, match="RHP"):
        model, metadata = fit_s_numpy(network.s[train], network.f[train], screening=screening)
    assert metadata["aaa_pole_count"] == 8
    assert metadata["pole_count"] == 4
    assert [row["poles"] for row in metadata["screen"]] == [2, 4, 6, 8]
    assert errors_numpy(model, network.s[~train], network.f[~train])[0] < 4e-5
    assert isinstance(model.residues, np.ndarray)


def test_failure_is_explicit_when_no_candidate_meets_thresholds():
    f = np.linspace(0.1, 1, 50)
    S = (0.3 * np.exp(-2j * np.pi * f))[:, None, None]
    with pytest.raises(ValueError, match="No refined candidate"):
        fit_s_numpy(S, f, mmax=2, iterations=0, normalized_rmse=1e-12, max_absolute_error=1e-12)


def test_refine_first_discovers_odd_order_at_tight_accuracy():
    skrf = pytest.importorskip("skrf")
    network = skrf.data.ring_slot
    train = np.arange(len(network.f)) % 5 != 0
    with pytest.warns(RuntimeWarning, match="RHP"):
        model, metadata = fit_s_numpy(
            network.s[train],
            network.f[train],
            reduction_stage="refined",
            normalized_rmse=8e-7,
            max_absolute_error=1e-5,
        )
    assert metadata["pole_count"] == 7
    assert metadata["training_nrmse"] < 8e-7
    assert errors_numpy(model, network.s[~train], network.f[~train])[0] < 8e-7
    assert all(row["nrmse"] > 8e-7 for row in metadata["refined_candidates"][:-1])
