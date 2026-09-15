"""Behavioral tests of sample-domain projections and their stopping checks."""

import numpy as np
import pytest

from circulax.fitting.conditioning_numpy import condition_sparameters, project_s_passive, project_s_reciprocal


def test_batched_passivity_matches_individual_svd_and_preserves_input():
    rng = np.random.default_rng(19)
    S = rng.normal(size=(3, 21, 2, 2)) + 1j * rng.normal(size=(3, 21, 2, 2))
    original = S.copy()
    actual = project_s_passive(S)
    expected = np.empty_like(S)
    for batch in range(3):
        for f in range(21):
            u, singular, vh = np.linalg.svd(S[batch, f])
            expected[batch, f] = (u * np.minimum(singular, 1)) @ vh
    np.testing.assert_allclose(actual, expected, atol=1e-13)
    assert np.linalg.svd(actual, compute_uv=False).max() <= 1 + 1e-12
    np.testing.assert_array_equal(S, original)


def test_reciprocity_is_transpose_not_hermitian():
    S = np.array([[0.1, 0.2 + 0.3j], [0.4 + 0.5j, 0.1]])
    projected = project_s_reciprocal(S)
    np.testing.assert_allclose(projected[0, 1], 0.3 + 0.4j)
    np.testing.assert_array_equal(projected, projected.T)


def test_causal_projection_removes_advance_and_preserves_dc():
    # Length 33 uses positive lags 0..16 and negative lags -16..-1.
    h = np.zeros((2, 33, 1, 1))
    h[:, 2, 0, 0] = [0.2, 0.3]
    h[:, -3, 0, 0] = [0.1, 0.15]
    S = np.fft.rfft(h, axis=-3)
    projected, report = condition_sparameters(S, np.arange(17.0), preserve_dc=True)
    corrected = np.fft.irfft(projected, n=33, axis=-3)
    assert report.converged
    np.testing.assert_allclose(corrected[:, 17:], 0, atol=1e-14)
    np.testing.assert_allclose(projected[:, 0], S[:, 0], atol=1e-14)
    assert report.negative_time_relative_norm < 1e-12


def test_causal_passive_delay_is_unchanged_including_complex_top_bin():
    f = np.arange(17.0)
    S = (0.7 * np.exp(-2j * np.pi * f * 3 / 33))[:, None, None]
    projected, report = condition_sparameters(S, f)
    np.testing.assert_allclose(projected, S, atol=1e-14)
    assert report.converged
    assert abs(projected[-1, 0, 0].imag) > 0.1


def test_iteration_limit_does_not_claim_both_constraints_pass():
    rng = np.random.default_rng(72)
    S = 5 * (rng.normal(size=(17, 2, 2)) + 1j * rng.normal(size=(17, 2, 2)))
    result, report = condition_sparameters(S, np.arange(17.0), max_iterations=1)
    sigma = np.linalg.svd(result, compute_uv=False).max()
    assert sigma > 1 + 1e-8
    assert not report.converged
    np.testing.assert_allclose(report.maximum_singular_value, sigma)
    final, complete = condition_sparameters(S, np.arange(17.0), max_iterations=500)
    assert complete.converged
    assert np.linalg.svd(final, compute_uv=False).max() <= 1 + 1e-8


@pytest.mark.parametrize("frequencies", [np.arange(1.0, 18.0), np.r_[0.0, np.arange(2.0, 18.0)]])
def test_fft_grid_requirements_are_explicit(frequencies):
    S = np.zeros((17, 1, 1), complex)
    with pytest.raises(ValueError, match="uniform grid starting at DC"):
        condition_sparameters(S, frequencies)


def test_infeasible_preserved_dc_is_rejected():
    S = np.ones((17, 1, 1), complex) * 2
    with pytest.raises(ValueError, match="DC matrix violates the passivity"):
        condition_sparameters(S, np.arange(17.0), preserve_dc=True)


def test_passivity_only_accepts_band_limited_nonuniform_data():
    S = np.ones((3, 1, 1), complex) * 2
    projected, report = condition_sparameters(S, np.array([75.0, 89.0, 110.0]), causality=False)
    np.testing.assert_allclose(projected, 1)
    assert report.converged
    assert report.negative_time_relative_norm is None
