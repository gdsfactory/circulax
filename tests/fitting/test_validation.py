"""Tests for the pre-simulation fit validation report."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from circulax.fitting.surface import RationalSurface, evaluate_surface
from circulax.fitting.validation import FitValidationError, validate_surface_fit


def _surface() -> tuple[RationalSurface, jnp.ndarray, jnp.ndarray]:
    features = jnp.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])
    freqs = jnp.linspace(1e8, 10e9, 40)
    omega_scale = 2 * jnp.pi * freqs[-1]
    poles = jnp.array([-0.08 + 0.6j, -0.08 - 0.6j, -0.25 + 0j])
    residues = jnp.zeros((2, 2, 2, 3), dtype=jnp.complex128)
    base = jnp.array([1e-3 + 4e-4j, 1e-3 - 4e-4j, 2e-3])
    residues = residues.at[0, 0, 0].set(base)
    residues = residues.at[0, 1, 1].set(base)
    residues = residues.at[0, 0, 1].set(-0.8 * base)
    residues = residues.at[0, 1, 0].set(-0.8 * base)
    D = jnp.array([[[0.02, -0.018], [-0.018, 0.02]], jnp.zeros((2, 2))])
    E = jnp.zeros_like(D)
    tau = jnp.array([[omega_scale * 0.8e-9, omega_scale * 0.8e-9], [0.1, 0.1]])
    return RationalSurface(poles, residues, D, E, tau, omega_scale, jnp.asarray(50 + 0j)), features, freqs


def test_validation_report_passes_and_summarizes_good_fit() -> None:
    model, features, freqs = _surface()
    measured = evaluate_surface(model, features, freqs)

    report = validate_surface_fit(
        model,
        measured,
        features,
        freqs,
        validation_S=measured,
        validation_features=features,
        validation_freqs=freqs,
        passivity_features=features,
        passivity_freqs=freqs,
        simulation_frequency_range=(2e8, 9e9),
    )

    assert report.status == "pass"
    assert report.simulation_ready
    assert report.normalized_rmse == pytest.approx(0.0)
    assert "Fit validation: PASS" in report.summary()
    report.raise_for_simulation()


def test_validation_report_blocks_unstable_nonpassive_fit() -> None:
    target, features, freqs = _surface()
    measured = evaluate_surface(target, features, freqs)
    bad_model = RationalSurface(
        target.poles.at[0].set(0.1 + 0.6j),
        target.residue_coeffs,
        target.D_coeffs.at[0].add(-0.08 * jnp.eye(2)),
        target.E_coeffs,
        target.tau_coeffs,
        target.omega_scale,
        target.z0,
    )

    report = validate_surface_fit(
        bad_model,
        measured,
        features,
        freqs,
        validation_S=measured,
        validation_features=features,
        validation_freqs=freqs,
        passivity_features=features,
        passivity_freqs=freqs,
    )

    assert report.status == "fail"
    assert {finding.code for finding in report.findings} >= {"unstable", "passivity", "holdout_max_error"}
    with pytest.raises(FitValidationError, match="fit validation fail"):
        report.raise_for_simulation()


def test_missing_holdout_is_warning_requiring_explicit_override() -> None:
    model, features, freqs = _surface()
    measured = evaluate_surface(model, features, freqs)
    report = validate_surface_fit(model, measured, features, freqs)

    assert report.status == "warn"
    assert not report.simulation_ready
    with pytest.raises(FitValidationError, match="no_holdout"):
        report.raise_for_simulation()
    report.raise_for_simulation(allow_warnings=True)


def test_simulation_range_outside_validation_fails() -> None:
    model, features, freqs = _surface()
    measured = evaluate_surface(model, features, freqs)

    report = validate_surface_fit(
        model,
        measured,
        features,
        freqs,
        validation_S=measured,
        validation_features=features,
        validation_freqs=freqs,
        passivity_features=features,
        passivity_freqs=freqs,
        simulation_frequency_range=(0.0, 20e9),
    )

    assert report.status == "fail"
    assert "simulation_band" in {finding.code for finding in report.findings}


def test_expected_active_model_does_not_require_passivity() -> None:
    target, features, freqs = _surface()
    active_model = RationalSurface(
        target.poles,
        target.residue_coeffs,
        target.D_coeffs.at[0].add(-0.08 * jnp.eye(2)),
        target.E_coeffs,
        target.tau_coeffs,
        target.omega_scale,
        target.z0,
    )
    measured = evaluate_surface(active_model, features, freqs)

    report = validate_surface_fit(
        active_model,
        measured,
        features,
        freqs,
        validation_S=measured,
        validation_features=features,
        validation_freqs=freqs,
        expected_passive=False,
    )

    assert report.status == "pass"
    assert not report.expected_passive
    assert "passivity" not in {finding.code for finding in report.findings}
