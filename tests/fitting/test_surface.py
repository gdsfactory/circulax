"""Tests for the differentiable rational-surface prototype."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from circulax.fitting.surface import (
    RationalSurface,
    evaluate_surface,
    project_surface_passive,
    refine_surface,
    surface_asymptotic_passivity_margins,
    surface_from_fit,
    surface_loss,
    surface_passivity_margins,
)
from circulax.fitting.types import VFModel, vfmodel_to_ss


def _surface() -> tuple[RationalSurface, jax.Array, jax.Array]:
    features = jnp.array([[1.0, -1.0], [1.0, 0.0], [1.0, 1.0]])
    freqs = jnp.linspace(1e8, 10e9, 80)
    omega_scale = 2 * jnp.pi * freqs[-1]
    poles = jnp.array([-0.08 + 0.6j, -0.08 - 0.6j, -0.25 + 0j])
    residue_coeffs = jnp.zeros((2, 2, 2, 3), dtype=jnp.complex128)
    base = jnp.array([1e-3 + 4e-4j, 1e-3 - 4e-4j, 2e-3])
    slope = jnp.array([2e-4 + 1e-4j, 2e-4 - 1e-4j, -1e-4])
    residue_coeffs = residue_coeffs.at[0, 0, 0].set(base)
    residue_coeffs = residue_coeffs.at[0, 1, 1].set(base)
    residue_coeffs = residue_coeffs.at[0, 0, 1].set(-0.8 * base)
    residue_coeffs = residue_coeffs.at[0, 1, 0].set(-0.8 * base)
    residue_coeffs = residue_coeffs.at[1, 0, 0].set(slope)
    residue_coeffs = residue_coeffs.at[1, 1, 1].set(slope)
    residue_coeffs = residue_coeffs.at[1, 0, 1].set(-0.8 * slope)
    residue_coeffs = residue_coeffs.at[1, 1, 0].set(-0.8 * slope)
    D = jnp.array(
        [
            [[0.02, -0.018], [-0.018, 0.02]],
            [[0.001, -0.0005], [-0.0005, -0.001]],
        ]
    )
    E = jnp.zeros_like(D)
    tau = jnp.array([[omega_scale * 0.8e-9, omega_scale * 0.8e-9], [0.1, 0.1]])
    model = RationalSurface(poles, residue_coeffs, D, E, tau, omega_scale, jnp.asarray(50 + 0j))
    return model, features, freqs


def test_surface_evaluation_is_batched_and_differentiable() -> None:
    model, features, freqs = _surface()
    S = evaluate_surface(model, features, freqs)
    assert S.shape == (3, 80, 2, 2)
    gradient = jax.grad(lambda x: jnp.real(evaluate_surface(model, jnp.array([[1.0, x]]), freqs)[0, 10, 0, 1]))(0.2)
    assert jnp.isfinite(gradient)
    assert jnp.abs(gradient) > 0


def test_refinement_reduces_complex_s_error() -> None:
    target, features, freqs = _surface()
    target_S = evaluate_surface(target, features, freqs)
    perturbed = RationalSurface(
        target.poles,
        target.residue_coeffs * 0.8,
        target.D_coeffs * 1.1,
        target.E_coeffs,
        target.tau_coeffs * 0.95,
        target.omega_scale,
        target.z0,
    )
    initial_loss = surface_loss(perturbed, features, freqs, target_S)
    fitted, losses = refine_surface(perturbed, features, freqs, target_S, steps=100, learning_rate=1e-3)
    final_loss = surface_loss(fitted, features, freqs, target_S)
    assert losses.shape == (100,)
    assert final_loss < initial_loss * 0.1


def test_passivity_projection_enforces_sampled_constraint() -> None:
    model, features, freqs = _surface()
    nonpassive = RationalSurface(
        model.poles,
        model.residue_coeffs,
        model.D_coeffs.at[0].add(-0.06 * jnp.eye(2)),
        model.E_coeffs,
        model.tau_coeffs,
        model.omega_scale,
        model.z0,
    )
    assert jnp.min(surface_passivity_margins(nonpassive, features, freqs)) < 0

    projected, shifts = project_surface_passive(nonpassive, features, freqs, minimum_conductance=1e-10)

    assert shifts.conductance > 0
    assert jnp.min(surface_passivity_margins(projected, features, freqs)) >= 0.99e-10


def test_passivity_projection_enforces_asymptotic_terms() -> None:
    model, features, freqs = _surface()
    nonpassive = RationalSurface(
        model.poles,
        model.residue_coeffs,
        model.D_coeffs.at[0].add(-0.06 * jnp.eye(2)),
        model.E_coeffs.at[0].add(-0.02 * jnp.eye(2)),
        model.tau_coeffs,
        model.omega_scale,
        model.z0,
    )

    projected, shifts = project_surface_passive(nonpassive, features, freqs, minimum_conductance=1e-10)
    D_margins, E_margins = surface_asymptotic_passivity_margins(projected, features)

    assert shifts.conductance > 0
    assert shifts.slope > 0
    assert jnp.min(D_margins) >= 0.99e-10
    assert jnp.min(E_margins) >= 0.99e-12


def test_refinement_can_remain_passive() -> None:
    target, features, freqs = _surface()
    target_S = evaluate_surface(target, features, freqs)
    nonpassive = RationalSurface(
        target.poles,
        target.residue_coeffs,
        target.D_coeffs.at[0].add(-0.04 * jnp.eye(2)),
        target.E_coeffs,
        target.tau_coeffs,
        target.omega_scale,
        target.z0,
    )

    fitted, _ = refine_surface(
        nonpassive,
        features,
        freqs,
        target_S,
        steps=30,
        learning_rate=1e-4,
        enforce_passive=True,
        minimum_conductance=1e-10,
    )

    assert jnp.min(surface_passivity_margins(fitted, features, freqs)) >= 0.99e-10


def test_surface_from_fit_preserves_single_model_response() -> None:
    model, _, freqs = _surface()
    omega_scale = model.omega_scale
    vf_model = VFModel(
        poles=model.poles * omega_scale,
        residues=model.residue_coeffs[0] * omega_scale,
        D=model.D_coeffs[0],
        E=model.E_coeffs[0] / omega_scale,
    )
    ss = vfmodel_to_ss(vf_model, 2)
    tau = model.tau_coeffs[0] / omega_scale

    wrapped = surface_from_fit(ss, tau, float(omega_scale), z0=50.0)

    expected = evaluate_surface(model, jnp.array([[1.0, 0.0]]), freqs)
    actual = evaluate_surface(wrapped, jnp.ones((1, 1)), freqs)
    assert jnp.allclose(actual, expected)
