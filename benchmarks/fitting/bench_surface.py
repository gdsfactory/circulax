"""Benchmark NumPy AAA-per-corner against JAX shared-pole surface refinement."""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from circulax.fitting import fit_with_delay
from circulax.fitting.sparam import _y_to_s, embed_delay
from circulax.fitting.surface import (
    RationalSurface,
    evaluate_surface,
    initialize_surface,
    refine_surface,
    surface_asymptotic_passivity_margins,
    surface_loss,
    surface_passivity_margins,
)
from circulax.fitting.types import eval_model


def _features(u: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(u), u, u**2], axis=1)


def _synthetic_data(n_corners: int, n_freqs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth two-port family with shared dynamics and varying delay/residues."""
    u = np.linspace(-1.0, 1.0, n_corners)
    features = _features(u)
    freqs = np.linspace(1e8, 12e9, n_freqs)
    scale = 2.0 * np.pi * freqs[-1]
    poles = jnp.array([-0.025 + 0.48j, -0.025 - 0.48j, -0.14 + 0j])
    R = np.zeros((3, 2, 2, 3), dtype=np.complex128)
    base = np.array([7e-4 + 3e-4j, 7e-4 - 3e-4j, 1.5e-3])
    for feature, factor in enumerate((1.0, 0.20, -0.08)):
        diag = factor * base
        offdiag = -0.82 * diag
        R[feature, 0, 0] = diag
        R[feature, 1, 1] = diag * (1.0 + 0.03 * feature)
        R[feature, 0, 1] = offdiag
        R[feature, 1, 0] = offdiag
    D = np.array(
        [
            [[0.020, -0.0185], [-0.0185, 0.020]],
            [[0.0010, -0.0006], [-0.0006, -0.0008]],
            [[0.0004, -0.0002], [-0.0002, 0.0004]],
        ]
    )
    E = np.zeros_like(D)
    tau_seconds = np.array([[0.85e-9, 0.85e-9], [0.08e-9, 0.08e-9], [0.03e-9, 0.03e-9]])
    truth = RationalSurface(
        poles=poles,
        residue_coeffs=jnp.asarray(R),
        D_coeffs=jnp.asarray(D),
        E_coeffs=jnp.asarray(E),
        tau_coeffs=jnp.asarray(tau_seconds * scale),
        omega_scale=jnp.asarray(scale),
        z0=jnp.asarray(50.0 + 0j),
    )
    S = np.asarray(evaluate_surface(truth, jnp.asarray(features), jnp.asarray(freqs)))
    return S, freqs, features


def _evaluate_numpy_fit(ss: object, tau: np.ndarray, freqs: np.ndarray, z0: float) -> np.ndarray:
    s = jnp.asarray(1j * 2.0 * np.pi * freqs)
    Y = np.asarray(eval_model(s, ss))
    S_deembedded = np.stack([_y_to_s(Yk, z0) for Yk in Y])
    return embed_delay(S_deembedded, freqs, tau)


def run_benchmark(
    n_corners: int = 8,
    n_freqs: int = 160,
    steps: int = 200,
    learning_rate: float = 1e-4,
    *,
    enforce_passive: bool = True,
) -> dict[str, float]:
    """Run the benchmark and return machine-readable metrics."""
    S, freqs, features = _synthetic_data(n_corners, n_freqs)

    def run_current_fitter() -> tuple[list[np.ndarray], list[int], object, dict, float, float]:
        predictions = []
        pole_counts = []
        reference_ss = None
        reference_meta = None
        reference_seconds = 0.0
        start = time.perf_counter()
        for S_corner in S:
            corner_start = time.perf_counter()
            ss, tau, metadata = fit_with_delay(
                S_corner,
                freqs,
                z0=50.0,
                tol=1e-7,
                enforce_passive=enforce_passive,
                verbose=False,
            )
            predictions.append(_evaluate_numpy_fit(ss, tau, freqs, 50.0))
            jax.block_until_ready(predictions[-1])
            corner_seconds = time.perf_counter() - corner_start
            pole_counts.append(metadata["pole_count"])
            if reference_ss is None:
                reference_ss, reference_meta = ss, metadata
                reference_seconds = corner_seconds
        return predictions, pole_counts, reference_ss, reference_meta, reference_seconds, time.perf_counter() - start

    # The current AAA path is NumPy/SciPy for topology discovery but invokes
    # JAX for its fixed-pole solve and evaluation, so report its cold and warm
    # behavior just as we do for the prototype.
    *_, numpy_cold_seconds = run_current_fitter()
    (
        numpy_predictions,
        numpy_poles,
        reference_ss,
        reference_meta,
        reference_aaa_seconds,
        numpy_warm_seconds,
    ) = run_current_fitter()
    numpy_rmse = float(np.sqrt(np.mean(np.abs(np.stack(numpy_predictions) - S) ** 2)))
    numpy_max_sigma = float(np.max(np.linalg.svd(np.stack(numpy_predictions), compute_uv=False)[..., 0]))

    if reference_ss is None or reference_meta is None:
        msg = "benchmark requires at least one corner"
        raise ValueError(msg)
    n_reference_poles = reference_meta["pole_count"]
    poles = np.asarray(reference_ss.A[:n_reference_poles])
    start = time.perf_counter()
    initial = initialize_surface(S, freqs, features, poles, z0=50.0)
    init_seconds = time.perf_counter() - start
    initial_loss = float(surface_loss(initial, features, freqs, S))
    validation_features = _features(np.linspace(-1.0, 1.0, max(4 * n_corners, 33)))
    validation_freqs = np.linspace(freqs[0], freqs[-1], 4 * n_freqs)

    start = time.perf_counter()
    refined, losses = refine_surface(
        initial,
        features,
        freqs,
        S,
        steps=steps,
        learning_rate=learning_rate,
        enforce_passive=enforce_passive,
        passivity_features=validation_features,
        passivity_freqs=validation_freqs,
    )
    jax.block_until_ready(losses)
    jax_cold_seconds = time.perf_counter() - start
    refined_loss = float(surface_loss(refined, features, freqs, S))
    refined_rmse = float(np.sqrt(np.mean(np.abs(np.asarray(evaluate_surface(refined, features, freqs)) - S) ** 2)))
    refined_passivity_margin = float(jnp.min(surface_passivity_margins(refined, features, freqs)))
    validation_passivity_margin = float(
        jnp.min(surface_passivity_margins(refined, validation_features, validation_freqs))
    )
    D_margins, E_margins = surface_asymptotic_passivity_margins(refined, validation_features)

    start = time.perf_counter()
    _, warm_losses = refine_surface(
        initial,
        features,
        freqs,
        S,
        steps=steps,
        learning_rate=learning_rate,
        enforce_passive=enforce_passive,
        passivity_features=validation_features,
        passivity_freqs=validation_freqs,
    )
    jax.block_until_ready(warm_losses)
    jax_warm_seconds = time.perf_counter() - start

    return {
        "corners": float(n_corners),
        "frequencies": float(n_freqs),
        "steps": float(steps),
        "learning_rate": learning_rate,
        "enforce_passive": float(enforce_passive),
        "numpy_cold_seconds": numpy_cold_seconds,
        "numpy_warm_seconds": numpy_warm_seconds,
        "numpy_rmse_S": numpy_rmse,
        "numpy_S_passivity_margin": 1.0 - numpy_max_sigma,
        "numpy_mean_poles": float(np.mean(numpy_poles)),
        "surface_init_seconds": init_seconds,
        "reference_aaa_seconds": reference_aaa_seconds,
        "surface_initial_relative_mse": initial_loss,
        "jax_cold_seconds": jax_cold_seconds,
        "jax_warm_seconds": jax_warm_seconds,
        "jax_total_cold_seconds": reference_aaa_seconds + init_seconds + jax_cold_seconds,
        "jax_total_warm_seconds": reference_aaa_seconds + init_seconds + jax_warm_seconds,
        "jax_refined_relative_mse": refined_loss,
        "jax_refined_rmse_S": refined_rmse,
        "jax_Y_passivity_margin": refined_passivity_margin,
        "jax_validation_Y_passivity_margin": validation_passivity_margin,
        "jax_validation_D_margin": float(jnp.min(D_margins)),
        "jax_validation_E_margin": float(jnp.min(E_margins)),
        "shared_poles": float(n_reference_poles),
    }


def main() -> None:
    """Run the command-line benchmark."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corners", type=int, default=8)
    parser.add_argument("--frequencies", type=int, default=160)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--enforce-passive", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    metrics = run_benchmark(
        args.corners,
        args.frequencies,
        args.steps,
        args.learning_rate,
        enforce_passive=args.enforce_passive,
    )
    for key, value in metrics.items():
        print(f"{key}: {value:.8g}")  # noqa: T201


if __name__ == "__main__":
    main()
