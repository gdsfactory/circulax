"""Compare full NumPy reduction, matched JAX kernels, and notebook workflow.

Run: pixi run python -m benchmarks.fitting.bench_reduction_backends
"""

# ruff: noqa: T201
import time
import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import skrf
from skrf.vectorFitting import VectorFitting

from circulax.fitting import (
    FitOptions,
    fit_with_delay,
    scattering_state_space_to_admittance,
    surface_from_fit,
    validate_surface_fit,
    vfdriver,
    vmap_pole_count_sweep,
)
from circulax.fitting.pole_sweep import prune_poles_by_contribution
from circulax.fitting.reduction_numpy import (
    candidate_subsets,
    discover_numpy,
    errors_numpy,
    fit_s_numpy,
    refit_numpy,
    screen_masked_numpy,
)
from circulax.fitting.types import SSModel, VFModel, vfmodel_to_ss


def measure[T](label: str, run: Callable[[], T], repeats: int = 6) -> T:
    """Report first-use and median warm time with device synchronization."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = run()
        jax.block_until_ready(result)
        times.append(time.perf_counter() - start)
    print(f"{label}: first={times[0] * 1000:.3f} ms, warm median={np.median(times[1:]) * 1000:.3f} ms")
    return result


def main() -> None:  # noqa: PLR0915 -- keep benchmark stages together
    """Run timing and accuracy comparisons with an untouched holdout."""
    network = skrf.data.ring_slot
    train = np.arange(len(network.f)) % 5 != 0
    S, f = network.s[train], network.f[train]
    held_S, held_f = network.s[~train], network.f[~train]
    s = 2j * np.pi * f

    numpy_model, numpy_metadata = measure("NumPy AAA + compact screen + VF", lambda: fit_s_numpy(S, f))
    masked_model, _ = measure("NumPy AAA + masked screen + VF", lambda: fit_s_numpy(S, f, screening="masked"))
    initial, _ = discover_numpy(S, f)
    subsets = candidate_subsets(initial, f)
    budgets = tuple(len(subset) for subset in subsets)
    for row in numpy_metadata["screen"]:
        print("  screen", row)

    def jax_workflow() -> VFModel:
        # Same discovery, poles, masks, error thresholds and six VF iterations.
        # Discovery is repeated once per end-to-end run, just as for NumPy.
        model, _ = discover_numpy(S, f)
        sweep = vmap_pole_count_sweep(model, f, S, budgets)
        count = sweep.smallest_passing(normalized_rmse=0.02, max_absolute_error=0.05)
        if count is None:
            count = len(model.poles)
        poles = prune_poles_by_contribution(model, s, count)
        result, _, _, _ = vfdriver(
            jnp.asarray(np.moveaxis(S, 0, -1)), jnp.asarray(s), poles, FitOptions(N=len(poles), Niter1=0, Niter2=6), verbose=False
        )
        return result

    jax_model = measure("Matched discovery + JAX screen + VF", jax_workflow)
    common = {
        "tol": 1e-8,
        "mmax": 12,
        "enforce_passive": False,
        "reciprocal": True,
        "delay_mode": "none",
        "causality": "ignore",
        "fit_domain": "s",
        "s_refinement_iterations": 6,
        "verbose": False,
    }

    def notebook_workflow() -> tuple[SSModel, np.ndarray, dict]:
        _, _, metadata = fit_with_delay(S, f, **common, pole_count_candidates=budgets)
        count = metadata["pole_sweep"].smallest_passing(normalized_rmse=0.02, max_absolute_error=0.05)
        return fit_with_delay(S, f, **common, max_poles=count)

    measure("Existing two-call JAX workflow (includes S-to-Y)", notebook_workflow)

    # Compare compact vs padded NumPy screening directly, sharing discovery.
    compact = measure("NumPy compact screening only", lambda: [refit_numpy(S, s, initial.poles[x]) for x in subsets])
    padded = measure("NumPy masked stacked-SVD screening only", lambda: screen_masked_numpy(S, f, initial, subsets))
    from circulax.fitting.reduction_numpy import evaluate_numpy

    np.testing.assert_allclose(
        np.stack([evaluate_numpy(m, f) for m in padded]), np.stack([evaluate_numpy(m, f) for m in compact]), atol=1e-10
    )
    for label, model in (("NumPy", numpy_model), ("NumPy masked", masked_model), ("JAX", jax_model)):
        print(label, "poles", len(model.poles), "holdout (NRMSE, max)", errors_numpy(model, held_S, held_f))
        # Identical circuit conversion + qualification, outside fit timings.
        admittance, _ = scattering_state_space_to_admittance(vfmodel_to_ss(model, 2))
        features = np.ones((1, 1))
        surface = surface_from_fit(admittance, np.zeros(2), 2 * np.pi * f[-1], z0=50.0)
        report = validate_surface_fit(
            surface,
            S[None],
            features,
            f,
            validation_S=held_S[None],
            validation_features=features,
            validation_freqs=held_f,
            passivity_features=features,
            passivity_freqs=np.linspace(network.f[0], network.f[-1], 401),
            simulation_frequency_range=(network.f[0], network.f[-1]),
        )
        print(label, "validation", report.status)
        report.raise_for_simulation()

    training_network = network[train]

    def skrf_fit(*, auto: bool = False) -> VectorFitting:
        vf = VectorFitting(training_network)
        if auto:
            vf.auto_fit()
        else:
            vf.vector_fit(n_poles_real=4, n_poles_cmplx=0)
        return vf

    for label, auto in (("scikit-rf prescribed order 4", False), ("scikit-rf auto (default target)", True)):
        vf = measure(label, lambda auto=auto: skrf_fit(auto=auto))
        prediction = np.stack([vf.get_model_response(r, c, held_f) for r in range(2) for c in range(2)], -1).reshape(-1, 2, 2)
        error = np.linalg.norm(prediction - held_S) / np.linalg.norm(held_S)
        print(label, "order", vf.get_model_order(vf.poles), "holdout NRMSE", error)
    print("NumPy stages (last run, ms):", {k: v * 1000 for k, v in numpy_metadata.items() if k.endswith("seconds")})
    print("First-use timings share caches across methods; use warm medians for this comparison.")
    print("scikit-rf timings exclude circuit conversion/validation; auto_fit uses its own stopping criterion.")
    print(f"Versions: NumPy {np.__version__}, JAX {jax.__version__}, scikit-rf {skrf.__version__}; {jax.devices()}")


if __name__ == "__main__":
    main()
