"""Compare automatic order selection at a tighter NumPy training target.

Run: pixi run python -m benchmarks.fitting.bench_accuracy_target
"""

# ruff: noqa: T201
import time
import warnings

import numpy as np
import skrf
from skrf.vectorFitting import VectorFitting

from circulax.fitting.reduction_numpy import errors_numpy, fit_s_numpy
from circulax.fitting.sparam import scattering_state_space_to_admittance
from circulax.fitting.surface import surface_from_fit
from circulax.fitting.types import vfmodel_to_ss
from circulax.fitting.validation import validate_surface_fit


def main() -> None:
    """Measure fits before evaluating untouched holdout and circuit physics."""
    network = skrf.data.ring_slot
    train = np.arange(len(network.f)) % 5 != 0
    S, f = network.s[train], network.f[train]
    training_network = network[train]
    numpy_model = None
    for method in ("numpy", "scikit-rf"):
        elapsed = []
        for _ in range(11):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                start = time.perf_counter()
                if method == "numpy":
                    model, metadata = fit_s_numpy(S, f, reduction_stage="refined", normalized_rmse=8e-7, max_absolute_error=1e-5)
                else:
                    model = VectorFitting(training_network)
                    model.auto_fit()
                elapsed.append(time.perf_counter() - start)
        print(method, "warm median ms", 1000 * np.median(elapsed[1:]))
        if method == "numpy":
            numpy_model = model
            print("refined training candidates:", metadata["refined_candidates"])
            print(
                "selected order",
                len(model.poles),
                "holdout (NRMSE, max)",
                errors_numpy(model, network.s[~train], network.f[~train]),
            )
        else:
            prediction = np.stack(
                [model.get_model_response(r, c, network.f[~train]) for r in range(2) for c in range(2)], -1
            ).reshape(-1, 2, 2)
            print(
                "selected order",
                model.get_model_order(model.poles),
                "holdout NRMSE",
                np.linalg.norm(prediction - network.s[~train]) / np.linalg.norm(network.s[~train]),
            )
    ss, _ = scattering_state_space_to_admittance(vfmodel_to_ss(numpy_model, 2))
    features = np.ones((1, 1))
    surface = surface_from_fit(ss, np.zeros(2), 2 * np.pi * f[-1], z0=50.0)
    report = validate_surface_fit(
        surface,
        S[None],
        features,
        f,
        validation_S=network.s[~train][None],
        validation_features=features,
        validation_freqs=network.f[~train],
        passivity_features=features,
        passivity_freqs=np.linspace(network.f[0], network.f[-1], 401),
        simulation_frequency_range=(network.f[0], network.f[-1]),
    )
    print("Circuit qualification (outside fit timers):")
    print(report.summary())
    print("A failing report means this accuracy experiment is not approved for circuit simulation.")


if __name__ == "__main__":
    main()
