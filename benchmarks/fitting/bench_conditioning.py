"""Apply vectorized sample conditioning before ring-slot rational fitting.

Run: pixi run python -m benchmarks.fitting.bench_conditioning
All extrapolation uses training samples only. Holdout stays measured and raw.
"""

# ruff: noqa: T201
import time
import warnings
from dataclasses import asdict

import numpy as np
import skrf

from circulax.fitting.conditioning_numpy import condition_sparameters, project_s_passive
from circulax.fitting.reduction_numpy import fit_s_numpy
from circulax.fitting.sparam import scattering_state_space_to_admittance
from circulax.fitting.surface import surface_from_fit
from circulax.fitting.types import vfmodel_to_ss
from circulax.fitting.validation import validate_surface_fit


def main() -> None:
    """Report correction size, speed and downstream fit failures explicitly."""
    network = skrf.data.ring_slot
    mask = np.arange(len(network.f)) % 5 != 0
    training = network[mask]
    extended = training.extrapolate_to_dc()
    start = time.perf_counter()
    cleaned, conditioning = condition_sparameters(extended.s, extended.f, max_iterations=500)
    print("conditioning ms", 1000 * (time.perf_counter() - start))
    print("sample conditioning:", asdict(conditioning))
    print("estimated DC before/after:", extended.s[0], cleaned[0])
    if not conditioning.converged:
        message = "Conditioning did not converge; do not proceed as if input is qualified"
        raise RuntimeError(message)

    # The same pointwise projection, independent loop versus stacked SVD.
    def loop_projection() -> np.ndarray:
        result = []
        for matrix in extended.s:
            u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
            result.append((u * np.minimum(singular, 1.0)) @ vh)
        return np.asarray(result)

    for label, operation in (("stacked passivity", lambda: project_s_passive(extended.s)), ("loop passivity", loop_projection)):
        times = []
        for _ in range(11):
            start = time.perf_counter()
            result = operation()
            times.append(time.perf_counter() - start)
        print(label, "median ms", 1000 * np.median(times[1:]))
        np.testing.assert_allclose(result, project_s_passive(extended.s), atol=1e-13)

    projected_band, band_report = condition_sparameters(training.s, training.f, causality=False)
    print("measured-band projection:", asdict(band_report))
    cases = [
        ("original, tight accuracy", training.s, training.f, 8e-7, 1e-5, 1e-8, 12),
        ("band passivity only, tight accuracy", projected_band, training.f, 8e-7, 1e-5, 1e-8, 12),
        (
            "conditioned DC + original band, default accuracy",
            np.concatenate([cleaned[:1], training.s]),
            np.r_[0.0, training.f],
            0.02,
            0.05,
            1e-6,
            20,
        ),
        ("conditioned extrapolated grid, default accuracy", cleaned, extended.f, 0.02, 0.05, 1e-6, 20),
    ]
    for label, S, freqs, target, maximum, tol, capacity in cases:
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            try:
                model, metadata = fit_s_numpy(
                    S, freqs, reduction_stage="refined", normalized_rmse=target, max_absolute_error=maximum, tol=tol, mmax=capacity
                )
            except ValueError as exc:
                print(label, "fit failed:", str(exc))
                continue
        elapsed = time.perf_counter() - start
        print(
            label, "poles", len(model.poles), "fit ms", 1000 * elapsed, "fit-to-supplied-target NRMSE", metadata["training_nrmse"]
        )
        for warning in recorded:
            print("fit diagnostic:", warning.message)
        try:
            ss, _ = scattering_state_space_to_admittance(vfmodel_to_ss(model, 2))
        except ValueError as exc:
            print("circuit conversion failed:", str(exc))
            continue
        features = np.ones((1, 1))
        surface = surface_from_fit(ss, np.zeros(2), 2 * np.pi * network.f[-1], z0=50.0)
        low = 0.0 if freqs[0] == 0 else network.f[0]
        report = validate_surface_fit(
            surface,
            training.s[None],
            features,
            training.f,
            validation_S=network.s[~mask][None],
            validation_features=features,
            validation_freqs=network.f[~mask],
            passivity_features=features,
            passivity_freqs=np.linspace(low, network.f[-1], 801),
            simulation_frequency_range=(low, network.f[-1]),
        )
        print(report.summary())
    print("Finite-grid enforcement does not certify rational-model passivity or stable S-to-Y conversion.")


if __name__ == "__main__":
    main()
