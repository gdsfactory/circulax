"""Fixed-pole rational enforcement; scikit-rf is an independent test oracle.

Run: pixi run python -m benchmarks.fitting.bench_rational_enforcement
"""

# ruff: noqa: T201
import time

import numpy as np
import skrf
from skrf.vectorFitting import VectorFitting

from circulax.fitting import scattering_state_space_to_admittance, surface_from_fit, validate_surface_fit
from circulax.fitting.enforcement_numpy import enforce_s_passivity_numpy
from circulax.fitting.reduction_numpy import errors_numpy, fit_s_numpy
from circulax.fitting.types import VFModel, vfmodel_to_ss


def oracle(model: VFModel, network: skrf.Network) -> np.ndarray:
    """Convert full conjugate storage to scikit-rf's one-pole-per-pair format."""
    vf = VectorFitting(network)
    indices = np.flatnonzero(model.poles.imag >= 0)
    vf.poles = model.poles[indices]
    vf.residues = model.residues[:, :, indices].reshape(network.nports**2, -1)
    vf.constant_coeff = model.D.ravel()
    vf.proportional_coeff = model.E.ravel()
    return vf.passivity_test()


def main() -> None:
    """Refine the constraint grid using independently identified violations."""
    network = skrf.data.ring_slot
    mask = np.arange(len(network.f)) % 5 != 0
    model, _ = fit_s_numpy(
        network.s[mask], network.f[mask], reduction_stage="refined", normalized_rmse=8e-7, max_absolute_error=1e-5
    )
    scale = max(network.f.max(), np.abs(model.poles).max() / (2 * np.pi))
    grid = np.unique(np.r_[0, network.f[mask], np.geomspace(scale * 1e-6, scale * 1e3, 500)])
    print("before holdout NRMSE, max error", errors_numpy(model, network.s[~mask], network.f[~mask]))
    print("before violation bands Hz", oracle(model, network))
    elapsed = 0
    for attempt in range(8):
        start = time.perf_counter()
        corrected, report = enforce_s_passivity_numpy(model, network.f[mask], enforcement_freqs=grid)
        elapsed += time.perf_counter() - start
        bands = oracle(corrected, network)
        print("attempt", attempt + 1, report, "violation bands Hz", bands)
        if not report["converged"]:
            message = "Constrained optimization failed"
            raise RuntimeError(message)
        if not len(bands):
            break
        for low, high in bands:
            high = high if np.isfinite(high) else max(2 * low, scale * 1e3)
            grid = np.unique(np.r_[grid, np.linspace(low, high, 31)])
    else:
        message = "Rational passivity test still fails after grid refinement"
        raise RuntimeError(message)
    print("enforcement only ms (all corrections)", elapsed * 1000)
    print(
        "after poles", len(corrected.poles), "holdout NRMSE, max error", errors_numpy(corrected, network.s[~mask], network.f[~mask])
    )
    ss, _ = scattering_state_space_to_admittance(vfmodel_to_ss(corrected, 2))
    print("Y max real pole", np.asarray(ss.A).real.max(), "D eigenvalues", np.linalg.eigvalsh(ss.D))
    features = np.ones((1, 1))
    surface = surface_from_fit(ss, np.zeros(2), 2 * np.pi * network.f[-1], z0=50.0)
    validation = validate_surface_fit(
        surface,
        network.s[mask][None],
        features,
        network.f[mask],
        validation_S=network.s[~mask][None],
        validation_features=features,
        validation_freqs=network.f[~mask],
        passivity_features=features,
        passivity_freqs=np.linspace(0, network.f[-1], 801),
        simulation_frequency_range=(0.0, network.f[-1]),
    )
    print(validation.summary())
    print("scikit-rf rational passivity test: PASS (numerical test, not formal certification)")


if __name__ == "__main__":
    main()
