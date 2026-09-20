"""Experimental fixed-pole, finite-grid rational S passivity enforcement.

Unlike sample clipping, this changes residues and D of the rational model.
The returned diagnostic is deliberately not a global passivity certificate.
All ports must use real, positive power-wave reference impedances.
"""

import numpy as np
from scipy.optimize import minimize

from .reduction_numpy import _model_from_coefficients, basis_numpy, evaluate_numpy, pole_groups
from .types import VFModel


def enforce_s_passivity_numpy(model, freqs, *, enforcement_freqs=None, limit=0.999999, max_iterations=300) -> tuple[VFModel, dict]:
    """Minimize measured-band perturbation subject to sampled S and D bounds.

    Requires a stable, proper, reciprocal real-rational model. Poles are held
    fixed; conjugacy, reality and reciprocity are preserved by construction.
    ``freqs`` defines the equally weighted preservation objective, not new data.
    Always inspect ``converged`` and independently validate the returned model.
    """
    freqs = np.asarray(freqs, float)
    poles = np.asarray(model.poles)
    residues, direct = np.asarray(model.residues), np.asarray(model.D)
    if freqs.ndim != 1 or not len(freqs) or not np.all(np.isfinite(freqs)) or np.any(freqs < 0):
        raise ValueError("freqs must be finite nonnegative frequencies")
    if not 0 < limit < 1 or max_iterations < 1:
        raise ValueError("require 0 < limit < 1 and positive max_iterations")
    if not all(np.all(np.isfinite(a)) for a in (poles, residues, direct, model.E)):
        raise ValueError("model must be finite")
    if np.any(poles.real >= 0) or np.any(np.asarray(model.E) != 0):
        raise ValueError("model must be stable and proper (E=0)")
    if not np.allclose(direct.imag, 0) or not np.allclose(direct, direct.T) or not np.allclose(residues, residues.swapaxes(0, 1)):
        raise ValueError("model must be reciprocal with real D")
    for group in pole_groups(poles):
        if len(group) == 1:
            valid = np.allclose(residues[..., group[0]].imag, 0)
        else:
            valid = np.allclose(residues[..., group[0]], residues[..., group[1]].conj())
        if not valid:
            raise ValueError("model residues must satisfy real-rational conjugacy")
    if enforcement_freqs is None:
        scale = max(float(freqs.max()), float(np.max(np.abs(poles), initial=0) / (2 * np.pi)), 1.0)
        enforcement_freqs = np.unique(np.r_[0, freqs, np.geomspace(scale * 1e-6, scale * 1e3, 500)])
    grid = np.asarray(enforcement_freqs, float)
    if grid.ndim != 1 or not len(grid) or not np.all(np.isfinite(grid)) or np.any(grid < 0):
        raise ValueError("enforcement_freqs must be finite and nonnegative")
    grid = np.unique(np.r_[0, grid])
    if not len(poles):
        # Exact Frobenius projection for static real reciprocal matrices avoids
        # an unnecessarily ill-conditioned constrained optimization at the bound.
        eigenvalues, vectors = np.linalg.eigh(direct.real)
        corrected_direct = (vectors * np.clip(eigenvalues, -limit, limit)) @ vectors.T
        corrected = VFModel(poles.copy(), residues.copy(), corrected_direct, np.zeros_like(corrected_direct))
        return corrected, {
            "converged": True,
            "optimizer_message": "Static symmetric spectral projection",
            "iterations": 0,
            "max_sigma_grid_and_infinity": float(np.linalg.norm(corrected_direct, 2)),
            "grid_size": len(grid),
            "global_passivity_certified": False,
            "relative_band_change": float(np.linalg.norm(corrected_direct - direct) / max(np.linalg.norm(direct), 1e-30)),
        }
    n = len(direct)
    rows, cols = np.triu_indices(n)
    mapping = np.zeros((len(rows), n, n))
    mapping[np.arange(len(rows)), rows, cols] = 1
    mapping[np.arange(len(rows)), cols, rows] = 1
    train = basis_numpy(2j * np.pi * freqs, poles)
    # Whiten the coefficient coordinates against the preservation objective.
    _, r = np.linalg.qr(np.concatenate([train.real, train.imag]), mode="reduced")
    if r.shape != (len(poles) + 1, len(poles) + 1) or np.linalg.matrix_rank(r / np.linalg.norm(r, axis=0)) < len(r):
        raise ValueError("preservation grid does not identify the fixed-pole coefficients")
    transform = np.linalg.solve(r, np.eye(len(r)))
    train = train @ transform
    basis = basis_numpy(2j * np.pi * grid, poles) @ transform
    basis = np.vstack([basis, transform[-1]])  # Infinity: S tends to D.
    target = evaluate_numpy(model, freqs)
    shape = (len(poles) + 1, len(rows))
    initial = np.linalg.lstsq(
        np.r_[train.real, train.imag], np.r_[target[:, rows, cols].real, target[:, rows, cols].imag], rcond=None
    )[0]
    weights = np.where(rows == cols, 1.0, 2.0)

    def objective(x):
        delta = x.reshape(shape) - initial
        return np.sum(delta**2 * weights), (2 * delta * weights).ravel()

    def constraint(x):
        response = np.einsum("fk,ka,aij->fij", basis, x.reshape(shape), mapping)
        return (limit - np.linalg.svd(response, compute_uv=False)).ravel()

    def jacobian(x):
        response = np.einsum("fk,ka,aij->fij", basis, x.reshape(shape), mapping)
        u, _, vh = np.linalg.svd(response)
        derivative = np.einsum("fis,aij,fsj->fsa", u.conj(), mapping, vh.conj())
        return -np.einsum("fsa,fk->fska", derivative, basis).real.reshape(-1, initial.size)

    result = minimize(
        objective,
        initial.ravel(),
        jac=True,
        constraints={"type": "ineq", "fun": constraint, "jac": jacobian},
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": max_iterations},
    )
    coeff = np.einsum("ka,aij->kij", transform @ result.x.reshape(shape), mapping)
    corrected = _model_from_coefficients(coeff.reshape(len(poles) + 1, -1), poles, n)
    peak = float(limit - constraint(result.x).min())
    return corrected, {
        "converged": bool(result.success and peak <= limit + 1e-9),
        "optimizer_message": result.message,
        "iterations": result.nit,
        "max_sigma_grid_and_infinity": peak,
        "grid_size": len(grid),
        "global_passivity_certified": False,
        "relative_band_change": float(
            np.linalg.norm(evaluate_numpy(corrected, freqs) - target) / max(np.linalg.norm(target), 1e-30)
        ),
    }
