"""NumPy/SciPy S-domain AAA, compact pole screening, and relaxed VF.

By default all numerical work here runs on the CPU without JAX operations.
Optional aaa_backend="jax" selects JAX support discovery only. Returned
VFModel fields are NumPy arrays; conversion to a Circulax realization is an
explicit downstream operation. Uniform complex-S error and proper models
(constant term, no proportional term) are used throughout.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import scipy.linalg

from .aaa import _aaa_poles, _collect_poles, aaa_scalar
from .pole_id import _build_companion_matrix
from .types import VFModel
from .utils import compute_cindex, sort_poles


def pole_groups(poles: np.ndarray) -> list[list[int]]:
    """Return real poles and complete conjugate pairs; reject broken pairs."""
    groups = []
    i = 0
    while i < len(poles):
        if poles[i].imag == 0:
            groups.append([i])
            i += 1
        else:
            if i + 1 == len(poles) or not np.isclose(poles[i + 1], poles[i].conjugate(), rtol=1e-8, atol=0):
                raise ValueError("poles must contain adjacent conjugate pairs")
            groups.append([i, i + 1])
            i += 2
    return groups


def basis_numpy(s: np.ndarray, poles: np.ndarray) -> np.ndarray:
    """Real-coefficient basis for conjugate poles, plus a constant column."""
    basis = np.ones((len(s), len(poles) + 1), complex)
    for group in pole_groups(poles):
        i = group[0]
        first = 1 / (s - poles[i])
        if len(group) == 1:
            basis[:, i] = first
        else:
            second = 1 / (s - poles[i + 1])
            basis[:, i] = first + second
            basis[:, i + 1] = 1j * (first - second)
    return basis


def _solve(A, b):
    scale = np.linalg.norm(A, axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    solution = np.linalg.lstsq(A / scale, b, rcond=None)[0]
    return solution / (scale[:, None] if solution.ndim == 2 else scale)


def refit_numpy(S: np.ndarray, s: np.ndarray, poles: np.ndarray) -> VFModel:
    """Fit all ordered responses together using one multi-RHS least squares."""
    basis = basis_numpy(s, poles)
    responses = S.reshape(len(s), -1)
    coeff = _solve(np.concatenate([basis.real, basis.imag]), np.concatenate([responses.real, responses.imag]))
    return _model_from_coefficients(coeff, poles, S.shape[1])


def _model_from_coefficients(coeff, poles, nc):
    residues = coeff[:-1].astype(complex)
    for group in pole_groups(poles):
        if len(group) == 2:
            i, j = group
            residues[i] = coeff[i] + 1j * coeff[j]
            residues[j] = coeff[i] - 1j * coeff[j]
    return VFModel(poles.copy(), residues.T.reshape(nc, nc, len(poles)), coeff[-1].reshape(nc, nc), np.zeros((nc, nc)))


def evaluate_numpy(model: VFModel, freqs: np.ndarray) -> np.ndarray:
    """Evaluate the S-domain pole-residue model without constructing states."""
    s = 2j * np.pi * np.asarray(freqs)
    return np.einsum("fk,ijk->fij", 1 / (s[:, None] - model.poles), model.residues) + model.D


def screen_masked_numpy(S: np.ndarray, freqs: np.ndarray, initial: VFModel, subsets: list[np.ndarray]) -> list[VFModel]:
    """Refit all candidates with a padded stacked SVD and compact the results."""
    basis = basis_numpy(2j * np.pi * freqs, initial.poles)
    masks = np.zeros((len(subsets), basis.shape[1]))
    for i, subset in enumerate(subsets):
        masks[i, subset] = 1
        masks[i, -1] = 1
    matrices = np.concatenate([basis.real, basis.imag])[None] * masks[:, None]
    scale = np.linalg.norm(matrices, axis=1)
    scale = np.where(scale == 0, 1.0, scale)
    responses = S.reshape(len(freqs), -1)
    target = np.concatenate([responses.real, responses.imag])
    u, singular, vh = np.linalg.svd(matrices / scale[:, None], full_matrices=False)
    cutoff = np.finfo(float).eps * max(matrices.shape[1:]) * singular[:, :1]
    inverse = np.zeros_like(singular)
    np.divide(1.0, singular, out=inverse, where=singular > cutoff)
    coeff = (vh.swapaxes(1, 2) @ (inverse[:, :, None] * (u.swapaxes(1, 2) @ target))) / scale[:, :, None]
    return [
        _model_from_coefficients(coeff[i, np.append(subset, len(initial.poles))], initial.poles[subset], S.shape[1])
        for i, subset in enumerate(subsets)
    ]


def discover_numpy(
    S: np.ndarray, freqs: np.ndarray, *, tol=1e-8, mmax=12, reciprocal=True, aaa_backend="numpy"
) -> tuple[VFModel, dict]:
    """Discover a common pole set from the largest response, as in the S path."""
    S = np.asarray(S, complex)
    freqs = np.asarray(freqs, float)
    if S.ndim != 3 or S.shape != (len(freqs), S.shape[1], S.shape[1]):
        raise ValueError("S must have shape (frequencies, ports, ports)")
    if not len(freqs) or not np.all(np.isfinite(S)) or not np.all(np.isfinite(freqs)):
        raise ValueError("finite nonempty input is required")
    if np.any(np.diff(freqs) <= 0) or freqs[0] < 0:
        raise ValueError("frequencies must be nonnegative and strictly increasing")
    if mmax < 1 or tol <= 0:
        raise ValueError("mmax and tol must be positive")
    if aaa_backend not in {"numpy", "jax"}:
        raise ValueError("AAA backend must be 'numpy' or 'jax'")
    if reciprocal and not np.allclose(S, S.swapaxes(1, 2), atol=1e-8, rtol=0):
        raise ValueError("reciprocal=True requires symmetric S data")
    nc = S.shape[1]
    indices = [(r, c) for c in range(nc) for r in range(c, nc)] if reciprocal else [(r, c) for r in range(nc) for c in range(nc)]
    s = 2j * np.pi * freqs
    candidates = []
    for r, c in indices:
        w, nodes, _ = aaa_scalar(S[:, r, c], s, tol=tol, mmax=mmax, backend=aaa_backend)
        candidates.append(_aaa_poles(w, nodes) if len(nodes) > 1 else np.array([], complex))
    best = int(np.argmax([np.linalg.norm(S[:, r, c]) for r, c in indices]))
    raw = candidates[best]
    if not len(raw):
        raw = next((p for p in candidates if len(p)), np.array([], complex))
    if not len(raw):
        return VFModel(raw, np.zeros((nc, nc, 0), complex), S.mean(axis=0).real, np.zeros((nc, nc))), {
            "pole_flips": 0,
            "aaa_pole_count": 0,
        }
    flips = int(np.sum(raw.real > 0))
    poles = _collect_poles([raw])
    # Filter/deduplicate whole groups so close conjugates cannot be split.
    keep = []
    for group in pole_groups(poles):
        p = poles[group[0]]
        if abs(p) >= 5 * np.max(np.abs(s)):
            continue
        if any(abs(p - poles[i]) / max(abs(p), 1.0) < 1e-2 for i in keep):
            continue
        keep.extend(group)
    if keep:
        poles = poles[keep]
    return refit_numpy(S, s, poles), {"pole_flips": flips, "aaa_pole_count": len(poles)}


def candidate_subsets(model: VFModel, freqs: np.ndarray) -> list[np.ndarray]:
    """Enumerate distinct contribution-ranked budgets, including full order."""
    s = 2j * np.pi * freqs
    groups = pole_groups(model.poles)
    ranked = sorted(
        groups,
        key=lambda g: np.linalg.norm(np.sum(model.residues[:, :, g][None] / (s[:, None, None, None] - model.poles[g]), axis=-1)),
        reverse=True,
    )
    subsets = set()
    for budget in range(1, len(model.poles) + 1):
        chosen = []
        for group in ranked:
            if len(chosen) + len(group) <= budget:
                chosen.extend(group)
        if chosen:
            subsets.add(tuple(sorted(chosen)))
    return [np.asarray(indices) for indices in sorted(subsets, key=lambda x: (len(x), x))]


def refine_numpy(S: np.ndarray, freqs: np.ndarray, poles: np.ndarray, *, iterations=6, reciprocal=True) -> VFModel:
    """Traditional fast relaxed VF using NumPy QR/LS and SciPy eigenvalues."""
    if iterations < 0:
        raise ValueError("iterations must be nonnegative")
    s = 2j * np.pi * freqs
    nc = S.shape[1]
    responses = np.stack([S[:, r, c] for c in range(nc) for r in range(c, nc)]) if reciprocal else S.reshape(len(s), -1).T
    poles = np.asarray(poles).copy()
    if not len(poles):
        return refit_numpy(S, s, poles)
    for _ in range(iterations):
        n = len(poles)
        basis = basis_numpy(s, poles)
        scale = np.linalg.norm(responses) / len(s)
        blocks, rhs = [], []
        for i, response in enumerate(responses):
            a = np.concatenate([basis, -basis * response[:, None]], axis=1)
            a = np.concatenate([a.real, a.imag, np.zeros((1, 2 * (n + 1)))])
            if i == len(responses) - 1:
                a[-1, n + 1 :] = (scale * basis.sum(axis=0)).real
            q, r = np.linalg.qr(a, mode="reduced")
            blocks.append(r[n + 1 :, n + 1 :])
            rhs.append(q[-1, n + 1 :] * len(s) * scale if i == len(responses) - 1 else np.zeros(n + 1))
        sigma = _solve(np.concatenate(blocks), np.concatenate(rhs))
        if abs(sigma[-1]) < 1e-18 or abs(sigma[-1]) > 1e18:
            # Non-relaxed fallback fixes the denominator constant to one.
            blocks, rhs = [], []
            for response in responses:
                a = np.concatenate([basis, -basis[:, :n] * response[:, None]], axis=1)
                q, r = np.linalg.qr(np.concatenate([a.real, a.imag]), mode="reduced")
                blocks.append(r[n + 1 :, n + 1 :])
                rhs.append(q[:, n + 1 :].T @ np.concatenate([response.real, response.imag]))
            sigma = np.append(_solve(np.concatenate(blocks), np.concatenate(rhs)), 1.0)
        residues = sigma[:n].astype(complex)
        for group in pole_groups(poles):
            if len(group) == 2:
                i, j = group
                residues[i], residues[j] = sigma[i] + 1j * sigma[j], sigma[i] - 1j * sigma[j]
        matrix = _build_companion_matrix(poles, residues, sigma[-1], compute_cindex(poles))
        poles = scipy.linalg.eigvals(matrix)
        poles = sort_poles(-np.abs(poles.real) + 1j * poles.imag)
    return refit_numpy(S, s, poles)


def errors_numpy(model: VFModel, S: np.ndarray, freqs: np.ndarray) -> tuple[float, float]:
    """Return normalized complex RMS and maximum absolute S error."""
    error = evaluate_numpy(model, freqs) - S
    return float(np.linalg.norm(error) / max(np.linalg.norm(S), 1e-30)), float(np.max(np.abs(error)))


def fit_s_numpy(
    S: np.ndarray,
    freqs: np.ndarray,
    *,
    tol=1e-8,
    mmax=12,
    normalized_rmse=0.02,
    max_absolute_error=0.05,
    iterations=6,
    reciprocal=True,
    screening="compact",
    reduction_stage="initial",
    aaa_backend="numpy",
) -> tuple[VFModel, dict]:
    """Discover once, screen compact subsets, refine passing candidates.

    Selection uses training S error only. If no screen passes, refine full
    order; if no refined candidate passes, raise instead of claiming success.
    Passivity and the eventual admittance realization must be checked outside
    this fitting routine. No held-out samples participate in order selection.
    ``screening='masked'`` uses padded stacked SVD instead of compact solves.
    ``reduction_stage='refined'`` first relocates the full pole set, then
    refines candidate budgets in ascending order regardless of screening
    error. This can expose real poles and smaller accurate odd-order models.
    ``aaa_backend`` selects support discovery only; all later work is NumPy/SciPy.
    """
    start = time.perf_counter()
    if screening not in {"compact", "masked"}:
        raise ValueError("screening must be 'compact' or 'masked'")
    if reduction_stage not in {"initial", "refined"}:
        raise ValueError("reduction_stage must be 'initial' or 'refined'")
    if normalized_rmse <= 0 or max_absolute_error <= 0 or iterations < 0:
        raise ValueError("positive accuracy thresholds and nonnegative iterations are required")
    S, freqs = np.asarray(S, complex), np.asarray(freqs, float)
    initial, metadata = discover_numpy(S, freqs, tol=tol, mmax=mmax, reciprocal=reciprocal, aaa_backend=aaa_backend)
    discovered = time.perf_counter()
    if metadata["pole_flips"]:
        warnings.warn(f"AAA reflected {metadata['pole_flips']} RHP poles before refinement", RuntimeWarning, stacklevel=2)
    if reduction_stage == "refined":
        initial = refine_numpy(S, freqs, initial.poles, iterations=iterations, reciprocal=reciprocal)
    prepared = time.perf_counter()
    subsets = candidate_subsets(initial, freqs) if len(initial.poles) else [np.array([], int)]
    scores = []
    eligible = []
    candidates = (
        screen_masked_numpy(S, freqs, initial, subsets)
        if screening == "masked"
        else [refit_numpy(S, 2j * np.pi * freqs, initial.poles[subset]) for subset in subsets]
    )
    for candidate in candidates:
        error, maximum = errors_numpy(candidate, S, freqs)
        scores.append({"poles": len(candidate.poles), "nrmse": error, "max_error": maximum})
        if reduction_stage == "refined" or (error <= normalized_rmse and maximum <= max_absolute_error):
            eligible.append(candidate)
    screened = time.perf_counter()
    refinement_scores = []
    for candidate in eligible or [initial]:
        refined = refine_numpy(S, freqs, candidate.poles, iterations=iterations, reciprocal=reciprocal)
        error, maximum = errors_numpy(refined, S, freqs)
        refinement_scores.append({"poles": len(refined.poles), "nrmse": error, "max_error": maximum})
        if error <= normalized_rmse and maximum <= max_absolute_error:
            metadata.update(
                screen=scores,
                screening=screening,
                reduction_stage=reduction_stage,
                refined_candidates=refinement_scores,
                pole_count=len(refined.poles),
                training_nrmse=error,
                discovery_seconds=discovered - start,
                initial_refinement_seconds=prepared - discovered,
                screening_seconds=screened - prepared,
                refinement_seconds=time.perf_counter() - screened,
            )
            return refined, metadata
    raise ValueError("No refined candidate met the training accuracy thresholds")
