"""Vectorized sample-domain conditioning, independently implemented in NumPy.

Alternating projections onto reciprocal, sampled-passive and discrete causal
responses. These finite-grid constraints do not certify a rational model or
its extrapolation. Frequencies occupy axis -3; optional batch axes precede it.
The causal projection uses an odd-length Hermitian FFT extension, avoiding an
artificial real-valued constraint at the highest measured frequency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _matrices(S):
    S = np.asarray(S, dtype=np.complex128)
    if S.ndim < 2 or S.shape[-1] != S.shape[-2] or S.shape[-1] == 0 or not np.all(np.isfinite(S)):
        raise ValueError("expected finite square response matrices")
    return S


def project_s_passive(S: np.ndarray, limit: float = 1.0) -> np.ndarray:
    """Nearest matrix in Frobenius norm with singular values at most limit.

    Applies to power-normalized S data (e.g. real positive reference
    impedances). All frequency and batch axes share one stacked SVD call.
    """
    S = _matrices(S)
    if not np.isfinite(limit) or not 0 < limit <= 1:
        raise ValueError("passivity limit must be in (0, 1]")
    u, singular, vh = np.linalg.svd(S, full_matrices=False)
    return (u * np.minimum(singular, limit)[..., None, :]) @ vh


def project_s_reciprocal(S: np.ndarray) -> np.ndarray:
    """Project onto transpose symmetry without complex conjugation."""
    S = _matrices(S)
    return 0.5 * (S + S.swapaxes(-1, -2))


def _check_grid(S, freqs, causal):
    f = np.asarray(freqs, dtype=float)
    if S.ndim < 3 or f.ndim != 1 or len(f) != S.shape[-3] or len(f) < 2:
        raise ValueError("S must have shape (..., frequencies, ports, ports) with at least two frequencies")
    if not np.all(np.isfinite(f)) or f[0] < 0 or np.any(np.diff(f) <= 0):
        raise ValueError("frequencies must be finite, nonnegative and strictly increasing")
    if causal and (f[0] != 0 or not np.allclose(np.diff(f), f[1], rtol=1e-8, atol=0)):
        raise ValueError("FFT causality requires a uniform grid starting at DC; supply extrapolation explicitly")


def _impulse(S):
    return np.fft.irfft(S, n=2 * S.shape[-3] - 1, axis=-3)


def _causal_projection(S, dc):
    impulse = _impulse(S)
    # For N=2F-1: samples 0..F-1 have nonnegative lag; F..N-1
    # represent negative lags under the centered periodic interpretation.
    count = S.shape[-3]
    impulse[..., count:, :, :] = 0
    if dc is not None:
        # Orthogonal affine correction on retained taps; unlike rescaling,
        # this remains defined when their sum is zero.
        correction = (dc - impulse.sum(axis=-3)) / count
        impulse[..., :count, :, :] += correction[..., None, :, :]
    return np.fft.rfft(impulse, axis=-3)


@dataclass(frozen=True)
class ConditioningReport:
    """Measured constraints on the returned grid, not a global certificate."""

    converged: bool
    iterations: int
    maximum_singular_value: float
    reciprocity_error: float
    negative_time_relative_norm: float | None
    dc_error: float | None
    relative_correction: float
    maximum_correction: float


def condition_sparameters(
    S: np.ndarray,
    freqs: np.ndarray,
    *,
    passivity: bool = True,
    reciprocity: bool = True,
    causality: bool = True,
    preserve_dc: bool = False,
    passivity_limit: float = 1.0,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> tuple[np.ndarray, ConditioningReport]:
    """Alternate sample-domain projections, checking every final constraint.

    Causality requires an explicitly supplied DC-to-fmax uniform grid. No
    extrapolation or resampling happens implicitly. The finite FFT assumes a
    periodic impulse window; its causality check is a grid-dependent proxy.
    DC preservation retains the original real DC matrix and rejects infeasible
    endpoints. Disabled constraints are reported but do not gate convergence.
    An iteration limit returns ``converged=False``, never a claimed success.
    """
    original = _matrices(S)
    _check_grid(original, freqs, causality)
    if not np.isfinite(tolerance) or tolerance <= 0 or max_iterations < 1:
        raise ValueError("positive tolerance and iteration limit are required")
    if not np.isfinite(passivity_limit) or not 0 < passivity_limit <= 1:
        raise ValueError("passivity limit must be in (0, 1]")
    dc = None
    if preserve_dc:
        if not causality:
            raise ValueError("DC preservation requires the causal projection")
        dc = original[..., 0, :, :]
        if np.max(np.abs(dc.imag)) > tolerance:
            raise ValueError("preserved DC matrix must be real")
        dc = dc.real
        if passivity and np.max(np.linalg.svd(dc, compute_uv=False)) > passivity_limit + tolerance:
            raise ValueError("preserved DC matrix violates the passivity limit")
        if reciprocity and np.max(np.abs(dc - dc.swapaxes(-1, -2))) > tolerance:
            raise ValueError("preserved DC matrix violates reciprocity")
    result = original.copy()
    converged = False
    for iteration in range(1, max_iterations + 1):
        if reciprocity:
            result = project_s_reciprocal(result)
        if passivity:
            result = project_s_passive(result, passivity_limit)
        if causality:
            result = _causal_projection(result, dc)
        sigma = float(np.max(np.linalg.svd(result, compute_uv=False)))
        symmetry = float(np.max(np.abs(result - result.swapaxes(-1, -2))))
        negative = None
        if causality:
            impulse = _impulse(result)
            negative = float(np.linalg.norm(impulse[..., result.shape[-3] :, :, :]) / max(np.linalg.norm(impulse), 1e-30))
        dc_error = None if dc is None else float(np.max(np.abs(result[..., 0, :, :] - dc)))
        converged = (
            (not passivity or sigma <= passivity_limit + tolerance)
            and (not reciprocity or symmetry <= tolerance)
            and (not causality or negative <= tolerance)
            and (dc_error is None or dc_error <= tolerance)
        )
        if converged:
            break
    correction = result - original
    report = ConditioningReport(
        converged,
        iteration,
        sigma,
        symmetry,
        negative,
        dc_error,
        float(np.linalg.norm(correction) / max(np.linalg.norm(original), 1e-30)),
        float(np.max(np.abs(correction))),
    )
    return result, report
