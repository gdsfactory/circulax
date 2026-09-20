"""Stage 1: Pole identification via Fast Relaxed Vector Fitting.

Ported from vectfit4.m (SINTEF MFT-NNLS toolbox, B. Gustavsen, 2026).
Covers lines 256–534 of vectfit4.m.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .types import FitOptions
from .utils import build_dk, compute_cindex, sort_poles

# Tolerances for relaxed sigma D term (matching vectfit4 line 169)
_TOL_LOW = 1e-18
_TOL_HIGH = 1e18


def _build_pole_id_ls(
    f: jnp.ndarray,  # (nnn, Ns) complex — elements being fitted
    s: jnp.ndarray,  # (Ns,) complex
    Dk: jnp.ndarray,  # (Ns, N+offs) complex — basis functions
    weight: jnp.ndarray,  # (nnn, Ns) or (1, Ns) complex
    N: int,
    Ns: int,
    nnn: int,
    offs: int,
    scale: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build the combined AA, bb LS system for pole identification.

    For each element n:
      Build A (2*Ns+1, N+offs + N+1) where:
        Left block  (cols 0..N+offs-1):   weighted Dk    [for f(s)]
        Right block (cols N+offs..end-1): -weighted Dk*f [for sigma(s)]
        Relaxation row (2*Ns) appended only to last element.
      QR-decompose A → extract R22 (N+1, N+1) block.
      Stack R22 blocks into AA (nnn*(N+1), N+1).
      bb is derived from the last element's Q matrix.

    Uses vmap over element index for the QR step.

    Returns:
        AA: (nnn*(N+1), N+1) real
        bb: (nnn*(N+1),) real

    """
    # --- build left and right blocks for all elements simultaneously ---
    # weight: (nnn, Ns) or (1, Ns) → (nnn, Ns)
    w = jnp.broadcast_to(weight, (nnn, Ns))  # (nnn, Ns)

    # Left block: w * Dk[:, :N+offs]   → (nnn, Ns, N+offs)
    left = w[:, :, None] * Dk[None, :, : N + offs]  # broadcast over nnn

    # Right block: -w * Dk[:, :N+1] * f → (nnn, Ns, N+1)
    right = -w[:, :, None] * Dk[None, :, : N + 1] * f[:, :, None]

    # Concatenate → (nnn, Ns, N+offs + N+1)
    A_cmplx = jnp.concatenate([left, right], axis=2)

    # Stack [real; imag] → (nnn, 2*Ns, N+offs+N+1)
    A_real = jnp.concatenate([A_cmplx.real, A_cmplx.imag], axis=1)

    # Relaxation row (only for the last element, appended as row 2*Ns):
    # values = real(scale * sum(Dk[:, :N+1], axis=0))  placed at cols N+offs..end
    relax_vals = jnp.real(scale * jnp.sum(Dk[:, : N + 1], axis=0))  # (N+1,)
    relax_row_full = jnp.zeros(N + offs + N + 1)
    relax_row_full = relax_row_full.at[N + offs :].set(relax_vals)  # (N+offs+N+1,)

    # For all but last element: zero relax row; for last: actual relax row
    # Build (nnn, 1, N+offs+N+1) relax block
    zero_row = jnp.zeros((1, N + offs + N + 1))
    # Concatenate: (nnn-1) zero rows then 1 actual row
    relax_block = jnp.concatenate(
        [
            jnp.broadcast_to(zero_row, (nnn - 1, 1, N + offs + N + 1)),
            relax_row_full[None, None, :],
        ],
        axis=0,
    )  # (nnn, 1, N+offs+N+1)

    # Full A with relax row: (nnn, 2*Ns+1, N+offs+N+1)
    A_full = jnp.concatenate([A_real, relax_block], axis=1)

    # --- QR decompose each element's A matrix ---
    def _qr_one(A_n):
        Q, R = jnp.linalg.qr(A_n)  # reduced QR: Q (2*Ns+1, k), R (k, k)
        return Q, R

    Q_all, R_all = jax.vmap(_qr_one)(A_full)
    # Q_all: (nnn, 2*Ns+1, N+offs+N+1)
    # R_all: (nnn, N+offs+N+1, N+offs+N+1)

    # Extract R22 block: rows and cols N+offs .. N+offs+N
    R22_all = R_all[:, N + offs :, N + offs :]  # (nnn, N+1, N+1)

    # Stack into combined system AA: (nnn*(N+1), N+1)
    AA = R22_all.reshape(nnn * (N + 1), N + 1)

    # bb: from last element's Q matrix, row 2*Ns (the relaxation row)
    # bb[-N-1:] = Q_all[-1, 2*Ns, N+offs:] * Ns * scale
    bb = jnp.zeros(nnn * (N + 1))
    last_q_row = Q_all[-1, 2 * Ns, N + offs :]  # (N+1,)
    bb = bb.at[-(N + 1) :].set(jnp.real(last_q_row) * Ns * scale)

    return AA, bb


def _build_pole_id_ls_norelax(
    f: jnp.ndarray,
    s: jnp.ndarray,
    Dk: jnp.ndarray,
    weight: jnp.ndarray,
    N: int,
    Ns: int,
    nnn: int,
    offs: int,
    Dnew: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Non-relaxed LS system (fallback when sigma D is degenerate).

    Equivalent to vectfit4.m lines 381–434.
    """
    w = jnp.broadcast_to(weight, (nnn, Ns))  # (nnn, Ns)

    left = w[:, :, None] * Dk[None, :, : N + offs]
    right = -w[:, :, None] * Dk[None, :, :N] * f[:, :, None]
    b_cmplx = Dnew * w * f  # (nnn, Ns)

    A_cmplx = jnp.concatenate([left, right], axis=2)  # (nnn, Ns, N+offs+N)
    A_real = jnp.concatenate([A_cmplx.real, A_cmplx.imag], axis=1)  # (nnn, 2*Ns, ...)
    b_real = jnp.concatenate([b_cmplx.real, b_cmplx.imag], axis=1)  # (nnn, 2*Ns)

    def _qr_one(A_n, b_n):
        Q, R = jnp.linalg.qr(A_n)
        R22 = R[N + offs :, N + offs :]  # (N, N)
        bb_n = Q[:, N + offs :].T @ b_n  # (N,)
        return R22, bb_n

    R22_all, bb_all = jax.vmap(_qr_one)(A_real, b_real)
    # R22_all: (nnn, N, N), bb_all: (nnn, N)

    AA = R22_all.reshape(nnn * N, N)
    bb = bb_all.reshape(nnn * N)
    return AA, bb


def _solve_scaled(AA: jnp.ndarray, bb: jnp.ndarray) -> jnp.ndarray:
    """Column-scale AA, solve AA x = bb, unscale."""
    scale = jnp.linalg.norm(AA, axis=0)  # (n_cols,)
    scale = jnp.where(scale == 0, 1.0, scale)
    x, _, _, _ = jnp.linalg.lstsq(AA / scale[None, :], bb, rcond=None)
    return x / scale


def _real_to_complex_sigma(x_real: np.ndarray, cindex: np.ndarray, N: int) -> np.ndarray:
    """Convert real LS solution for sigma into complex coefficients.

    Equivalent to vectfit4.m lines 443–453.
    """
    # Must allocate as complex128 so the conjugate-pair values can be written.
    C = x_real[:N].copy().astype(np.complex128)
    for m in range(N):
        if cindex[m] == 1 and m + 1 < N:
            r1, r2 = float(x_real[m]), float(x_real[m + 1])
            C[m] = r1 + 1j * r2
            C[m + 1] = r1 - 1j * r2
    return C  # (N,) complex128


def _build_companion_matrix(poles: np.ndarray, sigma_C: np.ndarray, sigma_D: float, cindex: np.ndarray) -> np.ndarray:
    """Build the real companion matrix ZER with 2×2 blocks for complex pairs.

    Equivalent to vectfit4.m lines 491–506:
        - LAMBD starts as diag(poles)
        - For each complex pair (cindex[m]==1):
            LAMBD block → [[Re(a), Im(a)], [-Im(a), Re(a)]]
            B[m] = 2, B[m+1] = 0
            sigma_C_real[m]   = Re(sigma_C[m])
            sigma_C_real[m+1] = Im(sigma_C[m])
        - ZER = LAMBD_block - outer(B, sigma_C_real) / sigma_D

    Building ZER as a REAL matrix guarantees eigenvalues come in conjugate pairs.
    """
    N = len(poles)
    LAMBD = np.zeros((N, N), dtype=np.float64)
    B_col = np.zeros(N, dtype=np.float64)
    C_mod = np.zeros(N, dtype=np.float64)

    m = 0
    while m < N:
        if cindex[m] == 0:
            # Real pole
            LAMBD[m, m] = poles[m].real
            B_col[m] = 1.0
            C_mod[m] = sigma_C[m].real
            m += 1
        else:
            # Complex pair (cindex[m]==1, cindex[m+1]==2)
            a = poles[m].real
            b = poles[m].imag
            LAMBD[m, m] = a
            LAMBD[m, m + 1] = b
            LAMBD[m + 1, m] = -b
            LAMBD[m + 1, m + 1] = a
            B_col[m] = 2.0
            B_col[m + 1] = 0.0
            C_mod[m] = sigma_C[m].real
            C_mod[m + 1] = sigma_C[m].imag
            m += 2

    ZER = LAMBD - np.outer(B_col, C_mod) / sigma_D
    return ZER


def identify_poles(
    f: jnp.ndarray,  # (nnn, Ns) complex — elements to fit
    s: jnp.ndarray,  # (Ns,) complex
    poles: np.ndarray,  # (N,) complex — current poles (NumPy, outside JAX)
    weight: jnp.ndarray,  # (nnn, Ns) or (1, Ns)
    opts: FitOptions,
) -> np.ndarray:
    """Identify new poles from the current poles using FRVF (Stage 1).

    Args:
        f:      (nnn, Ns) complex — frequency-domain data (upper-triangle elements).
        s:      (Ns,) complex.
        poles:  (N,) complex — current pole set (NumPy array, updated each call).
        weight: (nnn, Ns) or (1, Ns) — fitting weights.
        opts:   FitOptions.

    Returns:
        new_poles: (N,) complex NumPy array — updated pole locations.

    """
    N = len(poles)
    Ns = s.shape[0]
    nnn = f.shape[0]
    offs = opts.asymp - 1  # 0, 1, or 2

    cindex = compute_cindex(poles)
    Dk = build_dk(s, jnp.array(poles), cindex, offs)  # (Ns, N+offs)

    # --- compute scale for relaxation row ---
    w = jnp.broadcast_to(weight, (nnn, Ns))
    scale_sq = jnp.sum(jnp.array([jnp.linalg.norm(w[n] * f[n]) ** 2 for n in range(nnn)]))
    scale = float(jnp.sqrt(scale_sq) / Ns)

    # --- relaxed LS system ---
    if opts.relax:
        AA, bb = _build_pole_id_ls(f, s, Dk, weight, N, Ns, nnn, offs, scale)
        x = _solve_scaled(AA, bb)
        x_np = np.array(x)
        sigma_D = x_np[-1]
    else:
        sigma_D = 0.0  # force fallback

    # --- fallback: non-relaxed LS (degenerate sigma D) ---
    use_fallback = not opts.relax or abs(sigma_D) < _TOL_LOW or abs(sigma_D) > _TOL_HIGH
    if use_fallback:
        if sigma_D == 0.0:
            Dnew = 1.0
        elif abs(sigma_D) < _TOL_LOW:
            Dnew = float(np.sign(sigma_D) * _TOL_LOW)
        else:
            Dnew = float(np.sign(sigma_D) * _TOL_HIGH)

        AA2, bb2 = _build_pole_id_ls_norelax(f, s, Dk, weight, N, Ns, nnn, offs, Dnew)
        x2 = _solve_scaled(AA2, bb2)
        # Append Dnew at end to match expected format
        x_np = np.append(np.array(x2), Dnew)
        sigma_D = Dnew

    # --- convert sigma coefficients from real to complex ---
    sigma_C = _real_to_complex_sigma(x_np, cindex, N)

    # --- build companion matrix and compute eigenvalues ---
    ZER = _build_companion_matrix(poles, sigma_C, sigma_D, cindex)
    # Keep the eigensolve on the JAX side. This is the dense operation that
    # must be batchable when pole relocation is lifted into a fixed-shape
    # candidate kernel; the surrounding compatibility path still returns a
    # NumPy array for the existing VF driver.
    new_poles = np.array(jnp.linalg.eigvals(jnp.asarray(ZER)), copy=True)  # (N,) complex

    # --- enforce stability: flip poles with positive real part ---
    if opts.stable:
        unstable = new_poles.real > 0
        new_poles[unstable] = new_poles[unstable] - 2 * new_poles[unstable].real

    # --- sort and cleanup ---
    new_poles = sort_poles(new_poles)

    return new_poles.astype(np.complex128)
