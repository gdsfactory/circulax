"""Stage 2: Residue identification via weighted least squares.

Ported from vectfit4.m lines 542–750 (SINTEF MFT-NNLS, B. Gustavsen, 2026).

The core computation is a per-element LS solve: for each of the nnn upper-triangle
elements, find residues C, and optionally D and E terms, given fixed poles.
This is the primary vmap target: each element is independent.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .utils import build_dk, compute_cindex


def _residue_ls_one(
    f_n: jnp.ndarray,  # (Ns,) complex — one element's frequency response
    w_n: jnp.ndarray,  # (Ns,) real    — weights for this element
    Dk_uw: jnp.ndarray,  # (Ns, N+offs) complex — UNweighted basis (shared)
    s: jnp.ndarray,  # (Ns,) complex
    offs: int,  # 0, 1, or 2 (number of extra columns: D and/or E)
) -> jnp.ndarray:
    """Solve the residue LS problem for one element.

    Build: A (2*Ns, N+offs) with column scaling.
    Solve: A @ x = b  in real arithmetic.
    Return: x (N+offs,) real — [residue coefficients | D | E].

    The residue coefficients use the same real representation as the basis
    functions in Dk_uw (conjugate pairs stored as real+imag pairs).
    """
    # Apply per-element weight to basis and target
    Dk_w = w_n[:, None] * Dk_uw  # (Ns, N+offs) — weighted basis
    b_w = w_n * f_n  # (Ns,) complex — weighted target

    # Stack [real; imag] to convert complex LS to real
    A = jnp.concatenate([Dk_w.real, Dk_w.imag], axis=0)  # (2*Ns, N+offs)
    b = jnp.concatenate([b_w.real, b_w.imag], axis=0)  # (2*Ns,)

    # Column scaling to improve conditioning
    scale = jnp.linalg.norm(A, axis=0)  # (N+offs,)
    scale = jnp.where(scale == 0, 1.0, scale)

    x, _, _, _ = jnp.linalg.lstsq(A / scale[None, :], b, rcond=None)
    return x / scale  # (N+offs,) real


def _real_to_complex_residues(C_real: np.ndarray, cindex: np.ndarray, N: int) -> np.ndarray:
    """Convert real residue coefficients back to complex.

    For each complex pair (cindex[m]==1):
        c[m]   = r1 + i*r2
        c[m+1] = r1 - i*r2

    Equivalent to vectfit4.m lines 732–739.
    """
    C = C_real.copy().astype(np.complex128)
    for m in range(N):
        if cindex[m] == 1 and m + 1 < N:
            r1, r2 = float(C_real[m]), float(C_real[m + 1])
            C[m] = r1 + 1j * r2
            C[m + 1] = r1 - 1j * r2
    return C


def identify_residues(
    f: jnp.ndarray,  # (nnn, Ns) complex — upper-triangle elements
    s: jnp.ndarray,  # (Ns,) complex
    poles: np.ndarray,  # (N,) complex NumPy — current (fixed) poles
    weight: jnp.ndarray,  # (nnn, Ns) or (1, Ns) — fitting weights
    opts,  # FitOptions
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Identify residues (and D, E terms) for all upper-triangle elements.

    Returns:
        C:     (nnn, N) complex — residues for each element and each pole.
        D_vec: (nnn,) float    — constant terms (zero if opts.asymp == 1).
        E_vec: (nnn,) float    — linear terms   (zero if opts.asymp < 3).

    """
    N = len(poles)
    Ns = s.shape[0]
    nnn = f.shape[0]
    offs = opts.asymp - 1  # number of extra columns (0, 1, or 2)

    cindex = compute_cindex(poles)
    Dk_uw = build_dk(s, jnp.array(poles), cindex, offs)  # (Ns, N+offs)

    # Broadcast weight to (nnn, Ns)
    w = jnp.broadcast_to(weight, (nnn, Ns)).real  # weights are real

    # vmap the per-element LS solve over the nnn elements
    _solve_vmap = jax.vmap(
        lambda f_n, w_n: _residue_ls_one(f_n, w_n, Dk_uw, s, offs),
        in_axes=(0, 0),
    )
    x_all = _solve_vmap(f, w)  # (nnn, N+offs) real

    # Split into residue coefficients and D, E
    C_real = np.array(x_all[:, :N])  # (nnn, N) — real representation

    # Convert each element's residues from real to complex
    C_complex = np.stack(
        [_real_to_complex_residues(C_real[n], cindex, N) for n in range(nnn)],
        axis=0,
    )  # (nnn, N) complex

    # Extract D and E from the extra columns
    if opts.asymp >= 2:
        D_vec = jnp.real(x_all[:, N])  # (nnn,) — D term
    else:
        D_vec = jnp.zeros(nnn)

    if opts.asymp >= 3:
        E_vec = jnp.real(x_all[:, N + 1])  # (nnn,) — E term
    else:
        E_vec = jnp.zeros(nnn)

    return jnp.array(C_complex), D_vec, E_vec
