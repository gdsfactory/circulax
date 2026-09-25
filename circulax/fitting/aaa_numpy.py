"""NumPy-only scalar AAA discovery, preserved from vfitax.

No JAX computation or pole stabilization is performed in this module.
"""

from __future__ import annotations

import numpy as np


def aaa_scalar_numpy(
    f: np.ndarray,  # (Ns,) complex — function values at sample points
    z: np.ndarray,  # (Ns,) complex — sample points
    tol: float = 1e-10,
    mmax: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """AAA rational approximation of a scalar function.

    Implements Algorithm 1 from Nakatsukasa, Sète & Trefethen (2018).

    Args:
        f:    (Ns,) complex — function values at sample points.
        z:    (Ns,) complex — sample points.
        tol:  convergence tolerance (relative error norm).
        mmax: maximum number of support points.

    Returns:
        w:    (m,) complex — barycentric weights.
        zj:   (m,) complex — support nodes.
        fj:   (m,) complex — function values at support nodes.

    """
    f = np.asarray(f, dtype=np.complex128)
    z = np.asarray(z, dtype=np.complex128)
    Ns = len(z)

    # Support set indices (indices into original arrays)
    support_mask = np.zeros(Ns, dtype=bool)

    # Barycentric nodes / values (grown iteratively)
    zj_list = []
    fj_list = []

    # Initial residual is the function itself
    residual = f.copy()
    norm_f = np.linalg.norm(f)
    if norm_f == 0.0:
        norm_f = 1.0

    w = np.array([], dtype=np.complex128)

    for _m in range(mmax):
        # Step a: find index with maximum |residual| among non-support points
        non_support = ~support_mask
        if not np.any(non_support):
            break

        candidates = np.where(non_support)[0]
        j_star = candidates[np.argmax(np.abs(residual[candidates]))]

        # Step b: add to support set
        support_mask[j_star] = True
        zj_list.append(z[j_star])
        fj_list.append(f[j_star])

        zj = np.array(zj_list, dtype=np.complex128)
        fj = np.array(fj_list, dtype=np.complex128)
        m = len(zj)

        # Indices of non-support points
        non_sup_idx = np.where(~support_mask)[0]

        if m == 1:
            # With only one support point, rational approx is constant = f[j*]
            # weights are just [1]
            w = np.array([1.0], dtype=np.complex128)
            # Evaluate r at non-support points
            r = np.full(Ns, fj[0], dtype=np.complex128)
            r[support_mask] = fj  # exact at support
            residual = np.where(support_mask, 0.0, f - r)
            err = np.linalg.norm(residual) / norm_f
            if err < tol:
                break
            continue

        # Step c: build Loewner matrix for non-support rows
        # L[k, j] = (f[k] - fj[j]) / (z[k] - zj[j])
        z_nonsup = z[non_sup_idx]  # (n_nonsup,)
        f_nonsup = f[non_sup_idx]  # (n_nonsup,)

        # (n_nonsup, m) Loewner matrix
        denom = z_nonsup[:, None] - zj[None, :]  # (n_nonsup, m)
        numer = f_nonsup[:, None] - fj[None, :]  # (n_nonsup, m)
        # Avoid division by zero (shouldn't happen since support pts excluded)
        with np.errstate(divide="ignore", invalid="ignore"):
            L = np.where(np.abs(denom) == 0, 0.0, numer / denom)

        # Step d: SVD of L → weights = last right singular vector (conjugated)
        if L.shape[0] >= m:
            _, _, Vh = np.linalg.svd(L, full_matrices=False)
            w = Vh[-1, :].conj()
        else:
            # Underdetermined: use right null vector
            _, _, Vh = np.linalg.svd(L, full_matrices=True)
            w = Vh[-1, :].conj()

        # Step e: evaluate barycentric rational approximation at non-support points
        # r(z) = N(z)/D(z)  where N(z) = sum w_j*f_j/(z-z_j), D(z) = sum w_j/(z-z_j)
        with np.errstate(divide="ignore", invalid="ignore"):
            cauchy = 1.0 / (z_nonsup[:, None] - zj[None, :])  # (n_nonsup, m)

        N_vals = cauchy @ (w * fj)  # (n_nonsup,)
        D_vals = cauchy @ w  # (n_nonsup,)

        # Handle near-zero denominators (near support points, shouldn't occur)
        safe = np.abs(D_vals) > 1e-300
        r_nonsup = np.where(safe, N_vals / D_vals, f_nonsup)

        # Step f: update residual
        residual = np.zeros(Ns, dtype=np.complex128)
        residual[non_sup_idx] = f_nonsup - r_nonsup
        # support points have zero residual (exact)

        # Step g: check convergence
        err = np.linalg.norm(residual) / norm_f
        if err < tol:
            break

    return w, zj, fj
