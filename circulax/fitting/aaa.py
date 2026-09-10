"""AAA (Adaptive Antoulas-Anderson) rational approximation.

Implements the AAA algorithm (Nakatsukasa, Sète, Trefethen 2018) as an
alternative pole-finding stage compatible with the Circulax fitting infrastructure.

References:
    Nakatsukasa, Y., Sète, O., & Trefethen, L. N. (2018).
    The AAA algorithm for rational approximation.
    SIAM Journal on Scientific Computing, 40(3), A1494-A1522.

"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import scipy.linalg

from .residue_id import identify_residues
from .types import FitOptions, SSModel, VFModel, eval_model, vfmodel_to_ss
from .utils import (
    _upper_triangle_indices,
    compute_rmserr,
    compute_weights,
    sort_poles,
    stack_upper_triangle,
)

# ---------------------------------------------------------------------------
# Core AAA algorithm (scalar, numpy-based)
# ---------------------------------------------------------------------------


def aaa_scalar(
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


# ---------------------------------------------------------------------------
# Pole extraction from barycentric form
# ---------------------------------------------------------------------------


def _aaa_poles(w: np.ndarray, zj: np.ndarray) -> np.ndarray:
    """Extract poles of the AAA barycentric rational approximation.

    Uses the generalized eigenvalue formulation from Theorem 4.1 of
    Nakatsukasa et al. (2018), identical to chebfun's aaa.m implementation.

    Args:
        w:  (m,) complex — barycentric weights.
        zj: (m,) complex — support nodes.

    Returns:
        pol: finite poles (at most m-1).

    """
    m = len(zj)
    if m == 0:
        return np.array([], dtype=np.complex128)

    # Build (m+1) x (m+1) generalized eigenvalue pencil (E, B)
    B = np.eye(m + 1, dtype=np.complex128)
    B[0, 0] = 0.0

    E = np.zeros((m + 1, m + 1), dtype=np.complex128)
    E[0, 1:] = w
    E[1:, 0] = 1.0
    E[1:, 1:] = np.diag(zj)

    pol = scipy.linalg.eig(E, B, right=False)
    # Keep only finite eigenvalues
    pol = pol[np.isfinite(pol)]
    return pol


# ---------------------------------------------------------------------------
# Pole post-processing
# ---------------------------------------------------------------------------


def _collect_poles(all_poles: list[np.ndarray]) -> np.ndarray:
    """Merge poles from multiple AAA runs and post-process.

    Steps:
    1. Concatenate all poles.
    2. Flip unstable poles to left half-plane.
    3. Remove near-duplicate poles (O(n²) — n is small).
    4. Enforce exact conjugate pairing.
    5. Sort using sort_poles().

    Args:
        all_poles: list of pole arrays from individual aaa_scalar runs.

    Returns:
        poles: (N,) complex numpy array of cleaned, sorted poles.

    """
    if not all_poles or all(len(p) == 0 for p in all_poles):
        return np.array([], dtype=np.complex128)

    poles = np.concatenate([p for p in all_poles if len(p) > 0])

    if len(poles) == 0:
        return poles

    # Step 2: enforce stability — flip positive real-part poles
    poles = np.where(poles.real > 0, -poles.real + 1j * poles.imag, poles)

    # Step 3: remove near-duplicate poles (O(n²) cluster dedup — n is small)
    if len(poles) > 1:
        keep_flags = np.ones(len(poles), dtype=bool)
        for i in range(len(poles)):
            if not keep_flags[i]:
                continue
            for j in range(i + 1, len(poles)):
                if not keep_flags[j]:
                    continue
                if abs(poles[j] - poles[i]) / max(abs(poles[i]), 1.0) < 1e-4:
                    keep_flags[j] = False
        poles = poles[keep_flags]

    # Step 4: enforce exact conjugate pairing — after dedup, each complex pole
    # survivor gets an exact conj(p) partner if one is not already present.
    # Use relative tolerance to detect near-conjugates (poles carry FP noise).
    unique = list(poles)
    for p in poles:
        if p.imag == 0.0:
            continue
        conj_p = p.real - 1j * p.imag
        rtol = 1e-4
        present = any(abs(q - conj_p) / max(abs(conj_p), 1.0) < rtol for q in unique)
        if not present:
            unique.append(conj_p)
    poles = np.array(unique, dtype=np.complex128)

    # Step 5: sort
    poles = sort_poles(poles)

    return poles


# ---------------------------------------------------------------------------
# Main AAA driver
# ---------------------------------------------------------------------------


def aaa_driver(
    bigH: jnp.ndarray,  # (Nc, Nc, Ns) complex
    s: jnp.ndarray,  # (Ns,) complex
    opts: FitOptions,  # reuses asymp, weightparam; N is ignored
    tol: float = 1e-10,  # AAA convergence tolerance
    mmax: int = 100,  # max support points per element
    verbose: bool = True,
) -> tuple[VFModel, SSModel, float, jnp.ndarray]:
    """Fit bigH(s) using AAA rational approximation for pole finding.

    Replaces the VF pole-finding phase with AAA applied to each diagonal
    element. The poles found are then used with the standard residue
    identification step.

    Args:
        bigH:    (Nc, Nc, Ns) complex — frequency-domain matrix data.
        s:       (Ns,) complex — frequency points (= j*omega).
        opts:    FitOptions — asymp and weightparam are used; N is ignored.
        tol:     AAA convergence tolerance (per diagonal element).
        mmax:    maximum AAA support points per diagonal element.
        verbose: print progress.

    Returns:
        model:   VFModel — fitted pole-residue model.
        ss:      SSModel — state-space form.
        rmserr:  float   — final RMS fitting error.
        bigHfit: (Nc, Nc, Ns) — reconstructed frequency response.

    """
    Nc = bigH.shape[0]
    Ns = s.shape[0]

    bigH_np = np.asarray(bigH, dtype=np.complex128)
    s_np = np.asarray(s, dtype=np.complex128)

    # -------------------------------------------------------------------------
    # Step 1-2: Run AAA on each diagonal element and extract poles
    # -------------------------------------------------------------------------
    all_poles = []

    for i in range(Nc):
        f_diag = bigH_np[i, i, :]  # (Ns,) complex diagonal element

        w, zj, fj = aaa_scalar(f_diag, s_np, tol=tol, mmax=mmax)

        if len(zj) > 1:
            pols = _aaa_poles(w, zj)
        else:
            pols = np.array([], dtype=np.complex128)

        all_poles.append(pols)

        if verbose:
            print(f"  AAA diagonal [{i},{i}]: {len(zj)} support pts, {len(pols)} raw poles")

    # -------------------------------------------------------------------------
    # Step 3: Collect and clean all poles
    # -------------------------------------------------------------------------
    poles_np = _collect_poles(all_poles)

    if verbose:
        print(f"AAA driver: {len(poles_np)} poles after post-processing")

    if len(poles_np) == 0:
        raise RuntimeError("AAA found no poles. Try increasing mmax or reducing tol.")

    # -------------------------------------------------------------------------
    # Steps 5-7: Residue identification using standard VF machinery
    # -------------------------------------------------------------------------
    f_full = stack_upper_triangle(bigH)  # (nnn, Ns)
    w_full = compute_weights(bigH, opts.weightparam)  # (nnn, Ns) or (1, Ns)

    C_flat, D_vec, E_vec = identify_residues(f_full, s, poles_np, w_full, opts)
    # C_flat: (nnn, N) complex
    # D_vec:  (nnn,) float
    # E_vec:  (nnn,) float

    # -------------------------------------------------------------------------
    # Step 8: Reconstruct VFModel
    # -------------------------------------------------------------------------
    N = len(poles_np)
    nnn = f_full.shape[0]
    idx = _upper_triangle_indices(Nc)

    residues = jnp.zeros((Nc, Nc, N), dtype=jnp.complex128)
    D_mat = jnp.zeros((Nc, Nc), dtype=jnp.float64)
    E_mat = jnp.zeros((Nc, Nc), dtype=jnp.float64)

    for k, (r, c) in enumerate(idx):
        residues = residues.at[r, c, :].set(C_flat[k])
        D_mat = D_mat.at[r, c].set(float(D_vec[k]))
        E_mat = E_mat.at[r, c].set(float(E_vec[k]))
        if r != c:
            # Y-matrices are symmetric (not Hermitian): H[c,r] = H[r,c], not conj.
            residues = residues.at[c, r, :].set(C_flat[k])
            D_mat = D_mat.at[c, r].set(float(D_vec[k]))
            E_mat = E_mat.at[c, r].set(float(E_vec[k]))

    model = VFModel(
        poles=jnp.array(poles_np),
        residues=residues,
        D=D_mat,
        E=E_mat,
    )

    # -------------------------------------------------------------------------
    # Step 9: Build SSModel, evaluate, compute error
    # -------------------------------------------------------------------------
    ss = vfmodel_to_ss(model, Nc)
    bigHfit_stacked = eval_model(s, ss)  # (Ns, Nc, Nc)
    bigHfit = jnp.moveaxis(bigHfit_stacked, 0, -1)  # (Nc, Nc, Ns)

    H_data = jnp.moveaxis(bigH, -1, 0)  # (Ns, Nc, Nc)
    rmserr = compute_rmserr(H_data, bigHfit_stacked)

    return model, ss, rmserr, bigHfit
