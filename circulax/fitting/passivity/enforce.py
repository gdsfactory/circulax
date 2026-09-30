"""Passivity enforcement for Y-parameter models.

Ports:
  - ``RPdriver.m``      → enforce_passivity (outer loop)
  - ``RP_QRNNLS_Y.m``  → _rp_qrnnls_y (single RP iteration)

Algorithm reference:
  B. Gustavsen, "Passivity enforcement by residue perturbation via constrained
  non-negative least squares", IEEE Trans. Power Delivery, vol. 36, no. 5,
  pp. 2758–2767, October 2021.
"""

from __future__ import annotations

from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

from ..types import FitOptions, SSModel, VFModel, vfmodel_to_ss
from ..utils import compute_cindex
from .check import find_violation_bands, find_violation_extrema, passivity_sweep_Y
from .nnls import solve_passivity_nnls

# ---------------------------------------------------------------------------
# Upper-triangle element indexing (row-major, consistent with stack_upper_triangle)
# ---------------------------------------------------------------------------


def _triu_indices(Nc: int):
    """(rows, cols) for the upper triangle of an Nc×Nc matrix, row-major."""
    return np.triu_indices(Nc)


# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------


def _build_basis(
    s: np.ndarray,
    poles: np.ndarray,
    cindex: np.ndarray,
    Dflag: bool,
    Eflag: bool,
) -> np.ndarray:
    """Build the real-valued basis matrix Mmat_unw[k, m].

    For the Y-parameter LS problem, each column is:
      - Real pole aₘ:         1/(s_k − aₘ)
      - Complex pair first m:  1/(s−aₘ) + 1/(s−āₘ)
      - Complex pair second m: i/(s−aₘ) − i/(s−āₘ)
      - D term (if Dflag):     1
      - E term (if Eflag):     s_k

    Returns:
        Mmat_unw: (Ns, N+DEflag) complex array.

    """
    N = len(poles)
    DEflag = int(Dflag) + int(Eflag)
    Ndum = N + DEflag
    Ns = len(s)

    Mmat = np.zeros((Ns, Ndum), dtype=complex)
    m = 0
    while m < N:
        if cindex[m] == 0:  # real pole
            Mmat[:, m] = 1.0 / (s - poles[m])
            m += 1
        else:  # complex pair
            a, ac = poles[m], np.conj(poles[m])
            Mmat[:, m] = 1.0 / (s - a) + 1.0 / (s - ac)
            Mmat[:, m + 1] = 1j / (s - a) - 1j / (s - ac)
            m += 2

    if Dflag:
        Mmat[:, N] = 1.0
    if Eflag:
        Mmat[:, N + int(Dflag)] = s

    return Mmat


def _basis_real_at(
    sk: complex,
    poles: np.ndarray,
    cindex: np.ndarray,
    Dflag: bool,
    Eflag: bool,
) -> np.ndarray:
    """Real part of the basis vector at a single frequency sk.

    Used in the constraint matrix where only the real part contributes
    (residue perturbations are real-valued by symmetry of Y-parameters).
    """
    N = len(poles)
    DEflag = int(Dflag) + int(Eflag)
    Ndum = N + DEflag
    dum = np.zeros(Ndum)

    m = 0
    while m < N:
        if cindex[m] == 0:
            dum[m] = np.real(1.0 / (sk - poles[m]))
            m += 1
        else:
            a, ac = poles[m], np.conj(poles[m])
            dum[m] = np.real(1.0 / (sk - a) + 1.0 / (sk - ac))
            dum[m + 1] = np.real(1j / (sk - a) - 1j / (sk - ac))
            m += 2

    if Dflag:
        dum[N] = 1.0  # real(1) = 1
    # E term real(s_k) = real(jω) = 0 → stays zero

    return dum


# ---------------------------------------------------------------------------
# Auxiliary frequency extension (auxflag = 1 in MATLAB)
# ---------------------------------------------------------------------------


def _augment_frequencies(
    s: np.ndarray,
    poles: np.ndarray,
    cindex: np.ndarray,
    violpairs: list[dict],
    Eflag: bool,
    weightfactor: float = 1e-3,
) -> tuple[np.ndarray, int]:
    """Append auxiliary frequency samples to the sweep array.

    Adds:
    1. Pole resonant frequencies that lie outside the original sweep range.
    2. Violation frequencies from violpairs.
    3. DC (s = 0) unless an E term is present (which is improper at DC).

    The extra samples are downweighted by ``weightfactor`` in the LS system.

    Returns:
        s_full: Extended frequency array.
        Ns_orig: Length of original s (to identify extra samples for weighting).

    """
    Ns_orig = len(s)
    w_lo, w_hi = float(np.imag(s[0])), float(np.imag(s[-1]))
    s_extra = []

    # Add pole resonant frequencies outside sweep band.
    for m, p in enumerate(poles):
        if cindex[m] == 0:
            w_p = abs(p.real)  # real pole: use |re(a)|
        elif cindex[m] == 1:
            w_p = abs(p.imag)  # complex pair: use |im(a)|
        else:
            continue
        if w_p > 0 and (w_p > w_hi or w_p < w_lo):
            s_extra.append(1j * w_p)

    # Add violation frequencies.
    if violpairs:
        for p in violpairs:
            s_extra.append(1j * p["omega"])

    # DC unless the model has an E term.
    if not Eflag:
        s_extra.append(0.0 + 0j)

    if s_extra:
        s_extra_arr = np.unique(np.array(s_extra))
        s_full = np.concatenate([s, s_extra_arr])
    else:
        s_full = s

    return s_full, Ns_orig


# ---------------------------------------------------------------------------
# LS system — QR decomposition (built once, cached)
# ---------------------------------------------------------------------------


def _build_ls_system(
    s_full: np.ndarray,
    poles: np.ndarray,
    cindex: np.ndarray,
    Nc: int,
    Dflag: bool,
    Eflag: bool,
    nnn: int,
    Ns_orig: int,
    weightfactor: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the QR-factorised LS system for weighted uniform (weightparam=1) fitting.

    Port of the ``weightparam==1`` branch in RP_QRNNLS_Y.m.

    Returns:
        Rsub:      ``(Ndum, Ndum)`` upper-triangular R factor (one block for all elements).
        bigEscale: ``(nnn*Ndum,)`` column scale array (√2 for off-diagonal element blocks).

    """
    N = len(poles)
    DEflag = int(Dflag) + int(Eflag)
    Ndum = N + DEflag
    Ns_full = len(s_full)

    Mmat = _build_basis(s_full, poles, cindex, Dflag, Eflag)  # (Ns_full, Ndum)

    # Downweight out-of-band samples.
    Mmat_w = Mmat.copy()
    if Ns_full > Ns_orig:
        Mmat_w[Ns_orig:] *= weightfactor

    # Stack real and imaginary parts.
    koko = np.vstack([np.real(Mmat_w), np.imag(Mmat_w)])  # (2*Ns_full, Ndum)

    # Column scale for better conditioning.
    Escale = np.linalg.norm(koko, axis=0)  # (Ndum,)
    Escale = np.where(Escale < 1e-30, 1.0, Escale)
    _, Rsub = np.linalg.qr(koko / Escale, mode="reduced")  # (Ndum, Ndum)

    # Build bigEscale: replicate Escale for each of the nnn element blocks,
    # with √2 factor for off-diagonal elements (they appear twice in Y by symmetry).
    rows_tri, cols_tri = _triu_indices(Nc)
    sq2 = np.sqrt(2.0)
    bigEscale = np.empty(nnn * Ndum)
    for n, (r, c) in enumerate(zip(rows_tri, cols_tri)):
        factor = 1.0 if r == c else sq2
        bigEscale[n * Ndum : (n + 1) * Ndum] = factor * Escale

    return Rsub, bigEscale


# ---------------------------------------------------------------------------
# Constraint matrix construction
# ---------------------------------------------------------------------------


def _build_constraint_matrix(
    violpairs: list[dict],
    poles: np.ndarray,
    cindex: np.ndarray,
    Nc: int,
    N: int,
    Dflag: bool,
    Eflag: bool,
    nnn: int,
    TOLG: float,
    TOLD: float,
    TOLE: float,
    alpha: float,
    ss: SSModel,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build the constraint matrix bigB and RHS vector bigc.

    Port of the constraint-building loop in RP_QRNNLS_Y.m.

    For each violation pair (eigenvector, lambda_min, omega):
        bigB[row, block_n] = outer(v,v')[elem_n] × Mmat2[elem_n]
        bigc[row]          = −TOLG + α × λ_min

    Returns:
        bigB:   ``(offsB, nnn*Ndum)`` constraint matrix.
        bigc:   ``(offsB,)`` target vector.
        offsB:  Number of active constraint rows.

    """
    DEflag = int(Dflag) + int(Eflag)
    Ndum = N + DEflag
    rows_tri, cols_tri = _triu_indices(Nc)

    max_rows = len(violpairs) + Nc * int(Dflag) + Nc * int(Eflag) + 4
    bigB = np.zeros((max_rows, nnn * Ndum))
    bigc = np.zeros(max_rows)
    offsB = 0

    # --- Frequency-domain passivity violations ---
    pairs_by_group: dict[int, list] = defaultdict(list)
    for p in violpairs:
        pairs_by_group[p["group"]].append(p)

    for group, group_pairs in sorted(pairs_by_group.items()):
        # Use the frequency of the first (worst) pair in this group for dum.
        sk = 1j * group_pairs[0]["omega"]
        dum = _basis_real_at(sk, poles, cindex, Dflag, Eflag)  # (Ndum,)

        # Mmat2[elem_n] = factor × dum  (factor=1 diagonal, 2 off-diagonal)
        Mmat2 = np.zeros((nnn, Ndum))
        for n, (r, c) in enumerate(zip(rows_tri, cols_tri)):
            factor = 1.0 if r == c else 2.0
            Mmat2[n] = factor * dum

        for pair in group_pairs:
            v = np.asarray(pair["eigvec"], dtype=complex)
            outer_vv = np.real(np.outer(v, np.conj(v)))  # (Nc, Nc) real

            for n, (r, c) in enumerate(zip(rows_tri, cols_tri)):
                Q_n = outer_vv[r, c]
                bigB[offsB, n * Ndum : (n + 1) * Ndum] = Q_n * Mmat2[n]

            bigc[offsB] = -TOLG + alpha * pair["lambda_min"]
            offsB += 1

    # Apply MATLAB's bigc correction (lines 476-479 of RP_QRNNLS_Y.m).
    # For rows where bigc > 0, remap to keep perturbation proportional to violation depth.
    for i in range(offsB):
        if bigc[i] > 0:
            bigc[i] = (TOLG + bigc[i]) / alpha
            bigc[i] = (2 - alpha) * bigc[i] - TOLG

    # --- D-matrix eigenvalue constraints (if Dflag) ---
    if Dflag:
        D_arr = np.real(np.array(ss.D))
        eigD, VD = np.linalg.eigh(D_arr)

        Mmat5 = np.array([1.0 if r == c else 2.0 for r, c in zip(rows_tri, cols_tri)])

        for n_eig in range(Nc):
            v = VD[:, n_eig]
            outer_vv = np.real(np.outer(v, np.conj(v)))

            for n, (r, c) in enumerate(zip(rows_tri, cols_tri)):
                Q_n = outer_vv[r, c]
                # D-term is at position N within each Ndum block.
                bigB[offsB, n * Ndum + N] = Q_n * Mmat5[n]

            lam = float(eigD[n_eig])
            bigc[offsB] = (-TOLD + alpha * lam) if lam < 0 else (-TOLD + lam)
            offsB += 1

    # --- E-matrix eigenvalue constraints (if Eflag) ---
    if Eflag:
        E_arr = np.real(np.array(ss.E))
        eigE, VE = np.linalg.eigh(E_arr)
        D_col = N + int(Dflag)  # position of E term within Ndum block

        Mmat5 = np.array([1.0 if r == c else 2.0 for r, c in zip(rows_tri, cols_tri)])

        for n_eig in range(Nc):
            v = VE[:, n_eig]
            outer_vv = np.real(np.outer(v, np.conj(v)))

            for n, (r, c) in enumerate(zip(rows_tri, cols_tri)):
                Q_n = outer_vv[r, c]
                bigB[offsB, n * Ndum + D_col] = Q_n * Mmat5[n]

            lam = float(eigE[n_eig])
            bigc[offsB] = -TOLE + alpha * lam
            offsB += 1

    return bigB[:offsB], bigc[:offsB], offsB


# ---------------------------------------------------------------------------
# Solve NNLS and recover perturbation
# ---------------------------------------------------------------------------


def _solve_perturbation(
    bigB: np.ndarray,
    bigc: np.ndarray,
    offsB: int,
    Rsub: np.ndarray,
    bigEscale: np.ndarray,
    nnn: int,
    N: int,
    DEflag: int,
) -> np.ndarray:
    """Solve the NNLS problem and recover the residue perturbation vector dx.

    Steps (port of the NNLS solve in RP_QRNNLS_Y.m):

    1. Scale bigB columns by 1/bigEscale.
    2. For each element block n: E_block = bigB_scaled_block @ inv(Rsub).
    3. Solve augmented NNLS → xbar.
    4. Back-solve each block through Rsub: dx_block = Rsub⁻¹ xbar_block.
    5. Divide by bigEscale.

    Returns:
        dx: ``(nnn*Ndum,)`` perturbation in original (unscaled) space.

    """
    Ndum = N + DEflag
    n_vars = nnn * Ndum

    # Scale bigB by 1/bigEscale (element-wise per column).
    bigB_s = bigB / bigEscale[np.newaxis, :]  # (offsB, n_vars)

    # Compute E = bigB_s @ inv(Rsub) block-by-block.
    # bigB_s[:, block] / Rsub  ≡  bigB_s[:, block] @ inv(Rsub)
    # Use solve_triangular for numerical stability.
    E_mat = np.empty_like(bigB_s)
    for n in range(nnn):
        a, b = n * Ndum, (n + 1) * Ndum
        E_mat[:, a:b] = scipy.linalg.solve_triangular(Rsub, bigB_s[:, a:b].T, lower=False).T

    # Augmented NNLS → xbar in pre-conditioned space.
    xbar = solve_passivity_nnls(E_mat, bigc)  # (n_vars,)

    # Back-solve through Rsub: dx_block = Rsub⁻¹ xbar_block.
    dx = np.empty(n_vars)
    for n in range(nnn):
        a, b = n * Ndum, (n + 1) * Ndum
        dx[a:b] = scipy.linalg.solve_triangular(Rsub, xbar[a:b], lower=False)

    # Unscale.
    dx /= bigEscale

    return dx


# ---------------------------------------------------------------------------
# Apply perturbation to the model
# ---------------------------------------------------------------------------


def _apply_perturbation(
    dx: np.ndarray,
    cindex: np.ndarray,
    model: VFModel,
    N: int,
    Nc: int,
    nnn: int,
    Dflag: bool,
    Eflag: bool,
) -> VFModel:
    """Convert the real perturbation vector dx into residue/D/E updates.

    Port of the recovery block in RP_QRNNLS_Y.m (lines 713–752).

    For complex pole pairs, the real and imaginary perturbation components
    are combined so the residue update remains conjugate-symmetric.
    """
    DEflag = int(Dflag) + int(Eflag)
    Ndum = N + DEflag

    # Convert real-encoded complex-pole perturbations back to complex.
    # cindex_ext[m] tells which role each column plays:
    #   0 = real pole or D/E term, 1 = complex pair 1st, 2 = complex pair 2nd
    cindex_ext = np.concatenate([cindex, np.zeros(DEflag, dtype=int)])
    bigindex = np.tile(cindex_ext, nnn)  # (nnn*Ndum,)
    idx1 = np.where(bigindex == 1)[0]
    idx2 = np.where(bigindex == 2)[0]
    dx = dx.astype(complex)
    dx[idx1] = dx[idx1] + 1j * dx[idx2]
    dx[idx2] = np.conj(dx[idx1])

    rows_tri, cols_tri = _triu_indices(Nc)
    new_res = np.array(model.residues)  # (Nc, Nc, N) complex
    new_D = np.array(model.D, dtype=float)  # (Nc, Nc)
    new_E = np.array(model.E, dtype=float)  # (Nc, Nc)

    # --- Residue perturbation ---
    for m in range(N):
        # Gather the m-th coefficient from each of the nnn element blocks.
        rdum = dx[m::Ndum][:nnn]  # (nnn,)
        Rdum = np.zeros((Nc, Nc), dtype=complex)
        for n, (r, c) in enumerate(zip(rows_tri, cols_tri)):
            Rdum[r, c] = rdum[n]
            if r != c:
                Rdum[c, r] = rdum[n]  # symmetrise
        new_res[:, :, m] = new_res[:, :, m] + Rdum

    # --- D perturbation ---
    if Dflag:
        ddum = dx[N::Ndum][:nnn]  # (nnn,)
        Ddum = np.zeros((Nc, Nc), dtype=complex)
        for n, (r, c) in enumerate(zip(rows_tri, cols_tri)):
            Ddum[r, c] = ddum[n]
            if r != c:
                Ddum[c, r] = ddum[n]
        new_D = new_D + np.real(Ddum)

    # --- E perturbation ---
    if Eflag:
        edum = dx[N + int(Dflag) :: Ndum][:nnn]  # (nnn,)
        Edum = np.zeros((Nc, Nc), dtype=complex)
        for n, (r, c) in enumerate(zip(rows_tri, cols_tri)):
            Edum[r, c] = edum[n]
            if r != c:
                Edum[c, r] = edum[n]
        new_E = new_E + np.real(Edum)

    return VFModel(
        poles=model.poles,
        residues=jnp.array(new_res),
        D=jnp.array(new_D),
        E=jnp.array(new_E),
    )


# ---------------------------------------------------------------------------
# Single RP-NNLS iteration
# ---------------------------------------------------------------------------


def _rp_qrnnls_y(
    model: VFModel,
    ss: SSModel,
    s: np.ndarray,
    violpairs: list[dict],
    poles: np.ndarray,
    cindex: np.ndarray,
    opts: FitOptions,
    _cache: dict,
) -> tuple[VFModel, SSModel, dict]:
    """One residue-perturbation NNLS iteration (RP_QRNNLS_Y.m equivalent).

    Args:
        model:     Current VFModel.
        ss:        Corresponding SSModel.
        s:         Sweep frequencies (j*omega, purely imaginary).
        violpairs: Violation extrema from find_violation_extrema.
        poles:     Numpy array of poles.
        cindex:    Classification array.
        opts:      FitOptions.
        _cache:    Mutable dict for caching QR system across iterations.

    Returns:
        (new_model, new_ss, updated_cache)

    """
    Nc = int(ss.D.shape[0])
    N = len(poles)
    nnn = Nc * (Nc + 1) // 2  # full upper triangle (bw = Nc)

    # Determine which terms need active perturbation.
    D_arr = np.real(np.array(ss.D))
    E_arr = np.real(np.array(ss.E))
    Dflag = bool(np.any(np.linalg.eigvalsh(D_arr) < 0) and np.any(D_arr != 0))
    Eflag = bool(np.any(np.linalg.eigvalsh(E_arr) < 0) and np.any(E_arr != 0))
    DEflag = int(Dflag) + int(Eflag)
    Ndum = N + DEflag

    # Augment s with auxiliary frequencies.
    s_np = np.asarray(s, dtype=complex)
    s_full, Ns_orig = _augment_frequencies(s_np, poles, cindex, violpairs, Eflag, opts.nu)

    # Rebuild LS system only when Dflag/Eflag status changes.
    cache_key = (Dflag, Eflag)
    if _cache.get("key") != cache_key:
        Rsub, bigEscale = _build_ls_system(s_full, poles, cindex, Nc, Dflag, Eflag, nnn, Ns_orig, opts.nu)
        _cache["key"] = cache_key
        _cache["Rsub"] = Rsub
        _cache["bigEscale"] = bigEscale
    else:
        Rsub = _cache["Rsub"]
        bigEscale = _cache["bigEscale"]

    # Build constraint matrix.
    bigB, bigc, offsB = _build_constraint_matrix(
        violpairs,
        poles,
        cindex,
        Nc,
        N,
        Dflag,
        Eflag,
        nnn,
        opts.TOLG,
        opts.TOLD,
        opts.TOLE,
        alpha=1.0,
        ss=ss,
    )

    if offsB == 0:
        return model, ss, _cache

    # Solve NNLS → perturbation vector.
    dx = _solve_perturbation(bigB, bigc, offsB, Rsub, bigEscale, nnn, N, DEflag)

    # Apply perturbation to model.
    new_model = _apply_perturbation(dx, cindex, model, N, Nc, nnn, Dflag, Eflag)
    new_ss = vfmodel_to_ss(new_model, Nc)

    return new_model, new_ss, _cache


# ---------------------------------------------------------------------------
# Public API — outer enforcement loop
# ---------------------------------------------------------------------------


def enforce_passivity(
    model: VFModel,
    s: jnp.ndarray,
    opts: FitOptions,
    verbose: bool = True,
) -> tuple[VFModel, np.ndarray]:
    """Iteratively enforce passivity of a Y-parameter rational model.

    Port of ``RPdriver.m`` (outer loop) + ``RP_QRNNLS_Y.m`` (RP step).

    At each outer iteration:
      1. Sweep minimum eigenvalue of Re(Y(jω)) → detect violation bands.
      2. Find per-channel violation extrema → violpairs.
      3. If passive and D, E ≥ 0: stop.
      4. Solve the NNLS residue-perturbation problem.
      5. Repeat.

    Args:
        model:   Initial VFModel (pole-residue form).
        s:       ``(Ns,)`` purely imaginary sweep frequencies (s = j·ω).
        opts:    FitOptions controlling tolerances and iteration counts.
        verbose: Print per-iteration passivity summary if True.

    Returns:
        new_model: Passive (or best-effort passive) VFModel.
        gmin_final: ``(Ns,)`` minimum eigenvalue after enforcement.

    """
    # Pin everything to CPU: enforcement is a Python loop with scipy NNLS.
    # Callers may pass GPU-resident arrays (e.g. from vfdriver on CUDA);
    # mixing GPU matmuls with CPU NNLS causes device-mismatch divergence.
    cpu = jax.devices("cpu")[0]
    model = jax.device_put(model, cpu)
    s = jax.device_put(s, cpu)

    Nc = model.residues.shape[0]
    omega = np.array(jnp.imag(s), dtype=float)
    s_np = np.array(s, dtype=complex)

    ss = vfmodel_to_ss(model, Nc)
    poles = np.array(model.poles)
    cindex = compute_cindex(poles)

    _cache: dict = {}

    if verbose:
        print("--- enforce_passivity (Y-parameters) ---")

    for iter_out in range(opts.Niter_out):
        gmin, is_passive = passivity_sweep_Y(ss, s)
        gmin_np = np.array(gmin)

        d_eigs = np.linalg.eigvalsh(np.real(np.array(ss.D)))
        e_eigs = np.linalg.eigvalsh(np.real(np.array(ss.E)))
        d_ok = bool(np.all(d_eigs >= 0))
        e_ok = bool(np.all(e_eigs >= 0))

        g_min_val = float(np.min(gmin_np))
        if verbose:
            print(
                f"  iter {iter_out:2d}  gmin={g_min_val:+.3e}"
                f"  D_min={float(np.min(d_eigs)):+.3e}"
                f"  E_min={float(np.min(e_eigs)):+.3e}"
            )

        if is_passive and d_ok and e_ok:
            if verbose:
                print("  → passive ✓")
            break

        bands = find_violation_bands(gmin_np, omega, opts.TOLG)
        violpairs = find_violation_extrema(ss, omega, bands)

        if not violpairs and d_ok and e_ok:
            if verbose:
                print("  → no violation extrema found, stopping")
            break

        # RP-NNLS perturbation step.
        old_res = np.array(model.residues)
        old_D = np.array(model.D)
        old_E = np.array(model.E)
        model, ss, _cache = _rp_qrnnls_y(model, ss, s_np, violpairs, poles, cindex, opts, _cache)
        poles = np.array(model.poles)
        cindex = compute_cindex(poles)

        # Stall detection: stop if model did not change.
        if (
            np.allclose(np.array(model.residues), old_res, atol=1e-30)
            and np.allclose(np.array(model.D), old_D, atol=1e-30)
            and np.allclose(np.array(model.E), old_E, atol=1e-30)
        ):
            if verbose:
                print("  → stalled, stopping")
            break

    gmin_final, _ = passivity_sweep_Y(ss, s)
    return model, np.array(gmin_final)
