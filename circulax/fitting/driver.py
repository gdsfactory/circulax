"""VFdriver equivalent: two-phase iterative Vector Fitting.

Ported from VFdriver.m (SINTEF MFT-NNLS toolbox, B. Gustavsen, 2026).

Phase 1: fit diagonal elements only (fast pole refinement).
Phase 2: fit entire upper triangle with common pole set.
Optionally enforces passive D and E after fitting.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .types import FitOptions, SSModel, VFModel, eval_model, vfmodel_to_ss
from .utils import (
    compute_weights,
    init_poles_logcmplx,
    stack_upper_triangle,
)
from .vectfit import vectfit_iteration


def _project_psd(M: jnp.ndarray, tol: float) -> jnp.ndarray:
    """Project M to positive semi-definite by clamping eigenvalues below tol.

    Used for passive D and E enforcement.
    """
    vals, vecs = jnp.linalg.eigh(M)
    return (vecs * jnp.maximum(vals, tol)[None, :]) @ vecs.conj().T


def vfdriver(
    bigH: jnp.ndarray,
    s: jnp.ndarray,
    poles: np.ndarray | None,
    opts: FitOptions,
    reciprocal: bool = True,
    verbose: bool = True,
) -> tuple[VFModel, SSModel, float, jnp.ndarray]:
    """Fit bigH(s) with a rational model using the two-phase VF strategy.

    Args:
        bigH:    (Nc, Nc, Ns) complex — frequency-domain matrix data.
        s:       (Ns,) complex — frequency points (= j*omega).
        poles:   (N,) complex NumPy initial poles, or None to auto-generate.
        opts:    FitOptions.
        reciprocal: Fit one matrix triangle when true, otherwise fit every
            ordered response independently.
        verbose: print iteration progress.

    Returns:
        model:   VFModel — fitted pole-residue model.
        ss:      SSModel — state-space form.
        rmserr:  float   — final RMS fitting error.
        bigHfit: (Nc, Nc, Ns) — reconstructed frequency response.

    """
    Nc = bigH.shape[0]
    Ns = s.shape[0]
    N = opts.N

    # --- Initial poles ---
    if poles is None:
        poles = init_poles_logcmplx(s, N, nu=opts.nu)
    poles = np.asarray(poles, dtype=np.complex128)

    # --- Selected response stack and weights ---
    f_full = (
        stack_upper_triangle(bigH)
        if reciprocal
        else bigH.reshape(Nc * Nc, Ns)
    )
    w_full = compute_weights(bigH, opts.weightparam, reciprocal=reciprocal)

    # --- Diagonal-only data and weights (for phase 1) ---
    diag_idx = [(i, i) for i in range(Nc)]
    f_diags = jnp.stack([bigH[i, i, :] for i in range(Nc)], axis=0)  # (Nc, Ns)

    if opts.weightparam in (1, 4, 5):
        # Common weight — broadcast to diagonals
        w_diag = jnp.broadcast_to(w_full, (Nc, Ns))
    else:
        # Per-element weight — extract diagonal entries
        from .utils import _upper_triangle_indices

        if reciprocal:
            idx_all = _upper_triangle_indices(Nc)
            diag_positions = [idx_all.index((i, i)) for i in range(Nc)]
        else:
            diag_positions = [i * Nc + i for i in range(Nc)]
        w_diag = w_full[jnp.array(diag_positions), :]  # (Nc, Ns)

    # -------------------------------------------------------------------------
    # PHASE 1: Fit diagonal elements only → fast pole refinement
    # -------------------------------------------------------------------------
    if verbose:
        print(f"VFdriver: Phase 1 ({opts.Niter1} iterations on diagonal elements)")

    for it in range(opts.Niter1):
        poles, _, err = vectfit_iteration(
            f_diags,
            s,
            poles,
            w_diag,
            opts,
            Nc=Nc,
            skip_pole=False,
            skip_res=True,
        )
        if verbose:
            print(f"  Phase 1 iter {it + 1}: rmserr = {err:.4e}")

    # -------------------------------------------------------------------------
    # PHASE 2: Fit entire upper triangle with common poles
    # -------------------------------------------------------------------------
    if verbose:
        print(f"VFdriver: Phase 2 ({opts.Niter2} iterations on full matrix)")

    model = None
    rmserr = float("inf")

    for it in range(opts.Niter2):
        skip_res = it < opts.Niter2 - 1
        poles, model_new, err = vectfit_iteration(
            f_full,
            s,
            poles,
            w_full,
            opts,
            Nc=Nc,
            skip_pole=False,
            skip_res=skip_res,
            reciprocal=reciprocal,
        )
        if model_new is not None:
            model = model_new
            rmserr = err
        if verbose:
            print(f"  Phase 2 iter {it + 1}: rmserr = {err:.4e}")

    if model is None:
        raise RuntimeError("No residue identification was performed.")

    # -------------------------------------------------------------------------
    # Optional: enforce passive D and E (project to PSD)
    # -------------------------------------------------------------------------
    if opts.passive_DE and opts.asymp >= 2:
        if verbose:
            print("VFdriver: enforcing passive D")
        D_new = _project_psd(model.D, opts.TOLD)
        model = VFModel(model.poles, model.residues, D_new, model.E)

        if opts.asymp >= 3:
            if verbose:
                print("VFdriver: enforcing passive E")
            E_new = _project_psd(model.E, opts.TOLE)
            model = VFModel(model.poles, model.residues, model.D, E_new)

        # Recompute residues with modified D, E (re-run residue-id with skip_pole=True)
        poles, model, rmserr = vectfit_iteration(
            f_full,
            s,
            poles,
            w_full,
            opts,
            Nc=Nc,
            skip_pole=True,
            skip_res=False,
            reciprocal=reciprocal,
        )
        if verbose:
            print(f"  After passive D/E: rmserr = {rmserr:.4e}")

    # -------------------------------------------------------------------------
    # Build state-space model and reconstruct fitted response
    # -------------------------------------------------------------------------
    ss = vfmodel_to_ss(model, Nc)
    bigHfit_stacked = eval_model(s, ss)  # (Ns, Nc, Nc)
    bigHfit = jnp.moveaxis(bigHfit_stacked, 0, -1)  # (Nc, Nc, Ns)

    return model, ss, rmserr, bigHfit
