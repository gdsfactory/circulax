"""Single Vector Fitting iteration: pole identification + residue identification.

Combines pole_id and residue_id into one callable that mirrors the interface of
vectfit4.m. The outer driver (driver.py) calls this in a loop.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .pole_id import identify_poles
from .residue_id import identify_residues
from .types import FitOptions, VFModel, eval_model, vfmodel_to_ss
from .utils import (
    _upper_triangle_indices,
    compute_rmserr,
    unstack_upper_triangle,
)


def vectfit_iteration(
    f: jnp.ndarray,  # (nnn, Ns) complex — upper-triangle elements
    s: jnp.ndarray,  # (Ns,) complex
    poles: np.ndarray,  # (N,) complex NumPy — current poles
    weight: jnp.ndarray,  # (nnn, Ns) or (1, Ns)
    opts: FitOptions,
    Nc: int,  # number of ports
    skip_pole: bool = False,
    skip_res: bool = False,
    reciprocal: bool = True,
) -> tuple[np.ndarray, VFModel | None, float]:
    """One pass of Vector Fitting: pole ID then residue ID.

    Args:
        f:         (nnn, Ns) complex upper-triangle data.
        s:         (Ns,) complex frequency points.
        poles:     (N,) complex — current poles (updated in-place semantics).
        weight:    (nnn, Ns) or (1, Ns) — fitting weights.
        opts:      FitOptions.
        Nc:        number of ports (needed to reconstruct full residue matrices).
        skip_pole: skip pole identification; use current poles directly.
        skip_res:  skip residue identification; return None model.
        reciprocal: Reconstruct a mirrored triangle when true, or all ordered
            responses when false.

    Returns:
        new_poles: (N,) complex NumPy.
        model:     VFModel or None (if skip_res).
        rmserr:    float RMS fitting error (0.0 if skip_res).

    """
    N = opts.N

    # --- Stage 1: pole identification ---
    if skip_pole:
        new_poles = poles
    else:
        new_poles = identify_poles(f, s, poles, weight, opts)

    if skip_res:
        return new_poles, None, 0.0

    # --- Stage 2: residue identification ---
    C_flat, D_vec, E_vec = identify_residues(f, s, new_poles, weight, opts)
    # C_flat: (nnn, N) complex
    # D_vec:  (nnn,) float
    # E_vec:  (nnn,) float

    # --- Reconstruct full (Nc, Nc, N) residue tensor and (Nc, Nc) D, E ---
    nnn = f.shape[0]
    idx = (
        _upper_triangle_indices(Nc)
        if reciprocal
        else [(row, col) for row in range(Nc) for col in range(Nc)]
    )
    Ns = s.shape[0]

    residues = jnp.zeros((Nc, Nc, N), dtype=jnp.complex128)
    D_mat = jnp.zeros((Nc, Nc), dtype=jnp.float64)
    E_mat = jnp.zeros((Nc, Nc), dtype=jnp.float64)

    for k, (r, c) in enumerate(idx):
        residues = residues.at[r, c, :].set(C_flat[k])
        D_mat = D_mat.at[r, c].set(float(D_vec[k]))
        E_mat = E_mat.at[r, c].set(float(E_vec[k]))
        if reciprocal and r != c:
            # Reciprocity is transpose symmetry at a fixed complex frequency:
            # H[c, r](s) = H[r, c](s). Conjugating here would instead impose
            # an invalid Hermitian constraint and break S12 == S21.
            residues = residues.at[c, r, :].set(C_flat[k])
            D_mat = D_mat.at[c, r].set(float(D_vec[k]))
            E_mat = E_mat.at[c, r].set(float(E_vec[k]))

    model = VFModel(
        poles=jnp.array(new_poles),
        residues=residues,
        D=D_mat,
        E=E_mat,
    )

    # --- compute RMS error ---
    ss = vfmodel_to_ss(model, Nc)
    Hfit = eval_model(s, ss)  # (Ns, Nc, Nc)

    # Reconstruct H from upper-triangle elements for error comparison
    if reciprocal:
        H_data = unstack_upper_triangle(f, Nc)  # (Nc, Nc, Ns)
    else:
        H_data = f.reshape(Nc, Nc, Ns)
    H_data = jnp.moveaxis(H_data, -1, 0)  # (Ns, Nc, Nc)

    rmserr = compute_rmserr(H_data, Hfit)

    return new_poles, model, rmserr
