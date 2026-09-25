"""NNLS solver utilities for passivity enforcement.

Provides the Lawson-Hanson augmented NNLS formulation used in RP_QRNNLS_Y.m.
scipy.optimize.nnls is used as the underlying solver (LAPACK-based, no extra deps).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls as _scipy_nnls


def solve_passivity_nnls(
    E_mat: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    """Solve the augmented NNLS problem for minimum-norm residue perturbation.

    Implements the Lawson-Hanson formulation from RP_QRNNLS_Y.m::

        Augmented problem:
            A = [E_mat | −c]ᵀ          # (n+1, n_pairs)
            f = [0, …, 0, 1]            # (n+1,)
            u = argmin_{u≥0} ‖A u − f‖

        Recovery:
            residual = f − A u
            xbar     = −residual[:n] / residual[n]

    The formulation finds the smallest perturbation (in the pre-conditioned /
    R-factored space) that satisfies all passivity constraints.

    Args:
        E_mat: ``(n_pairs, n)`` constraint matrix already scaled by
               ``diag(1/bigEscale) @ inv(Rsub)``.
        c:     ``(n_pairs,)`` RHS target values (negative = must become positive).

    Returns:
        xbar: ``(n,)`` perturbation vector in the pre-conditioned space,
              or the zero vector if the system has no useful solution.

    """
    n = E_mat.shape[1]

    # Augment E with -c as an extra column, then transpose.
    A = np.column_stack([E_mat, -c]).T  # (n+1, n_pairs)
    f = np.zeros(n + 1)
    f[-1] = 1.0

    u, _ = _scipy_nnls(A, f)
    residual = f - A @ u

    if abs(residual[-1]) < 1e-30:
        return np.zeros(n)

    return -residual[:n] / residual[-1]
