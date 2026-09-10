"""Passivity assessment for Y-parameter models.

Ports:
  - ``pass_check_Y_sweep_new.m``     → passivity_sweep_Y + find_violation_bands
  - ``violextremaY_eigenpairs_globalminima2.m`` → find_violation_extrema
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..types import SSModel, eval_model
from .eigtrack import track_eigenvalues_all

# ---------------------------------------------------------------------------
# Passivity sweep
# ---------------------------------------------------------------------------


def passivity_sweep_Y(
    ss: SSModel,
    s: jnp.ndarray,
) -> tuple[jnp.ndarray, bool]:
    """Sweep the minimum eigenvalue of Re(Y(jω)) across all frequency points.

    Port of ``pass_check_Y_sweep_new.m``.

    For Y-parameters, passivity requires the Hermitian part
    ``G(jω) = (Y(jω) + Y(jω)ᴴ) / 2`` to be positive semi-definite at every
    frequency.  ``gmin[k] = λ_min(G(jω_k))`` is the passivity margin: negative
    values indicate violations.

    Args:
        ss: State-space model (``SSModel``).
        s:  ``(Ns,)`` purely imaginary frequency points ``s = j·ω``.

    Returns:
        gmin:       ``(Ns,)`` real — minimum eigenvalue of G at each frequency.
        is_passive: ``True`` if ``gmin ≥ 0`` everywhere.

    """
    # Run entirely on CPU: the round-trip cost of moving G to CPU for eigvalsh
    # (which cannot use GPU cuSolver portably) exceeds any gain from GPU matmuls
    # on the small (Nc×Nc) matrices.  Explicitly pin to CPU so callers with GPU
    # state-space models don't incur silent back-and-forth transfers.
    cpu = jax.devices("cpu")[0]
    ss_cpu = jax.device_put(ss, cpu)
    s_cpu = jax.device_put(s, cpu)

    Y_all = eval_model(s_cpu, ss_cpu)  # (Ns, Nc, Nc) CPU
    G_all = (Y_all + Y_all.conj().transpose(0, 2, 1)) / 2.0
    G_np = np.array(G_all)
    gmin_np = np.linalg.eigvalsh(G_np)[:, 0]  # (Ns,) min eigenvalue
    gmin = jnp.array(gmin_np)

    return gmin, bool(np.all(gmin_np >= 0.0))


# ---------------------------------------------------------------------------
# Violation band detection
# ---------------------------------------------------------------------------


def find_violation_bands(
    gmin: np.ndarray,
    omega: np.ndarray,
    TOLG: float = 1e-6,
) -> list[tuple[float, float]]:
    """Detect contiguous frequency bands where passivity is violated.

    Port of the band-detection logic in ``pass_check_Y_sweep_new.m``.

    A dummy stable point is prepended at ω = −1 with gmin = 777 so that
    violations that begin at the lowest sampled frequency are correctly
    detected.  If a violation extends past the last sample, the upper band
    limit is set to 1000 × ω_max.

    Args:
        gmin:  ``(Ns,)`` minimum eigenvalue array from :func:`passivity_sweep_Y`.
        omega: ``(Ns,)`` angular frequency array (rad/s, positive, increasing).
        TOLG:  Eigenvalue tolerance; values within ``TOLG`` of zero are treated
               as passive.

    Returns:
        List of ``(w_start, w_end)`` pairs (rad/s) — one per violation band.

    """
    gmin = np.asarray(gmin, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)

    # Shift so that margins within TOLG count as passive.
    g = gmin + TOLG

    # Prepend a guaranteed-stable dummy point.
    g_aug = np.concatenate([[777.0], g])
    w_aug = np.concatenate([[-1.0], omega])

    signs = np.sign(g_aug)
    signs[signs == 0] = 1  # exactly zero → passive
    ds = np.diff(signs)

    starts = np.where(ds < 0)[0]  # augmented-array index where gmin goes negative
    ends = np.where(ds > 0)[0]  # augmented-array index where gmin recovers

    bands: list[tuple[float, float]] = []
    for si in starts:
        w1 = float(w_aug[si + 1])
        w1 = max(w1, 0.0)

        later_ends = ends[ends > si]
        w2 = float(w_aug[later_ends[0] + 1]) if len(later_ends) > 0 else 1000.0 * float(omega[-1])

        bands.append((w1, w2))

    return bands


# ---------------------------------------------------------------------------
# Violation extrema
# ---------------------------------------------------------------------------


def find_violation_extrema(
    ss: SSModel,
    omega: np.ndarray,
    bands: list[tuple[float, float]],
    Nint: int = 21,
) -> list[dict]:
    """Find the worst-case violation point per eigenvalue channel per band.

    For each violation band a dense grid of ``2 × Nint`` (linear + log-spaced)
    test frequencies is generated, eigenpairs of ``Re(Y(jω))`` are computed,
    eigenvalue continuity is tracked across frequency using
    :func:`~circulax.fitting.passivity.eigtrack.track_eigenvalues`, and the global
    minimum per channel is recorded.

    Port of ``violextremaY_eigenpairs_globalminima2.m``.

    Args:
        ss:    State-space model.
        omega: ``(Ns,)`` reference frequency array (rad/s) from the original
               passivity sweep — used only for clamping band limits.
        bands: List of ``(w_start, w_end)`` pairs from
               :func:`find_violation_bands`.
        Nint:  Number of linear-spaced *and* log-spaced points per band
               (total ≤ 2 × Nint after deduplication).

    Returns:
        List of dicts, one per (channel, band) pair with a negative minimum
        eigenvalue.  Each dict has keys:

        * ``'group'``      — int: eigenvalue channel index (0-based).
        * ``'omega'``      — float: angular frequency of worst violation (rad/s).
        * ``'lambda_min'`` — float: minimum eigenvalue (negative).
        * ``'eigvec'``     — ``(Nc,)`` complex ndarray: associated eigenvector.

    """
    omega = np.asarray(omega, dtype=np.float64)
    Nc = int(ss.D.shape[0])
    violpairs: list[dict] = []

    for w1, w2 in bands:
        w1 = max(float(w1), 0.0)
        w2 = float(w2)
        if w2 <= w1:
            continue

        # Dense test grid: linear + log-spaced, deduplicated.
        om_lin = np.linspace(w1, w2, Nint)
        lo = np.log10(max(w1, 1e-10))
        hi = np.log10(max(w2, 10.0 * max(w1, 1e-10)))
        om_log = np.logspace(lo, hi, Nint)
        om_test = np.unique(np.sort(np.concatenate([om_lin, om_log])))
        s_test = 1j * jnp.array(om_test)

        # Run on CPU for the same reason as passivity_sweep_Y.
        cpu = jax.devices("cpu")[0]
        ss_cpu = jax.device_put(ss, cpu)
        s_test_cpu = jax.device_put(s_test, cpu)
        Y_all = eval_model(s_test_cpu, ss_cpu)  # (Ns_t, Nc, Nc) CPU
        G_all = (Y_all + Y_all.conj().transpose(0, 2, 1)) / 2.0
        all_vals_np, all_vecs_np = np.linalg.eigh(np.array(G_all))  # (Ns_t,Nc), (Ns_t,Nc,Nc)
        all_vals = jnp.array(all_vals_np)
        all_vecs = jnp.array(all_vecs_np)
        # all_vals: (Ns_t, Nc)   all_vecs: (Ns_t, Nc, Nc)

        # Track eigenvalue channels across all frequency points in one JIT call.
        vals_tracked, vecs_tracked = track_eigenvalues_all(all_vals, all_vecs)
        vals_np = np.array(vals_tracked)  # (Ns_t, Nc)
        vecs_np = np.array(vecs_tracked)  # (Ns_t, Nc, Nc)

        # Per-channel global minimum — find the worst violation index.
        min_idx = np.argmin(vals_np, axis=0)  # (Nc,)

        # Record channels with negative minimum eigenvalues.
        for ch in range(Nc):
            k_min = int(min_idx[ch])
            lam_min = float(vals_np[k_min, ch])
            if lam_min < 0.0:
                violpairs.append(
                    {
                        "group": ch,
                        "omega": float(om_test[k_min]),
                        "lambda_min": lam_min,
                        "eigvec": vecs_np[k_min, :, ch].copy(),
                    }
                )

    return violpairs
