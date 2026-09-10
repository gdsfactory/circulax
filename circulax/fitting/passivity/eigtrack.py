"""Eigenvalue continuity tracking (port of intercheig3.m).

Greedy algorithm that permutes new eigenpairs to match the channel ordering
of the previous frequency step, enabling smooth per-channel eigenvalue curves
across a frequency sweep.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def track_eigenvalues(
    vals_new: jnp.ndarray,
    vecs_new: jnp.ndarray,
    vecs_old: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Permute new eigenpairs to best match the old channel ordering.

    For each new eigenvector, finds the old eigenvector with the largest
    dot-product overlap.  Assignments are made greedily in descending order
    of confidence (highest max-overlap first) so the most certain matches are
    locked in before ambiguous ones are resolved.

    Port of ``intercheig3.m`` from the SINTEF MFT-NNLS toolbox.  Uses
    ``jax.lax.scan`` so the function is JIT-compilable.

    Args:
        vals_new: ``(Nc,)`` real eigenvalues at the current frequency step.
        vecs_new: ``(Nc, Nc)`` complex eigenvectors (columns) at the current step.
        vecs_old: ``(Nc, Nc)`` complex eigenvectors (columns) at the previous step.

    Returns:
        vals_tracked: ``(Nc,)`` eigenvalues reordered to match old channel ordering.
        vecs_tracked: ``(Nc, Nc)`` eigenvectors reordered to match old channel ordering.

    """
    Nc = vals_new.shape[0]

    # H[i, j] = |<vecs_new[:, i], vecs_old[:, j]>|
    # Row i = overlap of new eigenvector i with all old eigenvectors.
    H = jnp.abs(vecs_new.conj().T @ vecs_old)  # (Nc, Nc)

    # Process new cols in descending order of their best overlap — most
    # confident assignments are made first to avoid cascading errors.
    row_order = jnp.argsort(-jnp.max(H, axis=1))  # (Nc,) — new-col indices

    def _step(
        carry: tuple[jnp.ndarray, jnp.ndarray],
        new_col: jnp.ndarray,
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], None]:
        H_masked, idx = carry
        # Which old column best matches new column `new_col`?
        best_old = jnp.argmax(H_masked[new_col]).astype(jnp.int32)
        idx = idx.at[new_col].set(best_old)
        # Mark old column as consumed so it cannot be re-assigned.
        H_masked = H_masked.at[:, best_old].set(0.0)
        return (H_masked, idx), None

    (_, idx), _ = jax.lax.scan(
        _step,
        (H, jnp.zeros(Nc, dtype=jnp.int32)),
        row_order,
    )

    # idx[new_col] = old_col  (new col k was matched to old col idx[k])
    #
    # We want output channel k to hold the new eigenvector that was matched
    # to OLD channel k.  That requires the INVERSE permutation:
    #   inv_idx[old_col] = new_col
    inv_idx = jnp.zeros(Nc, dtype=jnp.int32).at[idx].set(jnp.arange(Nc, dtype=jnp.int32))

    return vals_new[inv_idx], vecs_new[:, inv_idx]


@jax.jit
def track_eigenvalues_all(
    all_vals: jnp.ndarray,
    all_vecs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Track eigenvalue channels across all frequency points in one JIT call.

    Args:
        all_vals: ``(Ns, Nc)`` real eigenvalues at each frequency, sorted ascending.
        all_vecs: ``(Ns, Nc, Nc)`` complex eigenvectors at each frequency.

    Returns:
        vals_tracked: ``(Ns, Nc)`` eigenvalues reordered for channel continuity.
        vecs_tracked: ``(Ns, Nc, Nc)`` eigenvectors reordered for channel continuity.

    """

    def step(vecs_carry, kdata):
        vals_k, vecs_k = kdata
        v_out, V_out = track_eigenvalues(vals_k, vecs_k, vecs_carry)
        return V_out, (v_out, V_out)

    _, (vals_rest, vecs_rest) = jax.lax.scan(step, all_vecs[0], (all_vals[1:], all_vecs[1:]))
    # Prepend the initial point (already in canonical order).
    vals_out = jnp.concatenate([all_vals[0:1], vals_rest], axis=0)
    vecs_out = jnp.concatenate([all_vecs[0:1], vecs_rest], axis=0)
    return vals_out, vecs_out
