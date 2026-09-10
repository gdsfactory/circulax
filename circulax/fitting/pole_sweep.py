"""Fixed-shape, vmapped screening of scattering-model pole budgets."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .types import SSModel, VFModel
from .utils import build_dk, compute_cindex


@jax.jit
def _solve_masked_responses(
    basis: jax.Array,
    responses: jax.Array,
    coefficient_masks: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Solve every response and pole mask with nested vectorization."""
    target_energy = jnp.maximum(jnp.sum(jnp.abs(responses) ** 2), 1e-30)

    def solve_response(
        active_basis: jax.Array,
        response: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        real_basis = jnp.concatenate([active_basis.real, active_basis.imag], axis=0)
        real_response = jnp.concatenate([response.real, response.imag], axis=0)
        scale = jnp.linalg.norm(real_basis, axis=0)
        scale = jnp.where(scale == 0, 1.0, scale)
        coefficients = jnp.linalg.lstsq(real_basis / scale, real_response, rcond=None)[0] / scale
        return active_basis @ coefficients, coefficients

    def solve_candidate(mask: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        active_basis = basis * mask[None, :]
        prediction, coefficients = jax.vmap(
            lambda response: solve_response(active_basis, response)
        )(responses)
        error = prediction - responses
        normalized_error = jnp.sqrt(jnp.sum(jnp.abs(error) ** 2) / target_energy)
        return normalized_error, jnp.max(jnp.abs(error)), coefficients

    return jax.vmap(solve_candidate)(coefficient_masks)


@dataclass(frozen=True)
class PoleCountSweep:
    """Errors and masks produced by :func:`vmap_pole_count_sweep`."""

    requested_counts: jax.Array
    retained_counts: jax.Array
    masks: jax.Array
    normalized_rmse: jax.Array
    max_absolute_error: jax.Array
    maximum_admittance_pole_real_part: jax.Array | None = None
    transform_condition: jax.Array | None = None

    def smallest_passing(
        self,
        *,
        normalized_rmse: float,
        max_absolute_error: float,
    ) -> int | None:
        """Return the smallest screened pole count satisfying both limits."""
        passing = np.asarray(
            (self.normalized_rmse <= normalized_rmse)
            & (self.max_absolute_error <= max_absolute_error)
        )
        if not np.any(passing):
            return None
        counts = np.asarray(self.retained_counts)
        return int(np.min(counts[passing]))


def _contribution_ordered_groups(
    model: VFModel,
    sample_points: np.ndarray,
) -> list[list[int]]:
    """Rank real poles and conjugate pairs by aggregate response contribution."""
    poles = np.asarray(model.poles)
    residues = np.asarray(model.residues)
    cindex = compute_cindex(poles)
    groups: list[tuple[float, list[int]]] = []
    index = 0
    while index < len(poles):
        indices = [index] if cindex[index] == 0 else [index, index + 1]
        contribution = sum(
            residues[:, :, pole_index][None, ...]
            / (sample_points[:, None, None] - poles[pole_index])
            for pole_index in indices
        )
        groups.append((float(np.linalg.norm(contribution)), indices))
        index += len(indices)
    return [indices for _, indices in sorted(groups, key=lambda item: item[0], reverse=True)]


def contribution_masks(
    model: VFModel,
    sample_points: np.ndarray,
    pole_counts: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build fixed-width masks that never split conjugate pole pairs."""
    counts = np.asarray(tuple(pole_counts), dtype=int)
    if counts.ndim != 1 or len(counts) == 0 or np.any(counts < 1):
        msg = "pole_counts must be a nonempty sequence of positive integers"
        raise ValueError(msg)
    groups = _contribution_ordered_groups(model, np.asarray(sample_points))
    masks = np.zeros((len(counts), len(model.poles)), dtype=np.float64)
    retained = np.zeros(len(counts), dtype=int)
    for row, budget in enumerate(counts):
        for indices in groups:
            if retained[row] + len(indices) <= budget:
                masks[row, indices] = 1.0
                retained[row] += len(indices)
    if np.any(retained == 0):
        msg = "a pole budget is too small to retain any real pole or conjugate pair"
        raise ValueError(msg)
    return masks, retained


def prune_poles_by_contribution(
    model: VFModel,
    sample_points: np.ndarray,
    max_poles: int,
) -> np.ndarray:
    """Return the strongest complete pole groups within ``max_poles``."""
    if len(model.poles) <= max_poles:
        return np.asarray(model.poles)
    masks, _ = contribution_masks(model, sample_points, [max_poles])
    return np.asarray(model.poles)[masks[0].astype(bool)]


def vmap_pole_count_sweep(
    model: VFModel,
    freqs: np.ndarray,
    target_S: np.ndarray,
    pole_counts: Iterable[int],
    *,
    asymp: int = 2,
    z0: float = 50.0,
) -> PoleCountSweep:
    """Refit and score many fixed-pole masks in one compiled ``vmap``.

    Every candidate retains the maximum pole-array shape. Excluded poles keep
    their locations but have zero coefficient columns; constant and optional
    proportional terms remain active. The returned screening errors precede
    pole relocation, so shortlisted candidates should still be refined and
    validated after physically compacting their pole arrays.
    """
    counts = tuple(pole_counts)
    if asymp not in {1, 2, 3}:
        msg = "asymp must be 1, 2, or 3"
        raise ValueError(msg)
    freqs = np.asarray(freqs, dtype=float)
    target_S = np.asarray(target_S, dtype=np.complex128)
    if target_S.ndim != 3:
        msg = "target_S must have shape (len(freqs), ports, ports)"
        raise ValueError(msg)
    nc = target_S.shape[-1]
    if target_S.shape != (len(freqs), nc, nc):
        msg = "target_S must have shape (len(freqs), ports, ports)"
        raise ValueError(msg)

    sample_points = 1j * 2.0 * np.pi * freqs
    masks, retained = contribution_masks(model, sample_points, counts)
    cindex = compute_cindex(np.asarray(model.poles))
    basis = build_dk(
        jnp.asarray(sample_points),
        jnp.asarray(model.poles),
        cindex,
        asymp - 1,
    )
    extra_mask = np.ones((len(masks), asymp - 1), dtype=float)
    coefficient_masks = jnp.asarray(np.concatenate([masks, extra_mask], axis=1))
    responses = jnp.asarray(np.moveaxis(target_S, 0, -1).reshape(nc * nc, len(freqs)))
    normalized_error, maximum_error, coefficients = _solve_masked_responses(
        basis,
        responses,
        coefficient_masks,
    )

    maximum_y_pole_real_part = None
    transform_condition = None
    if asymp <= 2:
        # Convert the real conjugate-pair coefficient representation back into
        # explicit residues without compacting the pole axis. The resulting
        # realization has one common fixed state shape for every candidate.
        coefficient_residues = coefficients[..., : len(model.poles)]
        cindex_jax = jnp.asarray(cindex)
        next_coefficient = jnp.roll(coefficient_residues, -1, axis=-1)
        previous_coefficient = jnp.roll(coefficient_residues, 1, axis=-1)
        complex_residues = jnp.where(
            cindex_jax == 1,
            coefficient_residues + 1j * next_coefficient,
            jnp.where(
                cindex_jax == 2,
                previous_coefficient - 1j * coefficient_residues,
                coefficient_residues + 0j,
            ),
        ).reshape(len(counts), nc, nc, len(model.poles))
        batch_size = len(counts)
        state_poles = jnp.tile(jnp.asarray(model.poles), nc)
        state_input = jnp.kron(
            jnp.eye(nc, dtype=jnp.complex128),
            jnp.ones((len(model.poles), 1), dtype=jnp.complex128),
        )
        direct = (
            coefficients[..., len(model.poles)].reshape(batch_size, nc, nc)
            if asymp == 2
            else jnp.zeros((batch_size, nc, nc), dtype=jnp.float64)
        )
        scattering_models = SSModel(
            A=jnp.broadcast_to(state_poles, (batch_size, len(state_poles))),
            B=jnp.broadcast_to(state_input, (batch_size, *state_input.shape)),
            C=complex_residues.reshape(batch_size, nc, nc * len(model.poles)),
            D=direct.astype(jnp.complex128),
            E=jnp.zeros((batch_size, nc, nc), dtype=jnp.complex128),
        )

        # Local import avoids a module cycle: sparam owns the public transform
        # and imports this screening helper for fit_with_delay.
        from .sparam import vmap_scattering_state_space_to_admittance

        admittance_models, transform_condition = vmap_scattering_state_space_to_admittance(
            scattering_models,
            jnp.asarray(z0, dtype=jnp.float64),
        )
        maximum_y_pole_real_part = jnp.max(jnp.real(admittance_models.A), axis=-1)

    return PoleCountSweep(
        requested_counts=jnp.asarray(counts),
        retained_counts=jnp.asarray(retained),
        masks=jnp.asarray(masks),
        normalized_rmse=normalized_error,
        max_absolute_error=maximum_error,
        maximum_admittance_pole_real_part=maximum_y_pole_real_part,
        transform_condition=transform_condition,
    )
