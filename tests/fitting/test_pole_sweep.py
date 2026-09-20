"""Tests for fixed-shape vmapped pole-count screening."""

import jax.numpy as jnp
import numpy as np

from circulax.fitting import VFModel, eval_model, vmap_pole_count_sweep
from circulax.fitting.types import vfmodel_to_ss


def test_vmap_pole_sweep_refits_nested_conjugate_pair_masks() -> None:
    freqs = np.linspace(1e8, 8e9, 120)
    poles = jnp.array(
        [
            -2e8 - 1j * 5e9,
            -2e8 + 1j * 5e9,
            -5e8 - 1j * 2e10,
            -5e8 + 1j * 2e10,
            -1e9 - 1j * 4e10,
            -1e9 + 1j * 4e10,
        ]
    )
    base = jnp.array([3e8 + 2e8j, 3e8 - 2e8j, 8e7 + 4e7j, 8e7 - 4e7j, 2e7 + 1e7j, 2e7 - 1e7j])
    residues = jnp.stack(
        [
            jnp.stack([base, -0.3 * base]),
            jnp.stack([0.6 * base, 0.2 * base]),
        ]
    )
    model = VFModel(
        poles,
        residues,
        jnp.array([[0.1, 0.02], [-0.04, 0.05]]),
        jnp.zeros((2, 2)),
    )
    target = np.asarray(eval_model(1j * 2 * jnp.pi * freqs, vfmodel_to_ss(model, 2)))

    sweep = vmap_pole_count_sweep(model, freqs, target, [2, 4, 6])

    assert sweep.masks.shape == (3, 6)
    np.testing.assert_array_equal(np.asarray(sweep.retained_counts), [2, 4, 6])
    np.testing.assert_array_equal(np.asarray(sweep.masks)[:, ::2], np.asarray(sweep.masks)[:, 1::2])
    assert np.all(np.diff(np.asarray(sweep.normalized_rmse)) <= 1e-12)
    assert float(sweep.normalized_rmse[-1]) < 1e-10
    assert sweep.maximum_admittance_pole_real_part is not None
    assert sweep.transform_condition is not None
    assert sweep.maximum_admittance_pole_real_part.shape == (3,)
    assert np.all(np.isfinite(np.asarray(sweep.transform_condition)))
    assert sweep.smallest_passing(normalized_rmse=1e-8, max_absolute_error=1e-8) == 6
