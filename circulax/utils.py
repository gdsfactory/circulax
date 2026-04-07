"""circulax utilities."""

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

if TYPE_CHECKING:
    from circulax.compiler import ComponentGroup


def update_params_dict(
    groups_dict: dict,
    group_name: str,
    instance_name: str,
    param_key: str,
    new_value: float,
) -> dict[str, "ComponentGroup"]:
    """Updates a parameter for a specific instance within a component group."""
    g = groups_dict[group_name]

    instance_idx = g.index_map[instance_name]

    # Handle Equinox Component (Batched)
    batched_comp = g.params
    current_val = getattr(batched_comp, param_key)
    new_vals = current_val.at[instance_idx].set(new_value)

    new_batched_comp = eqx.tree_at(lambda c: getattr(c, param_key), batched_comp, new_vals)
    new_g = eqx.tree_at(lambda g: g.params, g, new_batched_comp)

    # Return new dict (JAX helper to copy-and-modify dicts)
    return {**groups_dict, group_name: new_g}


def update_group_params(groups_dict: dict, group_name: str, param_key: str, new_value: float) -> dict[str, "ComponentGroup"]:
    """Updates a parameter for ALL instances in a component group."""
    g = groups_dict[group_name]

    # Handle Equinox Component (Batched)
    batched_comp = g.params
    current_val = getattr(batched_comp, param_key)

    new_vals = jnp.full_like(current_val, new_value)

    new_batched_comp = eqx.tree_at(lambda c: getattr(c, param_key), batched_comp, new_vals)
    new_g = eqx.tree_at(lambda g: g.params, g, new_batched_comp)

    return {**groups_dict, group_name: new_g}


def apply_global_params(groups: dict, params: dict) -> dict:
    """Forward global scalar params to all component groups that declare them.

    For each ``(param_name, value)`` pair in *params*, updates every group whose
    batched params object has an attribute with that name, broadcasting the value
    to all instances in that group.

    Works correctly under ``jax.jit`` and ``jax.vmap``: the dict walk is
    Python-level (static at trace time), and *value* is the only traced leaf.

    Args:
        groups: Compiled groups dict as returned by :func:`compile_netlist`.
        params: Mapping from parameter name to scalar JAX-traceable value.

    Returns:
        New groups dict with updated parameter values (immutable functional update).

    """
    updated = groups
    for param_name, value in params.items():
        for group_name in list(updated.keys()):
            if hasattr(updated[group_name].params, param_name):
                updated = update_group_params(updated, group_name, param_name, value)
    return updated
