"""Bind SPICE waveform sources to their analysis mode and default settings."""

from typing import Any

import equinox as eqx
import jax.numpy as jnp

from circulax.utils import update_group_params, update_params_dict


def record_dc_override(groups: dict[str, Any], param_name: str, instance: str | None = None) -> dict[str, Any]:
    """Remember DC parameter updates, including zero, as explicit overrides."""
    if param_name != "dc":
        return groups
    updated = groups
    for name, group in groups.items():
        if not getattr(type(group.params), "_is_spice_source", False):
            continue
        if instance is None:
            updated = update_group_params(updated, name, "dc_given", 1.0)
        elif instance in group.index_map:
            updated = update_params_dict(updated, name, instance, "dc_given", 1.0)
    return updated


def waveform_groups(groups: dict[str, Any], *, tstep: float, tstop: float) -> dict[str, Any]:
    """Select time-domain physics, preserving per-source analysis defaults.

    Numeric fields keep DC overrides and different waveform kinds in one batch.
    The returned groups are independent of the operating-point registrations.
    """
    updated = dict(groups)
    for name, group in groups.items():
        params = group.params
        if not getattr(type(params), "_is_spice_source", False):
            continue
        params = eqx.tree_at(
            lambda p: (p.source_mode, p.tstep, p.tstop),
            params,
            (
                jnp.ones_like(params.source_mode),
                jnp.where(params.tstep > 0, params.tstep, tstep),
                jnp.where(params.tstop > 0, params.tstop, tstop),
            ),
        )
        updated[name] = eqx.tree_at(lambda g: g.params, group, params)
    return updated
