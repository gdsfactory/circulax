# Copyright 2026 Circulax contributors

"""Configured OSDI topology, keeping native device buffers at their raw width."""

from __future__ import annotations

import dataclasses
from typing import Any, NoReturn

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from bosdi.circulax import OsdiComponentGroup, OsdiModelDescriptor


class ReducedOsdiGroup(OsdiComponentGroup):
    """Raw-size evaluation with physical equation selection and compact COO stamps."""

    physical_rows: jax.Array = eqx.field(kw_only=True)
    jac_take: jax.Array = eqx.field(kw_only=True)
    configured_params: jax.Array = eqx.field(kw_only=True)
    representatives: tuple[tuple[int, ...], ...] = eqx.field(static=True, kw_only=True)

    def check_binding(self, value: jax.Array) -> jax.Array:
        """Reject direct parameter swaps that leave a stale native snapshot."""
        # Equinox flattening does not preserve array alias identity during JIT.
        if self.params is self.configured_params:
            return value
        same = (self.params == self.configured_params) | (jnp.isnan(self.params) & jnp.isnan(self.configured_params))
        return eqx.error_if(value, ~jnp.all(same), "Reduced OSDI parameters require with_params(); recompile for topology changes")

    def without_handle(self) -> NoReturn:
        """Require full allocation for uncached, traced parameter evaluation."""
        msg = "Reduced OSDI requires a configured handle; compile with osdi_reduction=False for traced updates"
        raise ValueError(msg)

    def with_params(self, new_params: jax.Array) -> ReducedOsdiGroup:
        """Configure new native parameters and reject topology changes explicitly."""
        from osdi_jax import osdi_setup_batch

        if isinstance(new_params, jax.core.Tracer):
            msg = "Traced reduced OSDI updates require osdi_reduction=False"
            raise TypeError(msg)
        params = jnp.asarray(new_params, dtype=jnp.float64)
        handle = osdi_setup_batch(self.model_id, np.asarray(params))
        if tuple(map(tuple, handle.topology.raw_to_representative.tolist())) != self.representatives:
            handle.free()
            msg = "OSDI parameter update changes topology; recompile the circuit"
            raise ValueError(msg)
        return dataclasses.replace(self, params=params, configured_params=params, handle=handle)


def configure_batch(descriptor: OsdiModelDescriptor, items: list[dict], *, reduce: bool) -> tuple[jax.Array, Any, Any]:
    """Use final registration (temperature, analysis, settings) and instance params."""
    params = jnp.asarray([[item["params_dict"][key] for key in descriptor.param_names] for item in items], dtype=jnp.float64)
    try:
        from osdi_jax import osdi_setup_batch
    except ImportError:
        return params, None, None
    handle = osdi_setup_batch(descriptor.model.id, np.asarray(params))
    topology = getattr(handle, "topology", None)
    if topology is not None and not reduce:
        # Full raw internal allocation tolerates internal topology changes, but
        # cannot implement changing external shorts without global net unions.
        # Conservatively reject models whose candidate graph can join terminals
        # (including through internals), rather than silently stamp a false KCL.
        if _may_short_terminals(descriptor):
            handle.free()
            msg = "Full OSDI allocation cannot support terminal-short candidates; use osdi_reduction=True"
            raise ValueError(msg)
        topology = None
    if (
        topology is not None
        and np.array_equal(
            topology.raw_to_representative,
            np.broadcast_to(np.arange(descriptor.model.num_nodes), topology.raw_to_representative.shape),
        )
        and not _may_short_terminals(descriptor)
    ):
        topology = None  # Full internal allocation is safe only when terminal nets cannot change.
    return params, handle, topology


def _may_short_terminals(descriptor: OsdiModelDescriptor) -> bool:
    """Detect full-allocation hazards, without using candidates to infer a map."""
    parent = {-1: -1, **{i: i for i in range(descriptor.model.num_nodes)}}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first, second in descriptor.model.collapsible_pairs:
        first, second = int(first), int(second)
        first = -1 if first == 2**32 - 1 else first
        second = -1 if second == 2**32 - 1 else second
        parent[find(first)] = find(second)
    terminals = [find(i) for i in range(descriptor.model.num_pins)]
    return len(set(terminals)) != len(terminals) or find(-1) in terminals


def union_terminal_nets(count: int, buckets: dict, configured: dict) -> tuple[list[int], int]:
    """Apply terminal shorts across every device before global allocation."""
    parent = list(range(count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for name, items in buckets.items():
        topology = configured[name][2]
        if topology is None:
            continue
        for item, representatives in zip(items, topology.raw_to_representative, strict=True):
            for raw in range(topology.num_terminals):
                rep = int(representatives[raw])
                if rep >= topology.num_terminals:
                    msg = "BOSDI terminal representative must be a terminal or ground"
                    raise ValueError(msg)
                a, b = find(item["ports"][raw]), find(0 if rep < 0 else item["ports"][rep])
                parent[max(a, b)] = min(a, b)
    roots = sorted({find(i) for i in range(count)})
    compact = {root: i for i, root in enumerate(roots)}
    return [compact[find(i)] for i in range(count)], len(roots)


def allocate_raw_indices(  # noqa: PLR0917 -- mirrors the compiler allocation inputs
    items: list[dict],
    topology: Any,
    width: int,
    pins: int,
    size: int,
    port_map: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Allocate surviving internal representatives; zero is Circulax's ground slot."""
    indices = np.zeros((len(items), width), dtype=np.int32)
    physical = np.ones_like(indices, dtype=bool)
    for d, item in enumerate(items):
        reps = np.arange(width) if topology is None else topology.raw_to_representative[d]
        indices[d, :pins] = item["ports"]
        for raw in range(pins, width):
            if reps[raw] == raw:
                indices[d, raw] = size
                size += 1
        for raw, rep in enumerate(reps):
            indices[d, raw] = 0 if rep < 0 else indices[d, rep]
            if raw >= pins:
                port_map[f"{item['name']},__osdi_{raw}"] = int(indices[d, raw])
        if topology is not None:
            physical[d] = topology.physical_row_mask[d] & (indices[d] != 0)
    keep = physical[:, :, None] & (indices[:, None, :] != 0)
    return indices, physical, np.flatnonzero(np.broadcast_to(keep, (len(items), width, width)).reshape(-1)), size


def gather(group: OsdiComponentGroup, y: jax.Array) -> jax.Array:
    """Insert literal zero for grounded and inactive raw slots."""
    if isinstance(group, ReducedOsdiGroup):
        y = group.check_binding(y)
        return jnp.where(group.var_indices == 0, 0.0, y[group.var_indices]).astype(jnp.float64)
    return y[group.var_indices].astype(jnp.float64)


def physical_values(group: OsdiComponentGroup, values: jax.Array) -> jax.Array:
    """Drop equality and ground equations before summation."""
    if isinstance(group, ReducedOsdiGroup):
        return jnp.where(group.physical_rows, values, 0.0)
    return values


def matrix_values(group: OsdiComponentGroup, values: jax.Array) -> jax.Array:
    """Select physical rows and non-ground columns in raw row-major order."""
    flat = values.reshape(-1)
    return flat[group.jac_take] if isinstance(group, ReducedOsdiGroup) else flat


def checked_residual(
    group: OsdiComponentGroup,
    voltages: jax.Array,
    params: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Finite-difference evaluation must also check the perturbed topology."""
    from osdi_jax import osdi_residual_eval

    if isinstance(group, ReducedOsdiGroup):
        from osdi_jax import osdi_residual_eval_with_handle

        updated = group.with_params(params)
        f, q, states = osdi_residual_eval_with_handle(updated.handle, voltages, updated.states)
        return physical_values(group, f), physical_values(group, q), states
    return osdi_residual_eval(group.model_id, voltages, params, group.states)
