# Copyright (c) 2026 Circulax contributors.
"""Shared, dimensionally consistent nonlinear convergence checks."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from circulax.disciplines import Discipline, Nature, TolerancePolicy


class ConvergenceLayout(eqx.Module):
    """Resolved arrays in native order, before optional real/imag unrolling."""

    unknown_atol: jax.Array
    residual_atol: jax.Array
    residual_reference: jax.Array
    rtol: float = eqx.field(static=True)
    local_error_factor: float = eqx.field(static=True)
    names: tuple[str, ...] = eqx.field(static=True)
    unknown_natures: tuple[Nature | None, ...] = eqx.field(static=True)
    equation_natures: tuple[Nature | None, ...] = eqx.field(static=True)

    def magnitude(self, value: jax.Array) -> jax.Array:
        """Pair real/imag parts before taking magnitudes, including HB samples."""
        width = len(self.names)
        if value.shape[-1] == 2 * width:
            return jnp.hypot(value[..., :width], value[..., width:])
        return jnp.abs(value)

    def errors(self, old: jax.Array, new: jax.Array, delta: jax.Array, residual: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return normalized raw correction and physical equation errors."""
        scale = self.unknown_atol + self.rtol * jnp.maximum(self.magnitude(old), self.magnitude(new))
        update_error = jnp.max(self.magnitude(delta) / scale)
        # The reference slot is a potential constraint, not its penalty KCL.
        physical = self.magnitude(residual).at[..., 0].set(self.magnitude(new)[..., 0])
        residual_scale = self.residual_atol + self.rtol * self.residual_reference
        residual_error = jnp.max(physical / residual_scale)
        return update_error, residual_error

    def local_atol(self, *, is_complex: bool) -> jax.Array:
        """Return unknown tolerances in the controller's flat state layout."""
        values = self.unknown_atol * self.local_error_factor
        return jnp.concatenate((values, values)) if is_complex else values


@dataclass(frozen=True)
class _VariableMetadata:
    index: int
    name: str
    unknown: Nature
    equation: Nature
    discipline: Discipline | None


@dataclass
class _CircuitMetadata:
    nodes: dict[int, _VariableMetadata]
    all_names: dict[str, int]
    natures: dict[str, Nature]
    ground_natures: list[Nature]


def _metadata_entries(groups: dict) -> Iterator[_VariableMetadata]:
    """Read and validate annotations on each component-local variable."""
    for group in groups.values():
        unknowns = getattr(group, "unknown_natures", ())
        equations = getattr(group, "equation_natures", ())
        names = getattr(group, "variable_names", ())
        if not unknowns or not equations:
            msg = f"Group {group.name!r} has no discipline metadata; annotate ports and states before using TolerancePolicy"
            raise ValueError(msg)
        for instance, row in enumerate(np.asarray(group.var_indices)):
            for local, index in enumerate(row):
                name = names[instance][local]
                unknown, equation = unknowns[local], equations[local]
                if unknown is None or equation is None:
                    msg = f"Missing unknown/equation nature for {name}"
                    raise ValueError(msg)
                discipline = group.port_disciplines[local] if local < len(group.port_disciplines) else None
                yield _VariableMetadata(int(index), name, unknown, equation, discipline)


def _collect_metadata(groups: dict, size: int) -> _CircuitMetadata:
    """Merge connected variables, retaining every local name for overrides."""
    metadata = _CircuitMetadata({}, {}, {}, [])
    for variable in _metadata_entries(groups):
        metadata.all_names[variable.name] = variable.index
        for nature in (variable.unknown, variable.equation):
            if nature.name in metadata.natures and metadata.natures[nature.name] != nature:
                msg = f"Conflicting definitions for nature {nature.name!r}; use distinct nature names"
                raise ValueError(msg)
            metadata.natures[nature.name] = nature
        if variable.index == 0:
            # Shared zero reference may be connected to several domains.
            metadata.ground_natures.append(variable.unknown)
            continue
        previous = metadata.nodes.get(variable.index)
        if previous is not None:
            if previous.discipline != variable.discipline:
                msg = f"Incompatible disciplines at {previous.name} and {variable.name}"
                raise ValueError(msg)
            if previous.unknown != variable.unknown or previous.equation != variable.equation:
                msg = f"Incompatible natures at {previous.name} and {variable.name}"
                raise ValueError(msg)
        else:
            metadata.nodes[variable.index] = variable
    if any(index not in metadata.nodes for index in range(1, size)):
        msg = "Incomplete discipline metadata for global unknowns"
        raise ValueError(msg)
    return metadata


def _resolve_atol(
    natures: list[Nature | None],
    entries: tuple[tuple[str, float], ...],
    overrides: dict[str, float],
    all_names: dict[str, int],
    ground_atol: float,
) -> jax.Array:
    """Apply nature defaults, then validated per-variable overrides."""
    values = [overrides.get(nature.name, nature.atol) if nature is not None else ground_atol for nature in natures]
    index_overrides: dict[int, float] = {}
    for name, value in entries:
        if name not in all_names and name != "GND":
            msg = f"Unknown variable/equation tolerance override {name!r}"
            raise ValueError(msg)
        index = 0 if name == "GND" else all_names[name]
        if index in index_overrides and index_overrides[index] != value:
            msg = f"Conflicting tolerance overrides on shared node {name!r}"
            raise ValueError(msg)
        index_overrides[index] = value
    for index, value in index_overrides.items():
        if index == 0:
            msg = "Ground tolerance is derived from connected potential natures"
            raise ValueError(msg)
        values[index] = value
    return jnp.asarray(values)


def build_convergence_layout(groups: dict, size: int, policy: TolerancePolicy) -> ConvergenceLayout:
    """Validate native metadata and resolve tolerances outside traced solver loops."""
    metadata = _collect_metadata(groups, size)
    overrides = dict(policy.atol)
    unknown_keys = set(overrides) - metadata.natures.keys()
    if unknown_keys:
        msg = f"Unknown nature tolerance overrides: {sorted(unknown_keys)}"
        raise ValueError(msg)
    if not metadata.ground_natures:
        msg = "Discipline-aware circuits require an explicitly connected reference node"
        raise ValueError(msg)
    # A shared reference satisfies the strictest connected potential tolerance.
    ground_atol = min(overrides.get(n.name, n.atol) for n in metadata.ground_natures)
    nodes = [metadata.nodes[index] for index in range(1, size)]
    unknowns = [None, *[node.unknown for node in nodes]]
    equations = [None, *[node.equation for node in nodes]]
    return ConvergenceLayout(
        _resolve_atol(unknowns, policy.unknown_atol, overrides, metadata.all_names, ground_atol),
        _resolve_atol(equations, policy.residual_atol, overrides, metadata.all_names, ground_atol),
        jnp.asarray([nature.nominal if nature is not None else 0.0 for nature in equations]),
        policy.rtol,
        policy.local_error_factor,
        ("GND", *[node.name for node in nodes]),
        tuple(unknowns),
        tuple(equations),
    )


class PhysicalFixedPointIteration(optx.FixedPointIteration):
    """Retain Optimistix implicit AD while checking raw updates and equations."""

    layout: ConvergenceLayout = eqx.field(kw_only=True)
    residual_fn: Callable = eqx.field(kw_only=True)
    state_shape: tuple[int, ...] | None = eqx.field(static=True, default=None, kw_only=True)

    def step(  # noqa: PLR0917 -- Optimistix solver interface
        self,
        fn: Callable,
        y: jax.Array,
        args: Any,
        options: dict,
        state: Any,
        tags: frozenset,
    ) -> tuple[jax.Array, Any, Any]:
        """Apply a Newton step and replace iterate-only termination metrics."""
        new_y, new_state, aux = super().step(fn, y, args, options, state, tags)
        delta, linear_ok = aux
        residual = self.residual_fn(new_y, args)
        shape = self.state_shape or new_y.shape
        update_error, residual_error = self.layout.errors(
            y.reshape(shape), new_y.reshape(shape), delta.reshape(shape), residual.reshape(shape)
        )
        finite = jnp.all(jnp.isfinite(new_y)) & jnp.all(jnp.isfinite(delta)) & jnp.all(jnp.isfinite(residual)) & linear_ok
        error = jnp.where(finite, jnp.maximum(update_error, residual_error), jnp.inf)
        new_state = eqx.tree_at(lambda s: s.relative_error, new_state, error)
        return new_y, new_state, aux


def newton_solve(
    update: Callable,
    residual: Callable,
    y0: jax.Array,
    *,
    layout: ConvergenceLayout | None,
    rtol: float,
    atol: float,
    max_steps: int,
    args: Any = None,
    state_shape: tuple[int, ...] | None = None,
) -> optx.Solution:
    """Solve with legacy termination or explicit physical convergence checks.

    Updates return ``(next_y, (raw_delta, linear_ok))``. Physical residuals are
    evaluated at the accepted candidate, using the analysis' discretization.
    """
    if layout is None:
        solver = optx.FixedPointIteration(rtol=rtol, atol=atol)
    else:
        residual = eqx.filter_closure_convert(residual, y0, args)
        solver = PhysicalFixedPointIteration(rtol=rtol, atol=atol, layout=layout, residual_fn=residual, state_shape=state_shape)
    return optx.fixed_point(update, solver, y0, args=args, has_aux=True, max_steps=max_steps, throw=False)
