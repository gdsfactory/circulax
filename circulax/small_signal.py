# Copyright 2026 Circulax contributors

"""Reusable source-driven linear models and physical small-signal observables."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp

from circulax.solvers.ac_sweep import ac_admittance, linearize_ac

if TYPE_CHECKING:
    from circulax.circuit import Circuit


class QuadratureDrive(eqx.Module):
    """Independent complex RF peak phasors driving the two field coordinates.

    With the exp(+j omega t) convention, the source perturbation is
    Re(x exp(j omega t)) + j Re(y exp(j omega t)). A scalar source drive is
    shorthand for QuadratureDrive(x=amplitude, y=0), including optical sources.
    """

    x: jax.Array | complex = 0.0
    y: jax.Array | complex = 0.0


class SmallSignalResponse(eqx.Module):
    """Physical peak response phasors, retaining both optical field coordinates.

    The final axes of solution are (frequency, unknown). Complex-mode unknowns
    are stored as [x coordinates, y coordinates], each containing complex RF
    phasors. Additional leading axes may be introduced by external jax.vmap.
    y_dc uses the same block layout, but contains real DC coordinates.
    """

    freqs: jax.Array
    y_dc: jax.Array
    solution: jax.Array
    aliases: tuple[tuple[str, int], ...] = eqx.field(static=True)
    num_vars: int = eqx.field(static=True)
    is_complex: bool = eqx.field(static=True)
    wave_component_ports: tuple[str, ...] = eqx.field(static=True, default=())

    @property
    def port_map(self) -> dict[str, int]:
        """Return stable node and internal-state aliases."""
        return dict(self.aliases)

    def _index(self, node: str) -> int:
        try:
            return self.port_map[node]
        except KeyError:
            msg = f"Unknown small-signal node or state {node!r}"
            raise KeyError(msg) from None

    def quadratures(self, positive: str, negative: str | None = None) -> tuple[jax.Array, jax.Array]:
        """Extract complex RF phasors of both physical field coordinates.

        Optionally take the field difference between two nodes. Real circuits
        have only the x coordinate and return zeros for the y coordinate.
        """
        index = self._index(positive)
        x = self.solution[..., index]
        y = self.solution[..., index + self.num_vars] if self.is_complex else jnp.zeros_like(x)
        if negative is not None:
            index = self._index(negative)
            x = x - self.solution[..., index]
            if self.is_complex:
                y = y - self.solution[..., index + self.num_vars]
        return x, y

    def voltage(self, positive: str, negative: str | None = None) -> jax.Array:
        """Extract a physical electrical voltage peak phasor in volts.

        The caller declares these nodes electrical; no discipline is inferred.
        In a mixed complex circuit the electrical voltage is the x coordinate.
        """
        return self.quadratures(positive, negative)[0]

    def power(self, port: str, *, field_unit: str) -> jax.Array:
        """Return complex optical power-modulation peak phasors in watts.

        Declared power-wave component terminal potentials (including electrical
        terminals of mixed components) are rejected: use total-field
        Circuit.port_waves for DC/transient/HB samples. AC traveling-power
        reconstruction is not provided here. For other aliases the caller
        remains responsible for declaring a normalized traveling field.

        field_unit must be 'sqrt(W)', explicitly declaring the selected field
        power normalized, with P = |E|**2. This computes
        2 * (x_dc * dx + y_dc * dy), preserving RF phase. It is not DC power,
        modulation magnitude, or electrical absorbed power. The selected node
        must itself represent the intended optical field, e.g. a traveling wave;
        a total field at a mismatched boundary is not an incident power wave.
        """
        if field_unit != "sqrt(W)":
            msg = "power() requires field_unit='sqrt(W)' for a power-normalized optical field"
            raise ValueError(msg)
        if not self.is_complex:
            msg = "power() currently supports optical fields in complex-mode circuits only"
            raise NotImplementedError(msg)
        if port in self.wave_component_ports:
            msg = "power() cannot interpret a power-wave component terminal potential as traveling-wave power"
            raise ValueError(msg)
        index = self._index(port)
        dx, dy = self.quadratures(port)
        return 2 * (self.y_dc[..., index, None] * dx + self.y_dc[..., index + self.num_vars, None] * dy)

    def sidebands(self, port: str) -> tuple[jax.Array, jax.Array]:
        """Return actual envelope coefficients at +omega and -omega.

        For E(t) = E_dc + upper*exp(j omega t) + lower*exp(-j omega t),
        upper=(dx+j*dy)/2 and lower=(conj(dx)+j*conj(dy))/2. The labels refer
        to envelope frequency; optical upper/lower labeling additionally
        depends on the chosen optical carrier time convention.
        """
        dx, dy = self.quadratures(port)
        return (dx + 1j * dy) / 2, (jnp.conj(dx) + 1j * jnp.conj(dy)) / 2


class SmallSignalModel(eqx.Module):
    """Fixed-bias linearization callable as model(freqs=..., sources=...).

    response() is equivalent to calling the model. Parameters and DC bias are
    fixed at construction; source phasors and frequencies are runtime inputs.
    No matching loads or scattering normalization are introduced.
    """

    # Callable snapshots use identity for JIT's function cache. Numerical fields
    # remain dynamic leaves when the model is passed as a JAX argument.
    __hash__ = object.__hash__
    __eq__ = object.__eq__

    conductance: jax.Array
    capacitance: jax.Array
    y_dc: jax.Array
    source_stamps: jax.Array
    dynamic_groups: dict[str, Any]
    aliases: tuple[tuple[str, int], ...] = eqx.field(static=True)
    source_names: tuple[str, ...] = eqx.field(static=True)
    num_vars: int = eqx.field(static=True)
    is_complex: bool = eqx.field(static=True)
    wave_component_ports: tuple[str, ...] = eqx.field(static=True, default=())

    def __call__(
        self,
        *,
        freqs: jax.Array,
        sources: Mapping[str, complex | jax.Array | QuadratureDrive],
    ) -> SmallSignalResponse:
        """Solve physical source excitation at frequencies in Hz.

        Scalar drives are complex RF peak phasors in the source's physical
        units, applied to its x coordinate. Omitted sources have zero drive.
        Use QuadratureDrive for independent optical x/y excitation. Frequency
        arrays must be one dimensional; use external jax.vmap for batching.
        """
        frequencies = jnp.asarray(freqs)
        if frequencies.ndim != 1 or jnp.iscomplexobj(frequencies):
            msg = "Small-signal frequencies must be a one-dimensional real array in Hz"
            raise ValueError(msg)
        amplitudes_x = jnp.zeros(len(self.source_names), dtype=jnp.complex128)
        amplitudes_y = jnp.zeros_like(amplitudes_x)
        for name, value in sources.items():
            if name not in self.source_names:
                msg = f"Small-signal source {name!r} must identify an instance with an independent-source excitation stamp"
                raise ValueError(msg)
            index = self.source_names.index(name)
            if isinstance(value, QuadratureDrive):
                if not self.is_complex:
                    msg = "QuadratureDrive requires a complex-mode circuit"
                    raise ValueError(msg)
                x, y = jnp.asarray(value.x), jnp.asarray(value.y)
            else:
                x, y = jnp.asarray(value), jnp.asarray(0.0)
            if x.ndim != 0 or y.ndim != 0:
                msg = f"Small-signal source {name!r} needs scalar phasors; use jax.vmap for batches"
                raise ValueError(msg)
            amplitudes_x = amplitudes_x.at[index].set(x)
            amplitudes_y = amplitudes_y.at[index].set(y)
        rhs = self.source_stamps @ amplitudes_x
        if self.is_complex:
            rhs = jnp.concatenate((rhs, self.source_stamps @ amplitudes_y))

        def solve_frequency(freq: jax.Array) -> jax.Array:
            admittance = ac_admittance(
                self.conductance,
                self.capacitance,
                self.dynamic_groups,
                self.y_dc,
                freq,
                num_vars=self.num_vars,
                is_complex=self.is_complex,
                quadratures=self.is_complex,
            )
            return jnp.linalg.solve(admittance, rhs)

        return SmallSignalResponse(
            frequencies,
            self.y_dc,
            jax.vmap(solve_frequency)(frequencies),
            self.aliases,
            self.num_vars,
            self.is_complex,
            self.wave_component_ports,
        )

    def response(
        self,
        *,
        freqs: jax.Array,
        sources: Mapping[str, complex | jax.Array | QuadratureDrive],
    ) -> SmallSignalResponse:
        """Call the model; identical to model(freqs=freqs, sources=sources)."""
        return self(freqs=freqs, sources=sources)


def validate_small_signal_groups(groups: dict[str, Any], *, is_complex: bool) -> None:
    """Reject real-only native Jacobians in a doubled complex system."""
    if is_complex and any(hasattr(g, "model_id") or getattr(g, "combined_func", None) is not None for g in groups.values()):
        msg = "Complex-mode small-signal models with native/combined real-only Jacobians are not supported"
        raise NotImplementedError(msg)


def build_small_signal_model(
    circuit: Circuit,
    groups: dict[str, Any],
    y_dc: jax.Array,
    *,
    dc_groups: dict[str, Any] | None = None,
) -> SmallSignalModel:
    """Build an immutable numerical snapshot with static source and node names."""
    is_complex = circuit.solver.is_complex
    validate_small_signal_groups(groups, is_complex=is_complex)
    if dc_groups is not None:
        validate_small_signal_groups(dc_groups, is_complex=is_complex)
    bias = jnp.asarray(y_dc)
    expected = circuit.sys_size * (2 if is_complex else 1)
    if bias.shape != (expected,) or jnp.iscomplexobj(bias):
        msg = f"Small-signal y_dc must be a real coordinate array of shape ({expected},)"
        raise ValueError(msg)
    conductance, capacitance = linearize_ac(
        groups, circuit.sys_size, bias, is_complex=is_complex, quadratures=is_complex, dc_groups=dc_groups
    )
    source_names = tuple(
        sorted(
            name
            for group in groups.values()
            if getattr(group, "ac_source_stamp", ())
            for name in (getattr(group, "index_map", None) or {})
        )
    )
    stamps = jnp.zeros((circuit.sys_size, len(source_names)), dtype=jnp.complex128)
    for group in groups.values():
        for local_row, coefficient in getattr(group, "ac_source_stamp", ()):
            for name, instance in group.index_map.items():
                stamps = stamps.at[group.eq_indices[instance, local_row], source_names.index(name)].add(coefficient)
    dynamic_groups = {
        name: group for name, group in groups.items() if getattr(group, "is_fdomain", False) or getattr(group, "has_delay", False)
    }
    return SmallSignalModel(
        conductance,
        capacitance,
        bias,
        stamps,
        dynamic_groups,
        tuple(sorted(circuit.port_map.items())),
        source_names,
        circuit.sys_size,
        is_complex,
        tuple(
            sorted(
                {
                    f"{instance},{port}"
                    for group in groups.values()
                    if getattr(group, "wave_reference", None) is not None
                    for instance in (group.index_map or {})
                    for port in (*type(group.params).ports, *getattr(type(group.params), "_raw_to_sanitized_ports", {}))
                }
            )
        ),
    )
