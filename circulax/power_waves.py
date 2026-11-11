# Copyright 2026 Circulax contributors

"""Retained root-power-wave equations shared by scattering components.

Waves ``a`` and ``b`` are in sqrt(W); flow is positive into the device.
The reference is a common positive real coordinate normalization. These
helpers do not renormalize S-parameters or eliminate incident-wave states.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from circulax.components.base_component import PhysicsReturn, Signals


class PortWaves(NamedTuple):
    """Local traveling envelopes in sqrt(W), with flow positive into the component.

    Powers describe total physical envelopes, not AC perturbations or Fourier
    coefficients. Leading array dimensions are preserved.
    """

    incident: jax.Array
    outgoing: jax.Array

    @property
    def incident_power(self) -> jax.Array:
        """Power arriving at this component port, in W."""
        return jnp.abs(self.incident) ** 2

    @property
    def outgoing_power(self) -> jax.Array:
        """Power leaving this component port, in W."""
        return jnp.abs(self.outgoing) ** 2

    @property
    def net_power(self) -> jax.Array:
        """Signed net power into this component port, in W."""
        return self.incident_power - self.outgoing_power


def reconstruct_waves(potential: jax.Array, flow: jax.Array, *, r0: float = 1.0) -> PortWaves:
    """Reconstruct total waves from potential and signed terminal flow.

    ``flow`` must be positive into the component, including any dynamic
    contribution. The caller supplies compatible carrier, mode, and reference
    coordinates. This helper also works under jit/vmap; ``r0`` is static.
    """
    root = jnp.sqrt(validate_reference(r0))
    v, i = jnp.asarray(potential), jnp.asarray(flow)
    return PortWaves((v / root + root * i) / 2, (v / root - root * i) / 2)


def validate_reference(r0: float) -> float:
    """Validate a static reference normalization before tracing physics."""
    msg = "r0 must be finite, positive, and real"
    if jnp.iscomplexobj(r0):
        raise ValueError(msg)
    try:
        value = float(r0)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(msg) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(msg)
    return value


def power_wave_operator(s_matrix: jax.Array, *, r0: float = 1.0) -> jax.Array:
    """Local F operator on ``[potential, a]``; all Q entries are zero.

    The first rows give flows ``(a - S @ a) / sqrt(r0)``. The remaining
    rows impose ``a + S @ a - potential / sqrt(r0) = 0``. Keeping these
    constraints avoids an inverse of ``I + S``, even at exact losslessness.
    Callers validate the static ``r0`` once during component construction.
    """
    s_matrix = jnp.asarray(s_matrix, dtype=jnp.complex128)
    eye = jnp.eye(s_matrix.shape[-1], dtype=s_matrix.dtype)
    root = jnp.sqrt(r0)
    return jnp.concatenate(
        (
            jnp.concatenate((jnp.zeros_like(s_matrix), (eye - s_matrix) / root), axis=-1),
            jnp.concatenate((-eye / root, eye + s_matrix), axis=-1),
        ),
        axis=-2,
    )


def power_wave_stamp(signals: Signals, ports: tuple[str, ...], s_matrix: jax.Array, *, r0: float = 1.0) -> PhysicsReturn:
    """Return F/Q dictionaries with one ``wave_<port>`` incident state per port.

    ``ports`` must follow the S-matrix's row/column ordering. This includes
    diagonal reflections and arbitrary multiport coupling.
    """
    incident = jnp.array([getattr(signals, f"wave_{port}") for port in ports], dtype=jnp.complex128)
    return power_wave_constraints(signals, ports, jnp.asarray(s_matrix) @ incident, r0=r0)


def power_wave_constraints(
    signals: Signals,
    ports: tuple[str, ...],
    outgoing: jax.Array,
    *,
    r0: float = 1.0,
    algebraic_flow: bool = False,
    incident_states: tuple[str, ...] | None = None,
) -> PhysicsReturn:
    """Stamp retained incident constraints for arbitrary outgoing envelopes.

    ``outgoing`` may depend on delayed incident states. With ``algebraic_flow``,
    replace terminal flow by ``(2*a - potential/sqrt(r0))/sqrt(r0)``. This adds
    the retained constraint divided by sqrt(r0) to each terminal row;
    it preserves the physical solution for arbitrary reflections and delays.
    ``incident_states`` preserves directional state names in native models;
    its order must match ``ports`` (by default, ``wave_<port>``).
    Terminal F then depends only on current fields (Q=0), so complete flows
    can be reconstructed without the accepted-step history at solved samples.
    """
    states = tuple(f"wave_{port}" for port in ports) if incident_states is None else incident_states
    if len(states) != len(ports):
        msg = "incident_states must contain one state per optical port"
        raise ValueError(msg)
    incident = jnp.array([getattr(signals, state) for state in states], dtype=jnp.complex128)
    potential = jnp.array([getattr(signals, port) for port in ports], dtype=jnp.complex128)
    root = jnp.sqrt(r0)
    constraint = incident + outgoing - potential / root
    flow = (2 * incident - potential / root) / root if algebraic_flow else (incident - outgoing) / root
    return {**dict(zip(ports, flow, strict=True)), **dict(zip(states, constraint, strict=True))}, {}
