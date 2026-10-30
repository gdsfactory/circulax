# Copyright (c) 2026 Circulax contributors
"""Physical I/Q small-signal response of complex envelope circuits.

An I/Q phasor represents two real signals: delta E(t) = Re[I exp(jwt)]
+ i Re[Q exp(jwt)]. Thus the upper envelope sideband is (I+iQ)/2 and
its lower sideband is conj(I-iQ)/2. Electrical observables normally use I.
No extra matching loads are introduced: all loads belong in the schematic.
"""

# ruff: noqa: SLF001, C901, PLR0912, PLR0915
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from circulax.solvers.ac_sweep import _delay_admittance_2n
from circulax.solvers.assembly import assemble_gc_complex_2n, assemble_residual_only_complex
from circulax.solvers.linear import GROUND_STIFFNESS, _build_index_arrays


def quadrature_response(
    circuit: Any,
    *,
    freqs: jax.Array,
    sources: Mapping[str, Any] | None = None,
    parameter_drives: Mapping[str, complex] | None = None,
    y_dc: jax.Array | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Linearize native complex F/Q including nonholomorphic and delayed models.

    ``sources`` maps independent source instance names to ``(I, Q)`` peak
    phasors; a scalar drives I only. These use each source's excitation stamp,
    so an optical voltage source drive is in sqrt(W), not incident power.
    ``parameter_drives`` maps scalar real parameter names (e.g. ``LIGHT.power``)
    to peak phasors in that parameter's units; F and Q forcing are differentiated
    at fixed DC state. DC settings/waveforms are unchanged. Parameter drives
    of frequency-only models and OSDI groups are unsupported. Frequency-only
    circuits are refused until a two-sideband frequency-data contract is available.

    Returned ``in_phase``/``quadrature`` arrays have shape (frequency, sys_size).
    Their entries are complex modulation phasors, not physical complex states.
    Index these with ``port_map`` directly; do not pass them to Circuit.port().
    Upper/lower sidebands are returned separately with the convention above.
    Scalar parameter updates, differentiation and external JAX vmap are supported.
    """
    if not circuit.solver.is_complex:
        msg = "quadrature_response requires a complex-mode circuit"
        raise ValueError(msg)
    updates = dict(params or {})
    arrays = circuit._require_scalar_params(updates, "quadrature_response")
    variant = circuit._for_analysis("ac")
    groups = variant._with_param_values(arrays)
    if any(hasattr(g, "model_id") for g in groups.values()):
        msg = "quadrature_response currently supports native components only"
        raise ValueError(msg)
    if any(g.is_fdomain for g in groups.values()):
        msg = "Frequency-only components require a two-sideband realization; use native F/Q models"
        raise ValueError(msg)
    frequencies = jnp.asarray(freqs)
    if frequencies.ndim != 1:
        msg = "freqs must be a one-dimensional array"
        raise ValueError(msg)
    n = circuit.sys_size
    bias = circuit.dc(params=updates) if y_dc is None else jnp.asarray(y_dc)
    if bias.shape != (2 * n,):
        msg = "y_dc must have shape (2 * circuit.sys_size,)"
        raise ValueError(msg)
    rhs = jnp.zeros(2 * n, dtype=jnp.complex128)
    for name, value in (sources or {}).items():
        matches = [g for g in groups.values() if name in (getattr(g, "index_map", None) or {})]
        if len(matches) != 1 or not getattr(matches[0], "ac_source_stamp", ()):
            msg = f"Source {name!r} needs an independent-source excitation stamp"
            raise ValueError(msg)
        drive = jnp.asarray(value)
        if drive.shape == ():
            drive = jnp.array([drive, 0])
        if drive.shape != (2,):
            msg = "Source drive must be a scalar or an (I, Q) pair"
            raise ValueError(msg)
        group = matches[0]
        instance = group.index_map[name]
        for local_row, coefficient in group.ac_source_stamp:
            row = group.eq_indices[instance, local_row]
            # Complex coefficients act on physical quadratures, not modulation phase.
            rhs = rhs.at[row].add(jnp.real(coefficient) * drive[0] - jnp.imag(coefficient) * drive[1])
            rhs = rhs.at[row + n].add(jnp.imag(coefficient) * drive[0] + jnp.real(coefficient) * drive[1])
    forcing = []
    for key, amplitude in (parameter_drives or {}).items():
        if "." not in key:
            msg = "Parameter drives require an instance.parameter name"
            raise ValueError(msg)
        instance, parameter = key.split(".", 1)
        matches = [g for g in groups.values() if instance in (getattr(g, "index_map", None) or {})]
        if len(matches) != 1 or matches[0].is_fdomain:
            msg = f"Parameter drive {key!r} requires a native time-domain instance"
            raise ValueError(msg)
        group = matches[0]
        base = getattr(group.params, parameter)[group.index_map[instance]]
        if jnp.asarray(base).ndim or jnp.iscomplexobj(base) or jnp.asarray(amplitude).ndim:
            msg = "Parameter drives require a real scalar parameter and scalar phasor"
            raise ValueError(msg)

        def residual(offset: jax.Array, drive_key: str = key, value: Any = base) -> tuple[jax.Array, jax.Array]:
            changed = variant._with_param_values({**arrays, drive_key: value + offset})
            return assemble_residual_only_complex(bias, changed, 0.0, 1.0)

        fp, qp = jax.jacfwd(residual)(jnp.asarray(0.0))
        forcing.append((fp * amplitude, qp * amplitude))
    rows, cols, grounds, _ = _build_index_arrays(groups, n, is_complex=False)
    rows, cols = jnp.asarray(rows), jnp.asarray(cols)
    grounds = jnp.asarray(grounds)
    gb, cb = assemble_gc_complex_2n(bias, groups)
    if circuit._analysis_factory is not None:
        dc_groups = circuit._for_analysis("dc")._with_param_values(arrays)
        gb, _ = assemble_gc_complex_2n(bias, dc_groups)

    def solve(f: jax.Array) -> jax.Array:
        omega = 2 * jnp.pi * f
        matrix = jnp.zeros((2 * n, 2 * n), dtype=jnp.complex128)
        for k, (ro, co) in enumerate(((0, 0), (0, n), (n, 0), (n, n))):
            matrix = matrix.at[rows + ro, cols + co].add(gb[k] + 1j * omega * cb[k])
        for group in groups.values():
            rr, cc = jnp.asarray(group.jac_rows).reshape(-1), jnp.asarray(group.jac_cols).reshape(-1)
            if getattr(group, "has_delay", False):
                width = group.var_indices.shape[-1]
                values = _delay_admittance_2n(group, bias, f).reshape(-1, 2 * width, 2 * width)
                blocks = tuple(
                    x.reshape(-1)
                    for x in (
                        values[:, :width, :width],
                        values[:, :width, width:],
                        values[:, width:, :width],
                        values[:, width:, width:],
                    )
                )
            else:
                continue
            for block, (ro, co) in zip(blocks, ((0, 0), (0, n), (n, 0), (n, n)), strict=True):
                matrix = matrix.at[rr + ro, cc + co].add(block)
        matrix = matrix.at[grounds, grounds].add(GROUND_STIFFNESS)
        matrix = matrix.at[grounds + n, grounds + n].add(GROUND_STIFFNESS)
        drive = rhs
        for fp, qp in forcing:
            drive = drive - fp - 1j * omega * qp
        return jnp.linalg.solve(matrix, drive)

    result = jax.vmap(solve)(frequencies)
    ip, qp = result[:, :n], result[:, n:]
    return {
        "freqs": frequencies,
        "y_dc": bias,
        "in_phase": ip,
        "quadrature": qp,
        "upper_sideband": (ip + 1j * qp) / 2,
        "lower_sideband": jnp.conj(ip - 1j * qp) / 2,
        "port_map": dict(circuit.port_map),
    }
