# Copyright 2026 Circulax contributors

"""AC small-signal frequency sweep returning S-parameters.

AC analysis linearises the circuit DAE at a DC operating point and sweeps a
range of frequencies, returning the N-port scattering parameters S(f).

The DAE F(y) + dQ/dt = 0 is linearised at y_dc:

- G = ∂F/∂y|_{y_dc}  (conductance matrix)
- C = ∂Q/∂y|_{y_dc}  (capacitance matrix)

The complex nodal admittance matrix at angular frequency ω is::

    Y(jω) = G + jωC

For N circuit ports with reference impedance Z0::

    Y_total = Y(jω) + diag(1/Z0 at port nodes)
    RHS[:, p] = 2/Z0 at port_nodes[p], zero elsewhere
    V = solve(Y_total, RHS)        # shape (num_vars, N_ports)
    S = V[port_nodes, :] - I      # I = N_ports × N_ports identity

Frequency-domain components contribute Y_fdomain(f) to Y_total at each
frequency point and are evaluated inside the frequency sweep loop.

Example::

    groups, num_vars, pmap = compile_netlist(net_dict, models)
    linear_strat = analyze_circuit(groups, num_vars)
    y_dc = linear_strat.solve_dc(groups, jnp.zeros(num_vars))

    port_nodes = [pmap["R1,p1"]]
    run_ac = setup_ac_sweep(groups, num_vars, port_nodes, z0=50.0)

    freqs = jnp.logspace(6, 10, 100)  # 1 MHz to 10 GHz
    S = run_ac(y_dc, freqs)  # shape (100, 1, 1) complex

    # JIT for repeated sweeps:
    S = jax.jit(run_ac)(y_dc, freqs)
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from circulax.solvers.assembly import (
    assemble_gc_complex,
    assemble_gc_complex_2n,
    assemble_gc_real,
    evaluate_fdomain,
)
from circulax.solvers.linear import GROUND_STIFFNESS, _build_index_arrays


def _delay_admittance(group: Any, y_dc: Array, f: Array, *, is_complex: bool) -> Array:
    """Small-signal local admittance of ``physics(v, hist(v(t-tau)))``.

    Direct and delayed derivatives are kept separate, then the latter is
    multiplied by the exact frequency-domain delay. This is the AC analogue
    of including the interpolation derivative in transient Newton steps.
    """
    if is_complex:
        half = y_dc.shape[0] // 2
        v = y_dc[group.var_indices] + 1j * y_dc[group.var_indices + half]

        def _one(v_i: Array, p_i: Any, tau_i: Array) -> Array:
            def fq(v_arg: Array, h_arg: Array) -> tuple[Array, Array]:
                fv, qv = group.physics_func(y=v_arg, args=p_i, t=0.0, hist=h_arg)
                return fv.astype(v_arg.dtype), qv.astype(v_arg.dtype)

            df_v, dq_v = jax.jacfwd(lambda x: fq(x, v_i), holomorphic=True)(v_i)
            df_h, dq_h = jax.jacfwd(lambda x: fq(v_i, x), holomorphic=True)(v_i)
            phase = jnp.exp(-2j * jnp.pi * f * tau_i)
            return df_v + phase * df_h + 2j * jnp.pi * f * (dq_v + phase * dq_h)

    else:
        v = y_dc[group.var_indices]

        def _one(v_i: Array, p_i: Any, tau_i: Array) -> Array:
            def fq(v_arg: Array, h_arg: Array) -> tuple[Array, Array]:
                return group.physics_func(y=v_arg, args=p_i, t=0.0, hist=h_arg)

            df_v, dq_v = jax.jacfwd(lambda x: fq(x, v_i))(v_i)
            df_h, dq_h = jax.jacfwd(lambda x: fq(v_i, x))(v_i)
            phase = jnp.exp(-2j * jnp.pi * f * tau_i)
            return df_v + phase * df_h + 2j * jnp.pi * f * (dq_v + phase * dq_h)

    tau = jax.vmap(group.tau_func)(group.params)
    return jax.vmap(_one)(v, group.params, tau)


def _delay_admittance_2n(group: Any, y_dc: Array, f: Array) -> Array:
    """Full real-block small-signal admittance for non-holomorphic models."""
    half = y_dc.shape[0] // 2
    vr = y_dc[group.var_indices]
    vi = y_dc[group.var_indices + half]
    u = jnp.concatenate([vr, vi], axis=-1)

    def _one(u_i: Array, p_i: Any, tau_i: Array) -> Array:
        width = u_i.shape[0] // 2

        def fq(u_arg: Array, h_arg: Array) -> tuple[Array, Array]:
            v = u_arg[:width] + 1j * u_arg[width:]
            h = h_arg[:width] + 1j * h_arg[width:]
            fv, qv = group.physics_func(y=v, args=p_i, t=0.0, hist=h)
            return jnp.concatenate([fv.real, fv.imag]), jnp.concatenate([qv.real, qv.imag])

        df_v, dq_v = jax.jacfwd(lambda x: fq(x, u_i))(u_i)
        df_h, dq_h = jax.jacfwd(lambda x: fq(u_i, x))(u_i)
        phase = jnp.exp(-2j * jnp.pi * f * tau_i)
        return df_v + phase * df_h + 2j * jnp.pi * f * (dq_v + phase * dq_h)

    tau = jax.vmap(group.tau_func)(group.params)
    return jax.vmap(_one)(u, group.params, tau)


def _normalize_z0(z0: float | Array, n_ports: int) -> Array:
    """Broadcast z0 to shape ``(N_ports,)``."""
    z0 = jnp.atleast_1d(jnp.asarray(z0, dtype=jnp.complex128))
    if z0.ndim == 0 or (z0.ndim == 1 and z0.shape[0] == 1):
        return jnp.broadcast_to(z0.reshape(1), (n_ports,))
    if z0.ndim == 1 and z0.shape[0] == n_ports:
        return z0
    msg = f"z0 must be a scalar or shape ({n_ports},); got shape {z0.shape}"
    raise ValueError(msg)


def renormalize(
    S: Array,
    z0_from: float | Array,
    z0_to: float | Array,
) -> Array:
    """Renormalize S-parameters from one reference impedance to another.

    Transforms S-parameters computed at impedance ``z0_from`` to the
    equivalent S-parameters at impedance ``z0_to``, using the standard
    power-wave renormalization formula.

    Args:
        S: S-parameter array of shape ``(N_freqs, N_ports, N_ports)``.
        z0_from: Original reference impedance.  Scalar or ``(N_ports,)``
            for frequency-independent, or ``(N_freqs, N_ports)`` for
            frequency-dependent impedance.
        z0_to: Target reference impedance, same shape options as ``z0_from``.

    Returns:
        Renormalized S-parameter array, same shape as ``S``.

    """
    n_freqs, n_ports, _ = S.shape
    z0_from = jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(z0_from, dtype=jnp.complex128)), (n_freqs, n_ports))
    z0_to = jnp.broadcast_to(jnp.atleast_1d(jnp.asarray(z0_to, dtype=jnp.complex128)), (n_freqs, n_ports))

    def _renorm_one(S_f: Array, zf: Array, zt: Array) -> Array:
        I = jnp.eye(n_ports, dtype=jnp.complex128)
        gamma = (zt - zf) / (zt + zf)
        Gamma = jnp.diag(gamma)
        A = jnp.diag(jnp.sqrt(zt.real / zf.real) * zf / zt)
        return A @ (S_f - Gamma) @ jnp.linalg.solve(I - Gamma @ S_f, I) @ jnp.linalg.inv(A)

    return jax.vmap(_renorm_one)(S, z0_from, z0_to)


def linearize_ac(
    groups: dict[str, Any],
    num_vars: int,
    y_dc: Array,
    *,
    is_complex: bool = False,
    quadratures: bool = False,
    dc_groups: dict[str, Any] | None = None,
) -> tuple[Array, Array]:
    """Assemble frequency-independent derivatives with a shared COO layout.

    Quadrature assembly retains all four real derivative blocks. Native AC
    registrations supply C; optional DC registrations supply G.
    """
    rows, cols, _, size = _build_index_arrays(groups, num_vars, is_complex=quadratures)
    if quadratures:
        g_blocks, c_blocks = assemble_gc_complex_2n(y_dc, groups)
        if dc_groups is not None:
            g_blocks, _ = assemble_gc_complex_2n(y_dc, dc_groups)
        g_values, c_values = jnp.concatenate(g_blocks), jnp.concatenate(c_blocks)
    else:
        assemble = assemble_gc_complex if is_complex else assemble_gc_real
        g_values, c_values = assemble(y_dc, groups)
        if dc_groups is not None:
            g_values, _ = assemble(y_dc, dc_groups)
    rows, cols = jnp.asarray(rows), jnp.asarray(cols)
    conductance = jnp.zeros((size, size), dtype=jnp.complex128).at[rows, cols].add(g_values)
    capacitance = jnp.zeros_like(conductance).at[rows, cols].add(c_values)
    return conductance, capacitance


def ac_admittance(
    conductance: Array,
    capacitance: Array,
    groups: dict[str, Any],
    y_dc: Array,
    freq: Array,
    *,
    num_vars: int,
    is_complex: bool = False,
    quadratures: bool = False,
) -> Array:
    """Add frequency-dependent physics and reference constraints to G + jwC."""
    admittance = conductance + 2j * jnp.pi * freq * capacitance
    for group in groups.values():
        rows, cols = group.jac_rows.reshape(-1), group.jac_cols.reshape(-1)
        if getattr(group, "is_fdomain", False):
            values = evaluate_fdomain(group, freq).reshape(-1)
            if quadratures:
                # RF phase and optical-field phase are distinct. Recover the
                # coordinate transfer functions from both envelope sidebands;
                # taking real/imag at +f alone discards temporal phase (e.g.
                # a real capacitor must contribute jwC to both diagonals).
                negative = evaluate_fdomain(group, -freq).reshape(-1)
                direct = (values + jnp.conj(negative)) / 2
                cross = (values - jnp.conj(negative)) / (2j)
                blocks = (direct, -cross, cross, direct)
            else:
                admittance = admittance.at[rows, cols].add(values)
        elif getattr(group, "has_delay", False):
            if quadratures:
                values = _delay_admittance_2n(group, y_dc, freq)
                width = group.var_indices.shape[-1]
                blocks = tuple(
                    values[:, r : r + width, c : c + width].reshape(-1) for r, c in ((0, 0), (0, width), (width, 0), (width, width))
                )
            else:
                values = _delay_admittance(group, y_dc, freq, is_complex=is_complex)
                admittance = admittance.at[rows, cols].add(values.reshape(-1))
        else:
            continue
        if quadratures:
            for values, (row_offset, col_offset) in zip(
                blocks, ((0, 0), (0, num_vars), (num_vars, 0), (num_vars, num_vars)), strict=True
            ):
                admittance = admittance.at[rows + row_offset, cols + col_offset].add(values)
    ground = jnp.array([0, num_vars] if quadratures else [0])
    return admittance.at[ground, ground].add(GROUND_STIFFNESS)


def setup_ac_response(
    groups: dict[str, Any],
    num_vars: int,
    *,
    is_complex: bool = False,
    dc_groups: dict[str, Any] | None = None,
) -> Callable[[Array, Array, Array], Array]:
    """Linearize physical response without scattering-port terminations.

    This legacy interface retains the N-sized holomorphic complex response.
    SmallSignalModel uses the same assembly helpers with full quadratures.
    """

    def run_response(y_dc: Array, freqs: Array, rhs: Array) -> Array:
        conductance, capacitance = linearize_ac(groups, num_vars, y_dc, is_complex=is_complex, dc_groups=dc_groups)

        def solve_frequency(freq: Array) -> Array:
            admittance = ac_admittance(conductance, capacitance, groups, y_dc, freq, num_vars=num_vars, is_complex=is_complex)
            return jnp.linalg.solve(admittance, rhs)

        return jax.vmap(solve_frequency)(freqs)

    return run_response


def setup_ac_sweep(
    groups: dict[str, Any],
    num_vars: int,
    port_nodes: list[int],
    *,
    z0: float | Array = 50.0,
    is_complex: bool = False,
    holomorphic: bool = False,
    dc_groups: dict[str, Any] | None = None,
) -> Callable[[Array, Array], Array]:
    """Configure and return a callable for AC small-signal S-parameter sweep.

    Linearises the circuit DAE at the DC operating point and sweeps over an
    array of frequencies, returning the complex S-parameter matrix at each
    frequency.  The returned callable is compatible with :func:`jax.jit` and
    :func:`jax.vmap`.

    The analysis solves ``Y(jω) · V = RHS`` at each frequency, where::

        Y(jω) = G + jωC + Y_fdomain(f) + port_terminations + ground_penalty

    ``G = ∂F/∂y`` and ``C = ∂Q/∂y`` are extracted once at the DC operating
    point.  ``Y_fdomain(f)`` is the admittance contribution from
    frequency-domain components, re-evaluated at each frequency.

    **S-parameter convention** — matched-load verification:

    - Matched load (Z_circuit = Z0) → S11 = 0
    - Open circuit (Z_circuit → ∞) → S11 = +1
    - Short circuit (Z_circuit = 0) → S11 = −1

    Args:
        groups: Compiled component groups from :func:`~circulax.compile_netlist`.
        dc_groups: Optional groups registered for DC, providing conductance while
            ``groups`` provides AC charge derivatives. Both must share a scatter layout.
        num_vars: Total number of scalar unknowns (second return value of
            :func:`~circulax.compile_netlist`).
        port_nodes: Global node indices for each circuit port, in the desired
            port ordering.  Obtain from the port-to-node map returned by
            :func:`~circulax.compile_netlist`::

                _, _, pmap = compile_netlist(net_dict, models)
                port_nodes = [pmap["R1,p1"], pmap["C1,p1"]]

        z0: Reference impedance in ohms.  Accepts:

            - **scalar** — same impedance for all ports (default: 50.0).
            - **array of shape** ``(N_ports,)`` — per-port impedance.

            For frequency-dependent impedance, solve at a fixed z0 and use
            :func:`renormalize` afterwards.

        is_complex: If ``True``, use complex-valued assembly for photonic
            circuits.  The DC operating point ``y_dc`` is expected in unrolled
            block format (shape ``(2 * num_vars,)``).
        holomorphic: If ``True``, use the N×N Wirtinger system for complex
            circuits.  Defaults to ``False`` so the full 2N×2N real-block
            system is used safely when any component uses
            non-holomorphic operations (e.g. ``jnp.real()``, ``jnp.abs()``),
            which couple the field and its conjugate.  This option has no
            effect for real-valued circuits.

    Returns:
        A callable ``run_ac(y_dc, freqs) -> S`` where:

        - **y_dc** — DC operating point, shape ``(num_vars,)`` for real
        circuits or ``(2 * num_vars,)`` for complex circuits.
        - **freqs** — frequencies in Hz, shape ``(N_freqs,)``.
        - **S** — complex128 S-parameter matrix with shape
        ``(N_freqs, N_ports, N_ports)``.

        Compatible with :func:`jax.jit` and :func:`jax.vmap` over ``y_dc``.

    """
    if 0 in port_nodes:
        msg = "Port node cannot be the ground node (index 0)."
        raise ValueError(msg)

    quadratures = is_complex and not holomorphic
    port_nodes_arr = jnp.asarray(port_nodes, dtype=jnp.int32)
    n_ports = len(port_nodes)
    z0_arr = _normalize_z0(z0, n_ports)
    size = num_vars * (2 if quadratures else 1)

    def run_ac(y_dc: Array, freqs: Array) -> Array:
        conductance, capacitance = linearize_ac(
            groups, num_vars, y_dc, is_complex=is_complex, quadratures=quadratures, dc_groups=dc_groups
        )
        rhs = jnp.zeros((size, n_ports), dtype=jnp.complex128)
        rhs = rhs.at[port_nodes_arr, jnp.arange(n_ports)].set(2.0 / z0_arr)

        def solve_frequency(freq: Array) -> Array:
            admittance = ac_admittance(
                conductance,
                capacitance,
                groups,
                y_dc,
                freq,
                num_vars=num_vars,
                is_complex=is_complex,
                quadratures=quadratures,
            )
            admittance = admittance.at[port_nodes_arr, port_nodes_arr].add(1.0 / z0_arr)
            if quadratures:
                admittance = admittance.at[port_nodes_arr + num_vars, port_nodes_arr + num_vars].add(1.0 / z0_arr)
            solution = jnp.linalg.solve(admittance, rhs)
            fields = solution[port_nodes_arr, :]
            if quadratures:
                fields = fields + 1j * solution[port_nodes_arr + num_vars, :]
            return fields - jnp.eye(n_ports, dtype=jnp.complex128)

        return jax.vmap(solve_frequency)(freqs)

    return run_ac
