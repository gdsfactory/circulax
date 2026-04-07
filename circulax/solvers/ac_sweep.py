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

import functools
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from circulax.solvers.assembly import assemble_gc_real
from circulax.solvers.linear import GROUND_STIFFNESS, _build_index_arrays


def setup_ac_sweep(
    groups: dict[str, Any],
    num_vars: int,
    port_nodes: list[int],
    *,
    z0: float = 50.0,
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
        num_vars: Total number of scalar unknowns (second return value of
            :func:`~circulax.compile_netlist`).
        port_nodes: Global node indices for each circuit port, in the desired
            port ordering.  Obtain from the port-to-node map returned by
            :func:`~circulax.compile_netlist`::

                _, _, pmap = compile_netlist(net_dict, models)
                port_nodes = [pmap["R1,p1"], pmap["C1,p1"]]

        z0: Reference impedance in ohms, applied uniformly to all ports.
            Defaults to 50.0.

    Returns:
        A callable ``run_ac(y_dc, freqs) -> S`` where:

        - **y_dc** — DC operating point, shape ``(num_vars,)``.
        - **freqs** — frequencies in Hz, shape ``(N_freqs,)``.
        - **S** — S-parameter matrix, shape ``(N_freqs, N_ports, N_ports)``
          complex128.

        Compatible with :func:`jax.jit` and :func:`jax.vmap` over ``y_dc``.

    """
    if 0 in port_nodes:
        msg = "Port node cannot be the ground node (index 0)."
        raise ValueError(msg)

    # --- Pre-compute static COO index arrays (captured in closure) -----------
    static_rows, static_cols, ground_idxs, _ = _build_index_arrays(groups, num_vars, is_complex=False)
    static_rows_jax = jnp.array(static_rows)
    static_cols_jax = jnp.array(static_cols)
    ground_indices = jnp.array(ground_idxs)

    N_ports = len(port_nodes)
    port_nodes_arr = jnp.array(port_nodes, dtype=jnp.int32)

    # Pre-compute fdomain COO scatter index arrays (static integers — avoids
    # re-creating constant arrays on every trace inside vmap).
    fdomain_scatter: dict[str, tuple[Array, Array]] = {
        gk: (
            jnp.array(groups[gk].jac_rows).reshape(-1),
            jnp.array(groups[gk].jac_cols).reshape(-1),
        )
        for gk in sorted(groups)
        if groups[gk].is_fdomain
    }

    # -------------------------------------------------------------------------
    def run_ac(y_dc: Array, freqs: Array) -> Array:
        """Sweep AC frequencies and return the S-parameter matrix.

        Args:
            y_dc: DC operating point, shape ``(num_vars,)``.
            freqs: Frequencies in Hz, shape ``(N_freqs,)``.

        Returns:
            S-parameter matrix, shape ``(N_freqs, N_ports, N_ports)`` complex128.

        """
        # 1. Linearise once at y_dc — outside the frequency loop.
        G_vals, C_vals = assemble_gc_real(y_dc, groups)

        G_mat = jnp.zeros((num_vars, num_vars), dtype=jnp.float64)
        G_mat = G_mat.at[static_rows_jax, static_cols_jax].add(G_vals)

        C_mat = jnp.zeros((num_vars, num_vars), dtype=jnp.float64)
        C_mat = C_mat.at[static_rows_jax, static_cols_jax].add(C_vals)

        # 2. Build RHS: shape (num_vars, N_ports).
        #    Column p: 2/z0 at port_nodes[p], zero elsewhere.
        RHS = jnp.zeros((num_vars, N_ports), dtype=jnp.complex128)
        RHS = RHS.at[port_nodes_arr, jnp.arange(N_ports)].set(2.0 / z0)

        # 3. Single-frequency solve (vmapped over freqs in step 4).
        def _solve_one_freq(f: Array) -> Array:
            omega = 2.0 * jnp.pi * f
            Y = G_mat.astype(jnp.complex128) + 1j * omega * C_mat.astype(jnp.complex128)

            # Add frequency-domain component admittances.
            # The Python loop is over static strings — safe inside vmap.
            for gk, (rows_fd, cols_fd) in fdomain_scatter.items():
                group_fd = groups[gk]
                Y_mats = jax.vmap(functools.partial(group_fd.physics_func, f))(group_fd.params)
                Y = Y.at[rows_fd, cols_fd].add(Y_mats.reshape(-1))

            # Port terminations: Y[port, port] += 1/z0 for each port.
            Y = Y.at[port_nodes_arr, port_nodes_arr].add(1.0 / z0)

            # Ground stiffness: enforces V[ground] ≈ 0.
            Y = Y.at[ground_indices, ground_indices].add(GROUND_STIFFNESS)

            # Single batched solve: factor once, N_ports back-substitutions.
            V = jnp.linalg.solve(Y, RHS)  # (num_vars, N_ports)

            V_ports = V[port_nodes_arr, :]  # (N_ports, N_ports)
            return V_ports - jnp.eye(N_ports, dtype=jnp.complex128)

        # 4. Vmap over frequencies → (N_freqs, N_ports, N_ports).
        return jax.vmap(_solve_one_freq)(freqs)

    return run_ac
