"""Assembly functions for the transient circuit solver.

Provides functions for evaluating the residual vectors and effective
Jacobian of the discretised circuit equations at each Newton iteration.
Functions are provided in two variants:

- **Full assembly** (:func:`assemble_system_real`, :func:`assemble_system_complex`)
  — evaluates both the residual and the forward-mode Jacobian via
  ``jax.jacfwd``. Used once per timestep to assemble and factor the frozen
  Jacobian in :class:`~circulax.solver.FactorizedTransientSolver`.

- **Residual only** (:func:`assemble_residual_only_real`,
  :func:`assemble_residual_only_complex`) — evaluates only the primal
  residual, with no Jacobian computation. Used inside the Newton loop where
  the Jacobian has already been factored and only needs to be applied.

Each pair has a real and a complex variant. The complex variant operates on
state vectors in unrolled block format — real parts concatenated with imaginary
parts — allowing complex circuit analyses to reuse real-valued sparse linear
algebra kernels.
"""

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from circulax.components.osdi import OsdiComponentGroup


def _assemble_osdi_group(
    y: Array,
    group: OsdiComponentGroup,
    alpha: float,
    dt: float,
) -> tuple[Array, Array, Array]:
    """Evaluate one OSDI group via bosdi and return ``(f_l, q_l, j_eff)``.

    Args:
        y:     Global state vector.
        group: The :class:`~circulax.components.osdi.OsdiComponentGroup` to evaluate.
        alpha: Jacobian scaling factor (1.0 for Backward Euler).
        dt:    Timestep used to scale the reactive block.

    Returns:
        ``(f_l, q_l, j_eff)`` — ``f_l`` and ``q_l`` are shape
        ``(N, num_nodes)`` and ``j_eff`` is shape
        ``(N, num_nodes, num_nodes)``.  Both are stamped unchanged by
        the caller into the global system using ``group.eq_indices``.

        When ``group.use_schur_reduction`` is True, the internal-node
        rows of ``f_l`` enforce ``V_int = V_int_predicted`` via a
        local Schur back-substitution, the internal rows of ``q_l`` are
        zero (the reactive contribution is already folded into the
        terminal residual at the caller's ``alpha/dt``), and ``j_eff``
        has the reduced 4×4 in its terminal block with identity rows
        and the back-substitution coupling ``A_II⁻¹·A_IT`` in the
        internal rows so global Newton converges the internal slots in
        one step per iteration.
    """
    try:
        from osdi_jax import osdi_eval
    except ImportError as _bosdi_err:
        raise ImportError(
            "OSDI support requires the 'bosdi' package, which could not be imported. "
            "Ensure bosdi is installed or its src/ directory is on PYTHONPATH. "
            "Note: bosdi is not available on all platforms (e.g. Windows)."
        ) from _bosdi_err

    v_all = y[group.var_indices].astype(jnp.float64)  # (N, num_nodes) — terminals + internal
    cur, cond, chg, cap, _ = osdi_eval(group.model_id, v_all, group.params, group.states)

    G = cond.reshape(-1, group.num_nodes, group.num_nodes)   # (N, n, n)  dI/dV
    C = cap.reshape(-1, group.num_nodes, group.num_nodes)    # (N, n, n)  dQ/dV

    if group.use_schur_reduction:
        return _schur_reduce_osdi_stamp(
            v_all=v_all, cur=cur, chg=chg, G=G, C=C,
            alpha=alpha, dt=dt, group=group,
        )

    j_eff = G + (alpha / dt) * C + group.reg_diag           # reg_diag broadcasts over N
    return cur, chg, j_eff


def _schur_reduce_osdi_stamp(
    *,
    v_all: Array,
    cur: Array,
    chg: Array,
    G: Array,
    C: Array,
    alpha: float,
    dt: float,
    group: OsdiComponentGroup,
    gmin: float = 1e-12,
) -> tuple[Array, Array, Array]:
    """Schur-reduce the per-device stamp and pad back to num_nodes.

    Strategy (kept consistent with the caller's ``total_f + (total_q −
    q_prev)/dt`` contract):

    * Reduce ``cur`` and ``chg`` **separately** against the DC ``G_II``
      block.  This produces ``cur_eff_T`` and ``chg_eff_T`` terminal
      stamps that the caller can differently with its existing
      backward-Euler history logic.  The reductions share a single LU
      of ``G_II + gmin·I``.
    * Use the full ``α/dt``-weighted Schur complement for the Jacobian
      stamp so Newton sees the reactive coupling correctly.  This is
      an approximate Jacobian (differs slightly from the exact
      ``d(cur_eff_T + chg_eff_T/dt)/dV``) — that's fine; Newton is
      happy with any descent direction that points at the right root.
    * Pad internal rows with an identity pin to ``V_I_pred`` (computed
      at DC α=0) so the compiler-allocated internal slots stay
      self-consistent.  Internal reactive contributions are zero'd so
      ``(total_q − q_prev)/dt`` leaves them alone.
    """
    N = G.shape[0]
    T = group.num_pins
    I = group.num_nodes - T

    G_TT = G[:, :T, :T]
    G_TI = G[:, :T, T:]
    G_IT = G[:, T:, :T]
    G_II = G[:, T:, T:]
    C_TT = C[:, :T, :T]
    C_TI = C[:, :T, T:]
    C_IT = C[:, T:, :T]
    C_II = C[:, T:, T:]
    cur_T = cur[:, :T]
    cur_I = cur[:, T:]
    chg_T = chg[:, :T]
    chg_I = chg[:, T:]

    eye_I = jnp.eye(I, dtype=G.dtype)

    # ---- DC Schur (α=0) — for cur, chg, internal-node prediction ------------
    G_II_reg = G_II + gmin * eye_I
    # One LU, three RHSs: [G_IT | cur_I | chg_I]  shape (N, I, T+2).
    rhs_dc = jnp.concatenate(
        [G_IT, cur_I[..., None], chg_I[..., None]], axis=-1
    )
    sol_dc = jnp.linalg.solve(G_II_reg, rhs_dc)
    X_dc = sol_dc[..., :T]
    cur_back = sol_dc[..., T]
    chg_back = sol_dc[..., T + 1]

    cur_eff_T = cur_T - jnp.einsum("nij,nj->ni", G_TI, cur_back)
    chg_eff_T = chg_T - jnp.einsum("nij,nj->ni", G_TI, chg_back)

    # Internal-node prediction from DC Schur.
    v_T = v_all[:, :T]
    v_I = v_all[:, T:]
    v_I_pred = cur_back - jnp.einsum("nij,nj->ni", X_dc, v_T)

    # ---- Jacobian Schur at α/dt — for Newton descent direction --------------
    a_over_dt = alpha / dt
    A_TT = G_TT + a_over_dt * C_TT
    A_TI = G_TI + a_over_dt * C_TI
    A_IT = G_IT + a_over_dt * C_IT
    A_II = G_II + a_over_dt * C_II
    A_II_reg = A_II + gmin * eye_I
    X_jac = jnp.linalg.solve(A_II_reg, A_IT)
    j_eff_T = A_TT - A_TI @ X_jac

    # ---- Pad back to (N, num_nodes, num_nodes) -----------------------------
    n = group.num_nodes
    j_padded = jnp.zeros((N, n, n), dtype=j_eff_T.dtype)
    j_padded = j_padded.at[:, :T, :T].set(j_eff_T)
    j_padded = j_padded.at[:, T:, :T].set(X_dc)
    j_padded = j_padded.at[:, T:, T:].set(jnp.broadcast_to(eye_I, (N, I, I)))

    # Residual stamps.  Caller computes f + (q − q_prev)/dt.
    # Internal f row = V_I − V_I_pred  (one-shot pin).  Internal q row = 0.
    cur_padded = jnp.concatenate([cur_eff_T, v_I - v_I_pred], axis=-1)
    chg_padded = jnp.concatenate([chg_eff_T, jnp.zeros_like(v_I)], axis=-1)
    return cur_padded, chg_padded, j_padded


def _real_physics(v: Array, p: Array, group, t1: float) -> tuple[Array, Array]:
    return group.physics_func(y=v, args=p, t=t1)


def _complex_physics(vr: Array, vi: Array, p: Array, group, t1: float) -> tuple[Array, Array, Array, Array]:
    v = vr + 1j * vi
    f, q = group.physics_func(y=v, args=p, t=t1)
    return f.real, f.imag, q.real, q.imag


def _primal_and_jac_real(f, v: Array, p: Array) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
    """Compute f(v,p) and its Jacobian w.r.t. v in a single forward sweep.

    Sweeps n unit tangents via ``jax.jvp``, extracting the primal from the
    first sweep rather than computing it separately.  Jacobian is returned in
    ``(n_eqs, n_vars)`` shape to match ``jax.jacfwd`` convention.
    """
    n = v.shape[0]
    g = lambda v_: f(v_, p)  # close over p; differentiate w.r.t. v only
    (f_vals, q_vals), (dfs, dqs) = jax.vmap(lambda e: jax.jvp(g, (v,), (e,)))(jnp.eye(n))
    return (f_vals[0], q_vals[0]), (dfs.T, dqs.T)


def _primal_and_jac_complex(
    f, vr: Array, vi: Array, p: Array
) -> tuple[
    tuple[Array, Array, Array, Array],
    tuple[Array, Array, Array, Array],
    tuple[Array, Array, Array, Array],
]:
    """Compute f(vr,vi,p) and its Jacobian w.r.t. (vr, vi) in two forward sweeps.

    Mirrors ``jax.jacfwd(f, argnums=(0, 1))``: sweeps unit tangents for vr
    then vi, extracting the primal from the first sweep.  Each Jacobian block
    is returned in ``(n_eqs, n_vars)`` shape.
    """
    n = vr.shape[0]
    zeros_vr = jnp.zeros_like(vr)
    zeros_vi = jnp.zeros_like(vi)
    g = lambda vr_, vi_: f(vr_, vi_, p)  # close over p; differentiate w.r.t. vr, vi only
    (fr_s, fi_s, qr_s, qi_s), (dfr_r, dfi_r, dqr_r, dqi_r) = jax.vmap(lambda e: jax.jvp(g, (vr, vi), (e, zeros_vi)))(jnp.eye(n))
    _, (dfr_i, dfi_i, dqr_i, dqi_i) = jax.vmap(lambda e: jax.jvp(g, (vr, vi), (zeros_vr, e)))(jnp.eye(n))
    primal = (fr_s[0], fi_s[0], qr_s[0], qi_s[0])
    jac_r = (dfr_r.T, dfi_r.T, dqr_r.T, dqi_r.T)
    jac_i = (dfr_i.T, dfi_i.T, dqr_i.T, dqi_i.T)
    return primal, jac_r, jac_i


def assemble_system_real(
    y_guess: Array,
    component_groups: dict,
    t1: float,
    dt: float,
    source_scale: float = 1.0,
    alpha: float = 1.0,
) -> tuple[Array, Array, Array]:
    """Assemble the residual vectors and effective Jacobian values for a real system.

    For each component group, evaluates the physics at ``t1`` and computes the
    forward-mode Jacobian via ``jax.jacfwd``. The effective Jacobian combines
    the resistive and reactive contributions as ``J_eff = df/dy + (alpha/dt) * dq/dy``,
    where ``alpha=1`` recovers Backward Euler and ``alpha=3/2`` (uniform step) gives BDF2.

    Components are processed in sorted key order to ensure a deterministic
    non-zero layout in the sparse Jacobian, which is required for the
    factorisation step.

    Args:
        y_guess: Current state vector of shape ``(sys_size,)``.
        component_groups: Compiled component groups returned by
            :func:`compile_netlist`, keyed by group name.
        t1: Time at which the system is being evaluated.
        dt: Timestep duration, used to scale the reactive Jacobian block.
        source_scale: Multiplicative scale applied to source amplitudes
            (components whose ``amplitude_param`` is set).  Use ``1.0``
            for a standard evaluation and values in ``(0, 1)`` during
            DC homotopy source stepping.
        alpha: Jacobian scaling factor for the reactive block.  Use ``1.0``
            for Backward Euler, the variable-step BDF2 ``α₀`` coefficient for
            BDF2, or ``1/γ`` for SDIRK3 stages.

    Returns:
        A three-tuple ``(total_f, total_q, jac_vals)`` where:

        - **total_f** — assembled resistive residual, shape ``(sys_size,)``.
        - **total_q** — assembled reactive residual, shape ``(sys_size,)``.
        - **jac_vals** — concatenated non-zero values of the effective Jacobian
            in group-sorted order, ready to be passed to the sparse linear solver.

    """
    sys_size = y_guess.shape[0]
    total_f = jnp.zeros(sys_size, dtype=y_guess.dtype)
    total_q = jnp.zeros(sys_size, dtype=y_guess.dtype)
    vals_list = []

    for k in sorted(component_groups.keys()):
        group = component_groups[k]
        if isinstance(group, OsdiComponentGroup):
            f_l, q_l, j_eff = _assemble_osdi_group(y_guess, group, alpha, dt)
            total_f = total_f.at[group.eq_indices].add(f_l)
            total_q = total_q.at[group.eq_indices].add(q_l)
            vals_list.append(j_eff.reshape(-1))
            continue

        if group.is_fdomain:
            # F-domain component: evaluate admittance at f=0 (DC).
            v_locs = y_guess[group.var_indices]
            Y_mats = jax.vmap(lambda p: group.physics_func(0.0, p))(group.params)
            Y_real = Y_mats.real  # (N, n_ports, n_ports)
            f_l = jnp.einsum("nij,nj->ni", Y_real, v_locs)  # (N, n_ports)
            total_f = total_f.at[group.eq_indices].add(f_l)
            vals_list.append(Y_real.reshape(-1))  # Jacobian = Y at DC
            continue

        v_locs = y_guess[group.var_indices]

        ap = group.amplitude_param
        params = (
            eqx.tree_at(lambda p, _ap=ap: getattr(p, _ap), group.params, getattr(group.params, ap) * source_scale)
            if ap
            else group.params
        )

        physics_at_t1 = functools.partial(_real_physics, group=group, t1=t1)

        (f_l, q_l), (df_l, dq_l) = jax.vmap(functools.partial(_primal_and_jac_real, physics_at_t1))(v_locs, params)

        total_f = total_f.at[group.eq_indices].add(f_l)
        total_q = total_q.at[group.eq_indices].add(q_l)
        j_eff = df_l + (alpha / dt) * dq_l
        vals_list.append(j_eff.reshape(-1))

    return total_f, total_q, jnp.concatenate(vals_list)


def assemble_gc_real(
    y_guess: Array,
    component_groups: dict,
) -> tuple[Array, Array]:
    """Return separate G and C COO value arrays at the linearisation point.

    Mirrors :func:`assemble_system_real` but returns ``df/dy`` (conductance) and
    ``dq/dy`` (capacitance) separately instead of combining them as
    ``G + C/dt``.  Frequency-domain groups contribute zero-filled blocks so that
    the returned arrays align with the static COO index arrays produced by
    ``_build_index_arrays`` (which includes all groups).

    Args:
        y_guess: Linearisation point (DC operating point), shape ``(num_vars,)``.
        component_groups: Compiled component groups from :func:`compile_netlist`.

    Returns:
        A two-tuple ``(G_vals, C_vals)`` of real-valued 1-D JAX arrays.  Both
        have the same length as the concatenated ``jac_rows``/``jac_cols`` COO
        index arrays from ``_build_index_arrays``.

    """
    g_vals_list = []
    c_vals_list = []

    for k in sorted(component_groups.keys()):
        group = component_groups[k]
        n_entries = int(jnp.array(group.jac_rows).reshape(-1).shape[0])

        if isinstance(group, OsdiComponentGroup):
            # Use analytical G and C from bosdi directly (DC: alpha=1, dt=1 then separate).
            _, _, j_eff = _assemble_osdi_group(y_guess, group, alpha=1.0, dt=1.0)
            # j_eff = G + C; bodi currently zeros capacitances so G_vals = j_eff, C_vals = 0
            g_vals_list.append(j_eff.reshape(-1).astype(y_guess.dtype))
            c_vals_list.append(jnp.zeros(n_entries, dtype=y_guess.dtype))
            continue

        if group.is_fdomain:
            # Fdomain groups are re-evaluated per-frequency in ac_sweep.
            # Emit zero blocks so COO alignment with _build_index_arrays is preserved.
            g_vals_list.append(jnp.zeros(n_entries, dtype=y_guess.dtype))
            c_vals_list.append(jnp.zeros(n_entries, dtype=y_guess.dtype))
            continue

        v_locs = y_guess[group.var_indices]
        physics_at_dc = functools.partial(_real_physics, group=group, t1=0.0)

        (_, _), (df_l, dq_l) = jax.vmap(functools.partial(_primal_and_jac_real, physics_at_dc))(v_locs, group.params)

        g_vals_list.append(df_l.reshape(-1))
        c_vals_list.append(dq_l.reshape(-1))

    return jnp.concatenate(g_vals_list), jnp.concatenate(c_vals_list)


def assemble_residual_only_real(
    y_guess: Array,
    component_groups: dict,
    t1: float,
    dt: float,
) -> tuple[Array, Array]:
    """Assemble the residual vectors for a real system, without computing the Jacobian.

    Cheaper than :func:`assemble_system_real` as it performs only primal
    evaluations. Used inside the frozen-Jacobian Newton loop where the
    Jacobian has already been factored and only the residual needs to be
    recomputed at each iteration.

    Args:
        y_guess: Current state vector of shape ``(sys_size,)``.
        component_groups: Compiled component groups returned by
            :func:`compile_netlist`, keyed by group name.
        t1: Time at which the system is being evaluated.
        dt: Unused; present for signature symmetry with
            :func:`assemble_system_real` so the two functions are
            interchangeable at call sites.

    Returns:
        A two-tuple ``(total_f, total_q)`` where both arrays have shape
        ``(sys_size,)`` and ``dtype`` matching ``y_guess.dtype``.

    """
    sys_size = y_guess.shape[0]
    total_f = jnp.zeros(sys_size, dtype=y_guess.dtype)
    total_q = jnp.zeros(sys_size, dtype=y_guess.dtype)

    for k in sorted(component_groups.keys()):
        group = component_groups[k]

        if isinstance(group, OsdiComponentGroup):
            # dt is unused here (residual only); alpha/dt factor cancels out.
            f_l, q_l, _ = _assemble_osdi_group(y_guess, group, alpha=1.0, dt=1.0)
            total_f = total_f.at[group.eq_indices].add(f_l)
            total_q = total_q.at[group.eq_indices].add(q_l)
            continue

        if group.is_fdomain:
            # F-domain groups have no time-domain physics; their contribution is
            # added directly in the frequency domain by the HB solver.
            continue

        v = y_guess[group.var_indices]

        physics_at_t1 = functools.partial(_real_physics, group=group, t1=t1)

        f_l, q_l = jax.vmap(physics_at_t1)(v, group.params)

        total_f = total_f.at[group.eq_indices].add(f_l)
        total_q = total_q.at[group.eq_indices].add(q_l)

    return total_f, total_q


def assemble_system_complex(
    y_guess: Array,
    component_groups: dict,
    t1: float,
    dt: float,
    source_scale: float = 1.0,
    alpha: float = 1.0,
) -> tuple[Array, Array, Array]:
    """Assemble the residual vectors and effective Jacobian values for an unrolled complex system.

    The complex state vector is stored in unrolled (block) format: the first
    half of ``y_guess`` holds the real parts of all node voltages/states, the
    second half holds the imaginary parts. This avoids JAX's limited support
    for complex-valued sparse linear solvers by keeping all arithmetic real.

    The Jacobian is split into four real blocks — RR, RI, IR, II — representing
    the partial derivatives of the real and imaginary residual components with
    respect to the real and imaginary state components respectively. The blocks
    are concatenated in RR→RI→IR→II order to match the sparsity index layout
    produced during compilation.

    Args:
        y_guess: Unrolled state vector of shape ``(2 * num_vars,)``, where
            ``y_guess[:num_vars]`` are real parts and ``y_guess[num_vars:]``
            are imaginary parts.
        component_groups: Compiled component groups returned by
            :func:`compile_netlist`, keyed by group name.
        t1: Time at which the system is being evaluated.
        dt: Timestep duration, used to scale the reactive Jacobian blocks.
        source_scale: Multiplicative scale applied to source amplitudes
            (components whose ``amplitude_param`` is set).  Use ``1.0``
            for a standard evaluation and values in ``(0, 1)`` during
            DC homotopy source stepping.
        alpha: Jacobian scaling factor for the reactive blocks.  Use ``1.0``
            for Backward Euler, the variable-step BDF2 ``α₀`` coefficient for
            BDF2, or ``1/γ`` for SDIRK3 stages.

    Returns:
        A three-tuple ``(total_f, total_q, jac_vals)`` where:

        - **total_f** — assembled resistive residual in unrolled format,
            shape ``(2 * num_vars,)``.
        - **total_q** — assembled reactive residual in unrolled format,
            shape ``(2 * num_vars,)``.
        - **jac_vals** — concatenated non-zero values of the four effective
            Jacobian blocks (RR, RI, IR, II) in group-sorted order.

    """
    sys_size = y_guess.shape[0]
    half_size = sys_size // 2
    y_real, y_imag = y_guess[:half_size], y_guess[half_size:]

    total_f = jnp.zeros(sys_size, dtype=jnp.float64)
    total_q = jnp.zeros(sys_size, dtype=jnp.float64)

    vals_blocks: list[list[Array]] = [[], [], [], []]

    for k in sorted(component_groups.keys()):
        group = component_groups[k]

        if isinstance(group, OsdiComponentGroup):
            # OSDI models are real-valued; contribute only to the real residual/Jacobian block.
            f_l, q_l, j_eff = _assemble_osdi_group(y_guess[:half_size], group, alpha, dt)
            total_f = total_f.at[group.eq_indices].add(f_l)
            total_q = total_q.at[group.eq_indices].add(q_l)
            # RR block only; RI, IR, II are zero for real devices.
            vals_blocks[0].append(j_eff.reshape(-1))
            vals_blocks[1].append(jnp.zeros(j_eff.size, dtype=jnp.float64))
            vals_blocks[2].append(jnp.zeros(j_eff.size, dtype=jnp.float64))
            vals_blocks[3].append(jnp.zeros(j_eff.size, dtype=jnp.float64))
            continue

        if group.is_fdomain:
            # F-domain component: evaluate admittance at f=0 (DC) — complex circuit path.
            v_r, v_i = y_real[group.var_indices], y_imag[group.var_indices]
            v_c = v_r + 1j * v_i  # (N, n_ports) complex
            Y_mats = jax.vmap(lambda p: group.physics_func(0.0, p))(group.params)
            i_c = jnp.einsum("nij,nj->ni", Y_mats, v_c)  # (N, n_ports) complex
            idx_r, idx_i = group.eq_indices, group.eq_indices + half_size
            total_f = total_f.at[idx_r].add(i_c.real).at[idx_i].add(i_c.imag)
            # Jacobian blocks: dI/dVr = Y.real, dI/dVi = -Y.imag (by Cauchy-Riemann)
            # For general complex Y: dIr/dVr = Yr, dIr/dVi = -Yi, dIi/dVr = Yi, dIi/dVi = Yr
            Yr = Y_mats.real  # (N, n_ports, n_ports)
            Yi = Y_mats.imag
            vals_blocks[0].append(Yr.reshape(-1))  # RR: dIr/dVr
            vals_blocks[1].append((-Yi).reshape(-1))  # RI: dIr/dVi
            vals_blocks[2].append(Yi.reshape(-1))  # IR: dIi/dVr
            vals_blocks[3].append(Yr.reshape(-1))  # II: dIi/dVi
            continue

        v_r, v_i = y_real[group.var_indices], y_imag[group.var_indices]

        ap = group.amplitude_param
        params = (
            eqx.tree_at(lambda p, _ap=ap: getattr(p, _ap), group.params, getattr(group.params, ap) * source_scale)
            if ap
            else group.params
        )

        physics_split = functools.partial(_complex_physics, group=group, t1=t1)

        (fr, fi, qr, qi), (dfr_r, dfi_r, dqr_r, dqi_r), (dfr_i, dfi_i, dqr_i, dqi_i) = jax.vmap(
            functools.partial(_primal_and_jac_complex, physics_split)
        )(v_r, v_i, params)

        idx_r, idx_i = group.eq_indices, group.eq_indices + half_size
        total_f = total_f.at[idx_r].add(fr).at[idx_i].add(fi)
        total_q = total_q.at[idx_r].add(qr).at[idx_i].add(qi)

        vals_blocks[0].append((dfr_r + (alpha / dt) * dqr_r).reshape(-1))  # RR
        vals_blocks[1].append((dfr_i + (alpha / dt) * dqr_i).reshape(-1))  # RI
        vals_blocks[2].append((dfi_r + (alpha / dt) * dqi_r).reshape(-1))  # IR
        vals_blocks[3].append((dfi_i + (alpha / dt) * dqi_i).reshape(-1))  # II

    all_vals = jnp.concatenate([jnp.concatenate(b) for b in vals_blocks])
    return total_f, total_q, all_vals


def assemble_residual_only_complex(
    y_guess: Array,
    component_groups: dict,
    t1: float,
    dt: float,
) -> tuple[Array, Array]:
    """Assemble the residual vectors for an unrolled complex system, without computing the Jacobian.

    The complex counterpart of :func:`assemble_residual_only_real`. The state
    vector is expected in unrolled block format (real parts followed by imaginary
    parts) matching the layout used by :func:`assemble_system_complex`.

    Args:
        y_guess: Unrolled state vector of shape ``(2 * num_vars,)``.
        component_groups: Compiled component groups returned by
            :func:`compile_netlist`, keyed by group name.
        t1: Time at which the system is being evaluated.
        dt: Unused; present for signature symmetry with
            :func:`assemble_system_complex` so the two functions are
            interchangeable at call sites.

    Returns:
        A two-tuple ``(total_f, total_q)`` where both arrays have shape
        ``(2 * num_vars,)`` and ``dtype`` matching ``y_guess.dtype``.

    """
    sys_size = y_guess.shape[0]
    half_size = sys_size // 2
    y_real, y_imag = y_guess[:half_size], y_guess[half_size:]

    total_f = jnp.zeros(sys_size, dtype=y_guess.dtype)
    total_q = jnp.zeros(sys_size, dtype=y_guess.dtype)

    for k in sorted(component_groups.keys()):
        group = component_groups[k]

        if isinstance(group, OsdiComponentGroup):
            f_l, q_l, _ = _assemble_osdi_group(y_guess[:half_size], group, alpha=1.0, dt=1.0)
            total_f = total_f.at[group.eq_indices].add(f_l)
            total_q = total_q.at[group.eq_indices].add(q_l)
            continue

        if group.is_fdomain:
            # F-domain groups have no time-domain physics; their contribution is
            # added directly in the frequency domain by the HB solver.
            continue

        v_r, v_i = y_real[group.var_indices], y_imag[group.var_indices]

        physics_split = functools.partial(_complex_physics, group=group, t1=t1)

        fr, fi, qr, qi = jax.vmap(physics_split)(v_r, v_i, group.params)

        idx_r, idx_i = group.eq_indices, group.eq_indices + half_size
        total_f = total_f.at[idx_r].add(fr).at[idx_i].add(fi)
        total_q = total_q.at[idx_r].add(qr).at[idx_i].add(qi)

    return total_f, total_q
