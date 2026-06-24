"""Transient parameter sensitivity via discrete adjoint method.

Computes ∂loss/∂params through a transient simulation trajectory
using the discrete adjoint of the Backward Euler time-stepping scheme.
Avoids autodiff through the OSDI XLA FFI.

Mathematical background
-----------------------
Time-stepping residual at step k (Backward Euler):

    G[k](y[k], y[k-1], p) = F(y[k], p) + (Q(y[k], p) - Q(y[k-1], p)) / dt = 0

Implicit function: y[k] = Φ(y[k-1], p) defined by G[k] = 0.

Sensitivity through implicit differentiation:

    ∂y[k]/∂y[k-1]  =  J_eff[k]^{-1} · (J_q[k-1] / dt)
    ∂y[k]/∂p       = -J_eff[k]^{-1} · (∂F[k]/∂p + ∂Q[k]/∂p / dt)

where J_eff[k] = J_f[k] + J_q[k] / dt.

Discrete adjoint recurrence (backward sweep, k = N down to 0):

    ψ[N] = ∂L/∂y[N]|direct
    ψ[k] = ∂L/∂y[k]|direct + (J_q[k] / dt)^T · λ[k+1]

    J_eff[k]^T · λ[k] = ψ[k]     (adjoint linear solve at each step)

Parameter gradient accumulation:

    ∂loss/∂p = -Σ_{k=1}^{N} λ[k]^T · (∂F/∂p(y[k]) + ∂Q/∂p(y[k]) / dt)

Note on the J_q coupling term
------------------------------
The term (J_q[k]/dt)^T · λ[k+1] propagates sensitivity backward through
the capacitive coupling between time steps.  For purely resistive circuits
(J_q ≈ 0) this term is zero and the adjoint reduces to N independent
DC-like adjoint solves.  For RC/RLC circuits it is essential for accuracy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from circulax.solvers.assembly import assemble_gc_real, assemble_system_real
from circulax.solvers.linear import GROUND_STIFFNESS
from circulax.solvers.sensitivity import _build_klu_matrix_vals, _resolve_param_cols

if TYPE_CHECKING:
    from circulax.solvers.linear import CircuitLinearSolver


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_jeff_klu(
    component_groups: dict,
    y: jax.Array,
    dt: float,
    solver: CircuitLinearSolver,
) -> jax.Array:
    """Build J_eff coalesced KLU values at state y and timestep dt.

    J_eff = J_f + (1/dt) * J_q.
    """
    _, _, all_vals = assemble_system_real(y, component_groups, t1=0.0, dt=dt)
    return _build_klu_matrix_vals(solver, all_vals)


def _build_jeff_dense(
    component_groups: dict,
    y: jax.Array,
    dt: float,
    sys_size: int,
) -> jax.Array:
    """Build dense J_eff matrix at state y and timestep dt.

    Includes leakage conductance and ground constraint to match
    :class:`~circulax.solvers.linear.DenseSolver`.
    """
    _, _, all_vals = assemble_system_real(y, component_groups, t1=0.0, dt=dt)

    all_rows_list, all_cols_list = [], []
    for k in sorted(component_groups.keys()):
        g = component_groups[k]
        all_rows_list.append(jnp.array(g.jac_rows).reshape(-1))
        all_cols_list.append(jnp.array(g.jac_cols).reshape(-1))
    static_rows = jnp.concatenate(all_rows_list)
    static_cols = jnp.concatenate(all_cols_list)

    J = jnp.zeros((sys_size, sys_size), dtype=jnp.float64)
    J = J.at[static_rows, static_cols].add(all_vals)

    diag_idx = jnp.arange(sys_size)
    J = J.at[diag_idx, diag_idx].add(1e-9)  # g_leak
    J = J.at[0, 0].add(GROUND_STIFFNESS)
    return J


def _build_jq_total(
    component_groups: dict,
    y: jax.Array,
    sys_size: int,
) -> jax.Array:
    """Build the total capacitance matrix C (= dQ/dy) in dense form.

    For non-OSDI groups uses ``assemble_gc_real`` which returns G and C
    separately.  For OSDI groups, ``assemble_gc_real`` returns C=0 because the
    OSDI assembly does not separate G and C.  We patch this by directly calling
    ``osdi_eval`` to get the ``cap`` tensor and scatter it using the group's
    COO indices.

    Returns the unscaled C matrix (multiply by 1/dt at call site).
    """
    try:
        from bosdi.circulax import OsdiComponentGroup
        _has_osdi = True
    except ImportError:
        _has_osdi = False

    C = jnp.zeros((sys_size, sys_size), dtype=jnp.float64)

    all_rows_list, all_cols_list = [], []
    for k in sorted(component_groups.keys()):
        g = component_groups[k]
        all_rows_list.append(jnp.array(g.jac_rows).reshape(-1))
        all_cols_list.append(jnp.array(g.jac_cols).reshape(-1))
    static_rows = jnp.concatenate(all_rows_list)
    static_cols = jnp.concatenate(all_cols_list)

    # Non-OSDI contribution from assemble_gc_real
    _, c_vals_all = assemble_gc_real(y, component_groups)
    C = C.at[static_rows, static_cols].add(c_vals_all)

    # OSDI contribution: add cap (not included in assemble_gc_real's c_vals)
    if _has_osdi:
        for k in sorted(component_groups.keys()):
            g = component_groups[k]
            if not isinstance(g, OsdiComponentGroup):
                continue
            # Get cap from OSDI eval
            try:
                from osdi_jax import osdi_eval, osdi_eval_with_handle
                _has_handle = True
            except ImportError:
                from osdi_jax import osdi_eval
                _has_handle = False

            v_all = y[g.var_indices].astype(jnp.float64)
            try:
                if _has_handle and g.handle is not None:
                    from osdi_jax import osdi_eval_with_handle
                    _, _, _, cap, _ = osdi_eval_with_handle(g.handle, v_all, g.states)
                else:
                    _, _, _, cap, _ = osdi_eval(g.model_id, v_all, g.params, g.states)
            except Exception:  # noqa: BLE001,S112
                continue

            cap = cap.reshape(-1, g.num_nodes, g.num_nodes)
            # Schur reduction if needed (simplification: skip for now, use full cap)
            # Scatter into global C
            rows = jnp.array(g.jac_rows).reshape(-1)
            cols = jnp.array(g.jac_cols).reshape(-1)
            C = C.at[rows, cols].add(cap.reshape(-1))

    return C


def _build_jq_matvec(
    component_groups: dict,
    y: jax.Array,
    dt: float,
    sys_size: int,
) -> jax.Array:
    """Build (J_q / dt) in dense form, including OSDI capacitive contributions."""
    C = _build_jq_total(component_groups, y, sys_size)
    return C / dt


def _jq_matvec_klu(
    component_groups: dict,
    y: jax.Array,
    dt: float,
    v: jax.Array,
) -> jax.Array:
    """Compute (J_q/dt)^T · v using sparse matvec, including OSDI cap."""
    sys_size = v.shape[0]

    try:
        from bosdi.circulax import OsdiComponentGroup
        _has_osdi = True
    except ImportError:
        _has_osdi = False

    # Non-OSDI contribution
    _, c_vals_all = assemble_gc_real(y, component_groups)

    all_rows_list, all_cols_list = [], []
    for k in sorted(component_groups.keys()):
        g = component_groups[k]
        all_rows_list.append(jnp.array(g.jac_rows).reshape(-1))
        all_cols_list.append(jnp.array(g.jac_cols).reshape(-1))
    static_rows = jnp.concatenate(all_rows_list)
    static_cols = jnp.concatenate(all_cols_list)

    # sparse C^T · v for non-OSDI (c_vals_all already has OSDI = 0)
    result = jax.ops.segment_sum(
        c_vals_all * v[static_rows], static_cols, num_segments=sys_size
    )

    # OSDI contributions: add cap contributions directly
    if _has_osdi:
        for k in sorted(component_groups.keys()):
            g = component_groups[k]
            if not isinstance(g, OsdiComponentGroup):
                continue
            try:
                from osdi_jax import osdi_eval
                try:
                    from osdi_jax import osdi_eval_with_handle
                    if g.handle is not None:
                        _, _, _, cap, _ = osdi_eval_with_handle(g.handle, y[g.var_indices].astype(jnp.float64), g.states)
                    else:
                        _, _, _, cap, _ = osdi_eval(g.model_id, y[g.var_indices].astype(jnp.float64), g.params, g.states)
                except ImportError:
                    _, _, _, cap, _ = osdi_eval(g.model_id, y[g.var_indices].astype(jnp.float64), g.params, g.states)
            except Exception:  # noqa: BLE001,S112
                continue

            rows = jnp.array(g.jac_rows).reshape(-1)
            cols = jnp.array(g.jac_cols).reshape(-1)
            cap_flat = cap.reshape(-1)
            # C^T · v: result[cols] += cap_flat * v[rows]
            result = result + jax.ops.segment_sum(
                cap_flat * v[rows], cols, num_segments=sys_size
            )

    return result / dt


def _compute_transient_fd_gradients(  # noqa: PLR0915
    group: Any,
    y_cur: jax.Array,
    y_prev: jax.Array,
    lam: jax.Array,
    dt: float,
    sys_size: int,
    param_names: list[str],
    param_cols: list[int],
    eps: float,
    grad_accumulator: np.ndarray,
    *,
    shared_params: bool = False,
) -> None:
    """FD loop for one BE step: accumulate -λᵀ · ∂G[k]/∂p into grad_accumulator.

    The correct Backward Euler parameter derivative is:

        ∂G[k]/∂p = ∂F(y[k], p)/∂p + (∂Q(y[k], p)/∂p - ∂Q(y[k-1], p)/∂p) / dt

    Note the subtraction of the Q derivative at the *previous* state y[k-1].
    This is essential for parameters that affect Q (e.g. capacitance).
    For purely resistive parameters (no Q dependence), the term vanishes.

    Modifies ``grad_accumulator`` in place; shape ``(len(param_names),)``
    when ``shared_params=True``, or ``(len(param_names), n_devices)`` otherwise.

    Args:
        group: ``OsdiComponentGroup``.
        y_cur: State vector at step k, shape ``(sys_size,)``.
        y_prev: State vector at step k-1, shape ``(sys_size,)``.
        lam: Adjoint vector at step k, shape ``(sys_size,)``.
        dt: Timestep duration for this BE step.
        sys_size: Total system size.
        param_names: Parameter names (already validated).
        param_cols: Corresponding column indices in ``group.params``.
        eps: Relative FD step size.
        grad_accumulator: Pre-allocated array to accumulate gradients.
        shared_params: If True, all devices share the same parameter values
            (process params).  Perturbs all devices at once per parameter
            instead of one-at-a-time, reducing OSDI evals from
            ``n_params × n_devices`` to ``n_params``.

    """
    from osdi_jax import osdi_residual_eval

    try:
        from osdi_jax import osdi_residual_eval_with_handle
        _has_handle = True
    except ImportError:
        _has_handle = False

    mid = group.model_id
    params_np = np.array(jax.device_get(group.params))  # (N, num_params)
    n_devices = params_np.shape[0]

    v_all_cur = y_cur[group.var_indices].astype(jnp.float64)   # (N, num_nodes)
    v_all_prev = y_prev[group.var_indices].astype(jnp.float64)  # (N, num_nodes)

    # Base F(y[k]) and Q(y[k]) and Q(y[k-1])
    if _has_handle and group.handle is not None:
        cur_base, chg_base_cur, _ = osdi_residual_eval_with_handle(group.handle, v_all_cur, group.states)
        _, chg_base_prev, _ = osdi_residual_eval_with_handle(group.handle, v_all_prev, group.states)
    else:
        cur_base, chg_base_cur, _ = osdi_residual_eval(mid, v_all_cur, group.params, group.states)
        _, chg_base_prev, _ = osdi_residual_eval(mid, v_all_prev, group.params, group.states)

    f_base = jnp.zeros(sys_size, dtype=jnp.float64).at[group.eq_indices].add(cur_base)
    q_base_cur = jnp.zeros(sys_size, dtype=jnp.float64).at[group.eq_indices].add(chg_base_cur)
    q_base_prev = jnp.zeros(sys_size, dtype=jnp.float64).at[group.eq_indices].add(chg_base_prev)

    if shared_params:
        # Fast path: perturb ALL devices at once (they share the same param value).
        # One OSDI eval per param instead of n_devices evals.
        for pi, (_pname, pcol) in enumerate(zip(param_names, param_cols, strict=True)):
            params_perturbed = params_np.copy()
            p_val = params_np[0, pcol]
            h = eps * max(abs(p_val), 1.0)
            params_perturbed[:, pcol] = p_val + h
            params_jax_pert = jnp.array(params_perturbed, dtype=jnp.float64)

            cur_pert, chg_pert_cur, _ = osdi_residual_eval(mid, v_all_cur, params_jax_pert, group.states)
            _, chg_pert_prev, _ = osdi_residual_eval(mid, v_all_prev, params_jax_pert, group.states)

            f_pert = jnp.zeros(sys_size, dtype=jnp.float64).at[group.eq_indices].add(cur_pert)
            q_pert_cur = jnp.zeros(sys_size, dtype=jnp.float64).at[group.eq_indices].add(chg_pert_cur)
            q_pert_prev = jnp.zeros(sys_size, dtype=jnp.float64).at[group.eq_indices].add(chg_pert_prev)

            dF_dp = (f_pert - f_base) / h
            dQ_cur_dp = (q_pert_cur - q_base_cur) / h
            dQ_prev_dp = (q_pert_prev - q_base_prev) / h

            dG_dp = dF_dp + (dQ_cur_dp - dQ_prev_dp) / dt
            grad_accumulator[pi] -= float(jnp.dot(lam, dG_dp))
    else:
        for pi, (_pname, pcol) in enumerate(zip(param_names, param_cols, strict=True)):
            for i in range(n_devices):
                params_perturbed = params_np.copy()
                p_val = params_np[i, pcol]
                h = eps * max(abs(p_val), 1.0)
                params_perturbed[i, pcol] = p_val + h
                params_jax_pert = jnp.array(params_perturbed, dtype=jnp.float64)

                cur_pert, chg_pert_cur, _ = osdi_residual_eval(mid, v_all_cur, params_jax_pert, group.states)
                _, chg_pert_prev, _ = osdi_residual_eval(mid, v_all_prev, params_jax_pert, group.states)

                f_pert = jnp.zeros(sys_size, dtype=jnp.float64).at[group.eq_indices].add(cur_pert)
                q_pert_cur = jnp.zeros(sys_size, dtype=jnp.float64).at[group.eq_indices].add(chg_pert_cur)
                q_pert_prev = jnp.zeros(sys_size, dtype=jnp.float64).at[group.eq_indices].add(chg_pert_prev)

                dF_dp = (f_pert - f_base) / h
                dQ_cur_dp = (q_pert_cur - q_base_cur) / h
                dQ_prev_dp = (q_pert_prev - q_base_prev) / h

                dG_dp = dF_dp + (dQ_cur_dp - dQ_prev_dp) / dt
                grad_accumulator[pi, i] -= float(jnp.dot(lam, dG_dp))


# ---------------------------------------------------------------------------
# Public API (KLU-backed)
# ---------------------------------------------------------------------------


def transient_parameter_sensitivity(
    component_groups: dict,
    solver: CircuitLinearSolver,
    y_trajectory: jax.Array,
    ts: jax.Array,
    loss_fn: Callable,
    *,
    osdi_group_key: str,
    param_names: list[str],
    model_descriptor: Any | None = None,
    param_to_col: dict[str, int] | None = None,
    eps: float = 1e-6,
    shared_params: bool = False,
) -> dict[str, jax.Array | float]:
    """Compute ∂loss/∂params via discrete adjoint over a transient trajectory.

    Implements the correct discrete adjoint of the Backward Euler time-stepping
    scheme, with full inter-step coupling through the capacitance matrix (J_q).

    Note:
        This function uses host-side loops and ``jax.device_get`` for
        finite-difference perturbations and cannot be JIT-compiled.

    The adjoint recurrence is (k = N, N-1, ..., 1):

        ψ[k] = ∂L/∂y[k]|direct + (J_q[k] / dt)^T · λ[k+1]
        J_eff[k]^T · λ[k] = ψ[k]                  (linear solve)
        ∂loss/∂p -= λ[k]^T · (∂F[k]/∂p + ∂Q[k]/∂p / dt)   (gradient)

    Args:
        component_groups: Compiled circuit groups (dict returned by
            :func:`~circulax.compiler.compile_netlist`).
        solver: A :class:`~circulax.solvers.linear.CircuitLinearSolver`
            instance built from the same ``component_groups``.  Must expose
            KLU coalescing attributes (``u_rows``, ``u_cols``, ``map_idx``,
            ``n_unique``, ``sys_size``, ``g_leak``, ``ground_indices``).
            For DenseSolver, use :func:`transient_parameter_sensitivity_dense`.
        y_trajectory: Saved trajectory array of shape
            ``(n_checkpoints, sys_size)`` — the ``solution.ys`` from
            ``diffrax.diffeqsolve`` with ``SaveAt(ts=...)``.
        ts: Time points of the checkpoints, shape ``(n_checkpoints,)``.
        loss_fn: Callable ``(y_trajectory, ts) -> scalar`` or
            ``y_final -> scalar``.  Must be differentiable w.r.t.
            ``y_trajectory`` via ``jax.grad``.  If loss depends only on the
            final state, pass a single-argument callable.
        osdi_group_key: Key in ``component_groups`` identifying the
            :class:`~bosdi.circulax.OsdiComponentGroup` whose parameters are
            differentiated.
        param_names: List of canonical OSDI parameter names to differentiate.
        model_descriptor: The :class:`~bosdi.circulax.OsdiModelDescriptor`
            returned by :func:`~circulax.osdi_component`.  Either this or
            ``param_to_col`` must be provided.
        param_to_col: Explicit ``{param_name: column_index}`` mapping.
        eps: Relative finite difference step size.
        shared_params: If True, all devices share the same parameter values
            (process params). Perturbs all devices at once per parameter,
            reducing OSDI evals from ``n_params × n_devices`` to ``n_params``
            per checkpoint. Returns scalar gradients instead of per-device.

    Returns:
        Dict mapping each name in ``param_names`` to a gradient.
        When ``shared_params=False``: array of shape ``(n_devices,)``.
        When ``shared_params=True``: scalar ``float``.

    Raises:
        ValueError: If ``osdi_group_key`` is missing or a parameter name is
            not found.
        TypeError: If ``solver`` does not expose KLU coalescing attributes.
        ImportError: If bosdi / osdi_jax is not available.

    """
    try:
        from bosdi.circulax import OsdiComponentGroup
    except ImportError as err:
        msg = "transient_parameter_sensitivity requires bosdi."
        raise ImportError(msg) from err

    # --- Validate ---
    if osdi_group_key not in component_groups:
        available = list(component_groups.keys())
        msg = f"OSDI group key {osdi_group_key!r} not found. Available: {available}"
        raise ValueError(msg)

    group = component_groups[osdi_group_key]
    if not isinstance(group, OsdiComponentGroup):
        msg = f"Group {osdi_group_key!r} is not an OsdiComponentGroup (got {type(group).__name__})."
        raise TypeError(msg)

    if not hasattr(solver, "u_rows") or not hasattr(solver, "map_idx"):
        msg = (
            "transient_parameter_sensitivity requires a KLU-based solver. "
            f"Got {type(solver).__name__}. "
            "For DenseSolver, use transient_parameter_sensitivity_dense instead."
        )
        raise TypeError(msg)

    param_cols = _resolve_param_cols(
        group, param_names, model_descriptor=model_descriptor, param_to_col=param_to_col
    )

    import klujax

    sys_size = solver.sys_size
    n_checkpoints = y_trajectory.shape[0]
    n_devices = group.params.shape[0]

    import inspect
    sig = inspect.signature(loss_fn)
    _loss_takes_two_args = len(sig.parameters) >= 2

    # Precompute ∂L/∂y[k] for ALL checkpoints in a single jax.grad call.
    # The old per-checkpoint approach called jax.grad N times with different
    # compile-time constants, triggering N separate JIT compilations per step.
    dL_dy_all = (
        jax.grad(loss_fn)(y_trajectory, ts)
        if _loss_takes_two_args
        else jax.grad(lambda yt: loss_fn(yt[-1]))(y_trajectory)
    )

    if shared_params:
        grad_accum = np.zeros((len(param_names),), dtype=np.float64)
    else:
        grad_accum = np.zeros((len(param_names), n_devices), dtype=np.float64)

    lam_next = None

    for k in range(n_checkpoints - 1, 0, -1):
        y_cur = y_trajectory[k].astype(jnp.float64)
        y_prev = y_trajectory[k - 1].astype(jnp.float64)
        dt = float(ts[k]) - float(ts[k - 1])

        psi_k = dL_dy_all[k].astype(jnp.float64)

        if lam_next is not None:
            dt_next = float(ts[k + 1]) - float(ts[k]) if k + 1 < n_checkpoints else dt
            coupling = _jq_matvec_klu(component_groups, y_cur, dt_next, lam_next)
            psi_k = psi_k + coupling

        coalesced_vals = _build_jeff_klu(component_groups, y_cur, dt, solver)
        lam_k = klujax.tsolve_with_symbol(
            solver.u_rows,
            solver.u_cols,
            coalesced_vals,
            psi_k,
            solver._handle_wrapper,  # noqa: SLF001
        )

        _compute_transient_fd_gradients(
            group, y_cur, y_prev, lam_k, dt, sys_size,
            param_names, param_cols, eps, grad_accum,
            shared_params=shared_params,
        )

        lam_next = lam_k

    if shared_params:
        return {pname: float(grad_accum[pi]) for pi, pname in enumerate(param_names)}
    return {pname: jnp.array(grad_accum[pi]) for pi, pname in enumerate(param_names)}


# ---------------------------------------------------------------------------
# Public API (dense fallback)
# ---------------------------------------------------------------------------


def transient_parameter_sensitivity_dense(
    component_groups: dict,
    y_trajectory: jax.Array,
    ts: jax.Array,
    loss_fn: Callable,
    *,
    osdi_group_key: str,
    param_names: list[str],
    model_descriptor: Any | None = None,
    param_to_col: dict[str, int] | None = None,
    eps: float = 1e-6,
    shared_params: bool = False,
) -> dict[str, jax.Array | float]:
    """Dense-solver fallback for :func:`transient_parameter_sensitivity`.

    Uses ``jnp.linalg.solve`` instead of KLU for adjoint solves, so it works
    with any solver backend — including
    :class:`~circulax.solvers.linear.DenseSolver`.  Intended for small
    circuits and unit tests.

    Args match :func:`transient_parameter_sensitivity` except ``solver`` is
    not required.

    Returns:
        Same as :func:`transient_parameter_sensitivity`.

    """
    try:
        from bosdi.circulax import OsdiComponentGroup
    except ImportError as err:
        msg = "transient_parameter_sensitivity_dense requires bosdi."
        raise ImportError(msg) from err

    if osdi_group_key not in component_groups:
        available = list(component_groups.keys())
        msg = f"OSDI group key {osdi_group_key!r} not found. Available: {available}"
        raise ValueError(msg)

    group = component_groups[osdi_group_key]
    if not isinstance(group, OsdiComponentGroup):
        msg = f"Group {osdi_group_key!r} is not an OsdiComponentGroup (got {type(group).__name__})."
        raise TypeError(msg)

    param_cols = _resolve_param_cols(
        group, param_names, model_descriptor=model_descriptor, param_to_col=param_to_col
    )

    sys_size = y_trajectory.shape[1]
    n_checkpoints = y_trajectory.shape[0]
    n_devices = group.params.shape[0]

    import inspect
    sig = inspect.signature(loss_fn)
    _loss_takes_two_args = len(sig.parameters) >= 2

    dL_dy_all = (
        jax.grad(loss_fn)(y_trajectory, ts)
        if _loss_takes_two_args
        else jax.grad(lambda yt: loss_fn(yt[-1]))(y_trajectory)
    )

    if shared_params:
        grad_accum = np.zeros((len(param_names),), dtype=np.float64)
    else:
        grad_accum = np.zeros((len(param_names), n_devices), dtype=np.float64)

    lam_next = None

    for k in range(n_checkpoints - 1, 0, -1):
        y_cur = y_trajectory[k].astype(jnp.float64)
        y_prev = y_trajectory[k - 1].astype(jnp.float64)
        dt = float(ts[k]) - float(ts[k - 1])

        psi_k = dL_dy_all[k].astype(jnp.float64)

        if lam_next is not None:
            dt_next = float(ts[k + 1]) - float(ts[k]) if k + 1 < n_checkpoints else dt
            C_scaled = _build_jq_matvec(component_groups, y_cur, dt_next, sys_size)
            psi_k = psi_k + C_scaled.T @ lam_next

        J = _build_jeff_dense(component_groups, y_cur, dt, sys_size)
        lam_k = jnp.linalg.solve(J.T, psi_k)

        _compute_transient_fd_gradients(
            group, y_cur, y_prev, lam_k, dt, sys_size,
            param_names, param_cols, eps, grad_accum,
            shared_params=shared_params,
        )

        lam_next = lam_k

    if shared_params:
        return {pname: float(grad_accum[pi]) for pi, pname in enumerate(param_names)}
    return {pname: jnp.array(grad_accum[pi]) for pi, pname in enumerate(param_names)}


__all__ = ["transient_parameter_sensitivity", "transient_parameter_sensitivity_dense"]
