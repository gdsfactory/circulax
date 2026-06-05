"""DC parameter sensitivity via implicit differentiation.

Computes ∂loss/∂params at a DC operating point y* using the adjoint method:
    F(y*, p) = 0  (DC equilibrium)
    ∂loss/∂p = -λᵀ · ∂F/∂p    where J(y*)ᵀ λ = ∂L/∂y

The adjoint approach costs:
1. One linear solve for the adjoint vector λ  (J already factored from DC solve)
2. n_params × n_devices OSDI residual evaluations via finite differences
3. One dot product per parameter

No autodiff through OSDI FFI calls is required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import klujax
import numpy as np

from circulax.solvers.assembly import assemble_system_real
from circulax.solvers.linear import DC_DT, GROUND_STIFFNESS

if TYPE_CHECKING:
    from circulax.solvers.linear import CircuitLinearSolver


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_klu_matrix_vals(solver, all_vals: jax.Array) -> jax.Array:
    """Coalesce raw Jacobian values into the deduplicated KLU value array.

    Mirrors the KLUSplitSolver / KLUSolver ``_solve_impl`` coalescing logic.
    """
    g_vals = jnp.full(solver.ground_indices.shape[0], GROUND_STIFFNESS, dtype=all_vals.dtype)
    l_vals = jnp.full(solver.sys_size, solver.g_leak, dtype=all_vals.dtype)
    raw_vals = jnp.concatenate([all_vals, g_vals, l_vals])
    return jax.ops.segment_sum(raw_vals, solver.map_idx, num_segments=solver.n_unique)


def _resolve_param_cols(
    group,
    param_names: list[str],
    model_descriptor=None,
    param_to_col: dict[str, int] | None = None,
) -> list[int]:
    """Resolve param name strings to column indices in ``group.params``.

    Resolution priority:
    1. Explicit ``param_to_col`` dict (fastest; no model loading).
    2. ``model_descriptor._name_to_idx`` if a descriptor is provided.
    3. Load model param names from ``osdi_loader`` using the model registry
       (requires a locally cached path, best-effort).
    4. Raise with a helpful message.

    Args:
        group: ``OsdiComponentGroup`` instance.
        param_names: Names of parameters to resolve.
        model_descriptor: Optional ``OsdiModelDescriptor`` (from
            :func:`~circulax.osdi_component`); provides ``_name_to_idx``.
        param_to_col: Explicit ``{name: column_index}`` mapping (overrides
            descriptor lookup).

    Returns:
        List of integer column indices (one per name in ``param_names``).

    """
    # --- Strategy 1: explicit mapping ---
    if param_to_col is not None:
        name_to_col = {k.lower(): v for k, v in param_to_col.items()}
        result = []
        for pname in param_names:
            col = name_to_col.get(pname.lower())
            if col is None:
                msg = (
                    f"Parameter {pname!r} not found in param_to_col "
                    f"(available: {list(param_to_col.keys())})."
                )
                raise ValueError(msg)
            result.append(col)
        return result

    # --- Strategy 2: model descriptor ---
    if model_descriptor is not None:
        nti = getattr(model_descriptor, "_name_to_idx", None)
        if nti is not None:
            name_to_col = {k.lower(): v for k, v in nti.items()}
            result = []
            for pname in param_names:
                col = name_to_col.get(pname.lower())
                if col is None:
                    avail = list(nti.keys())
                    msg = (
                        f"Parameter {pname!r} not in OSDI model descriptor "
                        f"(available: {avail})."
                    )
                    raise ValueError(msg)
                result.append(col)
            return result

    # --- Strategy 3: load model from registry (best-effort) ---
    # The group.index_map maps instance_name -> device_index, not param -> col.
    # Try bosdi's module-level model registry if available.
    try:
        from bosdi.osdi_registry import get_model  # type: ignore[import]
        osdi_model = get_model(group.model_id)
        name_to_col = {n.lower(): i for i, n in enumerate(osdi_model.param_names)}
    except (ImportError, Exception):
        osdi_model = None
        name_to_col = None

    if name_to_col is None:
        msg = (
            "Cannot resolve parameter column indices for OSDI group.\n"
            "Pass one of:\n"
            "  param_to_col={'R': 1, 'N': 3, ...}  — explicit mapping\n"
            "  model_descriptor=my_descriptor        — OsdiModelDescriptor from osdi_component()\n"
            "\n"
            "The OsdiModelDescriptor is the object returned by osdi_component() "
            "when you defined the model — pass it as the 'model_descriptor' keyword."
        )
        raise ValueError(msg)

    result = []
    for pname in param_names:
        col = name_to_col.get(pname.lower())
        if col is None:
            avail = list(name_to_col.keys())
            msg = (
                f"Parameter {pname!r} not found in OSDI model "
                f"(available: {avail})."
            )
            raise ValueError(msg)
        result.append(col)
    return result


def _compute_fd_gradients(
    group,
    y_star: jax.Array,
    lam: jax.Array,
    param_names: list[str],
    param_cols: list[int],
    eps: float,
    model_id_override: int | None,
) -> dict[str, jax.Array]:
    """Core FD loop: compute ∂loss/∂p_k = -λᵀ · ∂F/∂p_k for each param.

    Args:
        group: ``OsdiComponentGroup``.
        y_star: DC operating point, shape ``(sys_size,)``.
        lam: Adjoint vector, shape ``(sys_size,)``.
        param_names: Parameter names (already validated).
        param_cols: Corresponding column indices in ``group.params``.
        eps: Base relative FD step size.
        model_id_override: If provided, use this model ID instead of
            ``group.model_id`` (for testing/comparison).

    Returns:
        Dict ``{param_name: gradient_array}`` with shapes ``(n_devices,)``.

    """
    from osdi_jax import osdi_residual_eval

    mid = group.model_id if model_id_override is None else model_id_override
    sys_size = y_star.shape[0]

    params_np = np.array(jax.device_get(group.params))  # (N, num_params)
    n_devices = params_np.shape[0]

    # Base residual: evaluate F(y*, p_base) for this group
    v_all = y_star[group.var_indices].astype(jnp.float64)  # (N, num_nodes)
    cur_base, _, _ = osdi_residual_eval(mid, v_all, group.params, group.states)

    # Scatter to global vector
    f_base_global = jnp.zeros(sys_size, dtype=jnp.float64)
    f_base_global = f_base_global.at[group.eq_indices].add(cur_base)

    gradients: dict[str, jax.Array] = {}

    for pname, pcol in zip(param_names, param_cols):
        grad_per_device = np.zeros(n_devices, dtype=np.float64)

        for i in range(n_devices):
            # Perturb device i's parameter p_k only
            params_perturbed = params_np.copy()
            p_val = params_np[i, pcol]
            # Relative step; use eps as absolute step when p_val ≈ 0
            h = eps * max(abs(p_val), 1.0)
            params_perturbed[i, pcol] = p_val + h

            params_jax_pert = jnp.array(params_perturbed, dtype=jnp.float64)
            cur_pert, _, _ = osdi_residual_eval(mid, v_all, params_jax_pert, group.states)

            f_pert_global = jnp.zeros(sys_size, dtype=jnp.float64)
            f_pert_global = f_pert_global.at[group.eq_indices].add(cur_pert)

            # ∂F/∂p_k[i] = (F_pert - F_base) / h
            dF_dpk = (f_pert_global - f_base_global) / h

            # ∂loss/∂p_k[i] = -λᵀ · ∂F/∂p_k
            grad_per_device[i] = float(-jnp.dot(lam, dF_dpk))

        gradients[pname] = jnp.array(grad_per_device)

    return gradients


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dc_parameter_sensitivity(
    component_groups: dict,
    solver: CircuitLinearSolver,
    y_star: jax.Array,
    loss_fn,
    *,
    osdi_group_key: str,
    param_names: list[str],
    model_descriptor=None,
    param_to_col: dict[str, int] | None = None,
    eps: float = 1e-6,
) -> dict[str, jax.Array]:
    """Compute ∂loss/∂params for named parameters of an OSDI group.

    Uses the implicit differentiation (adjoint) approach at the DC operating
    point ``y*``:

        F(y*, p) = 0
        J(y*)ᵀ λ = ∂L/∂y      [adjoint solve]
        ∂loss/∂p_k = -λᵀ · ∂F/∂p_k      [parameter gradient]

    ``∂F/∂p_k`` is computed via forward finite differences through
    ``osdi_residual_eval``.  This avoids autodiff through the XLA FFI.

    Args:
        component_groups: Compiled circuit groups (dict returned by
            :func:`~circulax.compiler.compile_netlist`).
        solver: A :class:`~circulax.solvers.linear.CircuitLinearSolver`
            instance built from the same ``component_groups``.  Must expose
            ``u_rows``, ``u_cols``, ``map_idx``, ``n_unique``, ``sys_size``,
            ``g_leak``, ``ground_indices`` (i.e. must be a KLU-based solver).
        y_star: DC operating point, shape ``(sys_size,)``.
        loss_fn: Callable ``y -> scalar``.  Must be differentiable w.r.t. ``y``
            via ``jax.grad``.
        osdi_group_key: Key in ``component_groups`` identifying the
            :class:`~bosdi.circulax.OsdiComponentGroup` whose parameters are
            to be differentiated.
        param_names: List of canonical OSDI parameter names to differentiate.
            Must be a subset of the model's parameter names.
        model_descriptor: The :class:`~bosdi.circulax.OsdiModelDescriptor`
            returned by :func:`~circulax.osdi_component` when the model was
            loaded.  Needed to map parameter names to column indices.  Either
            this or ``param_to_col`` must be provided.
        param_to_col: Explicit ``{param_name: column_index}`` mapping.
            Overrides ``model_descriptor`` lookup.  Column indices match the
            ordering of ``group.params`` columns (= ``descriptor.param_names``
            order).
        eps: Relative finite difference step size.  Default ``1e-6`` works
            for most parameters.

    Returns:
        Dict mapping each name in ``param_names`` to a gradient array of
        shape ``(n_devices,)`` — one scalar per device instance in the group.

    Raises:
        ValueError: If ``osdi_group_key`` is missing from ``component_groups``
            or if a requested parameter name is not found in the OSDI group.
        ImportError: If bosdi / osdi_jax is not available.
        TypeError: If ``solver`` does not expose the KLU coalescing attributes
            needed to build the Jacobian matrix.

    """
    try:
        from bosdi.circulax import OsdiComponentGroup
    except ImportError as err:
        raise ImportError("dc_parameter_sensitivity requires bosdi.") from err

    # -----------------------------------------------------------------------
    # Validate inputs
    # -----------------------------------------------------------------------
    if osdi_group_key not in component_groups:
        available = list(component_groups.keys())
        msg = f"OSDI group key {osdi_group_key!r} not found in component_groups. Available: {available}"
        raise ValueError(msg)

    group = component_groups[osdi_group_key]
    if not isinstance(group, OsdiComponentGroup):
        msg = f"Group {osdi_group_key!r} is not an OsdiComponentGroup (got {type(group).__name__})."
        raise TypeError(msg)

    if not hasattr(solver, "u_rows") or not hasattr(solver, "map_idx"):
        msg = (
            "dc_parameter_sensitivity requires a KLU-based solver "
            "(KLUSolver, KLUSplitLinear, KLUSplitQuadratic). "
            f"Got {type(solver).__name__}. "
            "For DenseSolver, use dc_parameter_sensitivity_dense instead."
        )
        raise TypeError(msg)

    # Resolve param names to column indices
    param_cols = _resolve_param_cols(
        group, param_names, model_descriptor=model_descriptor, param_to_col=param_to_col
    )

    # -----------------------------------------------------------------------
    # Step 1: ∂L/∂y  (via JAX autodiff — loss_fn must be JAX-differentiable)
    # -----------------------------------------------------------------------
    dL_dy = jax.grad(loss_fn)(y_star)  # shape: (sys_size,)

    # -----------------------------------------------------------------------
    # Step 2: Assemble J(y*) and coalesce into KLU format
    # -----------------------------------------------------------------------
    _, _, all_vals = assemble_system_real(
        y_star, component_groups, t1=0.0, dt=DC_DT
    )
    coalesced_vals = _build_klu_matrix_vals(solver, all_vals)

    # -----------------------------------------------------------------------
    # Step 3: Solve adjoint system J(y*)ᵀ λ = ∂L/∂y
    #
    # klujax.tsolve_with_symbol solves Aᵀ x = b using the KLU symbolic
    # analysis pre-computed during solver construction.
    # -----------------------------------------------------------------------
    lam = klujax.tsolve_with_symbol(
        solver.u_rows,
        solver.u_cols,
        coalesced_vals,
        dL_dy.astype(jnp.float64),
        solver._handle_wrapper,
    )  # shape: (sys_size,)

    # -----------------------------------------------------------------------
    # Step 4+5: FD ∂F/∂p and compute -λᵀ · ∂F/∂p for each param
    # -----------------------------------------------------------------------
    return _compute_fd_gradients(
        group, y_star, lam, param_names, param_cols, eps, model_id_override=None
    )


def dc_parameter_sensitivity_dense(
    component_groups: dict,
    y_star: jax.Array,
    loss_fn,
    *,
    osdi_group_key: str,
    param_names: list[str],
    model_descriptor=None,
    param_to_col: dict[str, int] | None = None,
    eps: float = 1e-6,
) -> dict[str, jax.Array]:
    """Dense-solver fallback for :func:`dc_parameter_sensitivity`.

    Uses ``jnp.linalg.solve`` instead of KLU for the adjoint solve, so it
    works with any solver backend — including
    :class:`~circulax.solvers.linear.DenseSolver`.  Intended for small circuits
    and unit tests.

    Args match :func:`dc_parameter_sensitivity` except ``solver`` is not
    required; the Jacobian is built densely from ``component_groups``.

    Returns:
        Same as :func:`dc_parameter_sensitivity`.

    """
    try:
        from bosdi.circulax import OsdiComponentGroup
    except ImportError as err:
        raise ImportError("dc_parameter_sensitivity_dense requires bosdi.") from err

    if osdi_group_key not in component_groups:
        available = list(component_groups.keys())
        msg = f"OSDI group key {osdi_group_key!r} not found. Available: {available}"
        raise ValueError(msg)

    group = component_groups[osdi_group_key]
    if not isinstance(group, OsdiComponentGroup):
        msg = f"Group {osdi_group_key!r} is not an OsdiComponentGroup."
        raise TypeError(msg)

    # Resolve param names to column indices
    param_cols = _resolve_param_cols(
        group, param_names, model_descriptor=model_descriptor, param_to_col=param_to_col
    )

    sys_size = y_star.shape[0]

    # -----------------------------------------------------------------------
    # Step 1: ∂L/∂y
    # -----------------------------------------------------------------------
    dL_dy = jax.grad(loss_fn)(y_star)

    # -----------------------------------------------------------------------
    # Step 2: Assemble dense J(y*)
    # -----------------------------------------------------------------------
    _, _, all_vals = assemble_system_real(y_star, component_groups, t1=0.0, dt=DC_DT)

    # Collect COO row/col from component_groups in sorted order (same as
    # _build_index_arrays in linear.py — must match to get consistent values)
    all_rows_list, all_cols_list = [], []
    for k in sorted(component_groups.keys()):
        g = component_groups[k]
        all_rows_list.append(jnp.array(g.jac_rows).reshape(-1))
        all_cols_list.append(jnp.array(g.jac_cols).reshape(-1))
    static_rows = jnp.concatenate(all_rows_list)
    static_cols = jnp.concatenate(all_cols_list)

    J = jnp.zeros((sys_size, sys_size), dtype=jnp.float64)
    J = J.at[static_rows, static_cols].add(all_vals)

    # Add leakage conductance (g_leak=1e-9 default matches DenseSolver default)
    diag_idx = jnp.arange(sys_size)
    J = J.at[diag_idx, diag_idx].add(1e-9)

    # Ground constraint: add GROUND_STIFFNESS to node 0 diagonal
    J = J.at[0, 0].add(GROUND_STIFFNESS)

    # -----------------------------------------------------------------------
    # Step 3: Solve adjoint Jᵀ λ = ∂L/∂y
    # -----------------------------------------------------------------------
    lam = jnp.linalg.solve(J.T, dL_dy.astype(jnp.float64))

    # -----------------------------------------------------------------------
    # Step 4+5: FD ∂F/∂p and compute -λᵀ · ∂F/∂p for each param
    # -----------------------------------------------------------------------
    return _compute_fd_gradients(
        group, y_star, lam, param_names, param_cols, eps, model_id_override=None
    )


__all__ = ["dc_parameter_sensitivity", "dc_parameter_sensitivity_dense"]
