"""A circulax component decorator that accepts a pre-computed Jacobian.

Like :func:`circulax.components.base_component.component`, ``@va_component``
turns a plain physics function into a :class:`CircuitComponent` subclass.
The difference is a ``jacobian_fn`` kwarg: when supplied, ``@va_component``
installs a :func:`jax.custom_jvp` on the component's ``_fast_physics`` that
uses the pre-computed Jacobian matrices to evaluate ``y``-tangents directly
instead of tracing through the physics body with forward-mode AD.

Intended for use by the :mod:`circulax.va` compiler, which lowers OpenVAF
MIR into a physics body plus a Jacobian body drawn from ``DaeSystem.jacobian``
(the values OpenVAF's ``mir_autodiff`` already computed via chain rule).

Parameter tangents still go through JAX AD, via a secondary
``jax.jvp`` call on the physics body inside the custom JVP rule, so
``jax.grad(loss, argnums=...)`` against model parameters still works.
"""

from __future__ import annotations

import inspect
from typing import Any

import jax
import jax.numpy as jnp

from circulax.components.base_component import (
    CircuitComponent,
    PhysicsReturn,
    _build_component,
    _extract_param,
)

JacobianReturn = tuple[jax.Array, jax.Array]


def va_component(
    ports: tuple[str, ...] = (),
    states: tuple[str, ...] = (),
    jacobian_fn: Any = None,
    combined_fn: Any = None,
    amplitude_param: str = "",
    setup_fn: Any = None,
    differentiable_params: tuple[str, ...] | None = (),
) -> Any:
    """Decorate a physics function into a :class:`CircuitComponent` subclass.

    Args:
        ports: Ordered tuple of port names, matching the netlist.
        states: Ordered tuple of internal state variable names.
        jacobian_fn: Optional function with the same signature as the
            decorated physics function that returns
            ``(J_resist, J_react)`` — two dense arrays of shape
            ``(n_ports + n_states, n_ports + n_states)``. When supplied,
            ``_fast_physics`` is wrapped in a :func:`jax.custom_jvp` that
            evaluates state-variable tangents via the provided matrices
            (parameter / time tangents still go through JAX AD).
            When ``None``, ``va_component`` behaves exactly like
            ``component`` — Jacobians fall through to ``jax.jacfwd``.
        combined_fn: Optional function with the same signature as the
            decorated physics function that returns
            ``(f_dict, q_dict, J_resist, J_react)`` — i.e. the residual
            and Jacobian computed in a single body sharing all hoisted
            subexpressions. When supplied alongside ``jacobian_fn``, the
            installed custom JVP routes the Newton hot path through
            ``combined_fn`` (one trace per Newton iter) instead of
            calling the physics body and ``jacobian_fn`` separately
            (two traces with duplicated hoists). At PSP103 scale the
            duplication dominates JIT time — the combined path is
            ~10× faster to compile and noticeably faster per step.
        amplitude_param: Name of the parameter representing source
            amplitude for DC-homotopy stepping. Leave empty for passives.
        differentiable_params: Controls which parameters remain JAX-traced
            leaves (and thus support ``jax.grad``).

            - ``()`` *(default)*: no parameters are differentiable.  All
              params are stored as ``eqx.field(static=True)`` so XLA sees
              their values as compile-time constants and folds ``jnp.where``
              guards — fastest execution, ~22× speedup on the hot path.
              All instances in a ``jax.vmap`` group must share the same
              static-param values (enforced implicitly by Equinox's treedef
              check).
            - ``None``: all parameters are differentiable (full ``jax.grad``
              support through every model parameter).  Equivalent to the
              previous unconditional behaviour.
            - ``("TOXE", "VTH0")``: only the named parameters remain JAX
              leaves; the rest are folded as constants.  Use for targeted
              model fitting while keeping the hot path fast.

    Returns:
        A decorator that accepts a physics function and returns a
        :class:`CircuitComponent` subclass with the Jacobian fast-path
        installed on ``_fast_physics``.

    """

    def decorator(fn: Any) -> type[CircuitComponent]:
        cls = _build_component(
            fn, ports, states, uses_time=False,
            amplitude_param=amplitude_param, setup_fn=setup_fn,
            differentiable_params=differentiable_params,
        )
        if jacobian_fn is None:
            return cls
        # Always install the custom JVP when a jacobian_fn is provided.
        # Even with differentiable_params=() (all static), the Newton hot path
        # needs the analytical Jacobian: jax.jacfwd through the raw physics
        # produces an ill-conditioned matrix because of the constraint rows
        # for collapsed internal nodes (1e40 diagonal entries).  The custom JVP
        # supplies J_f @ y_dot directly; with all-static params, JAX passes
        # symbolic zero tangents for params so the param-gradient branch is a
        # no-op (has_param_grad=False), adding essentially zero overhead.
        _install_custom_jvp(
            cls, fn, jacobian_fn, ports, states,
            uses_time=False, setup_fn=setup_fn, combined_fn=combined_fn,
        )
        return cls

    return decorator


def _install_custom_jvp(
    cls: type[CircuitComponent],
    user_fn: Any,
    jacobian_fn: Any,
    ports: tuple[str, ...],
    states: tuple[str, ...],
    *,
    uses_time: bool,
    setup_fn: Any = None,
    combined_fn: Any = None,
) -> None:
    """Wrap ``cls._fast_physics`` with a custom_jvp that uses ``jacobian_fn``.

    The JVP rule handles two concerns:

    1. ``y``-tangents (the Newton-iteration hot path) — evaluated directly
       via ``J_f @ y_dot`` and ``J_q @ y_dot``, avoiding any trace through
       the physics body.
    2. Parameter / time tangents — delegated to ``jax.jvp`` on an inner
       function that closes over ``y``. This keeps ``jax.grad`` w.r.t.
       parameters working without forcing users to pick between speed
       and differentiability.

    When both are zero (typical Newton step where params are static) the
    rule still calls ``jax.jvp`` for the primal output, which matters:
    JAX doesn't symbolically eliminate zero tangents, so there's one
    physics pass even in the fast path. That pass is what produces
    ``f`` / ``q`` themselves; the speedup comes from skipping the AD
    overhead on top.
    """
    n_p = len(ports)
    sig = inspect.signature(user_fn)
    reserved = ("signals", "s", "t") if uses_time else ("signals", "s")
    params_list = list(sig.parameters.values())[len(reserved):]

    # ``init`` is the new analog-initial positional slot; ``_init_cache`` is
    # the legacy name. Both are handled specially (not passed as kwargs).
    _has_init_arg = bool(params_list) and params_list[0].name == "init"
    if _has_init_arg:
        params_list = params_list[1:]

    param_names = tuple(p.name for p in params_list)

    # ``_init_cache`` is handled specially: extracted from the module instance
    # when available, recomputed via setup_fn for legacy dict callers, or
    # left as None when setup_fn is also absent.
    _has_cache = "_init_cache" in param_names
    _param_names_for_kw = tuple(n for n in param_names if n != "_init_cache")
    _jvp_setup_fn = setup_fn

    # Read specialisation metadata written by _build_component so _unpack
    # knows whether to recompute the cache (partial-diff mode) or reuse it.
    _jvp_dynamic_names: tuple[str, ...] = getattr(cls, "_diff_param_names", _param_names_for_kw)
    _jvp_is_specialized: bool = bool(getattr(cls, "_static_param_names", ()))

    _PortsType = cls._VarsType_P
    _StatesType = cls._VarsType_S

    def _unpack(vars_vec: jax.Array, params: Any) -> tuple[Any, Any, dict]:
        signals = _PortsType(*vars_vec[:n_p]) if _PortsType else ()
        s = _StatesType(*vars_vec[n_p:]) if _StatesType else ()
        kw = {name: _extract_param(params, name) for name in _param_names_for_kw}
        if _has_init_arg:
            # New @<Name>.setup API: read precomputed init from the instance
            # field (set at __init__ time by _custom_init_v2 in base_component).
            kw["init"] = params._init_cache_v2 if hasattr(params, "_init_cache_v2") else {}
        elif _has_cache:
            if _jvp_is_specialized and _jvp_dynamic_names:
                # Partial-diff: recompute so gradients flow through dynamic params.
                # Static params are concrete scalars via eqx.field(static=True),
                # so XLA constant-folds their contributions in _jvp_setup_fn.
                kw["_init_cache"] = _jvp_setup_fn(**{n: kw[n] for n in _param_names_for_kw})
            elif hasattr(params, "_init_cache"):
                kw["_init_cache"] = params._init_cache
            elif _jvp_setup_fn is not None:
                kw["_init_cache"] = _jvp_setup_fn(**{n: kw[n] for n in _param_names_for_kw})
            else:
                kw["_init_cache"] = None
        return signals, s, kw

    raw_physics = cls._fast_physics  # the unwrapped static closure built by _build_component

    def _fast_jacobian(vars_vec: jax.Array, params: Any, t: float) -> JacobianReturn:
        signals, s, kw = _unpack(vars_vec, params)
        if uses_time:
            return jacobian_fn(signals, s, t, **kw)
        return jacobian_fn(signals, s, **kw)

    # When ``combined_fn`` is supplied the lowered emitter has produced a
    # single function returning ``(f, q, J_f, J_q)`` from one hoist block.
    # Routing the Newton hot path through it gives JAX one trace per call
    # instead of two (raw_physics + _fast_jacobian) sharing intermediates
    # only via XLA's later HLO-level CSE — at PSP103 scale that doubles
    # JIT time and meaningfully inflates the per-step cost.
    _has_combined = combined_fn is not None

    def _fast_combined(
        vars_vec: jax.Array, params: Any, t: float,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        signals, s, kw = _unpack(vars_vec, params)
        if uses_time:
            f_dict, q_dict, j_f, j_q = combined_fn(signals, s, t, **kw)
        else:
            f_dict, q_dict, j_f, j_q = combined_fn(signals, s, **kw)
        f_vals = jnp.array([f_dict.get(k, 0.0) for k in (ports + states)])
        q_vals = jnp.array([q_dict.get(k, 0.0) for k in (ports + states)])
        return f_vals, q_vals, j_f, j_q

    @jax.custom_jvp
    def fast_physics(vars_vec: jax.Array, params: Any, t: float) -> tuple[jax.Array, jax.Array]:
        if _has_combined:
            f, q, _j_f, _j_q = _fast_combined(vars_vec, params, t)
            return f, q
        return raw_physics(vars_vec, params, t)

    def fast_physics_jvp(
        primals: tuple[jax.Array, Any, float],
        tangents: tuple[jax.Array, Any, float],
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        y, params, t = primals
        y_dot, params_dot, t_dot = tangents

        # With symbolic_zeros=True JAX passes ``SymbolicZero`` (or ``Zero``)
        # for unperturbed arguments.  Materialise only when needed.
        from jax._src.ad_util import SymbolicZero as _SymZ, Zero as _Zero  # noqa: PLC0415

        def _is_zero(leaf: Any) -> bool:
            return isinstance(leaf, (_SymZ, _Zero))

        def _mat(leaf: Any) -> Any:
            """Convert symbolic Zero to a concrete zero array."""
            return jnp.zeros(leaf.aval.shape, leaf.aval.dtype) if _is_zero(leaf) else leaf

        # Primal + voltage tangent: use the pre-computed Jacobian (hot path).
        # Materialise y_dot in case it's a Zero (only-param-grad scenario).
        if _has_combined:
            f, q, j_f, j_q = _fast_combined(y, params, t)
        else:
            f, q = raw_physics(y, params, t)
            j_f, j_q = _fast_jacobian(y, params, t)
        y_dot_m = _mat(y_dot)
        f_y_dot = j_f @ y_dot_m
        q_y_dot = j_q @ y_dot_m

        # Param/time tangent: only trace through the physics when params are
        # being actively differentiated.  With symbolic_zeros=True, JAX passes
        # ``Zero`` for closed-over (unperturbed) params, letting us skip the
        # expensive jax.jvp through hundreds of parameters during Newton
        # iterations where params are closed over.
        param_leaves = jax.tree_util.tree_leaves(params_dot)
        has_param_grad = not all(_is_zero(leaf) for leaf in param_leaves)

        if has_param_grad:
            params_dot_m = jax.tree_util.tree_map(_mat, params_dot)
            t_dot_m = _mat(t_dot)

            def _params_time_fn(p: Any, tt: Any) -> tuple[jax.Array, jax.Array]:
                return raw_physics(y, p, tt)

            _, (f_p_dot, q_p_dot) = jax.jvp(_params_time_fn, (params, t), (params_dot_m, t_dot_m))
        else:
            f_p_dot = jnp.zeros_like(f)
            q_p_dot = jnp.zeros_like(q)

        return (f, q), (f_p_dot + f_y_dot, q_p_dot + q_y_dot)

    fast_physics.defjvp(fast_physics_jvp, symbolic_zeros=True)

    cls._fast_physics = staticmethod(fast_physics)  # type: ignore[method-assign]


__all__ = ["JacobianReturn", "PhysicsReturn", "va_component"]


# Re-exported only to surface the ``PhysicsReturn`` type alias from
# ``base_component`` in one place (va_component users usually need it in
# their signature).
_ = (jnp,)  # retained import: emitted modules pull ``jnp`` via this module
