"""Base class and decorators for defining JAX-compatible circuit components.

Circuit components are defined as plain Python functions decorated with
:func:`component` or :func:`source`, which compile them into
:class:`CircuitComponent` subclasses — Equinox modules whose parameters are
JAX-traceable leaves. The resulting classes expose two entry points:

- ``__call__`` — a debug-friendly instance method that accepts port voltages
  and state values as keyword arguments and returns the physics dicts directly.
- ``solver_call`` — a class method used by the transient solver that operates
  on flat JAX arrays and a parameter container, and is compatible with
  ``jax.vmap`` and ``jax.jacfwd``.

Example::

    @component(ports=("p1", "p2"))
    def Resistor(signals: Signals, R: float = 1.0):
        i = (signals.p1 - signals.p2) / R
        return {"p1": i, "p2": -i}, {}


    r = Resistor(R=100.0)
    f, q = r(p1=1.0, p2=0.0)
"""

import inspect
from typing import Any, ClassVar

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
from jax import Array

PhysicsReturn = tuple[dict[str, Array], dict[str, Array]]


# ---------------------------------------------------------------------------
# Component variable view
# ---------------------------------------------------------------------------
class Signals:
    """Unified view of a component's ports and internal state variables.

    ``signals.name`` reads the value at the current analysis time, while
    ``signals.at_delay(tau).name`` reads it at ``t - tau``.  The analysis
    supplies the delayed values in its natural representation (history
    interpolation for transient, phase rotation for AC/HB, and identity for
    DC), so component physics stays independent of the solver.
    """

    __slots__ = ("_delay_recorder", "_delayed", "_indices", "_names", "_values")

    def __init__(
        self,
        values: Any,
        names: tuple[str, ...],
        *,
        delayed: Any | None = None,
        delay_recorder: list[Any] | None = None,
    ) -> None:
        """Create a named view over current and optionally delayed values."""
        self._values = values
        self._names = names
        self._indices = {name: i for i, name in enumerate(names)}
        self._delayed = values if delayed is None else delayed
        self._delay_recorder = delay_recorder

    def __getattr__(self, name: str) -> Any:
        try:
            index = self._indices[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return self._values[index]

    def at_delay(self, delay: Any) -> "Signals":
        """Return component-local values at ``t - delay``.

        Delays are causal durations and must therefore be non-negative.  The
        current implementation supports one unique delay per component.
        """
        delay = jnp.asarray(delay)
        if delay.ndim != 0:
            msg = "circulax delay: signals.at_delay(...) expects a scalar duration."
            raise ValueError(msg)
        delay = eqxi.error_if(delay, delay < 0, "circulax delay: delays must be non-negative.")
        if self._delay_recorder is not None:
            self._delay_recorder.append(delay)
        return Signals(self._delayed, self._names)


class States(Signals):
    """Legacy state-variable view retained for the Circulax 0.2.3 API.

    New components may read ports and states from a unified :class:`Signals`
    object. Components written against 0.2.3 continue to receive a separate
    ``States`` object when they declare ``(signals, s, ...)``.
    """


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class CircuitComponent(eqx.Module):
    """Base class for all JAX-compatible circuit components.

    Subclasses are not written by hand — they are generated at import time by
    the :func:`component` and :func:`source` decorators, which inspect the
    decorated function's signature to populate the class variables and wire up
    the two physics entry points.

    Class Variables:
        ports: Ordered tuple of port names, e.g. ``("p1", "p2")``.
        states: Ordered tuple of internal state variable names,
            e.g. ``("i_L",)``. Empty for purely algebraic components.
        _uses_time: ``True`` for components decorated with :func:`source`
            whose physics function accepts a ``t`` argument.
        _n_ports: Number of ports, cached to avoid repeated ``len`` calls
            in the hot path.
        _fast_physics: Static closure over the user-defined physics function,
            compatible with ``jax.vmap`` and ``jax.jacfwd``. Signature is
            ``(vars_vec, params, t) -> (f_vec, q_vec)``.
    """

    ports: ClassVar[tuple[str, ...]] = ()
    states: ClassVar[tuple[str, ...]] = ()

    _uses_time: ClassVar[bool] = False
    _is_fdomain: ClassVar[bool] = False
    _holomorphic: ClassVar[bool] = False
    _is_sax_wrapped: ClassVar[bool] = False

    _n_ports: ClassVar[int] = 0

    _fast_physics: ClassVar[Any] = None

    @staticmethod
    def delay_values(params: Any) -> tuple[Any, ...]:  # noqa: ARG004
        """Return inline delays used by a generated component, if any."""
        return ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Cache the port count for each generated subclass."""
        super().__init_subclass__(**kwargs)
        cls._n_ports = len(cls.ports)

    def __call__(
        self,
        t: Any = 0.0,
        y: jax.Array | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Evaluate the component physics (debug entry point).

        Accepts inputs either as a flat state vector ``y`` or as individual
        keyword arguments keyed by port and state name. A heuristic detects
        whether the first positional argument is a time value or a state vector,
        allowing the shorthand ``component(y)`` in addition to the explicit
        ``component(t=0.0, y=y)``.

        Args:
            t: Simulation time, or a state vector when called as
                ``component(y)`` without an explicit ``y`` keyword.
            y: Flat state vector of shape ``(n_ports + n_states,)``. When
                provided, port voltages are taken from ``y[:n_ports]`` and
                state values from ``y[n_ports:]``. Mutually exclusive with
                ``**kwargs``.
            **kwargs: Port voltages and state values by name, e.g.
                ``p1=1.0, p2=0.0``. Used when ``y`` is not provided.
                Missing names default to ``0.0``.

        Returns:
            A two-tuple ``(f, q)`` of dicts mapping port/state names to their
            resistive (``f``) and reactive (``q``) contributions respectively.

        """
        if y is None and not kwargs:
            is_scalar = isinstance(t, (int, float)) or (hasattr(t, "shape") and t.shape == ())
            if not is_scalar:
                y = t
                t = 0.0

        values = y if y is not None else [kwargs.get(name, 0.0) for name in self.ports + self.states]

        signals = Signals(values, self.ports + self.states)
        return self._invoke_physics(signals, t, self)

    @classmethod
    def solver_call(
        cls,
        t: float,
        y: jax.Array,
        args: Any,
        hist: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluate the component physics (solver entry point).

        Thin wrapper around the static ``_fast_physics`` closure. Called by
        the transient solver inside ``jax.vmap`` across all instances in a
        component group, and differentiated via ``jax.jacfwd`` to assemble
        the system Jacobian.

        Args:
            t: Current simulation time.
            y: Flat state vector of shape ``(n_ports + n_states,)`` containing
                port voltages followed by state variable values.
            args: Parameter container for this instance. May be a dict
                ``{"R": 100.0}`` or an object (e.g. the component instance
                itself) whose attributes match the parameter names. Must not
                be a raw scalar.
            hist: Internally supplied delayed local-variable vector of shape
                ``(n_ports + n_states,)``, i.e. this instance's ports followed
                by its internal states, evaluated at ``t - tau``. Only
                exposed through ``signals.at_delay(...)``; ``None`` otherwise.

        Returns:
            A two-tuple ``(f_vec, q_vec)`` of JAX arrays, each of shape
            ``(n_ports + n_states,)``, containing the resistive and reactive
            contributions for every port and state variable.

        """
        return cls._fast_physics(y, args, t, hist)

    # -----------------------------------------------------------------------
    # Internal Dispatchers (wired up by decorator)
    # -----------------------------------------------------------------------
    def physics(self, *args: Any, **kwargs: Any) -> tuple[dict, dict]:
        """Raw physics dispatch; overridden by the decorator-generated subclass."""
        raise NotImplementedError

    def _invoke_physics(
        self,
        signals: Any,
        t: float,
        params: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Trampoline used by ``__call__`` to dispatch to the user physics function.

        Overridden by the decorator-generated subclass with a closure that
        extracts named parameters from ``params`` and forwards them as keyword
        arguments to the original decorated function.

        Args:
            signals: Unified view of current port and internal-state values.
            t: Current simulation time.
            params: Parameter container (instance or dict).

        Returns:
            A two-tuple ``(f, q)`` of physics dicts.

        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helper: Parameter Extraction
# ---------------------------------------------------------------------------
def _extract_param(container: Any, name: str) -> Any:
    """Extract a named parameter from either a dict or an object.

    Args:
        container: A dict or any object with named attributes.
        name: The parameter name to look up.

    Returns:
        The parameter value.

    """
    if isinstance(container, dict):
        return container[name]
    return getattr(container, name)


# ---------------------------------------------------------------------------
# Holomorphic validation via jaxpr inspection
# ---------------------------------------------------------------------------
_NON_HOLOMORPHIC_PRIMITIVES = frozenset({"real", "imag", "conj", "abs"})


def _jaxpr_has_non_holomorphic(jaxpr: Any) -> set[str]:
    """Walk a jaxpr and return names of non-holomorphic primitives on traced variables."""
    from jax._src.core import ClosedJaxpr, Jaxpr, Literal

    found: set[str] = set()
    for eqn in jaxpr.eqns:
        if eqn.primitive.name in _NON_HOLOMORPHIC_PRIMITIVES:
            has_traced_complex_input = any(
                not isinstance(v, Literal) and hasattr(v.aval, "dtype") and jnp.issubdtype(v.aval.dtype, jnp.complexfloating)
                for v in eqn.invars
            )
            if has_traced_complex_input:
                found.add(eqn.primitive.name)
        for v in eqn.params.values():
            if isinstance(v, Jaxpr):
                found |= _jaxpr_has_non_holomorphic(v)
            elif isinstance(v, ClosedJaxpr):
                found |= _jaxpr_has_non_holomorphic(v.jaxpr)
    return found


# ---------------------------------------------------------------------------
# The Builder
# ---------------------------------------------------------------------------
def _build_component(  # noqa: C901, PLR0912, PLR0915
    fn: Any,
    ports: tuple[str, ...],
    states: tuple[str, ...],
    *,
    uses_time: bool,
    amplitude_param: str = "",
    setup_fn: Any = None,
    differentiable_params: "tuple[str, ...] | None" = None,
    port_aliases: "dict[str, str | tuple[str, ...]] | None" = None,
    holomorphic: bool = False,
) -> type[CircuitComponent]:
    """Compile a physics function into a :class:`CircuitComponent` subclass.

    Inspects ``fn``'s signature, validates it against the declared ``ports``
    and ``states``, performs a dry-run with default values to catch errors
    early, then builds two closures:

    - ``_fast_physics`` — a static function ``(vars_vec, params, t)`` suitable
      for ``jax.vmap`` / ``jax.jacfwd``, used by the solver.
    - ``_invoke_physics`` — a bound method used by the debug ``__call__`` path.

    Args:
        fn: The decorated physics function. New-style signatures begin with
            ``signals`` for :func:`component` or ``(signals, t)`` for
            :func:`source`. The 0.2.3 signatures ``(signals, s)`` and
            ``(signals, s, t)`` remain supported.
        ports: Ordered tuple of port names matching the netlist connections.
        states: Ordered tuple of internal state variable names.
        uses_time: ``True`` when compiling a :func:`source` component whose
            physics function accepts a time argument ``t``.
        amplitude_param: Name of the parameter representing the source
            amplitude for DC homotopy stepping.  Empty string for passives.
        port_aliases: Optional mapping from a canonical port name (must be a
            member of ``ports``) to one or more alternate port names that may
            appear in a netlist instead (e.g. ``{"p1": "P", "p2": "N"}``).
            Consumed by ``compiler.py``'s ``_resolve_port_index`` so callers
            can wire up a netlist using either name without duplicating the
            physics.

    Returns:
        A new :class:`CircuitComponent` subclass named after ``fn``.

    Raises:
        TypeError: If the function signature does not start with the required
            reserved arguments, if any parameter lacks a default value, if a
            non-source component declares a ``t`` parameter, or if the dry-run
            raises an exception.
        ValueError: If ``port_aliases`` references a canonical name that is
            not in ``ports``.

    """
    overlap = set(ports) & set(states)
    if overlap:
        names = ", ".join(sorted(overlap))
        msg = f"{fn.__name__}: ports and states must have distinct names; duplicated: {names}"
        raise ValueError(msg)

    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    legacy_states = len(params) > 1 and params[1].name == "s"
    if uses_time:
        reserved = ("signals", "s", "t") if legacy_states else ("signals", "t")
    else:
        reserved = ("signals", "s") if legacy_states else ("signals",)

    if len(params) < len(reserved):
        msg = f"Function '{fn.__name__}' must start with arguments {reserved}"
        raise TypeError(msg)
    for i, expected in enumerate(reserved):
        if params[i].name != expected:
            msg = f"Arg #{i + 1} must be '{expected}'"
            raise TypeError(msg)

    param_specs = params[len(reserved) :]

    # Optional analog-init slot. When the physics function declares ``init``
    # as the first non-reserved positional argument, the framework will
    # inject the return value of a ``@<Component>.setup``-registered
    # function there at evaluation time. The ``init`` arg is NOT a regular
    # JAX-traced parameter — it is computed from the other params each
    # call, allowing gradients to flow through the setup body and matching
    # Verilog-A's ``analog initial`` semantic.
    has_init_arg = bool(param_specs) and param_specs[0].name == "init"
    if has_init_arg:
        param_specs = param_specs[1:]

    if not uses_time:
        for p in param_specs:
            if p.name == "t":
                msg = "Use @source for time-dependent components."
                raise TypeError(msg)

    for p in param_specs:
        if p.default is inspect.Parameter.empty:
            msg = f"Parameter '{p.name}' must have a default."
            raise TypeError(msg)

    n_ports = len(ports)
    full_keys = ports + states
    _dummy_signals = Signals([0.0] * len(full_keys), full_keys)
    _dummy_legacy_signals = Signals([0.0] * len(ports), ports)
    _dummy_states = States([0.0] * len(states), states)
    _defaults = {p.name: p.default for p in param_specs}

    # Dry-run validates ordinary arguments. ``init`` is registered only after
    # class construction, so setup-backed components receive a placeholder.
    _dry_positional = [_dummy_legacy_signals, _dummy_states] if legacy_states else [_dummy_signals]
    if uses_time:
        _dry_positional.append(0.0)
    if has_init_arg:
        _dry_positional.append({})
    try:
        fn(*_dry_positional, **_defaults)
    except Exception as exc:
        # Bodies that index into init with concrete keys will fail the
        # placeholder dry-run; that's expected and harmless. Suppress only
        # KeyError/IndexError/AttributeError, which are the typical "init
        # was the empty placeholder" failure modes — anything else is a
        # real signature bug.
        if not (has_init_arg and isinstance(exc, (KeyError, IndexError, AttributeError, TypeError))):
            raise TypeError(f"Dry-run failed: {exc}") from exc

    _param_names = tuple(p.name for p in param_specs)
    _user_fn = fn

    # Mutable cell for the analog-init function — populated by the
    # ``@<Component>.setup`` classmethod after class construction. The
    # closure below reads from this list so re-registration works without
    # rebuilding ``_fast_physics``.
    _setup_cell: list[Any] = [None]

    def _resolve_init(kw: dict[str, Any]) -> Any:
        """Run the registered setup fn with the current params, returning init.

        Called from inside ``_fast_physics`` / ``_invoke_physics`` when the
        physics signature declares an ``init`` argument. The setup body is
        re-traced each call — XLA constant-folds it for static params,
        and gradients flow through it for differentiable ones.
        """
        setup_fn = _setup_cell[0]
        if setup_fn is None:
            msg = (
                f"{fn.__name__}: physics function declares an 'init' argument "
                f"but no setup function has been registered. "
                f"Use ``@{fn.__name__}.setup`` to register one."
            )
            raise RuntimeError(msg)
        # Pass through only kwargs the setup function declares — ergonomic
        # for setups that consume a subset of the physics params.  When the
        # setup wrapper uses **kwargs (e.g. emitted _register_setup), pass
        # everything so the inner function receives the actual param values.
        setup_sig = inspect.signature(setup_fn)
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in setup_sig.parameters.values())
        if has_var_kw:
            return setup_fn(**kw)
        accepts = {p.name for p in setup_sig.parameters.values()}
        sub_kw = {k: v for k, v in kw.items() if k in accepts}
        return setup_fn(**sub_kw)

    def _build_positional(signals: Signals, t: float, init_value: Any) -> list[Any]:
        if legacy_states:
            delayed = signals._delayed  # noqa: SLF001
            positional = [
                Signals(signals._values[:n_ports], ports, delayed=delayed[:n_ports]),  # noqa: SLF001
                States(signals._values[n_ports:], states, delayed=delayed[n_ports:]),  # noqa: SLF001
            ]
        else:
            positional = [signals]
        if uses_time:
            positional.append(t)
        if has_init_arg:
            positional.append(init_value)
        return positional

    if len(full_keys) == 0:
        _fast_physics = lambda v, p, t, hist=None: (jnp.zeros(0), jnp.zeros(0))  # noqa: E731
    else:

        def _fast_physics(
            vars_vec: jax.Array,
            params: Any,
            t: float,
            hist: jax.Array | None = None,
        ) -> tuple[jax.Array, jax.Array]:
            kw = {name: _extract_param(params, name) for name in _param_names}
            init_value = _resolve_init(kw) if has_init_arg else None
            signals = Signals(vars_vec, full_keys, delayed=vars_vec if hist is None else hist)
            positional = _build_positional(signals, t, init_value)
            f_dict, q_dict = _user_fn(*positional, **kw)
            f_vals = [f_dict.get(k, 0.0) for k in full_keys]
            q_vals = [q_dict.get(k, 0.0) for k in full_keys]
            return jnp.array(f_vals), jnp.array(q_vals)

    if uses_time:

        def _invoke_physics(
            self: CircuitComponent,
            signals: Any,
            t: float,
            params: Any,
        ) -> tuple[dict, dict]:
            kw = {name: _extract_param(params, name) for name in _param_names}
            init_value = _resolve_init(kw) if has_init_arg else None
            positional = _build_positional(signals, t, init_value)
            return _user_fn(*positional, **kw)
    else:

        def _invoke_physics(
            self: CircuitComponent,
            signals: Any,
            t: float,
            params: Any,
        ) -> tuple[dict, dict]:
            kw = {name: _extract_param(params, name) for name in _param_names}
            init_value = _resolve_init(kw) if has_init_arg else None
            positional = _build_positional(signals, t, init_value)
            return _user_fn(*positional, **kw)

    annotations = {p.name: (p.annotation if p.annotation is not inspect.Parameter.empty else Any) for p in param_specs}

    # Distinguish JAX-traced params (numeric scalars / arrays) from static
    # config fields (strings, None, bools). Static fields are carried on the
    # Equinox module but not stacked across instances or traced through
    # transformations — useful for absorbing GDSFactory-style "config"
    # settings like ``cross_section="strip"`` or ``width=None``.
    #
    # ``differentiable_params`` (from va_component) overrides the default
    # staticness for numeric params:
    #   - ``()``      → all numeric params are static (fastest; no jax.grad)
    #   - ``None``    → all numeric params are dynamic JAX leaves (full grad)
    #   - ``("W",)``  → only named params are dynamic; rest are static
    def _is_static(p: Any) -> bool:
        if p.default is None or isinstance(p.default, (str, bool)):
            return True
        if differentiable_params is None:
            return False
        return p.name not in differentiable_params

    _static_names: set[str] = set()
    defaults: dict[str, Any] = {}
    for p in param_specs:
        if _is_static(p):
            defaults[p.name] = eqx.field(default=p.default, static=True)
            _static_names.add(p.name)
        else:
            defaults[p.name] = p.default
    _diff_param_names_tuple = tuple(p.name for p in param_specs if p.name not in _static_names)

    def _register_setup(cls_inner: type, setup_fn: Any) -> type:
        """Register an analog-init / setup function on the component class.

        Used as ``@<Component>.setup`` decorator. The registered function
        will be called inside the JAX trace each time the component is
        evaluated, with the same parameters the physics function receives
        (filtered to those the setup function declares). Its return value
        is injected as the ``init`` argument of the physics function —
        the physics function MUST declare ``init`` as the first
        non-reserved positional argument for ``.setup`` to work.

        Returns ``cls_inner`` for decorator chaining.

        Raises:
            TypeError: physics function did not declare ``init`` arg.
            RuntimeError: ``.setup`` was already registered on this class.

        """
        if not has_init_arg:
            msg = (
                f"@{fn.__name__}.setup requires the physics function to declare "
                f"an 'init' argument as the first non-reserved positional "
                f"parameter. Add ``init`` to the signature, e.g. "
                f"``def {fn.__name__}(signals, init, R=1.0): ...``"
            )
            raise TypeError(msg)
        if _setup_cell[0] is not None:
            msg = (
                f"@{fn.__name__}.setup is already registered. Re-register "
                f"is intentionally rejected to catch typos; if you really "
                f"want to replace it, set ``cls._setup_fn_ref = None`` first."
            )
            raise RuntimeError(msg)
        if not callable(setup_fn):
            msg = f"@{fn.__name__}.setup expects a callable; got {type(setup_fn).__name__}"
            raise TypeError(msg)
        _setup_cell[0] = setup_fn
        cls_inner._setup_fn_ref = staticmethod(setup_fn)
        return cls_inner

    def _delay_values(params: Any) -> tuple[Any, ...]:
        """Evaluate physics with a recorder to discover inline delay reads."""
        kw = {name: _extract_param(params, name) for name in _param_names}
        init_value = _resolve_init(kw) if has_init_arg else None
        values = jnp.zeros(len(full_keys))
        recorded: list[Any] = []
        signals = Signals(values, full_keys, delayed=values, delay_recorder=recorded)
        if legacy_states:
            positional_signals = Signals(values[:n_ports], ports, delayed=values[:n_ports], delay_recorder=recorded)
            positional_states = States(values[n_ports:], states, delayed=values[n_ports:], delay_recorder=recorded)
            positional = [positional_signals, positional_states]
            if uses_time:
                positional.append(0.0)
            if has_init_arg:
                positional.append(init_value)
        else:
            positional = _build_positional(signals, 0.0, init_value)
        _user_fn(*positional, **kw)
        return tuple(recorded)

    def _tau_of(params: Any) -> Any:
        """Infer one instance's delay from its ``signals.at_delay`` call."""
        delays = _delay_values(params)
        if not delays:
            msg = f"{fn.__name__}: no signals.at_delay(...) call was found"
            raise RuntimeError(msg)
        first = delays[0]
        if len(delays) > 1:
            distinct = jnp.any(jnp.stack([delay != first for delay in delays[1:]]))
            first = eqxi.error_if(
                first,
                distinct,
                f"{fn.__name__}: multiple distinct delays are not supported; reuse one signals.at_delay(...) snapshot.",
            )
        return first

    namespace = {
        "__annotations__": annotations,
        "ports": ports,
        "states": states,
        "_fast_physics": staticmethod(_fast_physics),
        "_invoke_physics": _invoke_physics,
        "_uses_time": uses_time,
        "_holomorphic": holomorphic,
        "amplitude_param": amplitude_param,
        "_has_init_arg": has_init_arg,
        "_setup_fn_ref": None,
        "setup": classmethod(_register_setup),
        "delay_values": staticmethod(_delay_values),
        "tau_of": staticmethod(_tau_of),
        # Expose static/diff param name splits for _install_custom_jvp.
        "_static_param_names": tuple(sorted(_static_names)),
        "_diff_param_names": _diff_param_names_tuple,
        **defaults,
    }

    cls = type(fn.__name__, (CircuitComponent,), namespace)
    cls.__doc__ = fn.__doc__

    if port_aliases:
        normalized_aliases: dict[str, tuple[str, ...]] = {}
        for canonical, raw in port_aliases.items():
            if canonical not in ports:
                msg = f"{fn.__name__}: port_aliases key {canonical!r} is not in ports {ports}"
                raise ValueError(msg)
            normalized_aliases[canonical] = (raw,) if isinstance(raw, str) else tuple(raw)
        cls._sanitized_to_raw_ports = normalized_aliases

    return cls


def component(
    ports: tuple[str, ...] = (),
    states: tuple[str, ...] = (),
    amplitude_param: str = "",
    port_aliases: "dict[str, str | tuple[str, ...]] | None" = None,
    holomorphic: bool = False,
) -> Any:
    """Decorator for defining a time-independent circuit component.

    Compiles the decorated physics function into a :class:`CircuitComponent`
    subclass. The function must begin with ``signals`` followed by parameters
    with defaults, which become JAX-traceable Equinox fields on the resulting
    class. Ports and internal states share the ``signals`` namespace.

    Args:
        ports: Ordered tuple of port names. Must match the connection keys
            used in the netlist.
        states: Ordered tuple of internal state variable names. State
            variables are appended to the solver's state vector after the
            node voltages.
        amplitude_param: Name of the parameter that represents the source
            amplitude (e.g. ``"I"`` for a current source). When non-empty,
            DC homotopy solvers will scale this parameter during stepping.
            Leave empty (default) for passive components.
        port_aliases: Optional mapping from a canonical port name to one or
            more alternate names a netlist may use instead, e.g.
            ``{"p1": "P", "p2": "N"}``. Lets integrators (e.g. Mosaic/kfnetlist)
            reuse this component's physics under their own port-naming
            convention without redefining it.
        holomorphic: If ``True``, the component's physics function is
            holomorphic (complex-differentiable), enabling the faster N×N
            Wirtinger AC solve.  Defaults to ``False`` (safe: the full 2N×2N
            real-block system is always correct).  Set to ``True`` for
            components whose physics uses only holomorphic operations — if
            *every* component in a circuit is holomorphic, the AC sweep
            uses the faster N×N path automatically.

    Returns:
        A decorator that accepts a physics function and returns a
        :class:`CircuitComponent` subclass.

    Example::

        @component(ports=("p1", "p2"))
        def Resistor(signals: Signals, R: float = 1.0):
            i = (signals.p1 - signals.p2) / R
            return {"p1": i, "p2": -i}, {}

    """
    return lambda fn: _build_component(
        fn, ports, states, uses_time=False, amplitude_param=amplitude_param, port_aliases=port_aliases, holomorphic=holomorphic
    )


def source(
    ports: tuple[str, ...] = (),
    states: tuple[str, ...] = (),
    amplitude_param: str = "",
    port_aliases: "dict[str, str | tuple[str, ...]] | None" = None,
    holomorphic: bool = False,
) -> Any:
    """Decorator for defining a time-dependent circuit component.

    Identical to :func:`component` except the decorated physics function
    must accept ``t`` as its second argument (after ``signals``),
    and may use it to implement time-varying behaviour such as sinusoidal
    sources or delayed step functions.

    Args:
        ports: Ordered tuple of port names.
        states: Ordered tuple of internal state variable names.
        amplitude_param: Name of the parameter that represents the source
            amplitude (e.g. ``"V"`` for a voltage source). When non-empty,
            DC homotopy solvers will scale this parameter during stepping.
            Leave empty (default) for time-dependent components that are not
            primary excitation sources.
        port_aliases: Optional mapping from a canonical port name to one or
            more alternate names a netlist may use instead. See
            :func:`component`.
        holomorphic: If ``True``, the component's physics is holomorphic.
            Defaults to ``False`` (safe).  See :func:`component`.

    Returns:
        A decorator that accepts a physics function and returns a
        :class:`CircuitComponent` subclass.

    Example::

        @source(ports=("p1", "p2"), states=("i_src",))
        def VoltageSource(signals: Signals, t: float, V: float = 1.0):
            constraint = (signals.p1 - signals.p2) - V
            return {"p1": signals.i_src, "p2": -signals.i_src, "i_src": constraint}, {}

    """
    return lambda fn: _build_component(
        fn, ports, states, uses_time=True, amplitude_param=amplitude_param, port_aliases=port_aliases, holomorphic=holomorphic
    )
