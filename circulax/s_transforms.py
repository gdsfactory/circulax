"""Utilities for converting between S-parameters.

Utilities for converting between S-parameter and admittance representations,
and for wrapping SAX model functions as circulax components.
"""

import functools
import inspect
from typing import Any

import jax
import jax.numpy as jnp
import sax
from sax import get_ports, sdense
from sax.saxtypes import try_into

from circulax.components.base_component import CircuitComponent, Signals, _extract_param, component
from circulax.power_waves import power_wave_operator, power_wave_stamp, validate_reference


def _unwrap(fn: callable) -> callable:
    """Return the innermost wrapped callable of a ``functools.partial`` chain."""
    while isinstance(fn, functools.partial):
        fn = fn.func
    return fn


def _sanitize_port(name: str) -> str:
    """Coerce a SAX port label into a valid Python identifier.

    ``base_component`` builds a namedtuple from the port tuple, so port names
    must be valid identifiers. SAX PDKs occasionally use numeric labels
    (e.g. ``'1'``, ``'2'``); those are prefixed with ``'p'``. Other
    non-identifier characters are replaced with ``'_'``. Already-valid names
    pass through unchanged.
    """
    s = str(name)
    if s.isidentifier():
        return s
    if s and s[0].isdigit():
        s = "p" + s
    s = "".join(c if c.isalnum() or c == "_" else "_" for c in s)
    if not s.isidentifier():
        s = "p_" + s
    return s


@jax.jit
def s_to_y(S: jax.Array, z0: complex = 1.0 + 1e-12j) -> jax.Array:
    """Convert an S-parameter matrix to an admittance (Y) matrix.

    Kurokawa power-wave form: ``Y = (I - S) (z0 S + z0* I)^-1``. Reduces to
    ``(1/z0) (I - S) (I + S)^-1`` for real ``z0``. A small ``Im(z0)`` keeps
    the inverse well-conditioned when ``S`` has eigenvalues at -1 (e.g. an
    ideal lossless symmetric splitter).
    """
    n = S.shape[-1]
    eye = jnp.eye(n, dtype=jnp.complex128)
    Sc = S.astype(jnp.complex128)
    z0c = jnp.asarray(z0, dtype=jnp.complex128)
    if n == 2:
        # Two-port models can export to StableHLO/IREE without a host LAPACK
        # custom call. Preserve the Kurokawa normalization, including complex z0.
        a, b = z0c * Sc[..., 0, 0] + jnp.conj(z0c), z0c * Sc[..., 0, 1]
        c, d = z0c * Sc[..., 1, 0], z0c * Sc[..., 1, 1] + jnp.conj(z0c)
        det = a * d - b * c
        n00, n01 = 1 - Sc[..., 0, 0], -Sc[..., 0, 1]
        n10, n11 = -Sc[..., 1, 0], 1 - Sc[..., 1, 1]
        return jnp.stack((
            jnp.stack(((n00 * d - n01 * c) / det, (-n00 * b + n01 * a) / det), axis=-1),
            jnp.stack(((n10 * d - n11 * c) / det, (-n10 * b + n11 * a) / det), axis=-1),
        ), axis=-2)
    M = z0c * Sc + jnp.conj(z0c) * eye
    return jnp.linalg.solve(M.swapaxes(-1, -2), (eye - Sc).swapaxes(-1, -2)).swapaxes(-1, -2)


@jax.jit
def y_to_s(Y: jax.Array, z0: complex = 1.0 + 1e-12j) -> jax.Array:
    """Convert an admittance (Y) matrix to an S-parameter matrix.

    Exact inverse of :func:`s_to_y`: ``S = (I - z0 Y) (I + z0* Y)^-1``.
    Reduces to ``(I - z0 Y) (I + z0 Y)^-1`` for real ``z0``. ``(I + z0* Y)``
    and ``(I - z0 Y)`` are both polynomials in ``Y`` and so commute, making
    this well-defined regardless of multiplication order.
    """
    n = Y.shape[-1]
    eye = jnp.eye(n, dtype=jnp.complex128)
    Yc = Y.astype(jnp.complex128)
    z0c = jnp.asarray(z0, dtype=jnp.complex128)
    M = eye + jnp.conj(z0c) * Yc
    return jnp.linalg.solve(M.swapaxes(-1, -2), (eye - z0c * Yc).swapaxes(-1, -2)).swapaxes(-1, -2)


def fdomain_model(fn: callable, *, ports: tuple[str, ...], z0: complex = 1.0 + 1e-12j) -> callable:
    """Adapt a frequency-domain admittance function into a SAX-shaped S-parameter model.

    Bridges :func:`fdomain_component`-style physics (``fn(f, **params) ->
    Y-matrix``) into an ordinary SAX model function (``(**params) ->
    sax.SDict``) via :func:`y_to_s`. The result is meant to be passed to
    :func:`sax_component`, e.g.::

        @sax_component
        def my_rf_model(f=1e9, R0=50.0): ...  # returns a Y-matrix


        RfComp = sax_component(fdomain_model(my_rf_model, ports=("p1", "p2")))

    Once wrapped this way, the resulting component's frequency argument
    becomes an ordinary settable parameter (like ``wl`` on a photonic SAX
    model) rather than the circuit-level AC-sweep frequency: it is evaluated
    once per compile/param-update, not re-evaluated per point in a
    :meth:`~circulax.circuit.Circuit.sp` sweep. This makes it usable in the
    all-SAX, source-free :meth:`~circulax.circuit.Circuit.dc` S-parameter
    path (where the frequency becomes a batchable parameter, e.g.
    ``circuit.dc(f=jnp.linspace(...))``), but it is **not** a drop-in
    replacement for ``@fdomain_component`` inside a circuit that also has
    reactive elements (capacitors/inductors) or sources: there, the
    component's own frequency dependence is frozen at whatever value its
    parameter was last set to, decoupled from the ``freqs`` sweep argument
    driving the rest of the circuit. To explicitly restore live sweep
    semantics, wrap it as ``sax_component(fdomain_model(...),
    frequency_param="f")``; that creates a frequency-domain Circulax
    component and is consequently no longer eligible for SAX-circuit export.

    Note:
        Compiling the result with ``is_complex=True`` may emit a spurious
        "uses non-holomorphic operations: conj" warning from
        :func:`~circulax.circuit.compile_circuit`. This is a known false
        positive: ``s_to_y``/``y_to_s`` are individually ``jax.jit``-ed, so
        their default ``z0`` argument is traced as an abstract value inside
        that nested call, which trips the checker's literal-vs-traced
        heuristic. It does not indicate an actual bug — ``fn`` never
        receives port signals (only ``f`` and parameters), so its
        contribution is provably linear in the port voltages regardless.

    Args:
        fn: Admittance function with signature ``fn(f, **params) ->
            Y-matrix``, matching :func:`fdomain_component`'s contract (``f``
            first, every other parameter with a default).
        ports: Ordered tuple of port names, matching the shape of ``fn``'s
            returned Y-matrix.
        z0: Reference impedance used for the Y-to-S conversion. Defaults to
            SAX's own convention so the wrapped model composes correctly
            with other SAX models under :func:`sax_component`.

    Returns:
        A callable with the same signature as ``fn`` that returns a
        ``sax.SDict`` instead of a Y-matrix.

    Raises:
        TypeError: If ``fn``'s signature doesn't start with ``f`` or any
            other parameter lacks a default.

    """
    sig = inspect.signature(fn)
    params_list = list(sig.parameters.values())
    if not params_list or params_list[0].name != "f":
        msg = f"fdomain_model function '{fn.__name__}' must have 'f' as its first argument."
        raise TypeError(msg)
    for p in params_list[1:]:
        if p.default is inspect.Parameter.empty:
            msg = f"Parameter '{p.name}' in '{fn.__name__}' must have a default value."
            raise TypeError(msg)

    f_name = params_list[0].name

    def sax_model(*args, **kwargs) -> dict:  # noqa: ANN002, ANN003
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        f = arguments.pop(f_name)
        Y = fn(f, **arguments)
        S = y_to_s(Y, z0=z0)
        return {(a, b): S[j, i] for i, a in enumerate(ports) for j, b in enumerate(ports)}

    sax_model.__name__ = fn.__name__
    sax_model.__doc__ = fn.__doc__
    sax_model.__signature__ = sig
    return sax_model


def sax_component(
    fn: callable,
    *,
    name: str | None = None,
    frequency_param: str | None = None,
    r0: float = 1.0,
) -> callable:
    """Decorator to convert a SAX model function into a circulax component.

    Discover ports with a dry run, then retain one incident root-power-wave
    state per port. For ``b = S @ a``, terminal flows are
    ``(a - b) / sqrt(r0)`` and algebraic constraints are
    ``a + b - potential / sqrt(r0) = 0``. Both waves have units sqrt(W).
    No S-to-Y inversion or default-parameter conditioning decision is used.
    The dry run discovers the interface only; the same equations apply over
    the complete parameter range, including exactly lossless devices.
    This is also the default for automatically imported SAX models. No
    external matching or zero reflection at default parameters is assumed;
    every port retains its incident unknown even for reflectionless models.

    ``fn`` may be a plain function or a :class:`functools.partial` wrapping
    one (SAX PDKs typically use partials to bind fab-specific defaults).
    Partials are unwrapped to the innermost callable for ``__name__`` /
    ``__doc__`` recovery; :func:`inspect.signature` handles the parameter
    reduction.

    Args:
        fn: A SAX model function whose keyword arguments are scalar
            parameters and whose return value is a SAX S-parameter dict.
            All parameters must have defaults, or will be substituted with
            ``1.0`` during the dry run.
        name: Optional override for the resulting class name. Useful when
            wrapping :class:`functools.partial` objects where several
            partials share the same underlying ``__name__`` — e.g.
            ``{key: sax_component(val, name=key) for key, val in pdk.items()}``.
        frequency_param: Optional SAX parameter that is driven by
            :meth:`~circulax.circuit.Circuit.sp`'s frequency at every sweep
            point. This creates a Circulax frequency-domain component rather
            than a normal static SAX wrapper, so it cannot be exported via
            :meth:`~circulax.circuit.Circuit.to_sax_circuit`. Use this for an
            explicit round trip such as
            ``sax_component(fdomain_model(y_fn, ports=("p1", "p2")),
            frequency_param="f")``. Frequency-dependent models retain the
            same incident-wave states and constraints as static models.
        r0: Common finite positive real reference normalization. Defaults to
            1.0. This sets potential/flow coordinates; it does not renormalize
            the supplied S-matrix. Connected optical models and boundaries
            must use compatible coordinates.

    Returns:
        A :class:`~circulax.components.base_component.CircuitComponent`
        subclass named after ``name`` (if given) else after the unwrapped
        function.

    Raises:
        RuntimeError: If the dry run fails for any reason.

    """
    r0 = validate_reference(r0)
    if frequency_param is not None:
        return _build_frequency_sax_component(fn, frequency_param=frequency_param, name=name, r0=r0)

    sig = inspect.signature(fn)
    base_fn = _unwrap(fn)
    cls_name = name if name is not None else getattr(base_fn, "__name__", "SaxComponent")
    defaults = {
        param.name: param.default if param.default is not inspect.Parameter.empty else 1.0 for param in sig.parameters.values()
    }

    try:
        dummy_s_dict = fn(**defaults)
        detected_ports = get_ports(dummy_s_dict)
    except Exception as exc:
        msg = f"Failed to dry-run SAX component '{cls_name}': {exc}"
        raise RuntimeError(msg) from exc

    # Component variables use attribute access, which requires every port
    # name to be a valid Python identifier. Some SAX PDKs label
    # ports numerically ('1', '2'); coerce those to identifiers while keeping
    # the index ordering.
    raw_to_sanitized = {str(p): _sanitize_port(p) for p in detected_ports}
    sanitized_to_raw: dict[str, list[str]] = {}
    for raw, sanitized in raw_to_sanitized.items():
        sanitized_to_raw.setdefault(sanitized, []).append(raw)
    port_names = tuple(raw_to_sanitized[str(p)] for p in detected_ports)
    aux_state_names = tuple(f"wave_{p}" for p in port_names)

    def physics_wrapper(signals: Signals, **kwargs) -> tuple[dict, dict]:  # noqa: ANN003
        s_dict = fn(**kwargs)
        # `sdense` returns the S-matrix *and* a port_map keyed in dict-insertion
        # order (dict[raw_port, matrix_row_index]). `get_ports` returns ports
        # sorted alphabetically — the two can differ (e.g. an MMI2x2 SDict
        # built {(o1,o1):…,(o3,o3):…,(o4,o4):…,(o2,o2):…}). We must use the
        # port_map to line up v_vec with the rows/cols of the matrix; using the
        # sorted order silently scrambles coupling between non-adjacent rows.
        s_matrix, port_map = sdense(s_dict)
        matrix_order = sorted(port_map, key=port_map.get)  # raw names, matrix-row order
        sanitized_in_order = [_sanitize_port(p) for p in matrix_order]
        return power_wave_stamp(signals, tuple(sanitized_in_order), s_matrix, r0=r0)

    physics_wrapper.__name__ = cls_name
    physics_wrapper.__doc__ = getattr(base_fn, "__doc__", None)

    # Synthesise a signature that base_component._build_component can consume:
    # it must begin with the reserved ``signals`` arg and expose every SAX
    # parameter as a keyword-only entry with a default. The wrapper's runtime
    # body still accepts them via **kwargs.
    _sax_params = [
        inspect.Parameter(
            p.name,
            inspect.Parameter.KEYWORD_ONLY,
            default=defaults[p.name],
            annotation=p.annotation if p.annotation is not inspect.Parameter.empty else inspect.Parameter.empty,
        )
        for p in sig.parameters.values()
    ]
    physics_wrapper.__signature__ = inspect.Signature(
        parameters=[
            inspect.Parameter("signals", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            *_sax_params,
        ]
    )

    cls = component(ports=port_names, states=aux_state_names, holomorphic=True)(physics_wrapper)
    cls._is_sax_wrapped = True
    cls._wave_r0 = r0
    cls._raw_to_sanitized_ports = raw_to_sanitized
    cls._sanitized_to_raw_ports = {sanitized: tuple(raws) for sanitized, raws in sanitized_to_raw.items()}
    return cls


def _build_fdomain_component(
    fn: callable,
    ports: tuple[str, ...],
    *,
    states: tuple[str, ...] = (),
) -> type[CircuitComponent]:
    """Compile a frequency-domain local linear operator into a component class.

    The decorated function must have signature ``fn(f, **params) -> jnp.ndarray`` where
    the returned square operator acts on ``(*ports, *states)``. With no states
    it is the usual port admittance. Retained wave constraints use extra rows
    and columns, assembled by the same DC/AC/HB scatter layout.

    Args:
        fn: Local linear operator. First argument must be ``f`` (frequency in Hz);
            subsequent keyword arguments are scalar parameters with defaults.
        ports: Ordered tuple of port names matching the netlist connections.
        states: Internal algebraic unknowns appended to the port coordinates.

    Returns:
        A new :class:`CircuitComponent` subclass named after ``fn`` with
        ``_is_fdomain = True``.

    Raises:
        TypeError: If the signature is invalid, a parameter lacks a default, or
            the dry-run with ``f=0`` raises an exception.

    """
    sig = inspect.signature(fn)
    params_list = list(sig.parameters.values())

    if not params_list or params_list[0].name != "f":
        msg = f"fdomain_component function '{fn.__name__}' must have 'f' as its first argument."
        raise TypeError(msg)

    param_specs = params_list[1:]  # everything after 'f'
    for p in param_specs:
        if p.default is inspect.Parameter.empty:
            msg = f"Parameter '{p.name}' in '{fn.__name__}' must have a default value."
            raise TypeError(msg)

    _defaults = {p.name: p.default for p in param_specs}
    _param_names = tuple(p.name for p in param_specs)

    try:
        fn(0.0, **_defaults)
    except Exception as exc:
        raise TypeError(f"fdomain_component dry-run failed for '{fn.__name__}': {exc}") from exc

    _user_fn = fn

    def _fast_physics(f: float, args: Any) -> jnp.ndarray:
        """Evaluate the local linear operator at frequency *f*.

        Args:
            f: Frequency in Hz.
            args: Parameter container (Equinox module or dict) for this instance.

        Returns:
            Square operator over ports and states, dtype ``complex128``.

        """
        kw = {name: _extract_param(args, name) for name in _param_names}
        return _user_fn(f, **kw)

    annotations = {p.name: (p.annotation if p.annotation is not inspect.Parameter.empty else Any) for p in param_specs}

    namespace: dict[str, Any] = {
        "__annotations__": annotations,
        "ports": ports,
        "states": states,
        "_is_fdomain": True,
        "_holomorphic": True,
        "_uses_time": False,
        "_fast_physics": staticmethod(_fast_physics),
        **_defaults,
    }

    @classmethod  # type: ignore[misc]
    def solver_call(cls, f: float, args: Any) -> jnp.ndarray:  # noqa: ANN001
        """Evaluate admittance matrix at frequency *f* (solver entry point).

        Args:
            f: Frequency in Hz.
            args: Parameter container for this instance.

        Returns:
            Square operator over ports and states.

        """
        return cls._fast_physics(f, args)

    namespace["solver_call"] = solver_call

    cls = type(fn.__name__, (CircuitComponent,), namespace)
    cls.__doc__ = fn.__doc__
    return cls


def fdomain_component(ports: tuple[str, ...]) -> Any:
    """Decorator for frequency-domain (admittance) circuit components.

    Compiles the decorated admittance function into a
    :class:`~circulax.components.base_component.CircuitComponent` subclass
    that is evaluated in the frequency domain rather than the time domain.
    The component:

    - **DC analysis** — evaluated at ``f = 0`` Hz (e.g. skin-effect reduces to
      ``R₀`` at DC).
    - **Harmonic Balance** — evaluated at each harmonic frequency ``k · f₀``,
      contributing ``Y(k · f₀) @ V_k`` directly to the frequency-domain residual.
    - **Transient simulation** — raises :exc:`RuntimeError` at setup time
      (time-domain convolution not supported).

    The decorated function must accept ``f`` as its first positional argument
    (frequency in Hz) followed by any number of keyword parameters with defaults.
    It must return a square Y-matrix of shape ``(n_ports, n_ports)`` with
    ``dtype=complex128``.

    Complex-mode small-signal analysis evaluates both positive and negative
    frequencies to preserve temporal phase independently of field rotation.
    Real electrical models must obey ``Y(-f) = conj(Y(f))``; a static complex
    optical coupling retains the same complex value at either frequency.

    Args:
        ports: Ordered tuple of port names matching the netlist connection keys.

    Returns:
        A decorator that accepts an admittance function and returns a
        :class:`~circulax.components.base_component.CircuitComponent` subclass.

    Example::

        @fdomain_component(ports=("p1", "p2"))
        def SkinEffectResistor(f: float, R0: float = 1.0, a: float = 0.1):
            \"\"\"Z(f) = R0 + a * sqrt(|f|)\"\"\"
            Z = R0 + a * jnp.sqrt(jnp.abs(f) + 1e-30)
            Y = 1.0 / Z
            return jnp.array([[Y, -Y], [-Y, Y]], dtype=jnp.complex128)

    """
    return lambda fn: _build_fdomain_component(fn, ports)


def _build_frequency_sax_component(
    fn: callable,
    *,
    frequency_param: str,
    name: str | None,
    r0: float,
) -> type[CircuitComponent]:
    """Build the explicit, sweep-frequency-driven form of a SAX model.

    This is intentionally separate from the usual :func:`sax_component`
    path.  A SAX parameter has no intrinsic connection to the Circulax AC
    sweep, so only the caller's ``frequency_param=...`` opt-in permits that
    connection.
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    by_name = {param.name: param for param in params}
    if frequency_param not in by_name:
        msg = f"SAX model '{getattr(fn, '__name__', 'SaxComponent')}' has no parameter '{frequency_param}'."
        raise TypeError(msg)

    base_fn = _unwrap(fn)
    cls_name = name if name is not None else getattr(base_fn, "__name__", "SaxComponent")

    defaults = {
        param.name: param.default if param.default is not inspect.Parameter.empty else 1.0
        for param in params
        if param.name != frequency_param
    }
    try:
        initial_kwargs = {**defaults, frequency_param: 0.0}
        dummy_s_dict = fn(**initial_kwargs)
        detected_ports = get_ports(dummy_s_dict)
    except Exception as exc:
        msg = f"Failed to dry-run frequency-driven SAX component '{cls_name}': {exc}"
        raise RuntimeError(msg) from exc

    port_names = tuple(_sanitize_port(port) for port in detected_ports)
    states = tuple(f"wave_{port}" for port in port_names)

    def sax_operator(f: float, **kwargs: Any) -> jax.Array:
        s_dict = fn(**{**kwargs, frequency_param: f})
        s_matrix, port_map = sdense(s_dict)
        # The solver layout follows discovered ports, while sdense may use
        # dictionary insertion order. Align both operator axes explicitly.
        indices = jnp.array([port_map[port] for port in detected_ports])
        s_matrix = s_matrix[indices[:, None], indices[None, :]]
        return power_wave_operator(s_matrix, r0=r0)

    sax_operator.__name__ = cls_name
    sax_operator.__doc__ = getattr(base_fn, "__doc__", None)
    sax_operator.__signature__ = inspect.Signature(
        parameters=[
            inspect.Parameter("f", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            *[
                inspect.Parameter(
                    param.name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=defaults[param.name],
                    annotation=param.annotation if param.annotation is not inspect.Parameter.empty else inspect.Parameter.empty,
                )
                for param in params
                if param.name != frequency_param
            ],
        ]
    )
    cls = _build_fdomain_component(sax_operator, port_names, states=states)
    cls._is_sax_wrapped = True
    cls._wave_r0 = r0
    cls._raw_to_sanitized_ports = {str(p): _sanitize_port(p) for p in detected_ports}
    cls._sanitized_to_raw_ports = {p: (str(raw),) for p, raw in zip(port_names, detected_ports, strict=True)}
    return cls


# ---------------------------------------------------------------------------
# SAX model auto-detection (for compile_netlist)
# ---------------------------------------------------------------------------

# Annotated SAX return types that a model function may declare. All three
# single-mode S-matrix containers are supported because SAX PDKs use all of
# them, and `sax_component` wraps any of them via :func:`sax.sdense`.
_SAX_RETURN_TYPES: tuple = (sax.SDict, sax.SDense, sax.SCoo, sax.SType)
_SAX_RETURN_NAMES: frozenset = frozenset({"SDict", "SDense", "SCoo", "SType"})


def _is_sax_model(obj: Any) -> bool:
    """Return True if ``obj`` is a plain SAX model function.

    Two-layered check:

    1. **SAX structural validator** — ``sax.saxtypes.try_into[sax.Model](obj)``
       applies SAX's own Pydantic-based `val_model` (callable, no positional-only
       / `*args` / `**kwargs`, every parameter has a default, not a model
       factory). Delegating to SAX means we match its canonical conventions and
       inherit any future tightening automatically.
    2. **Return annotation check** — the raw return annotation must be one of
       :data:`sax.SDict`, :data:`sax.SDense`, :data:`sax.SCoo`, or
       :data:`sax.SType`. SAX's own validator does not enforce this at runtime,
       so we do it here to keep the auto-wrap pathway strict. Both PEP 563
       string annotations and evaluated TypeAlias objects are handled.

    Classes are rejected up-front so :class:`CircuitComponent` subclasses (and
    any user-authored classes) never match and fall through to the direct-use
    branch in :func:`_normalize_model`.
    """
    if inspect.isclass(obj):
        return False
    if try_into[sax.Model](obj) is None:
        return False
    try:
        ann = inspect.signature(obj).return_annotation
    except (TypeError, ValueError):
        return False
    if ann is inspect.Signature.empty:
        return False
    # PEP 563 / `from __future__ import annotations` — compare by last name.
    # JAX's `pjit` wrapper preserves the string form (e.g. ``'sax.SDict'``).
    if isinstance(ann, str):
        return ann.split(".")[-1] in _SAX_RETURN_NAMES
    # Evaluated form: compare identity against the SAX TypeAlias exports.
    # `get_type_hints` resolves aliases to their structural form and breaks
    # identity comparison, so we intentionally skip it here.
    return any(ann is t for t in _SAX_RETURN_TYPES)


def _normalize_model(obj: Any, *, name: str) -> type[CircuitComponent]:
    """Return a :class:`CircuitComponent` class for the given netlist model entry.

    * If ``obj`` is already a :class:`CircuitComponent` subclass, it is returned unchanged.
    * If ``obj`` is a SAX model function (see :func:`_is_sax_model`), it is wrapped
      via :func:`sax_component` and the resulting subclass is returned.
    * Otherwise, a :class:`TypeError` is raised with a clear message.

    Called by :func:`circulax.compiler.compile_netlist` to auto-wrap raw SAX
    model functions so they can be passed directly in the netlist ``models`` dict.
    """
    if inspect.isclass(obj) and issubclass(obj, CircuitComponent):
        return obj
    if _is_sax_model(obj):
        return sax_component(obj, name=name)
    raise TypeError(
        f"Model {name!r} must be a CircuitComponent subclass or a SAX model "
        f"function (callable with all-defaulted parameters and a "
        f"`-> sax.SDict | sax.SDense | sax.SCoo | sax.SType` return annotation). "
        f"Got {type(obj).__name__}: {obj!r}"
    )
