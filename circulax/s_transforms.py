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

from circulax.components.base_component import CircuitComponent, Signals, States, _extract_param, component


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
def s_to_y(S: jax.Array, z0: float = 1.0) -> jax.Array:
    """Convert an S-parameter matrix to an admittance (Y) matrix.

    Uses the formula ``Y = (1/z0) * (I - S) * (I + S)^-1``. Requires dense
    matrix inversion; if a component can be defined directly in terms of a
    Y-matrix it should be, to avoid the overhead of this conversion.

    Args:
        S: S-parameter matrix of shape ``(..., n, n)``.
        z0: Reference impedance in ohms. Defaults to ``1.0``.

    Returns:
        Y-matrix of the same shape and dtype as ``S``.

    """
    n = S.shape[-1]
    eye = jnp.eye(n, dtype=S.dtype)
    return (1.0 / z0) * (eye - S) @ jnp.linalg.inv(eye + S)


def sax_component(fn: callable, *, name: str | None = None) -> callable:
    """Decorator to convert a SAX model function into a circulax component.

    Inspects ``fn`` at decoration time to discover its port interface via a
    dry run, then wraps its S-matrix output in an admittance-based physics
    function compatible with the circulax nodal solver.

    The conversion proceeds in three stages:

    1. **Discovery** — ``fn`` is called once with its default (or dummy)
       parameter values and :func:`sax.get_ports` extracts the sorted port
       names from the resulting S-parameter dict.
    2. **Physics wrapper** — a closure is built that calls ``fn`` at runtime,
       converts the S-dict to a dense matrix via :func:`sax.sdense`, converts
       it to an admittance matrix via :func:`s_to_y`, and returns
       ``I = Y @ V`` as a port current dict.
    3. **Component registration** — the wrapper is passed to
       :func:`~circulax.components.base_component.component` with the
       discovered ports, producing a :class:`~circulax.components.base_component.CircuitComponent`
       subclass.

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

    Returns:
        A :class:`~circulax.components.base_component.CircuitComponent`
        subclass named after ``name`` (if given) else after the unwrapped
        function.

    Raises:
        RuntimeError: If the dry run fails for any reason.

    """
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

    # base_component builds a namedtuple over the port tuple, which requires
    # every port name to be a valid Python identifier. Some SAX PDKs label
    # ports numerically ('1', '2'); coerce those to identifiers while keeping
    # the index ordering.
    port_names = tuple(_sanitize_port(p) for p in detected_ports)

    def physics_wrapper(signals: Signals, s: States, **kwargs) -> tuple[dict, dict]:  # noqa: ANN003
        s_dict = fn(**kwargs)
        s_matrix, _ = sdense(s_dict)
        y_matrix = s_to_y(s_matrix)
        v_vec = jnp.array([getattr(signals, p) for p in port_names], dtype=jnp.complex128)
        i_vec = y_matrix @ v_vec
        return {p: i_vec[i] for i, p in enumerate(port_names)}, {}

    physics_wrapper.__name__ = cls_name
    physics_wrapper.__doc__ = getattr(base_fn, "__doc__", None)

    # Synthesise a signature that base_component._build_component can consume:
    # it must begin with the reserved (signals, s) args and expose every SAX
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
            inspect.Parameter("s", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            *_sax_params,
        ]
    )

    return component(ports=port_names)(physics_wrapper)


def _build_fdomain_component(
    fn: callable,
    ports: tuple[str, ...],
) -> type[CircuitComponent]:
    """Compile a frequency-domain admittance function into a :class:`CircuitComponent` subclass.

    The decorated function must have signature ``fn(f, **params) -> jnp.ndarray`` where
    the return value is a Y-matrix of shape ``(n_ports, n_ports)``.

    Args:
        fn: Admittance function.  First argument must be ``f`` (frequency in Hz);
            subsequent keyword arguments are scalar parameters with defaults.
        ports: Ordered tuple of port names matching the netlist connections.

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
        """Evaluate admittance matrix at frequency *f*.

        Args:
            f: Frequency in Hz.
            args: Parameter container (Equinox module or dict) for this instance.

        Returns:
            Y-matrix of shape ``(n_ports, n_ports)``, dtype ``complex128``.

        """
        kw = {name: _extract_param(args, name) for name in _param_names}
        return _user_fn(f, **kw)

    annotations = {p.name: (p.annotation if p.annotation is not inspect.Parameter.empty else Any) for p in param_specs}

    namespace: dict[str, Any] = {
        "__annotations__": annotations,
        "ports": ports,
        "states": (),
        "_is_fdomain": True,
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
            Y-matrix of shape ``(n_ports, n_ports)``.

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
