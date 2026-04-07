"""Utilities for converting between S-parameters.

Utilities for converting between S-parameter and admittance representations,
and for wrapping SAX model functions as circulax components.
"""

import inspect
from typing import Any

import jax
import jax.numpy as jnp
from sax import get_ports, sdense

from circulax.components.base_component import CircuitComponent, Signals, States, _extract_param, component


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


def sax_component(fn: callable) -> callable:
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

    Args:
        fn: A SAX model function whose keyword arguments are scalar
            parameters and whose return value is a SAX S-parameter dict.
            All parameters must have defaults, or will be substituted with
            ``1.0`` during the dry run.

    Returns:
        A :class:`~circulax.components.base_component.CircuitComponent`
        subclass named after ``fn``.

    Raises:
        RuntimeError: If the dry run fails for any reason.

    """
    sig = inspect.signature(fn)
    defaults = {
        param.name: param.default if param.default is not inspect.Parameter.empty else 1.0 for param in sig.parameters.values()
    }

    try:
        dummy_s_dict = fn(**defaults)
        detected_ports = get_ports(dummy_s_dict)
    except Exception as exc:
        msg = f"Failed to dry-run SAX component '{fn.__name__}': {exc}"
        raise RuntimeError(msg) from exc

    def physics_wrapper(signals: Signals, s: States, **kwargs) -> tuple[dict, dict]:  # noqa: ANN003
        s_dict = fn(**kwargs)
        s_matrix, _ = sdense(s_dict)
        y_matrix = s_to_y(s_matrix)
        v_vec = jnp.array([getattr(signals, p) for p in detected_ports], dtype=jnp.complex128)
        i_vec = y_matrix @ v_vec
        return {p: i_vec[i] for i, p in enumerate(detected_ports)}, {}

    physics_wrapper.__name__ = fn.__name__
    physics_wrapper.__doc__ = fn.__doc__
    physics_wrapper.__signature__ = sig

    return component(ports=detected_ports)(physics_wrapper)


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
