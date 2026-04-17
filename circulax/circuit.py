"""High-level callable circuit interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from circulax.utils import apply_global_params

if TYPE_CHECKING:
    from circulax.solvers.linear import CircuitLinearSolver


class Circuit:
    """A compiled, callable circuit with SAX-like parameter API.

    Wraps the output of :func:`compile_netlist` and :func:`analyze_circuit`
    into a single callable. Global parameters (e.g. ``wavelength_nm``) are
    forwarded to every component group that declares them. When array-valued
    parameters are passed, :meth:`__call__` applies ``jax.vmap`` over the
    batch dimension automatically.

    Attributes:
        solver: The configured linear solver strategy.
        groups: Compiled component groups dict.
        sys_size: Number of scalar unknowns in the real-valued system.
        port_map: Maps ``"Instance,port"`` strings to indices in the flat
            solution vector ``y``.

    """

    def __init__(  # noqa: D107
        self,
        solver: CircuitLinearSolver,
        groups: dict,
        sys_size: int,
        port_map: dict[str, int],
    ) -> None:
        self.solver = solver
        self.groups = groups
        self.sys_size = sys_size
        self.port_map = port_map

    def _n(self) -> int:
        return self.sys_size * (2 if self.solver.is_complex else 1)

    def __call__(self, y_guess: jax.Array | None = None, **params: Any) -> jax.Array:
        """Solve the circuit for the given global parameters.

        Scalar params produce a single solve returning shape ``(n,)``.
        Array params trigger ``jax.vmap`` over the leading dimension,
        returning shape ``(batch, n)``. All array params must share the
        same leading dimension size.

        Args:
            y_guess: Initial guess for the Newton solver. Defaults to zeros.
            **params: Global parameter values to forward to matching component
                groups, e.g. ``wavelength_nm=1310.0`` or
                ``wavelength_nm=jnp.linspace(1260, 1360, 1000)``.

        Returns:
            Flat solution vector of shape ``(n,)`` or ``(batch, n)``.

        Raises:
            ValueError: If multiple array params have different leading dims.

        """
        arrays = {k: jnp.asarray(v) for k, v in params.items()}
        batch_keys = [k for k, v in arrays.items() if v.ndim > 0]

        if y_guess is None:
            y_guess = jnp.zeros(self._n())

        if not batch_keys:
            return self.solver.solve_dc(apply_global_params(self.groups, arrays), y_guess)

        batch_sizes = {k: arrays[k].shape[0] for k in batch_keys}
        if len(set(batch_sizes.values())) > 1:
            msg = f"All batched params must share the same leading dim. Got: {batch_sizes}"
            raise ValueError(msg)

        scalar_params = {k: v for k, v in arrays.items() if k not in batch_keys}

        def solve_single(*batch_vals: jax.Array) -> jax.Array:
            kw = dict(zip(batch_keys, batch_vals, strict=True))
            kw.update(scalar_params)
            return self.solver.solve_dc(apply_global_params(self.groups, kw), y_guess)

        return jax.vmap(solve_single)(*[arrays[k] for k in batch_keys])

    def get_port_field(self, y: jax.Array, port: str) -> jax.Array:
        """Extract the (possibly complex) field at a named port.

        Args:
            y: Solution array of shape ``(n,)`` or ``(batch, n)``.
            port: Port key as ``"InstanceName,port_name"``.

        Returns:
            Real or complex scalar/array. For complex circuits, reconstructs
            the field from the unrolled block format.

        """
        idx = self.port_map[port]
        if self.solver.is_complex:
            return y[..., idx] + 1j * y[..., idx + self.sys_size]
        return y[..., idx]

    def with_groups(self, groups: dict) -> Circuit:
        """Return a new Circuit with replaced component groups.

        Use together with :func:`~circulax.utils.update_params_dict` for
        instance-specific parameter changes before solving.

        Args:
            groups: New groups dict.

        Returns:
            A new :class:`Circuit` with the updated groups.

        """
        return Circuit(self.solver, groups, self.sys_size, self.port_map)


def compile_circuit(
    net_dict: dict,
    models_map: dict,
    *,
    backend: str = "default",
    is_complex: bool = False,
    g_leak: float = 1e-9,
) -> Circuit:
    """Compile a netlist into a callable :class:`Circuit`.

    Convenience wrapper that runs :func:`~circulax.compiler.compile_netlist`
    and :func:`~circulax.solvers.linear.analyze_circuit` in one call.

    Args:
        net_dict: SAX-format netlist dict with ``instances``, ``connections``,
            and optionally ``ports`` keys.
        models_map: Mapping from component type name strings to
            :class:`~circulax.components.base_component.CircuitComponent`
            subclasses.
        backend: Linear solver backend. One of ``"default"``, ``"dense"``,
            ``"sparse"``, ``"klu"``, ``"klu_split"``. Defaults to
            ``"default"`` (``klu_split``).
        is_complex: If ``True``, treat the circuit as complex-valued (photonic).
            The solution vector will have length ``2 * sys_size``.
        g_leak: Leakage conductance for regularisation. Defaults to ``1e-9``.

    Returns:
        A :class:`Circuit` ready to call with ``circuit(**params)``.

    Example::

        circuit = compile_circuit(net_dict, models_map, is_complex=True)
        solutions = jax.jit(circuit)(wavelength_nm=jnp.linspace(1260, 1360, 2000))
        field_out = circuit.get_port_field(solutions, "Detector,p1")

    """
    from circulax.compiler import compile_netlist
    from circulax.solvers.linear import analyze_circuit

    groups, sys_size, port_map = compile_netlist(net_dict, models_map)
    solver = analyze_circuit(groups, sys_size, backend=backend, is_complex=is_complex, g_leak=g_leak)
    return Circuit(solver=solver, groups=groups, sys_size=sys_size, port_map=port_map)
