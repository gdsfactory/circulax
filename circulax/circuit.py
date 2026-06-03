"""High-level callable circuit interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import kfnetlist as kfnl

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
        rtol: float = 1e-6,
        atol: float = 1e-6,
        max_steps: int = 100,
    ) -> None:
        self.solver = solver
        self.groups = groups
        self.sys_size = sys_size
        self.port_map = port_map
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps

    def _n(self) -> int:
        return self.sys_size * (2 if self.solver.is_complex else 1)

    def __call__(
        self,
        y_guess: jax.Array | None = None,
        *,
        rtol: float | None = None,
        atol: float | None = None,
        max_steps: int | None = None,
        **params: Any,
    ) -> jax.Array:
        """Solve the circuit for the given global parameters.

        Scalar params produce a single solve returning shape ``(n,)``.
        Array params trigger ``jax.vmap`` over the leading dimension,
        returning shape ``(batch, n)``. All array params must share the
        same leading dimension size.

        Args:
            y_guess: Initial guess for the Newton solver. Defaults to zeros.
            rtol: Relative tolerance override. Defaults to value from
                :func:`compile_circuit` (``1e-6``).
            atol: Absolute tolerance override. Defaults to value from
                :func:`compile_circuit` (``1e-6``).
            max_steps: Max Newton iterations override. Defaults to value
                from :func:`compile_circuit` (``100``).
            **params: Global parameter values to forward to matching component
                groups, e.g. ``wavelength_nm=1310.0`` or
                ``wavelength_nm=jnp.linspace(1260, 1360, 1000)``.

        Returns:
            Flat solution vector of shape ``(n,)`` or ``(batch, n)``.

        Raises:
            ValueError: If multiple array params have different leading dims.

        """
        rtol = self.rtol if rtol is None else rtol
        atol = self.atol if atol is None else atol
        max_steps = self.max_steps if max_steps is None else max_steps

        arrays = {k: jnp.asarray(v) for k, v in params.items()}
        batch_keys = [k for k, v in arrays.items() if v.ndim > 0]

        if y_guess is None:
            y_guess = jnp.zeros(self._n())

        if not batch_keys:
            return self.solver.solve_dc(
                apply_global_params(self.groups, arrays), y_guess, rtol=rtol, atol=atol, max_steps=max_steps
            )

        batch_sizes = {k: arrays[k].shape[0] for k in batch_keys}
        if len(set(batch_sizes.values())) > 1:
            msg = f"All batched params must share the same leading dim. Got: {batch_sizes}"
            raise ValueError(msg)

        scalar_params = {k: v for k, v in arrays.items() if k not in batch_keys}

        def solve_single(*batch_vals: jax.Array) -> jax.Array:
            kw = dict(zip(batch_keys, batch_vals, strict=True))
            kw.update(scalar_params)
            return self.solver.solve_dc(
                apply_global_params(self.groups, kw), y_guess, rtol=rtol, atol=atol, max_steps=max_steps
            )

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
        return Circuit(
            self.solver, groups, self.sys_size, self.port_map,
            rtol=self.rtol, atol=self.atol, max_steps=self.max_steps,
        )


def compile_circuit(
    net_dict: dict | kfnl.Netlist,
    models_map: dict,
    *,
    backend: str = "default",
    is_complex: bool = False,
    g_leak: float = 1e-9,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    max_steps: int = 100,
) -> Circuit:
    """Compile a netlist into a callable :class:`Circuit`.

    Accepts either a ``kfnetlist.Netlist`` or a SAX-format dict.

    Args:
        net_dict: Netlist (kfnetlist.Netlist or SAX-format dict).
        models_map: Mapping from component type name strings to component classes.
        backend: Linear solver backend (``"default"``, ``"dense"``, ``"klu"`` etc.).
        is_complex: If ``True``, treat the circuit as complex-valued (photonic).
        g_leak: Leakage conductance for regularisation.
        rtol: Relative tolerance for the Newton solver.
        atol: Absolute tolerance for the Newton solver.
        max_steps: Max Newton iterations.

    Returns:
        A :class:`Circuit` ready to call with ``circuit(**params)``.

    """
    from circulax.compiler import compile_netlist
    from circulax.solvers.linear import analyze_circuit

    groups, sys_size, port_map = compile_netlist(net_dict, models_map)
    solver = analyze_circuit(groups, sys_size, backend=backend, is_complex=is_complex, g_leak=g_leak)
    return Circuit(
        solver=solver, groups=groups, sys_size=sys_size, port_map=port_map,
        rtol=rtol, atol=atol, max_steps=max_steps,
    )
