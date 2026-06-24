"""High-level callable circuit interface."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import kfnetlist as kfnl

from circulax.utils import apply_global_params, update_params_dict

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

    def __init__(
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

    def _coerce_param_updates(
        self,
        params: dict[str, Any] | None,
        param_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        updates = dict(params or {})
        updates.update(param_kwargs)
        return updates

    def _with_param_values(self, params: dict[str, Any]) -> dict:
        updated = self.groups
        for name, value in params.items():
            if "." not in name:
                updated = apply_global_params(updated, {name: value})
                continue

            instance_name, param_key = name.split(".", 1)
            for group_name, group in updated.items():
                index_map = getattr(group, "index_map", None)
                if index_map is None or instance_name not in index_map:
                    continue
                if not hasattr(group.params, param_key):
                    msg = f"Instance '{instance_name}' has no parameter '{param_key}'."
                    raise ValueError(msg)
                updated = update_params_dict(updated, group_name, instance_name, param_key, value)
                break
            else:
                msg = f"Instance '{instance_name}' not found in compiled circuit."
                raise ValueError(msg)
        return updated

    def _as_arrays(self, params: dict[str, Any]) -> dict[str, jax.Array]:
        return {k: jnp.asarray(v) for k, v in params.items()}

    def _require_scalar_params(self, params: dict[str, Any], method: str) -> dict[str, jax.Array]:
        arrays = self._as_arrays(params)
        batched = {k: v.shape for k, v in arrays.items() if v.ndim > 0}
        if batched:
            msg = f"Circuit.{method}() does not batch parameter arrays directly. Got batched params: {batched}"
            raise ValueError(msg)
        return arrays

    def _zero_guess(self) -> jax.Array:
        return jnp.zeros(self._n())

    def _resolve_port_node(self, port: str) -> int:
        if port not in self.port_map:
            msg = f"Unknown circuit port '{port}'. Available ports include: {sorted(self.port_map)[:10]}"
            raise KeyError(msg)
        return self.port_map[port]

    def dc(
        self,
        y_guess: jax.Array | None = None,
        *,
        params: dict[str, Any] | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        max_steps: int | None = None,
        **param_updates: Any,
    ) -> jax.Array:
        """Solve the DC operating point for the given parameters.

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
            params: Optional mapping of parameter updates. Keys without a dot
                are broadcast to every component group declaring that parameter.
                Keys like ``"R1.R"`` update one instance.
            **param_updates: Convenience form for global parameter values, e.g.
                ``wavelength_nm=1310.0`` or
                ``wavelength_nm=jnp.linspace(1260, 1360, 1000)``.

        Returns:
            Flat solution vector of shape ``(n,)`` or ``(batch, n)``.

        Raises:
            ValueError: If multiple array params have different leading dims.

        """
        rtol = self.rtol if rtol is None else rtol
        atol = self.atol if atol is None else atol
        max_steps = self.max_steps if max_steps is None else max_steps

        updates = self._coerce_param_updates(params, param_updates)
        arrays = self._as_arrays(updates)
        batch_keys = [k for k, v in arrays.items() if v.ndim > 0]

        if y_guess is None:
            y_guess = self._zero_guess()

        if not batch_keys:
            return self.solver.solve_dc(self._with_param_values(arrays), y_guess, rtol=rtol, atol=atol, max_steps=max_steps)

        batch_sizes = {k: arrays[k].shape[0] for k in batch_keys}
        if len(set(batch_sizes.values())) > 1:
            msg = f"All batched params must share the same leading dim. Got: {batch_sizes}"
            raise ValueError(msg)

        scalar_params = {k: v for k, v in arrays.items() if k not in batch_keys}

        def solve_single(*batch_vals: jax.Array) -> jax.Array:
            kw = dict(zip(batch_keys, batch_vals, strict=True))
            kw.update(scalar_params)
            return self.solver.solve_dc(self._with_param_values(kw), y_guess, rtol=rtol, atol=atol, max_steps=max_steps)

        return jax.vmap(solve_single)(*[arrays[k] for k in batch_keys])

    def __call__(
        self,
        y_guess: jax.Array | None = None,
        *,
        params: dict[str, Any] | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        max_steps: int | None = None,
        **param_updates: Any,
    ) -> jax.Array:
        """Backward-compatible alias for :meth:`dc`."""
        return self.dc(
            y_guess,
            params=params,
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
            **param_updates,
        )

    def transient(
        self,
        *,
        t0: float,
        t1: float,
        dt0: float,
        y0: jax.Array | None = None,
        saveat: Any = None,
        params: dict[str, Any] | None = None,
        transient_solver: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Run transient (time-domain) analysis.

        Args:
            t0: Start time.
            t1: End time.
            dt0: Initial time step.
            y0: Initial state vector. If ``None``, a DC solve is run first.
            saveat: Times at which to save the solution. Accepts an array of
                timestamps or a ``diffrax.SaveAt`` object.
            params: Parameter updates (same format as :meth:`dc`).
                Array-valued params are **not** supported — raises ``ValueError``.
            transient_solver: Override the Diffrax solver (default BDF2).
            **kwargs: Forwarded to ``setup_transient`` / Diffrax (e.g.
                ``max_steps``).

        Returns:
            A ``diffrax.Solution`` with ``.ts`` and ``.ys`` attributes.

        """
        from diffrax import SaveAt

        from circulax.solvers import setup_transient

        param_updates = kwargs.pop("param_updates", {})
        updates = self._coerce_param_updates(params, param_updates)
        arrays = self._require_scalar_params(updates, "transient")
        groups = self._with_param_values(arrays)
        if y0 is None:
            y0 = self.solver.solve_dc(
                groups,
                self._zero_guess(),
                rtol=self.rtol,
                atol=self.atol,
                max_steps=self.max_steps,
            )
        saveat_obj = SaveAt(ts=saveat) if saveat is not None and not isinstance(saveat, SaveAt) else saveat
        run_transient = setup_transient(groups=groups, linear_strategy=self.solver, transient_solver=transient_solver)
        return run_transient(t0=t0, t1=t1, dt0=dt0, y0=y0, saveat=saveat_obj, **kwargs)

    def ac(
        self,
        *,
        ports: str | Sequence[str],
        freqs: jax.Array,
        z0: float = 50.0,
        y_dc: jax.Array | None = None,
        params: dict[str, Any] | None = None,
        **param_updates: Any,
    ) -> jax.Array:
        """Run an AC small-signal S-parameter sweep.

        Linearises at the DC operating point and sweeps over ``freqs``,
        returning S-parameters for the given ports.

        Args:
            ports: Port name(s) to probe (e.g. ``"out"`` or ``["in", "out"]``).
            freqs: Frequency array in Hz.
            z0: Reference impedance (default 50 Ohm).
            y_dc: DC operating point. If ``None``, a DC solve is run first.
            params: Parameter updates (same format as :meth:`dc`).
                Array-valued params are **not** supported.
            **param_updates: Global parameter overrides.

        Returns:
            Complex S-parameter array of shape ``(len(freqs), n_ports, n_ports)``.

        Raises:
            ValueError: If the circuit is complex-valued (photonic). Use the
                low-level ``setup_ac_sweep`` API for complex circuits.

        """
        from circulax.solvers import setup_ac_sweep

        if self.solver.is_complex:
            msg = "Circuit.ac() currently supports real-valued circuits. Use the low-level AC API for custom paths."
            raise ValueError(msg)

        updates = self._coerce_param_updates(params, param_updates)
        arrays = self._require_scalar_params(updates, "ac")
        groups = self._with_param_values(arrays)
        if y_dc is None:
            y_dc = self.solver.solve_dc(
                groups,
                self._zero_guess(),
                rtol=self.rtol,
                atol=self.atol,
                max_steps=self.max_steps,
            )
        port_list = [ports] if isinstance(ports, str) else list(ports)
        port_nodes = [self._resolve_port_node(port) for port in port_list]
        run_ac = setup_ac_sweep(groups, self.sys_size, port_nodes, z0=z0)
        return run_ac(y_dc, jnp.asarray(freqs))

    def hb(
        self,
        *,
        freq: float,
        harmonics: int = 5,
        y0: jax.Array | None = None,
        params: dict[str, Any] | None = None,
        y_flat_init: jax.Array | None = None,
        rtol: float | None = None,
        atol: float | None = None,
        max_steps: int | None = None,
        osc_node: int | str | None = None,
        amplitude_tries: jax.Array | None = None,
        g_leak: float | None = None,
        **param_updates: Any,
    ) -> tuple[jax.Array, jax.Array]:
        """Run harmonic balance to find the periodic steady state.

        Args:
            freq: Fundamental frequency in Hz.
            harmonics: Number of harmonics (K = 2*harmonics + 1 time samples).
            y0: DC operating point. If ``None``, a DC solve is run first.
            params: Parameter updates (same format as :meth:`dc`).
                Array-valued params are **not** supported.
            y_flat_init: Flat initial waveform, shape ``(K * sys_size,)``.
                Overrides the automatic multi-start when ``osc_node`` is set.
            rtol: Relative tolerance override.
            atol: Absolute tolerance override.
            max_steps: Max Newton iterations override.
            osc_node: Port name or state index for oscillator multi-start.
            amplitude_tries: Amplitudes to try when ``osc_node`` is set.
            g_leak: Override the solver's regularisation conductance.
            **param_updates: Global parameter overrides.

        Returns:
            ``(y_time, y_freq)`` — time samples ``(K, n)`` and normalised
            Fourier coefficients ``(harmonics+1, n)`` complex.

        """
        from circulax.solvers import setup_harmonic_balance

        updates = self._coerce_param_updates(params, param_updates)
        arrays = self._require_scalar_params(updates, "hb")
        groups = self._with_param_values(arrays)
        if y0 is None:
            y0 = self.solver.solve_dc(
                groups,
                self._zero_guess(),
                rtol=self.rtol,
                atol=self.atol,
                max_steps=self.max_steps,
            )
        osc_idx = self._resolve_port_node(osc_node) if isinstance(osc_node, str) else osc_node
        run_hb = setup_harmonic_balance(
            groups,
            self.sys_size,
            freq=freq,
            num_harmonics=harmonics,
            is_complex=self.solver.is_complex,
            g_leak=getattr(self.solver, "g_leak", 1e-9) if g_leak is None else g_leak,
            osc_node=osc_idx,
            amplitude_tries=amplitude_tries,
        )
        rtol = self.rtol if rtol is None else rtol
        atol = self.atol if atol is None else atol
        max_steps = self.max_steps if max_steps is None else max_steps
        return run_hb(y0, y_flat_init=y_flat_init, max_steps=max_steps, rtol=rtol, atol=atol)

    def get_port_field(self, y: jax.Array, port: str) -> jax.Array:
        """Extract the (possibly complex) field at a named port.

        Args:
            y: Solution array of shape ``(n,)`` or ``(batch, n)``.
            port: Port key as ``"InstanceName,port_name"``.

        Returns:
            Real or complex scalar/array. For complex circuits, reconstructs
            the field from the unrolled block format.

        """
        idx = self._resolve_port_node(port)
        if self.solver.is_complex:
            return y[..., idx] + 1j * y[..., idx + self.sys_size]
        return y[..., idx]

    def port(self, y: jax.Array, port: str) -> jax.Array:
        """Alias for :meth:`get_port_field`."""
        return self.get_port_field(y, port)

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
            self.solver,
            groups,
            self.sys_size,
            self.port_map,
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
        )


def compile_circuit(
    net_dict: dict | kfnl.Netlist,
    models_map: dict,
    *,
    backend: str = "default",
    is_complex: bool | str = "auto",
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
            If ``"auto"`` (default), infer this from component outputs.
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
    if is_complex == "auto":
        is_complex = _infer_is_complex(groups)
    elif not isinstance(is_complex, bool):
        msg = "is_complex must be True, False, or 'auto'."
        raise ValueError(msg)
    solver = analyze_circuit(groups, sys_size, backend=backend, is_complex=is_complex, g_leak=g_leak)
    return Circuit(
        solver=solver,
        groups=groups,
        sys_size=sys_size,
        port_map=port_map,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
    )


def _infer_is_complex(groups: dict) -> bool:
    """Best-effort complex-mode inference from compiled component groups."""
    for group in groups.values():
        if _group_outputs_complex(group):
            return True
    return False


def _group_outputs_complex(group: Any) -> bool:
    try:
        count = group.var_indices.shape[0]
        y0 = jnp.zeros(group.var_indices.shape[1])
        params0 = jax.tree.map(
            lambda x: x[0] if hasattr(x, "shape") and x.shape[:1] == (count,) else x,
            group.params,
        )
        f_vals, q_vals = group.physics_func(0.0, y0, params0)
    except (AttributeError, TypeError, ValueError, IndexError, KeyError) as exc:
        warnings.warn(
            f"Complex-mode auto-detection failed for group ({exc!r}); "
            f"defaulting to real. Pass is_complex=True to compile_circuit() "
            f"if this is a photonic/complex-valued circuit.",
            stacklevel=2,
        )
        return False
    return bool(jnp.iscomplexobj(f_vals) or jnp.iscomplexobj(q_vals))
