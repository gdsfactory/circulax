"""High-level callable circuit interface."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import kfnetlist as kfnl

from circulax.sax_dispatch import SaxDispatch, build_sax_circuit, check_sax_dispatch
from circulax.utils import apply_global_params, update_params_dict

if TYPE_CHECKING:
    from circulax.solvers.linear import CircuitLinearSolver


def _resolve_osdi_param_col(group: Any, param_key: str) -> int:
    """Resolve a parameter name to its column index in an OSDI group's params array."""
    try:
        from bosdi.osdi_registry import get_model  # type: ignore[import]

        model = get_model(group.model_id)
        name_to_col = {n.lower(): i for i, n in enumerate(model.param_names)}
    except (ImportError, Exception) as exc:
        msg = f"Cannot resolve OSDI parameter '{param_key}': bosdi registry lookup failed ({exc!r})."
        raise ValueError(msg) from exc
    col = name_to_col.get(param_key.lower())
    if col is None:
        msg = f"Parameter '{param_key}' not found in OSDI model (available: {sorted(name_to_col)})."
        raise ValueError(msg)
    return col


def _update_osdi_param(
    groups_dict: dict,
    group_name: str,
    group: Any,
    instance_name: str,
    param_key: str,
    new_value: float,
) -> dict:
    """Update a single parameter for one OSDI instance and return new groups dict."""
    col = _resolve_osdi_param_col(group, param_key)
    instance_idx = group.index_map[instance_name]
    new_params = group.params.at[instance_idx, col].set(new_value)
    new_group = group.with_params(new_params)
    return {**groups_dict, group_name: new_group}


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
        _source_netlist: dict | None = None,
        _source_models: dict | None = None,
    ) -> None:
        self.solver = solver
        self.groups = groups
        self.sys_size = sys_size
        self.port_map = port_map
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps
        self._source_netlist = _source_netlist
        self._source_models = _source_models
        # Both are resolved on first use by `sax_dispatch` / `_sax_evaluator`:
        # detection inspects every instance, and building the SAX circuit is
        # only worth doing for callers that actually ask for S-parameters.
        self._sax_verdict: SaxDispatch | None = None
        self._sax_circuit: Any = None

    @property
    def ports(self) -> tuple[str, ...]:
        """External port names declared in the source netlist."""
        if self._source_netlist is None:
            return ()
        return tuple(self._source_netlist.get("ports", {}).keys())

    @property
    def source_netlist(self) -> dict | None:
        """The original netlist used to compile this circuit, if available."""
        return self._source_netlist

    @property
    def source_models(self) -> dict | None:
        """The leaf models used to compile this circuit, if available."""
        return self._source_models

    @property
    def sax_dispatch(self) -> SaxDispatch:
        """Whether :meth:`sdict` can shortcut this circuit through SAX.

        Truthy when every instantiated model is a SAX model; otherwise it
        carries the blocking reasons in ``.reasons``. See
        :func:`circulax.sax_dispatch.check_sax_dispatch`.
        """
        if self._sax_verdict is None:
            self._sax_verdict = check_sax_dispatch(self._source_netlist, self._source_models)
        return self._sax_verdict

    def _sax_evaluator(self) -> Any:
        """Return this circuit's SAX evaluator, building it once on first use."""
        if self._sax_circuit is None:
            self._sax_circuit = build_sax_circuit(self._source_netlist, self._source_models)
        return self._sax_circuit

    def _sdict_via_nodal(self, params: dict[str, Any]) -> dict[tuple[str, str], jax.Array]:
        """Compute the circuit S-matrix from the nodal system.

        The fallback path, used when the circuit is not SAX-dispatchable. The
        S-parameters come from the existing small-signal sweep at ``z0=1`` —
        the normalisation ``sax_component`` stamps its admittances with —
        evaluated at ``f = 0``: a scattering network carries no electrical
        storage, so its S-matrix does not depend on the small-signal
        frequency. Optical dispersion enters through model parameters (``wl``)
        instead, which is why this returns the same matrix SAX composes.
        """
        if self._source_netlist is None:
            msg = "Circuit has no retained source netlist, so its external ports are unknown."
            raise ValueError(msg)
        ports = self._source_netlist.get("ports") or {}
        if not ports:
            msg = "Circuit declares no external 'ports'; there is nothing to build an S-matrix over."
            raise ValueError(msg)

        port_names = list(ports)
        s = self.sp(
            ports=[ports[name] for name in port_names],
            freqs=jnp.zeros(1),
            z0=1.0,
            params=params,
        )[0]
        return {(pi, pj): s[i, j] for i, pi in enumerate(port_names) for j, pj in enumerate(port_names)}

    def sdict(
        self,
        *,
        params: dict[str, Any] | None = None,
        backend: str = "auto",
        **param_updates: Any,
    ) -> dict[tuple[str, str], jax.Array]:
        """Return the circuit's S-parameters as a SAX S-dict.

        One entry point for both kinds of circuit. When every model in the
        netlist is a SAX model, the circuit is linear and memoryless and its
        S-matrix is composed directly by SAX — far cheaper than assembling and
        solving the nodal system for the same answer, and with no XLA compile
        per call shape. Anything else (a nonlinear device, a source, a ground)
        takes the nodal path. Both return the same S-dict, so callers need not
        know which ran; :attr:`sax_dispatch` reports which one would.

        Args:
            params: Parameter updates (same format as :meth:`dc`).
            backend: ``"auto"`` (default) uses SAX when the circuit qualifies
                and the nodal solver otherwise. ``"sax"`` forces the SAX path
                and raises if it does not qualify. ``"nodal"`` forces the nodal
                path, which is how the two are cross-checked.
            **param_updates: Global parameter overrides, e.g. ``wl=1.55``.
                The SAX path broadcasts over array-valued parameters; the
                nodal path requires scalars.

        Returns:
            A SAX S-dict mapping ``(port_out, port_in)`` pairs to complex
            amplitudes, keyed by the netlist's external port names.

        Raises:
            ValueError: If *backend* is not one of the three accepted values,
                or if ``"sax"`` was forced on a circuit that does not qualify.

        """
        if backend not in ("auto", "sax", "nodal"):
            msg = f"backend must be 'auto', 'sax' or 'nodal'. Got {backend!r}."
            raise ValueError(msg)

        updates = self._coerce_param_updates(params, param_updates)
        if backend == "nodal" or (backend == "auto" and not self.sax_dispatch):
            return self._sdict_via_nodal(updates)
        return self._sax_evaluator()(**updates)

    def smatrix(
        self,
        *,
        params: dict[str, Any] | None = None,
        backend: str = "auto",
        **param_updates: Any,
    ) -> tuple[jax.Array, tuple[str, ...]]:
        """Return the circuit's S-parameters as a dense matrix plus its port order.

        Dense form of :meth:`sdict`, taking the same arguments.

        Returns:
            ``(s_matrix, port_order)`` — the matrix indexed ``[..., out, in]``
            and the port names giving its row/column order.

        """
        from sax import sdense

        s_dict = self.sdict(params=params, backend=backend, **param_updates)
        s_matrix, port_map = sdense(s_dict)
        return s_matrix, tuple(sorted(port_map, key=port_map.get))

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
                if hasattr(group, "model_id"):
                    updated = _update_osdi_param(updated, group_name, group, instance_name, param_key, value)
                else:
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

    def sp(
        self,
        *,
        ports: str | Sequence[str],
        freqs: jax.Array,
        z0: float | jax.Array = 50.0,
        y_dc: jax.Array | None = None,
        holomorphic: bool | str = "auto",
        params: dict[str, Any] | None = None,
        **param_updates: Any,
    ) -> jax.Array:
        """Run a small-signal S-parameter sweep.

        Linearises at the DC operating point and sweeps over ``freqs``,
        returning S-parameters for the given ports.

        Args:
            ports: Port name(s) to probe (e.g. ``"out"`` or ``["in", "out"]``).
            freqs: Frequency array in Hz.
            z0: Reference impedance in ohms.  Scalar (uniform) or array of
                shape ``(N_ports,)`` (per-port).  Default 50 Ohm.  For
                frequency-dependent impedance, solve at a fixed z0 and use
                :func:`~circulax.solvers.renormalize` afterwards.
            y_dc: DC operating point. If ``None``, a DC solve is run first.
            holomorphic: If ``"auto"`` (default), infer from component flags —
                if any component in the circuit has ``holomorphic=False``, the
                full 2N×2N real-block system is used.  Pass ``True`` to force
                the N×N Wirtinger system, or ``False`` to force the 2N×2N
                system.
            params: Parameter updates (same format as :meth:`dc`).
                Array-valued params are **not** supported.
            **param_updates: Global parameter overrides.

        Returns:
            Complex S-parameter array of shape ``(len(freqs), n_ports, n_ports)``.

        """
        from circulax.solvers import setup_ac_sweep

        updates = self._coerce_param_updates(params, param_updates)
        arrays = self._require_scalar_params(updates, "sp")
        groups = self._with_param_values(arrays)
        if holomorphic == "auto":
            holomorphic = _infer_holomorphic(groups)
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
        run_ac = setup_ac_sweep(
            groups, self.sys_size, port_nodes, z0=z0, is_complex=self.solver.is_complex, holomorphic=holomorphic
        )
        return run_ac(y_dc, jnp.asarray(freqs))

    def ac(
        self,
        *,
        ports: str | Sequence[str],
        freqs: jax.Array,
        z0: float | jax.Array = 50.0,
        y_dc: jax.Array | None = None,
        holomorphic: bool | str = "auto",
        params: dict[str, Any] | None = None,
        **param_updates: Any,
    ) -> jax.Array:
        """Run a small-signal S-parameter sweep.

        .. deprecated:: 0.2
            Use :meth:`sp` instead. Will be removed in 0.3.

        """
        warnings.warn("Circuit.ac() is deprecated, use Circuit.sp() instead.", DeprecationWarning, stacklevel=2)
        return self.sp(
            ports=ports, freqs=freqs, z0=z0, y_dc=y_dc, holomorphic=holomorphic, params=params, **param_updates
        )

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


def _embed_circuit_subcircuits(
    net_dict: dict | kfnl.Netlist,
    models_map: dict,
    circuit_models: dict[str, Circuit],
) -> dict:
    """Build a RecursiveNetlist from Circuit objects in *models_map* (mutates *models_map*)."""
    from circulax.netlist import _is_recursive_netlist

    if isinstance(net_dict, kfnl.Netlist):
        net_dict = net_dict.to_dict()
    recnet: dict[str, dict] = {}
    if isinstance(net_dict, dict) and _is_recursive_netlist(net_dict):
        recnet.update(net_dict)
    else:
        recnet["top"] = net_dict  # type: ignore[assignment]
    for name, circ in circuit_models.items():
        if circ.source_netlist is None:
            msg = f"Circuit '{name}' has no stored source netlist and cannot be used as a subcircuit."
            raise ValueError(msg)
        recnet[name] = circ.source_netlist
        for mk, mv in (circ.source_models or {}).items():
            existing = models_map.get(mk)
            if existing is not None and existing is not mv:
                msg = f"Model name conflict: '{mk}' maps to different objects in parent and subcircuit '{name}'."
                raise ValueError(msg)
            models_map[mk] = mv
        del models_map[name]
    return recnet


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
    params_map: dict[str, dict[str, str]] | None = None,
) -> Circuit:
    """Compile a netlist into a callable :class:`Circuit`.

    Accepts a ``kfnetlist.Netlist``, a SAX-format dict, or a
    ``RecursiveNetlist`` (``dict[str, Netlist]``).  When a recursive netlist
    is given, subcircuit instances are flattened before compilation.

    A compiled :class:`Circuit` may also appear as a value in *models_map*;
    its stored source netlist is inlined as a subcircuit automatically.

    Args:
        net_dict: Netlist (kfnetlist.Netlist, SAX-format dict, or
            RecursiveNetlist).
        models_map: Mapping from component type name strings to component
            classes, SAX model functions, or compiled :class:`Circuit`
            objects.
        backend: Linear solver backend (``"default"``, ``"dense"``, ``"klu"`` etc.).
        is_complex: If ``True``, treat the circuit as complex-valued (photonic).
            If ``"auto"`` (default), infer this from component outputs.
        g_leak: Leakage conductance for regularisation.
        rtol: Relative tolerance for the Newton solver.
        atol: Absolute tolerance for the Newton solver.
        max_steps: Max Newton iterations.
        params_map: Optional mapping from component type names to dicts that
            rename netlist setting keys to model field names, e.g.
            ``{"thermal_heater": {"length": "length_um"}}``.

    Returns:
        A :class:`Circuit` ready to call with ``circuit(**params)``.

    """
    from circulax.compiler import compile_netlist
    from circulax.netlist import _is_recursive_netlist, flatten_recursive_netlist
    from circulax.solvers.linear import analyze_circuit

    models_map = dict(models_map)
    source_netlist: dict | None = None
    source_models: dict | None = None

    circuit_models = {k: v for k, v in models_map.items() if isinstance(v, Circuit)}
    if circuit_models:
        net_dict = _embed_circuit_subcircuits(net_dict, models_map, circuit_models)

    if isinstance(net_dict, dict) and _is_recursive_netlist(net_dict):
        source_netlist = net_dict.get(next(iter(net_dict)))
        source_models = {k: v for k, v in models_map.items() if not isinstance(v, Circuit)}
        net_dict = flatten_recursive_netlist(net_dict)
    elif isinstance(net_dict, dict):
        source_netlist = net_dict
        source_models = {k: v for k, v in models_map.items() if not isinstance(v, Circuit)}

    groups, sys_size, port_map = compile_netlist(net_dict, models_map, params_map=params_map)
    if is_complex == "auto":
        is_complex = _infer_is_complex(groups)
    elif not isinstance(is_complex, bool):
        msg = "is_complex must be True, False, or 'auto'."
        raise ValueError(msg)
    if is_complex:
        _validate_holomorphic_flags(groups)
    solver = analyze_circuit(groups, sys_size, backend=backend, is_complex=is_complex, g_leak=g_leak)
    return Circuit(
        solver=solver,
        groups=groups,
        sys_size=sys_size,
        port_map=port_map,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
        _source_netlist=source_netlist,
        _source_models=source_models,
    )


def _infer_holomorphic(groups: dict) -> bool:
    """Return ``True`` if all component groups are holomorphic."""
    return all(getattr(group, "holomorphic", True) for group in groups.values())


def _validate_holomorphic_flags(groups: dict) -> None:
    """Warn about components marked holomorphic=True that use non-holomorphic ops.

    Only called for complex-valued circuits where holomorphicity matters.
    Traces each group's physics with complex inputs and inspects the jaxpr.
    """
    from circulax.components.base_component import _jaxpr_has_non_holomorphic

    for name, group in groups.items():
        if not getattr(group, "holomorphic", True):
            continue
        if getattr(group, "is_fdomain", False):
            continue
        try:
            n_vars = group.var_indices.shape[1]
            count = group.var_indices.shape[0]
            vars_vec = jnp.zeros(n_vars, dtype=jnp.complex128)
            params0 = jax.tree.map(
                lambda x: x[0] if hasattr(x, "shape") and x.shape[:1] == (count,) else x,
                group.params,
            )
            closed_jaxpr = jax.make_jaxpr(group.physics_func, static_argnums=(0,))(0.0, vars_vec, params0)
            bad_ops = _jaxpr_has_non_holomorphic(closed_jaxpr.jaxpr)
            if bad_ops:
                ops_str = ", ".join(sorted(bad_ops))
                warnings.warn(
                    f"Component group '{name}' is marked holomorphic=True but "
                    f"uses non-holomorphic operations: {ops_str}. Set "
                    f"holomorphic=False on its @component/@source decorator.",
                    stacklevel=3,
                )
        except Exception:
            pass


def _infer_is_complex(groups: dict) -> bool:
    """Best-effort complex-mode inference from compiled component groups."""
    for group in groups.values():
        if _group_outputs_complex(group):
            return True
    return False


def _group_outputs_complex(group: Any) -> bool:
    if getattr(group, "is_fdomain", False):
        return False
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
