"""OSDI device model integration for circulax.

Provides :func:`osdi_component` to load OpenVAF-compiled ``.osdi`` binaries and
use them as circuit components inside :func:`~circulax.compiler.compile_netlist`.

Requires the ``bosdi`` package to be installed or its ``src/`` directory present
on ``PYTHONPATH`` (e.g. via a ``.env`` file).  OSDI support is optional and not
available on all platforms (e.g. Windows).
"""

import difflib

import equinox as eqx
import jax.numpy as jnp

try:
    from osdi_loader import OsdiModel, load_osdi_model
    _BOSDI_AVAILABLE = True
    _BOSDI_ERR = None
except ImportError as _bosdi_err:
    _BOSDI_AVAILABLE = False
    _BOSDI_ERR = _bosdi_err
    OsdiModel = None  # type: ignore[assignment]
    load_osdi_model = None  # type: ignore[assignment]


class OsdiComponentGroup(eqx.Module):
    """Batch of N identical OSDI device instances, evaluated via bosdi.

    Unlike :class:`~circulax.compiler.ComponentGroup`, this group calls
    ``osdi_eval`` with all N instances at once (leveraging Rayon parallelism)
    and uses the analytical Jacobians (conductances/capacitances) that the
    OSDI model returns directly — no ``jax.jacfwd`` required.

    Internal OSDI nodes (e.g. PSP103's ``di``, ``si``) that are not collapsed
    onto a terminal are allocated as extra unknowns in the global state vector,
    exactly like VoltageSource's ``i_src``.  ``num_nodes`` covers all of them;
    ``num_pins`` is the external terminal count only.

    ``var_indices`` has shape ``(N, num_nodes)``: the first ``num_pins`` columns
    index terminal node voltages in the circuit node block; the remaining
    ``num_nodes - num_pins`` columns index the internal-node slots appended at
    the end of the global state vector.  ``eq_indices`` is identical — each
    node's KCL equation lives at the same global index as its voltage unknown.
    """

    name: str = eqx.field(static=True)
    model_id: int = eqx.field(static=True)   # bosdi registry ID — not differentiable
    num_pins: int = eqx.field(static=True)   # external terminals only
    num_nodes: int = eqx.field(static=True)  # terminals + non-collapsed internal nodes
    num_params: int = eqx.field(static=True)
    num_states: int = eqx.field(static=True)

    params: jnp.ndarray   # (N, num_params) float64 — batched device parameters
    states: jnp.ndarray   # (N, num_states) float64 — zeros for stateless models

    var_indices: jnp.ndarray  # (N, num_nodes) int32 — terminal + internal indices into global y
    eq_indices: jnp.ndarray   # (N, num_nodes) int32 — same as var_indices (eq lives at its own y slot)
    jac_rows: jnp.ndarray     # (N*num_nodes*num_nodes,) int32 — COO row indices
    jac_cols: jnp.ndarray     # (N*num_nodes*num_nodes,) int32 — COO col indices

    # Precomputed diagonal regularisation matrix (num_nodes, num_nodes).
    # Diagonal entry i is 1.0 if node i is reactive-only (G[i,:]=0 always, F[i]=0),
    # and 0.0 otherwise.  Added to j_eff in assembly so DC Newton stays non-singular.
    reg_diag: jnp.ndarray  # (num_nodes, num_nodes) float64

    index_map: dict | None = eqx.field(static=True, default=None)
    is_fdomain: bool = eqx.field(static=True, default=False)
    amplitude_param: str = eqx.field(static=True, default="")

    # Experimental: use bosdi.osdi_debug.schur_reduce to eliminate internal
    # nodes from the per-device stamp before handing it to global Newton.
    # See docs/bosdi_psp103_ring_oscillator_issue.md for motivation.  When
    # True, the assembly pads the reduced 4x4 stamp back to num_nodes with
    # identity rows on internal slots so the compiler-allocated internal
    # unknowns stay self-consistent.
    use_schur_reduction: bool = eqx.field(static=True, default=False)

    # Optional bosdi Tier-3 handle (``osdi_jax.OsdiBatchHandle``) pre-baked
    # with this group's params.  Stored as a static Equinox field so JAX
    # tracing treats it as a closure constant (the author's requirement —
    # handle is a Python object, not a JAX array, and its ``handle_id`` is
    # baked into the compiled XLA graph).  When non-None, the assembly
    # uses ``osdi_eval_with_handle`` / ``osdi_residual_eval_with_handle``
    # which skip per-call param upload (~20–40 % faster for PSP103).
    # None falls back to the model_id + params path.
    #
    # For parameter-optimisation loops (same topology, repeated param
    # updates), use :meth:`with_params` to build a new group with refreshed
    # handle and params without re-running ``compile_netlist``.
    handle: object | None = eqx.field(static=True, default=None)

    def with_params(self, new_params: jnp.ndarray) -> "OsdiComponentGroup":
        """Return a copy of this group with updated params and a fresh handle.

        Use this in parameter-optimisation loops — topology is fixed so
        the expensive parts of ``compile_netlist`` (union-find, index
        arrays, Jacobian sparsity) are reused unchanged; only the params
        array and the Tier-3 handle are swapped.  The handle rebuild is
        a ~microsecond C++ call (:func:`osdi_jax.osdi_setup_batch`).

        Args:
            new_params: array of shape ``(N, num_params)`` in the same
                column order as ``self.params``.  Must be a NumPy-coerceable
                value (not a traced JAX array — bosdi's Tier-3 API is a
                one-time C++ call and doesn't participate in tracing).

        Returns:
            A new :class:`OsdiComponentGroup` with ``params`` = ``new_params``
            and ``handle`` rebuilt from ``new_params``.  Every other field
            (var_indices, eq_indices, jac_rows, reg_diag, …) is shared
            with the original group (no copy).

        Example::

            for iter in range(n_opt):
                theta = optimizer.ask()
                groups["nmos"] = groups["nmos"].with_params(theta_to_params(theta))
                sol = run(...)
        """
        import numpy as _np

        new_params = jnp.asarray(new_params, dtype=jnp.float64)
        new_handle = None
        try:
            from osdi_jax import osdi_setup_batch
            new_handle = osdi_setup_batch(self.model_id, _np.asarray(new_params))
        except ImportError:
            pass  # older bosdi without Tier-3; legacy path still works
        return OsdiComponentGroup(
            name=self.name,
            model_id=self.model_id,
            num_pins=self.num_pins,
            num_nodes=self.num_nodes,
            num_params=self.num_params,
            num_states=self.num_states,
            params=new_params,
            states=self.states,
            var_indices=self.var_indices,
            eq_indices=self.eq_indices,
            jac_rows=self.jac_rows,
            jac_cols=self.jac_cols,
            reg_diag=self.reg_diag,
            index_map=self.index_map,
            is_fdomain=self.is_fdomain,
            amplitude_param=self.amplitude_param,
            use_schur_reduction=self.use_schur_reduction,
            handle=new_handle,
        )


class OsdiModelDescriptor:
    """Descriptor returned by :func:`osdi_component`, consumed by ``compile_netlist``.

    Behaves like a component class from ``models_map``'s perspective but carries
    OSDI-specific metadata instead of Equinox fields.

    Two construction modes:

    1. **Canonical (recommended)** — ``param_names`` is ``None``, so the
       descriptor uses ``model.param_names`` directly.  Settings dicts may
       reference any canonical parameter name case-insensitively; unknown
       names raise a ``ValueError`` with near-match suggestions.  This is
       the mode used for models with large parameter sets (e.g. PSP103 with
       783 parameters).

    2. **Legacy positional** — ``param_names`` is an explicit tuple whose
       *order* must match the OSDI model's internal parameter ordering.
       Names are treated as opaque keys into ``default_params`` / settings
       dicts; they do not need to match ``model.param_names``.  Retained
       for backward compatibility with early OSDI tests that shipped
       pre-inference aliases like ``"m"`` for ``"$mfactor"``.
    """

    _is_osdi_descriptor: bool = True

    def __init__(
        self,
        model: OsdiModel,
        ports: tuple,
        param_names: tuple | None,
        default_params: dict,
        use_schur_reduction: bool = False,
    ) -> None:
        self.model = model
        self.ports = ports
        self.states: tuple = ()   # stateless for now; stateful support requires bodi changes
        self.use_schur_reduction = use_schur_reduction

        if param_names is None:
            self.param_names = tuple(model.param_names)
            self.is_canonical = True
            self._name_to_idx = {
                n.lower(): i for i, n in enumerate(model.param_names) if n
            }
        else:
            self.param_names = param_names
            self.is_canonical = False
            self._name_to_idx = {n.lower(): i for i, n in enumerate(param_names)}

        # Canonicalise default_params keys (case-insensitive) so the merge in
        # make_instance can be case-insensitive too.
        self.default_params = self._canonicalise(default_params, source="default_params")

    def _canonicalise(self, d: dict, *, source: str) -> dict:
        """Case-insensitive: rewrite ``d``'s keys to match ``self.param_names``.

        Unknown keys raise with a difflib "did you mean" suggestion.
        """
        out = dict.fromkeys(self.param_names, 0.0)
        for k, v in d.items():
            idx = self._name_to_idx.get(k.lower())
            if idx is None:
                candidates = list(self._name_to_idx.keys())
                close = difflib.get_close_matches(k.lower(), candidates, n=5)
                msg = (
                    f"Unknown OSDI parameter {k!r} in {source}. "
                    f"Did you mean one of: {close}?"
                )
                raise ValueError(msg)
            out[self.param_names[idx]] = v
        return out

    def make_instance(self, settings: dict) -> dict:
        """Merge per-instance ``settings`` into ``default_params`` by canonical name."""
        merged = dict(self.default_params)
        for k, v in settings.items():
            idx = self._name_to_idx.get(k.lower())
            if idx is None:
                candidates = list(self._name_to_idx.keys())
                close = difflib.get_close_matches(k.lower(), candidates, n=5)
                msg = (
                    f"Unknown OSDI parameter {k!r} in instance settings. "
                    f"Did you mean one of: {close}?"
                )
                raise ValueError(msg)
            merged[self.param_names[idx]] = v
        return {k: merged[k] for k in self.param_names}


def osdi_component(
    osdi_path: str,
    ports: tuple,
    param_names: tuple | None = None,
    default_params: dict | None = None,
    use_schur_reduction: bool = False,
) -> OsdiModelDescriptor:
    """Load a compiled ``.osdi`` binary and return a descriptor for ``compile_netlist``.

    Args:
        osdi_path:      Absolute path to the OpenVAF-compiled ``.osdi`` file.
        ports:          Ordered tuple of port names matching the Verilog-A terminals.
        param_names:    *(Optional, legacy)* Ordered tuple of parameter names.
                        If ``None`` (recommended), the descriptor uses
                        ``OsdiModel.param_names`` — the canonical names read
                        from the OSDI binary — and every setting key is
                        resolved against them case-insensitively.  If an
                        explicit tuple is supplied, its order must match the
                        OSDI model's internal parameter order and the names
                        are opaque dict keys (legacy positional mode).
        default_params: Default values for selected parameters.  Keys may be
                        any subset of canonical parameter names; unspecified
                        params default to ``0.0``.  In legacy positional mode
                        the keys must match the supplied ``param_names``.

    Returns:
        :class:`OsdiModelDescriptor` — pass this as a value in the
        ``models_map`` argument of :func:`~circulax.compiler.compile_netlist`.

    Raises:
        ValueError:            If ``ports`` length or a legacy ``param_names``
            length doesn't match the OSDI model's declared pin/param counts,
            or if an unknown parameter name appears in ``default_params``.
        NotImplementedError:   If the OSDI model has internal state variables
            (``num_states > 0``); stateful models are not yet supported.
        ImportError:           If ``bosdi`` is not available.

    Example (canonical mode, recommended)::

        OsdiPSP103N = osdi_component(
            osdi_path="psp103v4_psp103.osdi",
            ports=("D", "G", "S", "B"),
            default_params={"TYPE": 1.0, "L": 1e-6, "W": 10e-6},
        )

    Example (legacy positional mode)::

        OsdiResistor = osdi_component(
            osdi_path="resistor.osdi",
            ports=("A", "B"),
            param_names=("m", "R"),
            default_params={"m": 1.0, "R": 1000.0},
        )

    """
    if not _BOSDI_AVAILABLE:
        raise ImportError(
            "OSDI support requires the 'bosdi' package, which could not be imported. "
            "Ensure bosdi is installed or its src/ directory is on PYTHONPATH. "
            "Note: bosdi is not available on all platforms (e.g. Windows)."
        ) from _BOSDI_ERR

    model = load_osdi_model(osdi_path)

    if model.num_pins != len(ports):
        msg = f"OSDI model has {model.num_pins} pins but {len(ports)} port names given"
        raise ValueError(msg)
    if param_names is not None and model.num_params != len(param_names):
        msg = f"OSDI model has {model.num_params} params but {len(param_names)} param names given"
        raise ValueError(msg)
    if model.num_states > 0:
        msg = "Stateful OSDI models (num_states > 0) are not yet supported"
        raise NotImplementedError(msg)

    return OsdiModelDescriptor(
        model=model,
        ports=ports,
        param_names=param_names,
        default_params=default_params or {},
        use_schur_reduction=use_schur_reduction,
    )
