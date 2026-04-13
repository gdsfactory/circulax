"""OSDI device model integration for circulax.

Provides :func:`osdi_component` to load OpenVAF-compiled ``.osdi`` binaries and
use them as circuit components inside :func:`~circulax.compiler.compile_netlist`.

Requires the ``bosdi`` package to be installed or its ``src/`` directory present
on ``PYTHONPATH`` (e.g. via a ``.env`` file).  OSDI support is optional and not
available on all platforms (e.g. Windows).
"""

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


class OsdiModelDescriptor:
    """Descriptor returned by :func:`osdi_component`, consumed by ``compile_netlist``.

    Behaves like a component class from ``models_map``'s perspective but carries
    OSDI-specific metadata instead of Equinox fields.
    """

    _is_osdi_descriptor: bool = True

    def __init__(self, model: OsdiModel, ports: tuple, param_names: tuple, default_params: dict) -> None:
        self.model = model
        self.ports = ports
        self.states: tuple = ()   # stateless for now; stateful support requires bodi changes
        self.param_names = param_names
        self.default_params = default_params

    def make_instance(self, settings: dict) -> dict:
        """Merge per-instance ``settings`` into ``default_params``, return ordered dict."""
        merged = {**self.default_params, **settings}
        return {k: merged[k] for k in self.param_names}


def osdi_component(
    osdi_path: str,
    ports: tuple,
    param_names: tuple,
    default_params: dict | None = None,
) -> OsdiModelDescriptor:
    """Load a compiled ``.osdi`` binary and return a descriptor for ``compile_netlist``.

    Args:
        osdi_path:      Absolute path to the OpenVAF-compiled ``.osdi`` file.
        ports:          Ordered tuple of port names matching the Verilog-A terminals.
        param_names:    Ordered tuple of parameter names matching the OSDI model.
        default_params: Default values for each parameter. If ``None``, all
                        parameters default to ``0.0`` — be sure to pass real
                        values in the netlist ``settings``.

    Returns:
        :class:`OsdiModelDescriptor` — pass this as a value in the
        ``models_map`` argument of :func:`~circulax.compiler.compile_netlist`.

    Raises:
        AssertionError: If ``ports`` or ``param_names`` lengths don't match the
            OSDI model's declared pin/parameter counts.
        NotImplementedError: If the OSDI model has internal state variables
            (``num_states > 0``); stateful models are not yet supported.

    Example::

        OsdiResistor = osdi_component(
            osdi_path="/path/to/resistor.osdi",
            ports=("A", "B"),
            param_names=("R", "m"),
            default_params={"R": 1000.0, "m": 1.0},
        )
        models = {"res": OsdiResistor, "vsrc": VoltageSource}

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
    if model.num_params != len(param_names):
        msg = f"OSDI model has {model.num_params} params but {len(param_names)} param names given"
        raise ValueError(msg)
    if model.num_states > 0:
        msg = "Stateful OSDI models (num_states > 0) are not yet supported"
        raise NotImplementedError(msg)

    return OsdiModelDescriptor(
        model=model,
        ports=ports,
        param_names=param_names,
        default_params=default_params or dict.fromkeys(param_names, 0.0),
    )
