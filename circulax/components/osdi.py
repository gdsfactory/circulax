"""OSDI device model integration for circulax.

Provides :func:`osdi_component` to load OpenVAF-compiled ``.osdi`` binaries and
use them as circuit components inside :func:`~circulax.compiler.compile_netlist`.

Requires the ``bodi`` package (available at ``/home/cdaunt/code/bodi``).
"""

import sys

import equinox as eqx
import jax.numpy as jnp

# bodi ships as a local editable install; add its src to the path if needed.
_BODI_SRC = "/home/cdaunt/code/bodi/src"
if _BODI_SRC not in sys.path:
    sys.path.insert(0, _BODI_SRC)

from osdi_loader import OsdiModel, load_osdi_model  # noqa: E402


class OsdiComponentGroup(eqx.Module):
    """Batch of N identical OSDI device instances, evaluated via bodi.

    Unlike :class:`~circulax.compiler.ComponentGroup`, this group calls
    ``osdi_eval`` with all N instances at once (leveraging Rayon parallelism)
    and uses the analytical Jacobians (conductances/capacitances) that the
    OSDI model returns directly — no ``jax.jacfwd`` required.
    """

    name: str = eqx.field(static=True)
    model_id: int = eqx.field(static=True)   # bodi registry ID — not differentiable
    num_pins: int = eqx.field(static=True)
    num_params: int = eqx.field(static=True)
    num_states: int = eqx.field(static=True)

    params: jnp.ndarray   # (N, num_params) float64 — batched device parameters
    states: jnp.ndarray   # (N, num_states) float64 — zeros for stateless models

    var_indices: jnp.ndarray  # (N, num_pins) int32 — indices into global y vector
    eq_indices: jnp.ndarray   # (N, num_pins) int32 — indices into global residual
    jac_rows: jnp.ndarray     # (N*num_pins*num_pins,) int32 — COO row indices
    jac_cols: jnp.ndarray     # (N*num_pins*num_pins,) int32 — COO col indices

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
