"""OSDI-based circuit components using compiled Verilog-A models via JAX.

This module provides a framework for using OpenVAF-compiled OSDI device models
in Circulax circuits. OSDI models are dynamically loaded from .osdi binary files,
wrapped in CircuitComponent subclasses, and made fully differentiable via JAX.

Key Features:
- Load OSDI models dynamically (supports OSDI 0.4 and 0.5)
- Automatic batched evaluation with Rayon parallelization
- Analytical Jacobians (dI/dV, dQ/dV) from OSDI models
- Full JAX integration: jax.grad(), jax.vmap(), jax.jacobian() work directly
- Zero-copy data flow where possible

Example:
    >>> from circulax.components.osdi_component import osdi_component
    >>> OsdiResistor = osdi_component(
    ...     "resistor.osdi",
    ...     port_names=("p1", "p2"),
    ...     param_names=("R", "m"),
    ...     param_defaults={"R": 1e3, "m": 1.0}
    ... )
    >>> resistor = OsdiResistor(R=100.0, m=1.0)
"""

import sys
sys.path.insert(0, "/home/cdaunt/code/bodi/src")

from osdi_loader import load_osdi_model, OsdiModel
from osdi_jax import osdi_eval

from typing import Any, ClassVar, Optional
import jax
import jax.numpy as jnp
from circulax.components.base_component import CircuitComponent


def _extract_param(container: Any, name: str) -> Any:
    """Extract a parameter from either a dict or an object with attributes.

    Args:
        container: A dict or any object with named attributes
        name: The parameter name to look up

    Returns:
        The parameter value
    """
    if isinstance(container, dict):
        return container[name]
    return getattr(container, name)


class OsdiComponent(CircuitComponent):
    """Base class for OSDI-based circuit components.

    OSDI components wrap compiled Verilog-A models (via OpenVAF) in the Circulax
    component system. They automatically handle:

    - Loading OSDI binaries dynamically
    - Mapping between Circulax's named port/state format and OSDI's batched arrays
    - Parameter extraction from Equinox fields or dict
    - JAX tracing and differentiation via osdi_eval()

    Subclasses are typically created via the osdi_component() factory function,
    which inspects OSDI metadata to populate class variables.
    """

    # Class variables set by factory function
    osdi_model: ClassVar[Optional[OsdiModel]] = None
    param_names: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def solver_call(
        cls,
        t: float,
        y: jax.Array,
        args: Any,
    ) -> tuple[jax.Array, jax.Array]:
        """Evaluate the component physics via OSDI model.

        Maps Circulax's solver interface (flat arrays, parameter container)
        to OSDI's batched evaluation interface, handles shape transformations,
        and converts OSDI's 5-tuple output (currents, conductances, charges,
        capacitances, new_state) to Circulax's 2-tuple format (f_vec, q_vec).

        Args:
            t: Current simulation time (unused for time-independent OSDI models)
            y: Flat state vector of shape (n_ports + n_states,), containing
               port voltages followed by internal state variables:
               [v_0, v_1, ..., v_{n-1}, s_0, s_1, ..., s_{m-1}]
            args: Parameter container (dict or object) with attributes matching
                  param_names. Used to extract current parameter values.

        Returns:
            A 2-tuple (f_vec, q_vec) of JAX arrays:
            - f_vec: Resistive contributions, shape (n_ports + n_states,)
            - q_vec: Reactive contributions, shape (n_ports + n_states,)

        Notes:
            - Always operates on single device (batch size = 1)
            - OSDI outputs are reshaped to match Circulax's flat format
            - Charge (q) is mapped to reactive port contributions
            - Updated internal state is mapped to state derivatives (f contributions)
        """
        if cls.osdi_model is None:
            raise RuntimeError(
                f"{cls.__name__} has no OSDI model loaded. "
                "Did you use osdi_component() factory to create this class?"
            )

        num_pins = len(cls.ports)
        num_params = len(cls.param_names)
        num_states = len(cls.states)

        # ─────────────────────────────────────────────────────────────
        # 1. Extract port voltages and internal states from y
        # ─────────────────────────────────────────────────────────────
        voltages_flat = y[:num_pins]  # shape (num_pins,)
        states_flat = y[num_pins:] if num_states > 0 else jnp.array([])  # shape (num_states,)

        # ─────────────────────────────────────────────────────────────
        # 2. Extract parameters from args container
        # ─────────────────────────────────────────────────────────────
        param_values = [_extract_param(args, pn) for pn in cls.param_names]

        # ─────────────────────────────────────────────────────────────
        # 3. Reshape for OSDI: add batch dimension (batch size = 1)
        # ─────────────────────────────────────────────────────────────
        voltages_batch = jnp.expand_dims(voltages_flat, 0)    # (1, num_pins)
        params_batch = jnp.array([param_values])              # (1, num_params)
        states_batch = jnp.expand_dims(states_flat, 0)        # (1, num_states)

        # ─────────────────────────────────────────────────────────────
        # 4. Call OSDI model evaluation
        # ─────────────────────────────────────────────────────────────
        # Returns: (currents, conductances, charges, capacitances, new_state)
        # Shapes: [(1, P), (1, P²), (1, P), (1, P²), (1, S)] where P=pins, S=states
        currents, conductances, charges, capacitances, new_state = osdi_eval(
            cls.osdi_model.id,
            voltages_batch,
            params_batch,
            states_batch
        )

        # ─────────────────────────────────────────────────────────────
        # 5. Reshape outputs back to Circulax format
        # ─────────────────────────────────────────────────────────────
        # Extract single device (batch index 0) from all outputs
        currents_single = currents[0]      # shape (num_pins,)
        charges_single = charges[0]        # shape (num_pins,)
        new_state_single = new_state[0]    # shape (num_states,)

        # Build f_vec: [terminal_currents, state_derivatives]
        f_vec = jnp.concatenate([currents_single, new_state_single])

        # Build q_vec: [terminal_charges, zeros_for_states]
        # Note: OSDI doesn't provide state charge, so use zeros
        q_vec = jnp.concatenate([charges_single, jnp.zeros(num_states)])

        return f_vec, q_vec


def osdi_component(
    osdi_path: str,
    port_names: Optional[tuple[str, ...]] = None,
    param_names: Optional[tuple[str, ...]] = None,
    param_defaults: Optional[dict[str, float]] = None,
    osdi_version: str = "0.4",
) -> type[OsdiComponent]:
    """Factory function to create an OsdiComponent subclass from a .osdi binary.

    Loads an OSDI device model, extracts metadata, and returns a new CircuitComponent
    subclass that wraps the model. The returned class can be instantiated with
    parameter values and used in Circulax circuits.

    Args:
        osdi_path: Path to the compiled OSDI binary (.osdi file from OpenVAF)
        port_names: Tuple of port names. If None, defaults to ("p0", "p1", ...)
                   matching the OSDI terminal count.
        param_names: Tuple of parameter names in the order OSDI expects.
                    If None, defaults to ("param_0", "param_1", ...).
        param_defaults: Dict mapping parameter names to default values.
                       Used to initialize Equinox fields. If None, defaults to 1.0
                       for all parameters.
        osdi_version: OSDI standard version ("0.4" or "0.5"). Determines ABI layout
                     for descriptor parsing. Defaults to "0.4".

    Returns:
        A new OsdiComponent subclass with:
        - ports: ClassVar tuple of port names
        - states: ClassVar tuple of state names (empty for now; stateless models only)
        - param_names: ClassVar tuple of parameter names
        - Parameter attributes (Equinox fields) for each parameter with defaults
        - osdi_model: ClassVar reference to loaded OsdiModel

    Raises:
        FileNotFoundError: If osdi_path doesn't exist
        RuntimeError: If OSDI binary is invalid or version unsupported
        ValueError: If metadata extraction fails

    Example:
        >>> from circulax.components.osdi_component import osdi_component
        >>> # Create a resistor component from OSDI binary
        >>> OsdiResistor = osdi_component(
        ...     "/path/to/resistor_va.osdi",
        ...     port_names=("p1", "p2"),
        ...     param_names=("R", "m"),
        ...     param_defaults={"R": 1e3, "m": 1.0}
        ... )
        >>> # Instantiate with specific parameter values
        >>> r1 = OsdiResistor(R=100.0, m=1.0)
        >>> # Use in circuit...
    """

    # ─────────────────────────────────────────────────────────────────────
    # 1. Load OSDI model and extract metadata
    # ─────────────────────────────────────────────────────────────────────
    try:
        model = load_osdi_model(osdi_path, version=osdi_version)
    except (FileNotFoundError, RuntimeError) as e:
        raise RuntimeError(
            f"Failed to load OSDI binary at '{osdi_path}' as OSDI {osdi_version}. "
            f"Error: {e}"
        ) from e

    # ─────────────────────────────────────────────────────────────────────
    # 2. Set up port names (default: p0, p1, p2, ...)
    # ─────────────────────────────────────────────────────────────────────
    if port_names is None:
        port_names = tuple(f"p{i}" for i in range(model.num_pins))

    if len(port_names) != model.num_pins:
        raise ValueError(
            f"Port count mismatch: provided {len(port_names)} names but "
            f"OSDI model has {model.num_pins} terminals"
        )

    # ─────────────────────────────────────────────────────────────────────
    # 3. Set up parameter names (default: param_0, param_1, ...)
    # ─────────────────────────────────────────────────────────────────────
    if param_names is None:
        param_names = tuple(f"param_{i}" for i in range(model.num_params))

    if len(param_names) != model.num_params:
        raise ValueError(
            f"Parameter count mismatch: provided {len(param_names)} names but "
            f"OSDI model has {model.num_params} parameters"
        )

    # ─────────────────────────────────────────────────────────────────────
    # 4. Set up parameter defaults (default: 1.0 for all)
    # ─────────────────────────────────────────────────────────────────────
    if param_defaults is None:
        param_defaults = {pn: 1.0 for pn in param_names}
    else:
        # Ensure all parameters have defaults
        for pn in param_names:
            if pn not in param_defaults:
                param_defaults[pn] = 1.0

    # ─────────────────────────────────────────────────────────────────────
    # 5. Build namespace for new class
    # ─────────────────────────────────────────────────────────────────────
    # For now, stateless models only (num_states = 0)
    if model.num_states > 0:
        raise NotImplementedError(
            f"Stateful models (num_states={model.num_states}) not yet supported. "
            "BODI Rust implementation requires extension for state handling."
        )

    states = ()  # Empty for stateless models

    # Build class namespace
    namespace = {
        # Metadata
        "osdi_model": model,
        "param_names": param_names,

        # Circulax component metadata
        "ports": port_names,
        "states": states,

        # Type hints for parameters
        "__annotations__": {pn: float for pn in param_names},

        # Parameter default values (become Equinox fields)
        **param_defaults,
    }

    # ─────────────────────────────────────────────────────────────────────
    # 6. Create and return new class
    # ─────────────────────────────────────────────────────────────────────
    class_name = f"OsdiComponent_{model.id}_{osdi_path.split('/')[-1].split('.')[0]}"

    new_class = type(class_name, (OsdiComponent,), namespace)

    return new_class
