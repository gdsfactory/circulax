"""Analysis-owned matched boundaries for normalized optical DC characterization."""

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import kfnetlist as kfnl
import numpy as np

from circulax.components.base_component import PhysicsReturn, Signals, component
from circulax.testbench import attach_testbench


@component(ports=("p1", "p2"), holomorphic=True)
def _matched_port(signals: Signals, incident: complex = 0j, z0: float = 1.0) -> PhysicsReturn:
    """Apply a matched shunt and incident-wave excitation without clamping."""
    current = (signals.p1 - signals.p2) / z0 - 2 * incident / jnp.sqrt(z0)
    return {"p1": current, "p2": -current}, {}


def _validate_reference(z0: float) -> None:
    """Require a finite positive reference impedance."""
    if not np.isfinite(z0) or z0 <= 0:
        msg = "Optical incident amplitude and reference z0 must be finite and positive"
        raise ValueError(msg)


def _validate_coordinates(
    wavelengths: jax.Array,
    amplitude: float,
    amplitude_unit: str,
    z0: float,
) -> jax.Array:
    """Validate the caller's declared physical optical normalization and grid."""
    if amplitude_unit not in {"sqrt(W)", "sqrt(mW)"}:
        msg = "Optical incident amplitude_unit must explicitly declare sqrt(W) or sqrt(mW) normalization"
        raise ValueError(msg)
    if not np.isfinite(amplitude) or amplitude <= 0:
        msg = "Optical incident amplitude and reference z0 must be finite and positive"
        raise ValueError(msg)
    _validate_reference(z0)
    coordinates = jnp.asarray(wavelengths)
    # Value checks need concrete numbers; a traced grid is the caller's responsibility.
    concrete = None if isinstance(wavelengths, jax.core.Tracer) else np.asarray(wavelengths)
    if (
        coordinates.ndim != 1
        or not len(coordinates)
        or (concrete is not None and not np.all(np.isfinite(concrete) & (concrete > 0)))
    ):
        msg = "Wavelength coordinates must be a nonempty positive finite one-dimensional array"
        raise ValueError(msg)
    return coordinates


def _validate_boundary(circuit: Any, selected: tuple[str, ...]) -> None:
    """Reject internal aliases, duplicate nodes and ideal source clamps."""
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(circuit.ports):
        msg = "Wavelength scattering requires unique selected declared external ports"
        raise ValueError(msg)
    if circuit.source_netlist is None or circuit.source_models is None:
        msg = "Wavelength scattering requires canonical source topology and reusable models"
        raise ValueError(msg)
    nodes = [circuit.port_map[port] for port in selected]
    if 0 in nodes or len(set(nodes)) != len(nodes):
        msg = "Selected optical ports must identify distinct non-ground boundary nodes"
        raise ValueError(msg)
    for group in circuit.groups.values():
        for local in getattr(group, "ideal_voltage_constraint_ports", ()):
            if np.any(np.isin(np.asarray(group.var_indices[:, local]), nodes)):
                msg = "An ideal voltage/field constraint clamps a selected optical characterization port"
                raise ValueError(msg)


_COMPLEX_LINEAR_RTOL = 0.05


def _check_complex_linear(response: jax.Array, rotated: jax.Array) -> None:
    """Require the response to a quadrature incident wave to be the rotated in-phase response.

    A response that mixes the field with its conjugate is real-linear only; reporting it as a
    complex S-parameter would be wrong. Power-dependent phase-invariant models pass.
    """
    scale = jnp.maximum(jnp.max(jnp.abs(response)), jnp.finfo(jnp.float64).tiny)
    if not bool(jnp.max(jnp.abs(rotated - 1j * response)) <= _COMPLEX_LINEAR_RTOL * scale):
        msg = (
            "Wavelength scattering response is not complex-linear in the incident wave: "
            "the model is non-holomorphic or outside its linear regime at this amplitude"
        )
        raise ValueError(msg)


class WavelengthBench(NamedTuple):
    """A compiled matched-boundary bench and the excitation instance for each selected port."""

    circuit: Any
    fixtures: dict[str, str]


def build_wavelength_bench(circuit: Any, compile_fn: Callable, selected: tuple[str, ...], z0: float) -> WavelengthBench:
    """Clone the source topology, attach matched-port fixtures and compile the bench.

    The bench depends only on the circuit's source topology and models, the selected
    ports and ``z0``; parameter values are runtime inputs of the solves.
    """
    _validate_boundary(circuit, selected)
    _validate_reference(z0)
    device = (
        kfnl.Netlist.from_dict(circuit.source_netlist.to_dict())
        if isinstance(circuit.source_netlist, kfnl.Netlist)
        else deepcopy(circuit.source_netlist)
    )
    instances = device.instances if isinstance(device, kfnl.Netlist) else device["instances"]
    models = dict(circuit.source_models)
    model_key = "__circulax_matched_port__"
    while model_key in models:
        model_key += "_"
    models[model_key] = _matched_port
    fixtures = {}
    for index, port in enumerate(selected):
        name = f"__circulax_analysis_{index}"
        while name in instances:
            name += "_"
        fixtures[port] = {"name": name, "component": model_key, "settings": {"z0": z0}}
    # Topology is static: compile eagerly even when the caller is tracing so index arrays stay concrete.
    with jax.ensure_compile_time_eval():
        bench = compile_fn(
            attach_testbench(device, sources=fixtures),
            models,
            is_complex=True,
            g_leak=getattr(circuit.solver, "g_leak", 1e-9),
            rtol=circuit.rtol,
            atol=circuit.atol,
            max_steps=circuit.max_steps,
        )
    return WavelengthBench(bench, {port: fixture["name"] for port, fixture in fixtures.items()})


def wavelength_scattering(
    circuit: Any,
    bench_for: Callable[[tuple[str, ...], float], WavelengthBench],
    *,
    ports: Sequence[str],
    wavelengths: jax.Array,
    amplitude: float,
    amplitude_unit: str,
    params: Mapping[str, Any] | None,
    z0: float,
    wl_param: str,
) -> dict[str, Any]:
    """Run independent port excitations with wavelength continuation.

    Optical discipline and wave normalization are declared by the caller; no
    classification is inferred from solver dtype. For SAX optical models the
    reference is z0=1 and amplitude_unit identifies sqrt(W) or sqrt(mW).
    ``bench_for(selected, z0)`` supplies the (cached) compiled matched-boundary bench.
    """
    selected = tuple(ports)
    _validate_boundary(circuit, selected)
    coordinates = _validate_coordinates(wavelengths, amplitude, amplitude_unit, z0)
    bench, fixtures = bench_for(selected, z0)
    count = len(selected)
    excitations = jnp.concatenate([jnp.zeros((1, count)), amplitude * jnp.eye(count)])
    guesses = jnp.zeros((count + 1, 2 * bench.sys_size))

    def solve(guess: jax.Array, incident: jax.Array, wavelength: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Solve the DUT with fixed bias and this incident-wave column."""
        updates = {**(params or {}), wl_param: wavelength}
        updates.update({f"{fixtures[port]}.incident": incident[index] for index, port in enumerate(selected)})
        return bench.dc_checked(guess, params=updates)

    def sweep_point(previous: jax.Array, wavelength: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Maintain separate continuation guesses for baseline and each column."""
        solved, converged = jax.vmap(solve, in_axes=(0, 0, None))(previous, excitations, wavelength)
        return solved, (solved, converged)

    _, (solved, converged) = jax.lax.scan(sweep_point, guesses, coordinates)
    size = bench.sys_size
    physical = solved[..., :size] + 1j * solved[..., size:]
    baseline = physical[:, 0]
    excited = physical[:, 1:]
    if not isinstance(solved, jax.core.Tracer):
        # Concrete results are validated here; under jit callers inspect ``converged`` instead.
        if not bool(jnp.all(converged)):
            msg = "Wavelength scattering DC solve did not converge for at least one wavelength or excitation"
            raise ValueError(msg)
        if not bool(jnp.all(jnp.isfinite(solved))):
            msg = "Wavelength scattering produced nonfinite baseline or excited solutions"
            raise ValueError(msg)
        # One quadrature probe at the first wavelength detects responses that are not complex-linear.
        probe, probe_ok = solve(solved[0, 1], 1j * excitations[1], coordinates[0])
        if not bool(probe_ok):
            msg = "Wavelength scattering quadrature probe did not converge"
            raise ValueError(msg)
        _check_complex_linear(excited[0, 0] - baseline[0], probe[:size] + 1j * probe[size:] - baseline[0])
    port_nodes = jnp.asarray([bench.port_map[port] for port in selected])
    outgoing = (excited[..., port_nodes] - baseline[:, None, port_nodes]) / jnp.sqrt(z0) / amplitude
    scattering = jnp.swapaxes(outgoing, 1, 2) - jnp.eye(count)
    return {
        "wavelengths": coordinates,
        "ports": selected,
        "s": scattering,
        "amplitude": amplitude,
        "amplitude_unit": amplitude_unit,
        "converged": converged,
        "baseline_solution": baseline,
        "solution": excited,
        "port_map": dict(bench.port_map),
    }
