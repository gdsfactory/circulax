"""Analysis-owned matched boundaries for normalized optical DC characterization."""

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

import jax
import jax.numpy as jnp
import kfnetlist as kfnl
import numpy as np

from circulax.components.base_component import PhysicsReturn, Signals, component
from circulax.testbench import attach_testbench


@component(ports=("p1", "p2"), holomorphic=True)
def _matched_port(signals: Signals, incident: complex = 0j, z0: float = 1.0) -> PhysicsReturn:
    """Apply a matched shunt and incident-wave excitation without clamping.

    @tags circulax simulation
    """
    current = (signals.p1 - signals.p2) / z0 - 2 * incident / jnp.sqrt(z0)
    return {"p1": current, "p2": -current}, {}


def _validate_coordinates(
    wavelengths: jax.Array,
    amplitude: float,
    amplitude_unit: str,
    z0: float,
) -> jax.Array:
    """Validate the caller's declared physical optical normalization and grid.

    @tags circulax simulation
    """
    if amplitude_unit not in {"sqrt(W)", "sqrt(mW)"}:
        msg = "Optical incident amplitude_unit must explicitly declare sqrt(W) or sqrt(mW) normalization"
        raise ValueError(msg)
    if not np.isfinite(amplitude) or amplitude <= 0 or not np.isfinite(z0) or z0 <= 0:
        msg = "Optical incident amplitude and reference z0 must be finite and positive"
        raise ValueError(msg)
    coordinates = jnp.asarray(wavelengths)
    if coordinates.ndim != 1 or not len(coordinates) or not np.all(np.isfinite(coordinates)) or np.any(coordinates <= 0):
        msg = "Wavelength coordinates must be a nonempty positive finite one-dimensional array"
        raise ValueError(msg)
    return coordinates


def _validate_boundary(circuit: Any, selected: tuple[str, ...]) -> None:
    """Reject internal aliases, duplicate nodes and ideal source clamps.

    @tags circulax simulation
    """
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


def wavelength_scattering(
    circuit: Any,
    compile_fn: Callable,
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

    @tags circulax simulation
    """
    selected = tuple(ports)
    _validate_boundary(circuit, selected)
    coordinates = _validate_coordinates(wavelengths, amplitude, amplitude_unit, z0)
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
    bench = compile_fn(
        attach_testbench(device, sources=fixtures),
        models,
        is_complex=True,
        g_leak=0,
        rtol=circuit.rtol,
        atol=circuit.atol,
        max_steps=circuit.max_steps,
    )
    count = len(selected)
    excitations = jnp.concatenate([jnp.zeros((1, count)), amplitude * jnp.eye(count)])
    guesses = jnp.zeros((count + 1, 2 * bench.sys_size))

    def sweep_point(previous: jax.Array, wavelength: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Maintain separate continuation guesses for baseline and each column.

        @tags circulax simulation
        """

        def solve(guess: jax.Array, incident: jax.Array) -> jax.Array:
            """Solve the DUT with fixed bias and this incident-wave column.

            @tags circulax simulation
            """
            updates = {**(params or {}), wl_param: wavelength}
            updates.update({f"{fixtures[port]['name']}.incident": incident[index] for index, port in enumerate(selected)})
            return bench.dc(guess, params=updates)

        solved = jax.vmap(solve)(previous, excitations)
        return solved, solved

    _, solved = jax.lax.scan(sweep_point, guesses, coordinates)
    if not bool(jnp.all(jnp.isfinite(solved))):
        msg = "Wavelength scattering produced nonfinite baseline or excited solutions"
        raise ValueError(msg)
    physical = solved[..., : bench.sys_size] + 1j * solved[..., bench.sys_size :]
    baseline = physical[:, 0]
    excited = physical[:, 1:]
    port_nodes = jnp.asarray([bench.port_map[port] for port in selected])
    outgoing = (excited[..., port_nodes] - baseline[:, None, port_nodes]) / jnp.sqrt(z0) / amplitude
    scattering = jnp.swapaxes(outgoing, 1, 2) - jnp.eye(count)
    return {
        "wavelengths": coordinates,
        "ports": selected,
        "s": scattering,
        "amplitude": amplitude,
        "amplitude_unit": amplitude_unit,
        "baseline_solution": baseline,
        "solution": excited,
        "port_map": dict(bench.port_map),
    }
