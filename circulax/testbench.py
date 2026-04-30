"""Test-bench helpers for wrapping device netlists with sources and loads.

A device netlist (e.g. from GDSFactory or hand-written SAX) describes the
device only — its connectivity and external ``ports`` mapping. Running it
in circulax additionally requires every external port to be terminated:
driven by a source, loaded by a detector, or tied to GND.

:func:`attach_testbench` wraps the device into a runnable netlist by
adding the requested terminations and wiring them to the device's
external ports.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


def attach_testbench(
    device: dict,
    *,
    sources: dict[str, dict[str, Any]] | None = None,
    loads: dict[str, dict[str, Any]] | None = None,
    gnd: Iterable[str] | None = None,
) -> dict:
    """Wrap a device netlist with sources, loads, and GND terminations.

    Each entry in ``sources`` and ``loads`` is a 2-port SAX instance spec.
    Its ``p1`` port is wired to the named device port; its ``p2`` is tied
    to ``GND,p1``. Entries listed in ``gnd`` are wired directly to GND
    with no intervening component.

    The output netlist uses the ``nets`` list format and inherits all
    instances and connectivity from the device. A ``"GND"`` instance is
    added if not already present.

    Args:
        device: Device netlist. Must have ``instances`` and ``ports``,
            and either ``connections`` (SAX) or ``nets`` (GDSFactory)
            describing internal connectivity.
        sources: ``{device_port: instance_spec}`` for source-like
            terminations (e.g. optical or voltage sources). Each
            ``instance_spec`` is a normal SAX instance dict
            ``{"component": ..., "settings": ...}`` with an optional
            ``"name"`` field. If ``"name"`` is omitted, the instance is
            named ``f"src_{port}"``.
        loads: Same format as ``sources``. Loads are typically resistors
            acting as photodetectors or matched terminations. Auto-name:
            ``f"load_{port}"``.
        gnd: Device ports to tie directly to GND.  The ``"GND"``
            instance and the ``"ground"`` component type are reserved in
            circulax — no ``"ground"`` entry is required in the ``models``
            dict passed to :func:`~circulax.circuit.compile_circuit`.

    Returns:
        A new netlist with ``instances`` and ``nets`` keys, ready for
        :func:`~circulax.circuit.compile_circuit`.

    Raises:
        ValueError: If a device port appears in more than one of
            ``sources``/``loads``/``gnd``, if a listed port is not in
            ``device["ports"]``, or if an auto-generated or
            user-supplied instance name collides with an existing
            instance in the device.

    Example:
        >>> mzi_nl = mzi.get_netlist()  # GDSFactory device netlist
        >>> bench = attach_testbench(
        ...     mzi_nl,
        ...     sources={"o1": {"name": "Laser",
        ...                     "component": "source",
        ...                     "settings": {"power": 1.0}}},
        ...     loads={"o3": {"name": "PD",
        ...                   "component": "resistor",
        ...                   "settings": {"R": 1.0}}},
        ...     gnd=["o2", "o4"],
        ... )
        >>> circuit = compile_circuit(bench, models, is_complex=True)

    """
    sources = sources or {}
    loads = loads or {}
    gnd_list = list(gnd or [])

    device_ports = device.get("ports", {})

    used = [*sources, *loads, *gnd_list]
    duplicates = [p for p, c in Counter(used).items() if c > 1]
    if duplicates:
        msg = f"Device ports appear in multiple roles: {duplicates}"
        raise ValueError(msg)

    unknown = [p for p in used if p not in device_ports]
    if unknown:
        msg = f"Ports {unknown} not in device['ports']; available: {list(device_ports)}"
        raise ValueError(msg)

    instances = dict(device.get("instances", {}))
    if "GND" not in instances:
        instances["GND"] = {"component": "ground"}

    nets: list[dict] = []
    for src, tgts in device.get("connections", {}).items():
        if isinstance(tgts, str):
            tgts = (tgts,)
        nets.extend({"p1": src, "p2": t} for t in tgts)
    nets.extend(device.get("nets", []))

    def _add_terminator(device_port: str, default_prefix: str, spec: dict[str, Any]) -> None:
        name = spec.get("name") or f"{default_prefix}_{device_port}"
        if name in instances:
            msg = f"Instance name '{name}' already exists in device netlist"
            raise ValueError(msg)
        instances[name] = {k: v for k, v in spec.items() if k != "name"}
        nets.append({"p1": f"{name},p1", "p2": device_ports[device_port]})
        nets.append({"p1": f"{name},p2", "p2": "GND,p1"})

    for port, spec in sources.items():
        _add_terminator(port, "src", spec)
    for port, spec in loads.items():
        _add_terminator(port, "load", spec)
    for port in gnd_list:
        nets.append({"p1": device_ports[port], "p2": "GND,p1"})

    return {"instances": instances, "nets": nets}
