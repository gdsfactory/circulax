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
from collections.abc import Iterable
from typing import Any

import kfnetlist as kfnl


def _attach_testbench_kfnetlist(
    device: kfnl.Netlist,
    *,
    sources: dict[str, dict[str, Any]],
    loads: dict[str, dict[str, Any]],
    gnd_list: list[str],
) -> kfnl.Netlist:
    """kfnetlist-native testbench attachment."""
    device_port_names = {p.name for p in device.ports}

    used = [*sources, *loads, *gnd_list]
    duplicates = [p for p, c in Counter(used).items() if c > 1]
    if duplicates:
        msg = f"Device ports appear in multiple roles: {duplicates}"
        raise ValueError(msg)

    unknown = [p for p in used if p not in device_port_names]
    if unknown:
        msg = f"Ports {unknown} not in device ports; available: {sorted(device_port_names)}"
        raise ValueError(msg)

    # Build a map from top-level port name → the PortRef it connects to
    port_to_ref: dict[str, kfnl.PortRef] = {}
    for net in device.nets:
        port_member = None
        ref_member = None
        for m in net:
            if isinstance(m, kfnl.NetlistPort):
                port_member = m
            elif isinstance(m, kfnl.PortRef):
                ref_member = m
        if port_member and ref_member:
            port_to_ref[port_member.name] = ref_member

    nl = kfnl.Netlist()

    # Copy existing instances
    for name, inst in device.instances.items():
        nl.create_inst(name=name, kcl=inst.kcl, component=inst.component, settings=inst.settings)

    if not device.has_instance("GND"):
        nl.create_inst(name="GND", kcl="", component="ground")

    # Copy existing nets (skip top-level NetlistPort members — they become internal)
    for net in device.nets:
        refs = [m for m in net if isinstance(m, kfnl.PortRef)]
        if len(refs) >= 2:
            nl.create_net(*refs)

    gnd_ref = kfnl.PortRef(instance="GND", port="p1")

    def _add_terminator(device_port: str, default_prefix: str, spec: dict[str, Any]) -> None:
        name = spec.get("name") or f"{default_prefix}_{device_port}"
        if nl.has_instance(name):
            msg = f"Instance name '{name}' already exists in device netlist"
            raise ValueError(msg)
        component = spec["component"]
        settings = {k: v for k, v in spec.items() if k not in ("name", "component")}
        settings_for_inst = spec.get("settings", settings or None)
        nl.create_inst(name=name, kcl="", component=component, settings=settings_for_inst)

        # p1 → device port, p2 → GND
        target_ref = port_to_ref.get(device_port)
        if target_ref:
            nl.create_net(kfnl.PortRef(instance=name, port="p1"), target_ref)
        nl.create_net(kfnl.PortRef(instance=name, port="p2"), gnd_ref)

    for port, spec in sources.items():
        _add_terminator(port, "src", spec)
    for port, spec in loads.items():
        _add_terminator(port, "load", spec)
    for port in gnd_list:
        target_ref = port_to_ref.get(port)
        if target_ref:
            nl.create_net(target_ref, gnd_ref)

    nl.sort()
    return nl


def attach_testbench(
    device: dict | kfnl.Netlist,
    *,
    sources: dict[str, dict[str, Any]] | None = None,
    loads: dict[str, dict[str, Any]] | None = None,
    gnd: Iterable[str] | None = None,
) -> dict | kfnl.Netlist:
    """Wrap a device netlist with sources, loads, and GND terminations.

    Accepts either a SAX-format dict or a ``kfnetlist.Netlist``.
    Returns the same type as the input.

    """
    sources = sources or {}
    loads = loads or {}
    gnd_list = list(gnd or [])

    if isinstance(device, kfnl.Netlist):
        return _attach_testbench_kfnetlist(
            device, sources=sources, loads=loads, gnd_list=gnd_list,
        )

    # --- Legacy SAX dict path ---
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
