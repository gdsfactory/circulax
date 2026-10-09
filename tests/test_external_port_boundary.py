"""Declared characterization boundaries survive physical testbench attachment.

Contract: GF+ specs/circulax-simulation.md, Circuit and model boundary.
Sources and loads remain instances; their terminals and internal DUT nodes
never become additional declared analysis ports.
"""

import kfnetlist as kfnl
import pytest

from circulax import attach_testbench, compile_circuit
from circulax.components.electronic import Resistor, VoltageSource
from circulax.components.photonic import OpticalSource
from circulax.netlist import sax_to_kfnetlist


def _device(representation: str) -> dict | kfnl.Netlist:
    """Build a boundary with an internal node and physical bias/load instances.

    @tags circulax simulation
    """
    device = {
        "instances": {
            "Ra": {"component": "resistor", "settings": {"R": 50.0}},
            "Rb": {"component": "resistor", "settings": {"R": 50.0}},
            "physical_load": {"component": "resistor", "settings": {"R": 100.0}},
            "physical_source": {"component": "optical_source", "settings": {"power": 1e-6}},
            "bias": {"component": "voltage_source", "settings": {"V": 1.0}},
            "bias_load": {"component": "resistor", "settings": {"R": 1000.0}},
            "GND": {"component": "ground"},
        },
        "connections": {
            "Ra,p2": ("Rb,p1", "physical_source,p1"),
            "bias,p1": "bias_load,p1",
            "Rb,p2": "physical_load,p1",
            "GND,p1": ("physical_source,p2", "bias,p2", "physical_load,p2", "bias_load,p2"),
        },
        "ports": {"input": "Ra,p1", "output": "Rb,p2"},
    }
    if representation == "kfnetlist":
        return sax_to_kfnetlist(device)[0]
    return device


def _declared_ports(device: dict | kfnl.Netlist) -> set[str]:
    """Read external declarations without inferring ports from terminals.

    @tags circulax simulation
    """
    if isinstance(device, kfnl.Netlist):
        return {port.name for port in device.ports}
    return set(device.get("ports", {}))


def _attach(device: dict | kfnl.Netlist) -> dict | kfnl.Netlist:
    """Attach named analysis fixtures separately from physical DUT components.

    @tags circulax simulation
    """
    return attach_testbench(
        device,
        sources={"input": {"name": "analysis_source", "component": "optical_source", "settings": {"power": 1e-9}}},
        loads={"output": {"name": "analysis_load", "component": "resistor", "settings": {"R": 50.0}}},
    )


@pytest.mark.parametrize("representation", ["kfnetlist", "sax"])
def test_attachment_preserves_declared_boundary(representation: str) -> None:
    """Attaching source/load fixtures preserves exactly the declared ports.

    @tags circulax simulation
    """
    device = _device(representation)
    attached = _attach(device)
    assert _declared_ports(attached) == {"input", "output"}
    assert _declared_ports(device) == {"input", "output"}
    if isinstance(attached, kfnl.Netlist):
        for name in ("input", "output"):
            assert any(kfnl.NetlistPort(name) in net for net in attached.nets)
    else:
        assert attached["ports"] == device["ports"]


def test_unselected_disconnected_port_remains_declared_and_connected_to_its_net() -> None:
    """An unselected boundary marker survives even without a device terminal.

    @tags circulax simulation
    """
    device = _device("kfnetlist")
    device.create_port("unconnected")
    device.create_net(kfnl.NetlistPort("unconnected"))
    attached = _attach(device)
    assert _declared_ports(attached) == {"input", "output", "unconnected"}
    assert any(list(net) == [kfnl.NetlistPort("unconnected")] for net in attached.nets)


@pytest.mark.parametrize("representation", ["kfnetlist", "sax"])
def test_physical_sources_and_loads_remain_distinct_instances(representation: str) -> None:
    """Attachment neither consumes nor rewrites DUT source/load components.

    @tags circulax simulation
    """
    device = _device(representation)
    attached = _attach(device)
    original_instances = device.instances if isinstance(device, kfnl.Netlist) else device["instances"]
    attached_instances = attached.instances if isinstance(attached, kfnl.Netlist) else attached["instances"]
    for name in ("physical_source", "physical_load", "bias"):
        assert attached_instances[name] == original_instances[name]
    assert {"analysis_source", "analysis_load"} <= set(attached_instances)


@pytest.mark.parametrize("representation", ["kfnetlist", "sax"])
@pytest.mark.parametrize("attached", [False, True])
def test_compiled_ports_exclude_internal_nodes_and_source_terminals(representation: str, *, attached: bool) -> None:
    """Compilation retains the external boundary rather than all node aliases.

    @tags circulax simulation
    """
    device = _device(representation)
    if attached:
        device = _attach(device)
    circuit = compile_circuit(
        device,
        {
            "resistor": Resistor,
            "voltage_source": VoltageSource,
            "optical_source": OpticalSource,
        },
    )
    assert "Ra,p2" in circuit.port_map  # Internal nodes still exist for observables.
    assert set(circuit.ports) == {"input", "output"}
