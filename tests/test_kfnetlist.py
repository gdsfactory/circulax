"""Tests for kfnetlist integration: converter, build_net_map_kfnetlist, compile_netlist."""

import jax.numpy as jnp
import kfnetlist as kfnl
import pytest

from circulax.compiler import compile_netlist
from circulax.netlist import build_net_map_kfnetlist, sax_to_kfnetlist


def test_build_net_map_kfnetlist_basic():
    nl = kfnl.Netlist()
    nl.create_inst(name="GND", kcl="", component="ground")
    nl.create_inst(name="R1", kcl="", component="resistor", settings={"R": 100.0})
    nl.create_inst(name="R2", kcl="", component="resistor", settings={"R": 200.0})

    gnd = kfnl.PortRef(instance="GND", port="p1")
    nl.create_net(gnd, kfnl.PortRef(instance="R1", port="p1"))
    nl.create_net(kfnl.PortRef(instance="R1", port="p2"), kfnl.PortRef(instance="R2", port="p1"))
    nl.create_net(kfnl.PortRef(instance="R2", port="p2"), gnd)

    pmap, num_nets = build_net_map_kfnetlist(nl)

    assert pmap["GND,p1"] == 0
    assert pmap["R1,p1"] == 0
    assert pmap["R2,p2"] == 0
    assert pmap["R1,p2"] == pmap["R2,p1"]
    assert pmap["R1,p2"] > 0
    assert num_nets == 2


def test_gnd_handling_kfnetlist():
    nl = kfnl.Netlist()
    nl.create_inst(name="GND", kcl="", component="ground")
    nl.create_inst(name="R1", kcl="", component="resistor")

    gnd = kfnl.PortRef(instance="GND", port="p1")
    nl.create_net(gnd, kfnl.PortRef(instance="R1", port="p1"))
    nl.create_net(kfnl.PortRef(instance="R1", port="p2"), gnd)

    pmap, num_nets = build_net_map_kfnetlist(nl)
    assert all(v == 0 for v in pmap.values())
    assert num_nets == 1


def test_ground_component_name_not_required():
    """A component='ground' instance is ground even when its name is not GND."""
    nl = kfnl.Netlist()
    nl.create_inst(name="g0", kcl="", component="ground")
    nl.create_inst(name="R1", kcl="", component="resistor")

    gnd = kfnl.PortRef(instance="g0", port="p1")
    nl.create_net(gnd, kfnl.PortRef(instance="R1", port="p1"))
    nl.create_net(kfnl.PortRef(instance="R1", port="p2"), gnd)

    pmap, num_nets = build_net_map_kfnetlist(nl)
    assert pmap["g0,p1"] == 0
    assert pmap["R1,p1"] == 0
    assert pmap["R1,p2"] == 0
    assert num_nets == 1


def test_sax_to_kfnetlist_preserves_top_level_ports_as_nets():
    device = {
        "instances": {"R": {"component": "resistor", "settings": {"R": 100.0}}},
        "connections": {},
        "ports": {"a": "R,p1", "b": "R,p2"},
    }

    nl, _override = sax_to_kfnetlist(device)
    pmap, _ = build_net_map_kfnetlist(nl)

    assert [p.name for p in nl.ports] == ["a", "b"]
    assert pmap["a"] == pmap["R,p1"]
    assert pmap["b"] == pmap["R,p2"]
    assert len(nl.nets) == 2
    assert any(any(isinstance(m, kfnl.NetlistPort) and m.name == "a" for m in net) for net in nl.nets)
    assert any(any(isinstance(m, kfnl.PortRef) and m.instance == "R" and m.port == "p1" for m in net) for net in nl.nets)


def test_sax_to_kfnetlist_round_trip(simple_lrc_netlist):
    """SAX dict and converted kfnetlist produce identical compilation output."""
    sax_dict, models_map = simple_lrc_netlist

    groups_sax, sys_size_sax, pmap_sax = compile_netlist(dict(sax_dict), models_map)

    nl, _override = sax_to_kfnetlist(sax_dict)
    groups_kf, sys_size_kf, pmap_kf = compile_netlist(nl, models_map)

    assert sys_size_sax == sys_size_kf
    assert set(groups_sax.keys()) == set(groups_kf.keys())

    # Port maps should have the same keys and structure (node IDs may differ
    # due to ordering, but the equivalence classes should match)
    sax_ports = {k for k in pmap_sax if "," in k}
    kf_ports = {k for k in pmap_kf if "," in k}
    assert sax_ports == kf_ports


def test_compile_netlist_accepts_kfnetlist(simple_lrc_kfnetlist):
    nl, models_map = simple_lrc_kfnetlist
    groups, sys_size, pmap = compile_netlist(nl, models_map)

    assert sys_size > 0
    assert len(groups) > 0
    assert "V1,p1" in pmap
    assert pmap["GND,p1"] == 0

    for g in groups.values():
        assert g.var_indices.shape[0] > 0


def test_compile_netlist_kfnetlist_dc_solve():
    """Full DC solve with kfnetlist-constructed netlist (no delay on source)."""
    from circulax.components.electronic import Capacitor, Inductor, Resistor, VoltageSource
    from circulax.solvers.linear import analyze_circuit

    models_map = {
        "resistor": Resistor,
        "capacitor": Capacitor,
        "inductor": Inductor,
        "source_voltage": VoltageSource,
    }

    nl = kfnl.Netlist()
    nl.create_inst(name="GND", kcl="", component="ground")
    nl.create_inst(name="V1", kcl="", component="source_voltage", settings={"V": 5.0})
    nl.create_inst(name="R1", kcl="", component="resistor", settings={"R": 10.0})

    gnd = kfnl.PortRef(instance="GND", port="p1")
    nl.create_net(gnd, kfnl.PortRef(instance="V1", port="p1"), kfnl.PortRef(instance="R1", port="p2"))
    nl.create_net(kfnl.PortRef(instance="V1", port="p2"), kfnl.PortRef(instance="R1", port="p1"))
    nl.sort()

    groups, sys_size, pmap = compile_netlist(nl, models_map)
    solver = analyze_circuit(groups, sys_size, backend="dense")

    y0 = jnp.zeros(sys_size)
    y_dc = solver.solve_dc(groups, y0)

    v_node = pmap["V1,p2"]
    assert jnp.abs(jnp.abs(y_dc[v_node]) - 5.0) < 0.1


def test_compile_netlist_ground_component_named_g0_solves():
    """Compilation skips component='ground' instances regardless of name."""
    from circulax.components.electronic import Resistor, VoltageSource
    from circulax.solvers.linear import analyze_circuit

    models_map = {
        "resistor": Resistor,
        "source_voltage": VoltageSource,
    }

    nl = kfnl.Netlist()
    nl.create_inst(name="g0", kcl="", component="ground")
    nl.create_inst(name="V1", kcl="", component="source_voltage", settings={"V": 2.0})
    nl.create_inst(name="R1", kcl="", component="resistor", settings={"R": 10.0})

    gnd = kfnl.PortRef(instance="g0", port="p1")
    nl.create_net(gnd, kfnl.PortRef(instance="V1", port="p1"))
    nl.create_net(kfnl.PortRef(instance="V1", port="p2"), kfnl.PortRef(instance="R1", port="p1"))
    nl.create_net(kfnl.PortRef(instance="R1", port="p2"), gnd)

    groups, sys_size, pmap = compile_netlist(nl, models_map)
    solver = analyze_circuit(groups, sys_size, backend="dense")
    y_dc = solver.solve_dc(groups, jnp.zeros(sys_size))

    assert pmap["g0,p1"] == 0
    assert jnp.isclose(jnp.abs(y_dc[pmap["R1,p1"]]), 2.0, atol=1e-6)


def test_sax_connections_reject_malformed_port_refs():
    with pytest.raises(ValueError, match="instance,port"):
        sax_to_kfnetlist(
            {
                "instances": {"R": {"component": "resistor"}},
                "ports": {"bad": "R"},
            }
        )
