"""Tests for kfnetlist integration: converter, build_net_map_kfnetlist, compile_netlist."""

import jax.numpy as jnp
import kfnetlist as kfnl

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
        "resistor": Resistor, "capacitor": Capacitor,
        "inductor": Inductor, "source_voltage": VoltageSource,
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
