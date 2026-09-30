"""Tests for circulax.netlist.prune_unreachable_instances (gdsfactory/circulax#60).

An instance that cannot influence any of the netlist's exposed ports (no
connection path, direct or transitive) must be dropped before model
resolution. Otherwise a structural/decorative cell with no simulation model
(a die frame, logo, marker rectangle, ...) aborts compilation with a
spurious "Model not found" error, even though excluding it cannot change any
simulation result.
"""

import kfnetlist as kfnl
import pytest

from circulax.compiler import compile_netlist
from circulax.netlist import prune_unreachable_instances


def _kfnl_lrc_with_port():
    """A working R-C divider exposing one top-level port, ``"out"``."""
    nl = kfnl.Netlist()
    nl.create_inst(name="GND", kcl="", component="ground")
    nl.create_inst(name="R1", kcl="", component="resistor", settings={"R": 100.0})
    nl.create_inst(name="C1", kcl="", component="capacitor", settings={"C": 1e-9})
    nl.create_port("out")

    gnd = kfnl.PortRef(instance="GND", port="p1")
    nl.create_net(kfnl.NetlistPort("out"), kfnl.PortRef(instance="R1", port="p1"))
    nl.create_net(kfnl.PortRef(instance="R1", port="p2"), kfnl.PortRef(instance="C1", port="p1"))
    nl.create_net(kfnl.PortRef(instance="C1", port="p2"), gnd)
    nl.sort()
    return nl


class TestPruneUnreachableInstances:
    def test_no_top_ports_is_a_noop(self):
        """With no exposed ports, every instance looks unreachable; skip entirely."""
        nl = kfnl.Netlist()
        nl.create_inst(name="GND", kcl="", component="ground")
        nl.create_inst(name="R1", kcl="", component="resistor", settings={"R": 100.0})
        nl.create_net(kfnl.PortRef(instance="GND", port="p1"), kfnl.PortRef(instance="R1", port="p1"))

        pruned = prune_unreachable_instances(nl)
        assert set(pruned.instances) == {"GND", "R1"}

    def test_drops_instance_with_no_ports_and_no_nets(self):
        """A structural cell with `"ports": []` and no nets never appears in any net."""
        nl = _kfnl_lrc_with_port()
        nl.create_inst(name="die_frame", kcl="", component="rectangle", settings={})

        pruned = prune_unreachable_instances(nl)
        assert set(pruned.instances) == {"GND", "R1", "C1"}

    def test_drops_floating_subcircuit_wired_only_to_itself(self):
        """Two instances wired to each other, with no path to any port, are unreachable."""
        nl = _kfnl_lrc_with_port()
        nl.create_inst(name="X1", kcl="", component="wg", settings={})
        nl.create_inst(name="X2", kcl="", component="wg", settings={})
        nl.create_net(kfnl.PortRef(instance="X1", port="p2"), kfnl.PortRef(instance="X2", port="p1"))

        pruned = prune_unreachable_instances(nl)
        assert set(pruned.instances) == {"GND", "R1", "C1"}

    def test_keeps_instance_reachable_only_transitively(self):
        """An instance two hops from the port (bridged through another instance) is kept."""
        nl = _kfnl_lrc_with_port()
        # C1 is already two hops from "out" (out -> R1 -> C1 -> GND); assert nothing
        # downstream of the port gets dropped just for being indirectly connected.
        pruned = prune_unreachable_instances(nl)
        assert set(pruned.instances) == {"GND", "R1", "C1"}

    def test_prunes_dangling_nets_left_behind(self):
        """Nets that only touched removed instances are cleaned up, not left empty."""
        nl = _kfnl_lrc_with_port()
        nl.create_inst(name="X1", kcl="", component="wg", settings={})
        nl.create_inst(name="X2", kcl="", component="wg", settings={})
        nl.create_net(kfnl.PortRef(instance="X1", port="p2"), kfnl.PortRef(instance="X2", port="p1"))

        pruned = prune_unreachable_instances(nl)
        assert all(len(list(net)) > 0 for net in pruned.nets)

    def test_original_netlist_is_not_mutated(self):
        nl = _kfnl_lrc_with_port()
        nl.create_inst(name="die_frame", kcl="", component="rectangle", settings={})

        prune_unreachable_instances(nl)
        assert "die_frame" in nl.instances

    def test_compile_netlist_drops_unmodeled_inert_instance(self):
        """The end-to-end case from gdsfactoryplus#4882/#4883: an unmodeled,
        topologically inert component must not abort compilation.
        """
        from circulax.components.electronic import Capacitor, Resistor

        nl = _kfnl_lrc_with_port()
        nl.create_inst(name="die_frame", kcl="", component="rectangle", settings={})
        models_map = {"resistor": Resistor, "capacitor": Capacitor, "ground": lambda: 0}

        groups, sys_size, pmap = compile_netlist(nl, models_map)

        assert sys_size > 0
        assert "rectangle" not in groups
        assert not any(key.startswith("die_frame,") for key in pmap)

    def test_compile_netlist_still_errors_on_reachable_unmodeled_instance(self):
        """The prune must not mask a genuine missing-model error for a live instance."""
        from circulax.components.electronic import Resistor

        nl = _kfnl_lrc_with_port()
        models_map = {"resistor": Resistor, "ground": lambda: 0}  # capacitor model missing

        with pytest.raises(ValueError, match="Model 'capacitor' not found"):
            compile_netlist(nl, models_map)

    def test_sax_dict_path_drops_inert_instance(self):
        """Same behaviour when compile_netlist is called with a SAX-format dict."""
        from circulax.components.electronic import Resistor

        sax_dict = {
            "instances": {
                "GND": {"component": "ground"},
                "R1": {"component": "resistor", "settings": {"R": 100.0}},
                "die_frame": {"component": "rectangle", "settings": {}},
            },
            "connections": {"GND,p1": "R1,p2"},
            "ports": {"out": "R1,p1"},
        }
        models_map = {"resistor": Resistor, "ground": lambda: 0}

        _groups, sys_size, pmap = compile_netlist(sax_dict, models_map)
        assert sys_size > 0
        assert not any(key.startswith("die_frame,") for key in pmap)

    def test_kept_instances_still_solve(self):
        """Pruning an inert cell doesn't disturb the DC solution of the live circuit."""
        import jax.numpy as jnp

        from circulax.components.electronic import Resistor, VoltageSource
        from circulax.solvers.linear import analyze_circuit

        nl = kfnl.Netlist()
        nl.create_inst(name="GND", kcl="", component="ground")
        nl.create_inst(name="V1", kcl="", component="source_voltage", settings={"V": 5.0})
        nl.create_inst(name="R1", kcl="", component="resistor", settings={"R": 10.0})
        nl.create_inst(name="die_frame", kcl="", component="rectangle", settings={})
        nl.create_port("out")

        gnd = kfnl.PortRef(instance="GND", port="p1")
        nl.create_net(kfnl.NetlistPort("out"), gnd, kfnl.PortRef(instance="V1", port="p1"), kfnl.PortRef(instance="R1", port="p2"))
        nl.create_net(kfnl.PortRef(instance="V1", port="p2"), kfnl.PortRef(instance="R1", port="p1"))
        nl.sort()

        models_map = {"resistor": Resistor, "source_voltage": VoltageSource, "ground": lambda: 0}
        groups, sys_size, pmap = compile_netlist(nl, models_map)
        solver = analyze_circuit(groups, sys_size, backend="dense")
        y_dc = solver.solve_dc(groups, jnp.zeros(sys_size))

        assert jnp.isclose(jnp.abs(y_dc[pmap["V1,p2"]]), 5.0, atol=1e-6)


class TestPruneUnreachableInstancesDictAndRecursive:
    """The dict-facing half of prune_unreachable_instances (gdsfactory/circulax#60).

    This is the entry point a consumer like gdsfactoryplus should call
    directly on its recursive netlist *before* its own composite-expansion /
    model-resolution walk — the same call site that previously needed a
    local ``_prune_inert_instances`` workaround
    (github.com/doplaydo/gdsfactoryplus/pull/4883).
    """

    def test_flat_dict_drops_inert_instance(self):
        sax_dict = {
            "instances": {
                "R1": {"component": "resistor", "settings": {"R": 100.0}},
                "die_frame": {"component": "rectangle", "settings": {}},
            },
            "connections": {},
            "ports": {"p1": "R1,p1"},
        }
        pruned = prune_unreachable_instances(sax_dict)
        assert set(pruned["instances"]) == {"R1"}

    def test_flat_dict_no_ports_is_a_noop(self):
        sax_dict = {
            "instances": {
                "R1": {"component": "resistor", "settings": {"R": 100.0}},
                "die_frame": {"component": "rectangle", "settings": {}},
            },
            "connections": {},
        }
        pruned = prune_unreachable_instances(sax_dict)
        assert set(pruned["instances"]) == {"R1", "die_frame"}

    def test_recursive_netlist_drops_top_level_inert_instance(self):
        """The exact shape of gdsfactoryplus#4882: a top circuit with a
        connected `mmi` and an unmodeled, unconnected `die_half`.
        """
        recnet = {
            "top": {
                "instances": {
                    "X1": {"component": "mmi", "settings": {}},
                    "X6": {"component": "die_half", "settings": {"size": [5190, 4850]}},
                },
                "ports": {"P1": "X1,p1"},
            },
        }
        pruned = prune_unreachable_instances(recnet)
        assert set(pruned["top"]["instances"]) == {"X1"}

    def test_recursive_netlist_drops_nested_inert_instance(self):
        """An inert cell nested inside a reachable composite is dropped too."""
        recnet = {
            "top": {
                "instances": {"C1": {"component": "comp", "settings": {}}},
                "connections": {},
                "ports": {"P1": "C1,p1"},
            },
            "comp": {
                "instances": {
                    "wg1": {"component": "wg", "settings": {}},
                    "X6": {"component": "die_half", "settings": {"size": [1, 1]}},
                },
                "connections": {},
                "ports": {"p1": "wg1,p1"},
            },
        }
        pruned = prune_unreachable_instances(recnet)
        assert set(pruned["comp"]["instances"]) == {"wg1"}

    def test_recursive_netlist_no_top_ports_is_a_noop(self):
        """With no top-level ports, every instance would look unreachable;
        skip the whole recursive walk rather than emptying an
        internally-driven circuit.
        """
        recnet = {
            "top": {
                "instances": {
                    "V1": {"component": "source_voltage", "settings": {}},
                    "R1": {"component": "resistor", "settings": {}},
                },
                "connections": {"V1,p1": "R1,p1", "V1,p2": "R1,p2"},
            },
        }
        pruned = prune_unreachable_instances(recnet)
        assert set(pruned["top"]["instances"]) == {"V1", "R1"}

    def test_guard_uses_declared_ports_not_synthetic_hierarchy_stubs(self):
        """sax_to_kfnetlist synthesizes a top-level port for any connection
        that targets an unknown instance name (used for cross-hierarchy net
        labels like "vdd,p1"). The no-declared-ports guard must be evaluated
        on the *original* dict, before that synthesis, or a circuit with no
        real "ports" key would still get pruned against those synthetic
        labels (gdsfactory/circulax#60 review).
        """
        sax_dict = {
            "instances": {
                "V1": {"component": "source_voltage", "settings": {"V": 1.2}},
                "R1": {"component": "resistor", "settings": {"R": 100.0}},
                "GND": {"component": "ground"},
                "die_frame": {"component": "rectangle", "settings": {}},
            },
            "connections": {
                "V1,p1": "vdd,p1",
                "V1,p2": "GND,p1",
                "R1,p1": "vdd,p1",
                "R1,p2": "GND,p1",
            },
        }
        pruned = prune_unreachable_instances(sax_dict)
        assert set(pruned["instances"]) == {"V1", "R1", "GND", "die_frame"}

    def test_compile_netlist_guard_matches_dict_level_guard(self):
        """compile_netlist must apply the same no-declared-ports guard as
        calling prune_unreachable_instances directly on the dict — pruning
        must happen before sax_to_kfnetlist's synthetic port creation, not
        after.
        """
        from circulax.components.electronic import Resistor, VoltageSource

        sax_dict = {
            "instances": {
                "V1": {"component": "source_voltage", "settings": {"V": 1.2}},
                "R1": {"component": "resistor", "settings": {"R": 100.0}},
                "GND": {"component": "ground"},
            },
            "connections": {
                "V1,p1": "vdd,p1",
                "V1,p2": "GND,p1",
                "R1,p1": "vdd,p1",
                "R1,p2": "GND,p1",
            },
        }
        models_map = {"resistor": Resistor, "source_voltage": VoltageSource, "ground": lambda: 0}

        _groups, _sys_size, pmap = compile_netlist(sax_dict, models_map)
        assert "R1,p1" in pmap
        assert "V1,p1" in pmap

    def test_drops_dangling_connection_off_an_unreachable_hub_label(self):
        """A connection sourced from a hierarchy-stub hub label (not a real
        instance, e.g. "n1,p1") must be dropped by the *root* it belongs to,
        not by checking whether the source string happens to name an
        instance. Otherwise the dangling entry survives pruning, and
        sax_to_kfnetlist later turns it into a synthetic port for an
        instance that no longer exists.
        """
        from circulax.components.electronic import Resistor

        sax_dict = {
            "instances": {
                "GND": {"component": "ground"},
                "R1": {"component": "resistor", "settings": {"R": 100.0}},
                "X1": {"component": "resistor", "settings": {"R": 50.0}},
                "X2": {"component": "resistor", "settings": {"R": 50.0}},
            },
            "connections": {
                "GND,p1": "R1,p2",
                "n1,p1": ("X1,p1", "X2,p1"),
            },
            "ports": {"out": "R1,p1"},
        }

        pruned = prune_unreachable_instances(sax_dict)
        assert set(pruned["instances"]) == {"GND", "R1"}
        assert "n1,p1" not in pruned["connections"]

        models_map = {"resistor": Resistor, "ground": lambda: 0}
        _groups, _sys_size, pmap = compile_netlist(sax_dict, models_map)
        assert not any(key.startswith("X1,") or key.startswith("X2,") for key in pmap)

    def test_tuple_target_connections_do_not_crash(self):
        """circulax's own `connections` extension allows a tuple of targets
        for one shared net (`"GND,p1": ("V1,p1", "R1,p2")`). A dict-level
        prune that shells out to sax.netlists.remove_unused_instances chokes
        on this (`AttributeError: 'tuple' object has no attribute 'split'`);
        the native implementation must handle it directly.
        """
        sax_dict = {
            "instances": {
                "GND": {"component": "ground"},
                "V1": {"component": "source_voltage", "settings": {"V": 5.0}},
                "R1": {"component": "resistor", "settings": {"R": 10.0}},
                "die_frame": {"component": "rectangle", "settings": {}},
            },
            "connections": {"GND,p1": ("V1,p1", "R1,p2")},
            "ports": {"out": "V1,p2"},
        }
        pruned = prune_unreachable_instances(sax_dict)
        assert set(pruned["instances"]) == {"GND", "V1", "R1"}

    def test_tuple_target_connections_via_compile_netlist(self):
        from circulax.components.electronic import Resistor, VoltageSource

        sax_dict = {
            "instances": {
                "GND": {"component": "ground"},
                "V1": {"component": "source_voltage", "settings": {"V": 5.0}},
                "R1": {"component": "resistor", "settings": {"R": 10.0}},
                "die_frame": {"component": "rectangle", "settings": {}},
            },
            "connections": {"GND,p1": ("V1,p1", "R1,p2"), "V1,p2": "R1,p1"},
            "ports": {"out": "V1,p2"},
        }
        models_map = {"resistor": Resistor, "source_voltage": VoltageSource, "ground": lambda: 0}
        _groups, sys_size, pmap = compile_netlist(sax_dict, models_map)
        assert sys_size > 0
        assert not any(key.startswith("die_frame,") for key in pmap)

    def test_recursive_subnet_with_no_ports_is_emptied_not_skipped(self):
        """Once the top-level guard passes, per-subnet pruning has no further
        guard: a subnet with no ports of its own is emptied entirely, matching
        sax.netlists.remove_unused_instances (not left untouched).
        """
        recnet = {
            "top": {
                "instances": {"C1": {"component": "comp", "settings": {}}},
                "connections": {},
                "ports": {"P1": "C1,p1"},
            },
            "comp": {
                # No "ports" key at all on this subcircuit body.
                "instances": {"wg1": {"component": "wg", "settings": {}}},
                "connections": {},
            },
        }
        pruned = prune_unreachable_instances(recnet)
        assert pruned["comp"]["instances"] == {}


class TestAttachTestbenchPrunesInertInstances:
    """attach_testbench is where circulax last sees the device's own declared
    ports before wrapping it into a portless, source/load-terminated netlist
    (gdsfactory/circulax#60): pruning must happen here, not only inside
    compile_netlist, or it never fires for the canonical
    attach_testbench -> compile_circuit pipeline.
    """

    @classmethod
    def _models(cls) -> dict:
        from circulax.components.electronic import Resistor, VoltageSource

        return {"resistor": Resistor, "source_voltage": VoltageSource, "ground": lambda: 0}

    def test_kfnetlist_device_drops_inert_instance_before_wrapping(self):
        from circulax import attach_testbench, compile_circuit

        device = kfnl.Netlist()
        device.create_inst(name="R1", kcl="", component="resistor", settings={"R": 100.0})
        device.create_inst(name="die_frame", kcl="", component="rectangle", settings={})
        device.create_port("a")
        device.create_port("b")
        device.create_net(kfnl.NetlistPort("a"), kfnl.PortRef(instance="R1", port="p1"))
        device.create_net(kfnl.NetlistPort("b"), kfnl.PortRef(instance="R1", port="p2"))
        device.sort()

        wired = attach_testbench(
            device,
            sources={"a": {"name": "V1", "component": "source_voltage", "settings": {"V": 1.0}}},
            gnd=["b"],
        )

        circuit = compile_circuit(wired, self._models())
        assert circuit is not None

    def test_sax_dict_device_drops_inert_instance_before_wrapping(self):
        from circulax import attach_testbench, compile_circuit

        device = {
            "instances": {
                "R1": {"component": "resistor", "settings": {"R": 100.0}},
                "die_frame": {"component": "rectangle", "settings": {}},
            },
            "connections": {},
            "ports": {"a": "R1,p1", "b": "R1,p2"},
        }
        wired = attach_testbench(
            device,
            sources={"a": {"name": "V1", "component": "source_voltage", "settings": {"V": 1.0}}},
            gnd=["b"],
        )

        circuit = compile_circuit(wired, self._models())
        assert circuit is not None
