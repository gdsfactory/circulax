"""Tests for hierarchical subcircuit composition (RecursiveNetlist flattening)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from circulax import attach_testbench, compile_circuit
from circulax.netlist import _is_recursive_netlist, flatten_recursive_netlist

jax.config.update("jax_enable_x64", True)


def _leaf_models() -> dict:
    from circulax.components.electronic import Capacitor, Inductor, Resistor, VoltageSource

    return {
        "Resistor": Resistor,
        "Capacitor": Capacitor,
        "Inductor": Inductor,
        "VDC": VoltageSource,
        "ground": lambda: 0,
    }


class TestIsRecursiveNetlist:
    """Detection of RecursiveNetlist vs flat Netlist."""

    def test_flat_netlist(self) -> None:
        """A dict with 'instances' is a flat netlist."""
        flat = {"instances": {"R1": {"component": "R"}}, "connections": {}}
        assert not _is_recursive_netlist(flat)

    def test_recursive_netlist(self) -> None:
        """A dict-of-dicts where values have 'instances' is recursive."""
        recnet = {
            "top": {"instances": {"R1": {"component": "sub"}}, "connections": {}},
            "sub": {"instances": {"R1": {"component": "R"}}, "ports": {"p1": "R1,p1"}},
        }
        assert _is_recursive_netlist(recnet)

    def test_empty_dict(self) -> None:
        """An empty dict is not recursive."""
        assert not _is_recursive_netlist({})


class TestFlattenRecursiveNetlist:
    """Unit tests for the flattening algorithm."""

    def test_no_subcircuits(self) -> None:
        """A RecursiveNetlist with only leaf instances passes through unchanged."""
        recnet = {
            "top": {
                "instances": {
                    "R1": {"component": "Resistor", "settings": {"R": 100.0}},
                },
                "connections": {},
            },
        }
        flat = flatten_recursive_netlist(recnet)
        assert "R1" in flat["instances"]
        assert len(flat["instances"]) == 1

    def test_basic_flattening(self) -> None:
        """Subcircuit instances are inlined with prefixed names."""
        recnet = {
            "top": {
                "instances": {
                    "SC1": {"component": "sub"},
                    "GND": {"component": "ground"},
                },
                "connections": {"SC1,p1": "GND,p1"},
            },
            "sub": {
                "instances": {
                    "R1": {"component": "Resistor", "settings": {"R": 50.0}},
                },
                "connections": {},
                "ports": {"p1": "R1,p1", "p2": "R1,p2"},
            },
        }
        flat = flatten_recursive_netlist(recnet)
        assert "SC1" not in flat["instances"]
        assert "SC1~R1" in flat["instances"]
        assert flat["instances"]["SC1~R1"]["settings"]["R"] == 50.0

    def test_connection_rewriting(self) -> None:
        """Parent connections to subcircuit ports are rewritten to internal refs."""
        recnet = {
            "top": {
                "instances": {
                    "SC1": {"component": "sub"},
                    "R_ext": {"component": "Resistor"},
                },
                "connections": {"SC1,p2": "R_ext,p1"},
            },
            "sub": {
                "instances": {
                    "R1": {"component": "Resistor"},
                },
                "connections": {},
                "ports": {"p1": "R1,p1", "p2": "R1,p2"},
            },
        }
        flat = flatten_recursive_netlist(recnet)
        assert "SC1~R1,p2" in flat["connections"] or any("SC1~R1,p2" in str(v) for v in flat["connections"].values())

    def test_tuple_targets(self) -> None:
        """Circulax tuple-target connections are rewritten correctly."""
        recnet = {
            "top": {
                "instances": {
                    "SC1": {"component": "sub"},
                    "GND": {"component": "ground"},
                    "V1": {"component": "VDC"},
                },
                "connections": {"GND,p1": ("V1,p1", "SC1,p2")},
            },
            "sub": {
                "instances": {"R1": {"component": "Resistor"}},
                "connections": {},
                "ports": {"p1": "R1,p1", "p2": "R1,p2"},
            },
        }
        flat = flatten_recursive_netlist(recnet)
        gnd_targets = flat["connections"]["GND,p1"]
        assert isinstance(gnd_targets, tuple)
        assert "SC1~R1,p2" in gnd_targets

    def test_nested_subcircuits(self) -> None:
        """Subcircuits within subcircuits are flattened recursively."""
        recnet = {
            "top": {
                "instances": {"outer": {"component": "level1"}},
                "connections": {},
            },
            "level1": {
                "instances": {"inner": {"component": "level2"}},
                "connections": {},
                "ports": {"p1": "inner,p1"},
            },
            "level2": {
                "instances": {"R1": {"component": "Resistor"}},
                "connections": {},
                "ports": {"p1": "R1,p1"},
            },
        }
        flat = flatten_recursive_netlist(recnet)
        assert "outer~inner~R1" in flat["instances"]

    def test_6_level_deep_nesting(self) -> None:
        """Six levels of nesting flatten correctly — beyond SAX circuit() depth."""
        recnet = {
            "top": {
                "instances": {"a": {"component": "L1"}},
                "connections": {},
                "ports": {"p1": "a,p1", "p2": "a,p2"},
            },
        }
        for depth in range(1, 7):
            parent_key = f"L{depth}"
            child_key = f"L{depth + 1}" if depth < 6 else "Resistor"
            recnet[parent_key] = {
                "instances": {"x": {"component": child_key}},
                "connections": {},
                "ports": {"p1": "x,p1", "p2": "x,p2"},
            }
        flat = flatten_recursive_netlist(recnet)
        expected = "a~x~x~x~x~x~x"
        assert expected in flat["instances"]
        assert flat["instances"][expected]["component"] == "Resistor"
        assert len(flat["instances"]) == 1

    def test_6_level_deep_nesting_dc_solve(self) -> None:
        """Six-level nested subcircuit compiles and solves correctly."""
        models = _leaf_models()
        recnet = {
            "top": {
                "instances": {
                    "chain": {"component": "L1"},
                    "V1": {"component": "VDC", "settings": {"V": 1.0}},
                    "GND": {"component": "ground"},
                },
                "connections": {
                    "V1,p2": "chain,p1",
                    "GND,p1": ("V1,p1", "chain,p2"),
                },
            },
        }
        for depth in range(1, 7):
            child = f"L{depth + 1}" if depth < 6 else "Resistor"
            recnet[f"L{depth}"] = {
                "instances": {"x": {"component": child, "settings": {"R": 100.0}} if child == "Resistor" else {"component": child}},
                "connections": {},
                "ports": {"p1": "x,p1", "p2": "x,p2"},
            }
        circuit = compile_circuit(recnet, models)
        y = circuit.dc()
        v = circuit.port(y, "chain~x~x~x~x~x~x,p1")
        assert jnp.isclose(jnp.abs(v), 1.0, rtol=1e-3)

    def test_ground_not_prefixed(self) -> None:
        """GND instances inside subcircuits are not prefixed."""
        recnet = {
            "top": {
                "instances": {"SC1": {"component": "sub"}},
                "connections": {},
            },
            "sub": {
                "instances": {
                    "R1": {"component": "Resistor"},
                    "GND": {"component": "ground"},
                },
                "connections": {"R1,p2": "GND,p1"},
                "ports": {"p1": "R1,p1"},
            },
        }
        flat = flatten_recursive_netlist(recnet)
        assert "GND" in flat["instances"]
        assert "SC1~GND" not in flat["instances"]

    def test_multiple_instances_of_same_subcircuit(self) -> None:
        """Two instances of the same subcircuit get distinct prefixes."""
        recnet = {
            "top": {
                "instances": {
                    "A": {"component": "sub"},
                    "B": {"component": "sub"},
                },
                "connections": {"A,p2": "B,p1"},
            },
            "sub": {
                "instances": {"R1": {"component": "Resistor"}},
                "connections": {},
                "ports": {"p1": "R1,p1", "p2": "R1,p2"},
            },
        }
        flat = flatten_recursive_netlist(recnet)
        assert "A~R1" in flat["instances"]
        assert "B~R1" in flat["instances"]
        assert "A" not in flat["instances"]
        assert "B" not in flat["instances"]

    def test_parent_ports_rewritten(self) -> None:
        """Top-level ports referencing a subcircuit are rewritten."""
        recnet = {
            "top": {
                "instances": {"SC1": {"component": "sub"}},
                "connections": {},
                "ports": {"out": "SC1,p2"},
            },
            "sub": {
                "instances": {"R1": {"component": "Resistor"}},
                "connections": {},
                "ports": {"p1": "R1,p1", "p2": "R1,p2"},
            },
        }
        flat = flatten_recursive_netlist(recnet)
        assert flat["ports"]["out"] == "SC1~R1,p2"

    def test_nets_list_rewriting(self) -> None:
        """The 'nets' list format is rewritten correctly."""
        recnet = {
            "top": {
                "instances": {"SC1": {"component": "sub"}},
                "connections": {},
            },
            "sub": {
                "instances": {
                    "R1": {"component": "Resistor"},
                    "R2": {"component": "Resistor"},
                },
                "nets": [{"p1": "R1,p1", "p2": "R2,p1"}],
                "ports": {"p1": "R1,p1", "p2": "R2,p2"},
            },
        }
        flat = flatten_recursive_netlist(recnet)
        assert any(n["p1"] == "SC1~R1,p1" and n["p2"] == "SC1~R2,p1" for n in flat.get("nets", []))


class TestCompileCircuitRecursive:
    """End-to-end tests: compile_circuit with RecursiveNetlist."""

    def test_parallel_resistors_dc(self) -> None:
        """Two resistors in parallel via RecursiveNetlist, DC solve."""
        models = _leaf_models()
        recnet = {
            "top": {
                "instances": {
                    "RP1": {"component": "parallel_R"},
                    "V1": {"component": "VDC", "settings": {"V": 1.0}},
                    "GND": {"component": "ground"},
                },
                "connections": {
                    "V1,p2": "RP1,p1",
                    "GND,p1": ("V1,p1", "RP1,p2"),
                },
            },
            "parallel_R": {
                "instances": {
                    "R1": {"component": "Resistor", "settings": {"R": 100.0}},
                    "R2": {"component": "Resistor", "settings": {"R": 100.0}},
                },
                "connections": {"R1,p1": "R2,p1", "R1,p2": "R2,p2"},
                "ports": {"p1": "R1,p1", "p2": "R1,p2"},
            },
        }
        circuit = compile_circuit(recnet, models)
        y = circuit.dc()
        v_node = circuit.port(y, "RP1~R1,p1")
        expected_I = 1.0 / 50.0
        v_across = jnp.abs(v_node)
        i_total = v_across / 50.0
        assert jnp.isclose(i_total, expected_I, rtol=1e-3)

    def test_subcircuit_with_state_transient(self) -> None:
        """Subcircuit containing an Inductor (stateful), transient analysis."""
        models = _leaf_models()
        recnet = {
            "top": {
                "instances": {
                    "SC1": {"component": "rl_series"},
                    "V1": {"component": "VDC", "settings": {"V": 5.0, "delay": 0.0}},
                    "GND": {"component": "ground"},
                },
                "connections": {
                    "V1,p2": "SC1,p1",
                    "GND,p1": ("V1,p1", "SC1,p2"),
                },
            },
            "rl_series": {
                "instances": {
                    "R1": {"component": "Resistor", "settings": {"R": 10.0}},
                    "L1": {"component": "Inductor", "settings": {"L": 1e-6}},
                },
                "connections": {"R1,p2": "L1,p1"},
                "ports": {"p1": "R1,p1", "p2": "L1,p2"},
            },
        }
        circuit = compile_circuit(recnet, models)
        y0 = circuit.dc()
        sol = circuit.transient(
            t0=0,
            t1=1e-4,
            dt0=1e-7,
            y0=y0,
            saveat=jnp.linspace(0.0, 1e-4, 4),
            max_steps=2000,
        )
        assert sol.ys.shape[0] == 4
        assert jnp.isfinite(sol.ys).all()

    def test_ports_property(self) -> None:
        """Circuit.ports returns external port names from source netlist."""
        models = _leaf_models()
        netlist = {
            "instances": {
                "R1": {"component": "Resistor", "settings": {"R": 100.0}},
            },
            "connections": {},
            "ports": {"p1": "R1,p1", "p2": "R1,p2"},
        }
        circuit = compile_circuit(netlist, models)
        assert set(circuit.ports) == {"p1", "p2"}

    def test_ports_property_empty(self) -> None:
        """Circuit.ports is empty when no ports declared."""
        models = _leaf_models()
        netlist = {
            "instances": {
                "R1": {"component": "Resistor", "settings": {"R": 100.0}},
                "GND": {"component": "ground"},
            },
            "connections": {"R1,p1": "GND,p1", "R1,p2": "GND,p1"},
        }
        circuit = compile_circuit(netlist, models)
        assert circuit.ports == ()


class TestCircuitInModelsMap:
    """Tests for passing a compiled Circuit as a value in models_map."""

    def test_circuit_as_subcircuit(self) -> None:
        """A compiled Circuit can be used as a subcircuit in a parent."""
        models = _leaf_models()

        sub_netlist = {
            "instances": {
                "R1": {"component": "Resistor", "settings": {"R": 100.0}},
                "R2": {"component": "Resistor", "settings": {"R": 100.0}},
            },
            "connections": {"R1,p1": "R2,p1", "R1,p2": "R2,p2"},
            "ports": {"p1": "R1,p1", "p2": "R1,p2"},
        }
        sub_circuit = compile_circuit(sub_netlist, models)

        parent_netlist = {
            "instances": {
                "RP1": {"component": "parallel_R"},
                "V1": {"component": "VDC", "settings": {"V": 1.0}},
                "GND": {"component": "ground"},
            },
            "connections": {
                "V1,p2": "RP1,p1",
                "GND,p1": ("V1,p1", "RP1,p2"),
            },
        }
        parent_models = {**models, "parallel_R": sub_circuit}
        parent = compile_circuit(parent_netlist, parent_models)

        y = parent.dc()
        v_node = parent.port(y, "RP1~R1,p1")
        assert jnp.abs(v_node) > 0.0

    def test_circuit_without_source_raises(self) -> None:
        """A Circuit with no stored source netlist cannot be used as subcircuit."""
        models = _leaf_models()
        netlist = {
            "instances": {
                "R1": {"component": "Resistor"},
                "GND": {"component": "ground"},
            },
            "connections": {"R1,p1": "GND,p1", "R1,p2": "GND,p1"},
        }
        dummy = compile_circuit(netlist, models)
        object.__setattr__(dummy, "_source_netlist", None)

        parent_netlist = {
            "instances": {"SC1": {"component": "sub"}},
            "connections": {},
        }
        with pytest.raises(ValueError, match="no stored source netlist"):
            compile_circuit(parent_netlist, {"sub": dummy, "ground": lambda: 0})

    def test_model_name_collision_same_object(self) -> None:
        """Same model name mapping to same object merges silently."""
        models = _leaf_models()
        sub_netlist = {
            "instances": {"R1": {"component": "Resistor"}},
            "connections": {},
            "ports": {"p1": "R1,p1", "p2": "R1,p2"},
        }
        sub = compile_circuit(sub_netlist, models)

        parent_netlist = {
            "instances": {
                "SC1": {"component": "sub"},
                "GND": {"component": "ground"},
            },
            "connections": {"SC1,p1": "GND,p1", "SC1,p2": "GND,p1"},
        }
        circuit = compile_circuit(parent_netlist, {**models, "sub": sub})
        y = circuit.dc()
        assert y is not None

    def test_model_name_collision_different_object_raises(self) -> None:
        """Different objects under the same model name raises ValueError."""
        from circulax.components.electronic import Capacitor, Resistor

        sub_models = {"R": Resistor, "ground": lambda: 0}
        sub_netlist = {
            "instances": {"R1": {"component": "R"}},
            "connections": {},
            "ports": {"p1": "R1,p1", "p2": "R1,p2"},
        }
        sub = compile_circuit(sub_netlist, sub_models)

        parent_netlist = {
            "instances": {"SC1": {"component": "sub"}},
            "connections": {},
        }
        parent_models = {"R": Capacitor, "sub": sub, "ground": lambda: 0}
        with pytest.raises(ValueError, match="Model name conflict"):
            compile_circuit(parent_netlist, parent_models)


@pytest.mark.parametrize("nested", [False, True])
def test_attach_testbench_to_compiled_subcircuit(*, nested: bool) -> None:
    """Parent nets terminate compiled subcircuits, including nested packages."""
    models = _leaf_models()
    leaf = compile_circuit(
        {"instances": {"R": {"component": "Resistor", "settings": {"R": 100.0}}}, "ports": {"p1": "R,p1", "p2": "R,p2"}},
        models,
        backend="dense",
    )
    if nested:
        leaf = compile_circuit(
            {"instances": {"INNER": {"component": "leaf"}}, "ports": {"p1": "INNER,p1", "p2": "INNER,p2"}},
            {**models, "leaf": leaf},
            backend="dense",
        )
    device = {
        "instances": {"A": {"component": "package"}, "B": {"component": "package"}},
        "nets": [{"p1": "A,p2", "p2": "B,p1"}],
        "ports": {"in": "A,p1", "out": "B,p2"},
    }
    bench = attach_testbench(
        device,
        sources={"in": {"name": "V1", "component": "VDC", "settings": {"V": 3.0}}},
        loads={"out": {"name": "LOAD", "component": "Resistor", "settings": {"R": 100.0}}},
    )
    circuit = compile_circuit(bench, {**models, "package": leaf}, backend="dense")
    op = circuit.dc()
    # Two series 100-ohm packages and a 100-ohm load divide 3 V into thirds.
    assert jnp.isclose(circuit.port(op, "LOAD,p1"), 1.0, rtol=1e-6)
    assert jnp.isclose(circuit.port(op, "V1,i_src"), -0.01, rtol=1e-6)
    # Flattening must not mutate the original device or compiled package.
    assert device["nets"] == [{"p1": "A,p2", "p2": "B,p1"}]
    assert leaf.ports == ("p1", "p2")
