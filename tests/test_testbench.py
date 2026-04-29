"""Tests for circulax.testbench.attach_testbench."""

import jax.numpy as jnp
import pytest

from circulax import attach_testbench, compile_circuit
from circulax.components.electronic import Resistor, VoltageSource


_MODELS = {
    "resistor": Resistor,
    "source_voltage": VoltageSource,
    "ground": lambda: 0,
}


def _device_with_connections() -> dict:
    """A simple 2-port resistor 'device' with SAX connections format."""
    return {
        "instances": {"R": {"component": "resistor", "settings": {"R": 100.0}}},
        "connections": {},
        "ports": {"a": "R,p1", "b": "R,p2"},
    }


def _device_with_nets() -> dict:
    """Same device, but using GDSFactory 'nets' list format internally."""
    return {
        "instances": {
            "Ra": {"component": "resistor", "settings": {"R": 50.0}},
            "Rb": {"component": "resistor", "settings": {"R": 50.0}},
        },
        "nets": [{"p1": "Ra,p2", "p2": "Rb,p1"}],
        "ports": {"a": "Ra,p1", "b": "Rb,p2"},
    }


def test_attach_source_and_load_solves():
    """Full chain: device → attach → compile → solve."""
    device = _device_with_connections()
    bench = attach_testbench(
        device,
        sources={"a": {"name": "V1", "component": "source_voltage", "settings": {"V": 5.0}}},
        loads={"b": {"name": "Rload", "component": "resistor", "settings": {"R": 100.0}}},
    )

    assert "V1" in bench["instances"]
    assert "Rload" in bench["instances"]
    assert "GND" in bench["instances"]
    assert "nets" in bench

    # 5 V across V1, R=100 series, Rload=100 → I = 5/200 = 25 mA, V_node = 2.5
    circuit = compile_circuit(bench, _MODELS)
    y = circuit()
    # The shared node between R and Rload should be at 2.5V (half of source)
    mid_idx = circuit.port_map["R,p2"]
    assert jnp.isclose(y[mid_idx], 2.5, atol=1e-6)


def test_gnd_termination_only():
    """A pure-GND-terminated load gives a known voltage divider."""
    device = _device_with_connections()
    bench = attach_testbench(
        device,
        sources={"a": {"name": "V1", "component": "source_voltage", "settings": {"V": 3.0}}},
        gnd=["b"],
    )
    circuit = compile_circuit(bench, _MODELS)
    y = circuit()
    # b = R,p2 is grounded; full 3V across the resistor; node V_R,p2 = 0.
    assert jnp.isclose(y[circuit.port_map["R,p2"]], 0.0, atol=1e-6)


def test_nets_format_passes_through():
    """Device with 'nets' list works the same as 'connections' dict."""
    device = _device_with_nets()
    bench = attach_testbench(
        device,
        sources={"a": {"component": "source_voltage", "settings": {"V": 4.0}}},
        gnd=["b"],
    )
    # auto-named source
    assert "src_a" in bench["instances"]
    # the original internal net should be carried over
    assert {"p1": "Ra,p2", "p2": "Rb,p1"} in bench["nets"]
    circuit = compile_circuit(bench, _MODELS)
    y = circuit()
    # 4V across Ra+Rb=100Ω, midpoint at 2V
    assert jnp.isclose(y[circuit.port_map["Ra,p2"]], 2.0, atol=1e-6)


def test_auto_naming():
    device = _device_with_connections()
    bench = attach_testbench(
        device,
        sources={"a": {"component": "source_voltage", "settings": {"V": 1.0}}},
        loads={"b": {"component": "resistor", "settings": {"R": 1.0}}},
    )
    assert "src_a" in bench["instances"]
    assert "load_b" in bench["instances"]


def test_duplicate_port_role_raises():
    device = _device_with_connections()
    with pytest.raises(ValueError, match="multiple roles"):
        attach_testbench(
            device,
            sources={"a": {"component": "source_voltage", "settings": {"V": 1.0}}},
            gnd=["a"],
        )


def test_unknown_port_raises():
    device = _device_with_connections()
    with pytest.raises(ValueError, match="not in device"):
        attach_testbench(device, gnd=["nonexistent"])


def test_name_collision_raises():
    device = _device_with_connections()
    with pytest.raises(ValueError, match="already exists"):
        attach_testbench(
            device,
            sources={"a": {"name": "R", "component": "source_voltage", "settings": {"V": 1.0}}},
        )


def test_existing_GND_preserved():
    device = _device_with_connections()
    device["instances"]["GND"] = {"component": "ground"}
    bench = attach_testbench(device, gnd=["a", "b"])
    # Only one GND instance, no collision
    gnd_count = sum(1 for k in bench["instances"] if k == "GND")
    assert gnd_count == 1
