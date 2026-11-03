"""Independent electrical and power checks for the modal adapter."""

import jax.numpy as jnp
import numpy as np
from skrf.network import y2s

from circulax import compile_circuit
from circulax.components import DifferentialAdapter
from circulax.components.electronic import Capacitor, Resistor, VoltageSource


def test_loaded_adapter_matches_conductor_admittances() -> None:
    net = {
        "instances": {
            "X": {"component": "adapter"},
            "GND": {"component": "ground"},
            "RA": {"component": "R", "settings": {"R": 100.0}},
            "RB": {"component": "R", "settings": {"R": 200.0}},
            "CA": {"component": "C", "settings": {"C": 50e-15}},
            "CB": {"component": "C", "settings": {"C": 80e-15}},
        },
        "connections": {
            "RA,p1": "X,a",
            "RB,p1": "X,b",
            "CA,p1": "X,a",
            "CB,p1": "X,b",
            "RA,p2": "GND,p1",
            "RB,p2": "GND,p1",
            "CA,p2": "GND,p1",
            "CB,p2": "GND,p1",
        },
        "ports": {"d": "X,d", "c": "X,c"},
    }
    circuit = compile_circuit(net, {"adapter": DifferentialAdapter, "R": Resistor, "C": Capacitor}, g_leak=0)
    f = np.array([1e6, 10e9, 40e9])
    transform = np.array([[0.5, 1.0], [-0.5, 1.0]])
    y = np.stack(
        [transform.T @ np.diag([1 / 100 + 2j * np.pi * x * 50e-15, 1 / 200 + 2j * np.pi * x * 80e-15]) @ transform for x in f]
    )
    np.testing.assert_allclose(circuit.sp(ports=["d", "c"], freqs=jnp.asarray(f), z0=50), y2s(y, z0=50), atol=1e-8)


def test_dc_voltages_and_power_conservation() -> None:
    circuit = compile_circuit(
        {
            "instances": {
                "X": {"component": "adapter"},
                "D": {"component": "V", "settings": {"V": 1.2}},
                "CM": {"component": "V", "settings": {"V": -0.9}},
                "GND": {"component": "ground"},
                "RA": {"component": "R", "settings": {"R": 100}},
                "RB": {"component": "R", "settings": {"R": 200}},
            },
            "connections": {
                "D,p1": "X,d",
                "CM,p1": "X,c",
                "D,p2": "GND,p1",
                "CM,p2": "GND,p1",
                "RA,p1": "X,a",
                "RB,p1": "X,b",
                "RA,p2": "GND,p1",
                "RB,p2": "GND,p1",
            },
        },
        {"adapter": DifferentialAdapter, "V": VoltageSource, "R": Resistor},
        g_leak=0,
    )
    dc = circuit.dc()
    vd, vc, va, vb = [float(circuit.port(dc, "X," + p)) for p in ["d", "c", "a", "b"]]
    np.testing.assert_allclose([va, vb], [vc + vd / 2, vc - vd / 2], atol=1e-10)
    ia, ib = [float(circuit.port(dc, "X," + p)) for p in ["ia", "ib"]]
    np.testing.assert_allclose([ia, ib], [-va / 100, -vb / 200], atol=1e-10)
    power = -vd * (ia - ib) / 2 - vc * (ia + ib) + va * ia + vb * ib
    assert abs(power) < 1e-12
