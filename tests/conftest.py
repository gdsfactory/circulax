import sys
from pathlib import Path

import jax
import kfnetlist as kfnl
import pytest

# Ensure project root is on sys.path so tests can import the local package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Shared fixtures for tests
jax.config.update("jax_enable_x64", True)


@pytest.fixture
def simple_lrc_netlist():
    """Returns (net_dict, models_map) for a small LRC example."""
    from circulax.components.electronic import (
        Capacitor,
        Inductor,
        Resistor,
        VoltageSource,
    )

    models_map = {
        "resistor": Resistor,
        "capacitor": Capacitor,
        "inductor": Inductor,
        "source_voltage": VoltageSource,
        "ground": lambda: 0,
    }

    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "V1": {
                "component": "source_voltage",
                "settings": {"V": 5.0, "delay": 0.2e-8},
            },
            "R1": {"component": "resistor", "settings": {"R": 10.0}},
            "C1": {"component": "capacitor", "settings": {"C": 1e-11}},
            "L1": {"component": "inductor", "settings": {"L": 5e-9}},
        },
        "connections": {
            "GND,p1": ("V1,p1", "C1,p2"),
            "V1,p2": "R1,p1",
            "R1,p2": "L1,p1",
            "L1,p2": "C1,p1",
        },
    }

    return net_dict, models_map


@pytest.fixture
def simple_optical_netlist():
    from circulax.components.electronic import Resistor
    from circulax.components.photonic import OpticalSourcePulse, OpticalWaveguide

    models_map = {
        "waveguide": OpticalWaveguide,
        "source": OpticalSourcePulse,
        "resistor": Resistor,
        "ground": lambda: 0,
    }
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "I1": {
                "component": "source",
                "settings": {"power": 1.0 + 0j, "delay": 0.1e-9},
            },
            "WG1": {"component": "waveguide", "settings": {"length_um": 100.0}},
            "R1": {
                "component": "resistor",
                "settings": {"R": 1.0},
            },  # circulax.components.Resistor defaults to 1k, we set 1.0
        },
        "connections": {
            "GND,p1": ("I1,p2", "R1,p2"),
            "I1,p1": "WG1,p1",
            "WG1,p2": "R1,p1",
        },
    }

    return net_dict, models_map


@pytest.fixture
def simple_lrc_kfnetlist():
    """Returns (kfnetlist.Netlist, models_map) for a small LRC example."""
    from circulax.components.electronic import (
        Capacitor,
        Inductor,
        Resistor,
        VoltageSource,
    )

    models_map = {
        "resistor": Resistor,
        "capacitor": Capacitor,
        "inductor": Inductor,
        "source_voltage": VoltageSource,
        "ground": lambda: 0,
    }

    nl = kfnl.Netlist()
    nl.create_inst(name="GND", kcl="", component="ground")
    nl.create_inst(name="V1", kcl="", component="source_voltage", settings={"V": 5.0, "delay": 0.2e-8})
    nl.create_inst(name="R1", kcl="", component="resistor", settings={"R": 10.0})
    nl.create_inst(name="C1", kcl="", component="capacitor", settings={"C": 1e-11})
    nl.create_inst(name="L1", kcl="", component="inductor", settings={"L": 5e-9})

    gnd = kfnl.PortRef(instance="GND", port="p1")
    nl.create_net(gnd, kfnl.PortRef(instance="V1", port="p1"), kfnl.PortRef(instance="C1", port="p2"))
    nl.create_net(kfnl.PortRef(instance="V1", port="p2"), kfnl.PortRef(instance="R1", port="p1"))
    nl.create_net(kfnl.PortRef(instance="R1", port="p2"), kfnl.PortRef(instance="L1", port="p1"))
    nl.create_net(kfnl.PortRef(instance="L1", port="p2"), kfnl.PortRef(instance="C1", port="p1"))

    nl.sort()
    return nl, models_map
