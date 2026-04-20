"""Test OSDI resistor ladder against Python resistor ladder.

This test demonstrates that OSDI-based components produce identical results
to Python-defined components. It reproduces the resistor ladder example from
examples/electrical/dc/resistor_ladder.ipynb using OSDI resistors instead of
Python resistors.

The R-2R ladder network is a precision voltage divider where each successive
node voltage is half of the previous node due to the recursive equivalent
resistance property. This provides an excellent analytical benchmark.

Test Configuration:
  - Input voltage: 8.0 V
  - Series resistors (Rs): 1000 Ω
  - Shunt resistors (Rp): 2000 Ω (2*Rs)
  - 3-stage ladder with termination

Expected Results:
  - Node 1: 4.0 V (exactly V_ref / 2)
  - Node 2: 2.0 V (exactly V_ref / 4)
  - Node 3: 1.0 V (exactly V_ref / 8)
"""


import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax import compile_circuit
from circulax.components.electronic import Resistor, VoltageSource
from circulax.components.osdi_component import _BOSDI_AVAILABLE, osdi_component

pytestmark = pytest.mark.skipif(not _BOSDI_AVAILABLE, reason="bosdi package not available")


# Enable 64-bit precision for SPICE accuracy
jax.config.update("jax_enable_x64", True)


# ═════════════════════════════════════════════════════════════════════════════
# FIXTURES: Resistor ladder configuration
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def resistor_ladder_netlist():
    """R-2R ladder network configuration."""
    return {
        "instances": {
            "GND": {"component": "ground"},
            "V_REF": {"component": "source_voltage", "settings": {"V": 8.0}},
            "R_S1": {"component": "resistor", "settings": {"R": 1000.0}},
            "R_P1": {"component": "resistor", "settings": {"R": 2000.0}},
            "R_S2": {"component": "resistor", "settings": {"R": 1000.0}},
            "R_P2": {"component": "resistor", "settings": {"R": 2000.0}},
            "R_S3": {"component": "resistor", "settings": {"R": 1000.0}},
            "R_P3": {"component": "resistor", "settings": {"R": 2000.0}},
            "R_TERM": {"component": "resistor", "settings": {"R": 2000.0}},
        },
        "connections": {
            "GND,p1": ("V_REF,p2", "R_P1,p2", "R_P2,p2", "R_P3,p2", "R_TERM,p2"),
            "V_REF,p1": "R_S1,p1",
            "R_S1,p2": ("R_P1,p1", "R_S2,p1"),
            "R_S2,p2": ("R_P2,p1", "R_S3,p1"),
            "R_S3,p2": ("R_P3,p1", "R_TERM,p1"),
        },
    }


@pytest.fixture
def osdi_resistor_ladder_netlist():
    """R-2R ladder network using OSDI resistors instead of Python resistors."""
    return {
        "instances": {
            "GND": {"component": "ground"},
            "V_REF": {"component": "source_voltage", "settings": {"V": 8.0}},
            "R_S1": {"component": "osdi_resistor", "settings": {"R": 1000.0, "m": 1.0}},
            "R_P1": {"component": "osdi_resistor", "settings": {"R": 2000.0, "m": 1.0}},
            "R_S2": {"component": "osdi_resistor", "settings": {"R": 1000.0, "m": 1.0}},
            "R_P2": {"component": "osdi_resistor", "settings": {"R": 2000.0, "m": 1.0}},
            "R_S3": {"component": "osdi_resistor", "settings": {"R": 1000.0, "m": 1.0}},
            "R_P3": {"component": "osdi_resistor", "settings": {"R": 2000.0, "m": 1.0}},
            "R_TERM": {"component": "osdi_resistor", "settings": {"R": 2000.0, "m": 1.0}},
        },
        "connections": {
            "GND,p1": ("V_REF,p2", "R_P1,p2", "R_P2,p2", "R_P3,p2", "R_TERM,p2"),
            "V_REF,p1": "R_S1,p1",
            "R_S1,p2": ("R_P1,p1", "R_S2,p1"),
            "R_S2,p2": ("R_P2,p1", "R_S3,p1"),
            "R_S3,p2": ("R_P3,p1", "R_TERM,p1"),
        },
    }


@pytest.fixture
def osdi_resistor_class():
    """Create and cache the OSDI resistor component class.

    OSDI param ordering for resistor_va.osdi: [$mfactor(INST), R(MODEL)].
    """
    return osdi_component(
        "tests/resistor_va.osdi",
        port_names=("p1", "p2"),
        param_names=("m", "R"),
        param_defaults={"m": 1.0, "R": 1e3},
    )


# ═════════════════════════════════════════════════════════════════════════════
# TESTS
# ═════════════════════════════════════════════════════════════════════════════

class TestOsdiResistorLadder:
    """Test suite for OSDI resistor ladder implementation."""

    def test_python_resistor_ladder(self, resistor_ladder_netlist):
        """Baseline: Verify Python resistor ladder works correctly."""
        models_map = {
            "resistor": Resistor,
            "source_voltage": VoltageSource,
            "ground": lambda: 0,
        }

        circuit = compile_circuit(resistor_ladder_netlist, models_map)
        y_dc = circuit()

        # Extract node voltages
        v_n1 = float(circuit.get_port_field(y_dc, "R_S1,p2"))
        v_n2 = float(circuit.get_port_field(y_dc, "R_S2,p2"))
        v_n3 = float(circuit.get_port_field(y_dc, "R_S3,p2"))

        # Verify against expected values
        np.testing.assert_allclose(v_n1, 4.0, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(v_n2, 2.0, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(v_n3, 1.0, rtol=1e-6, atol=1e-10)

    def test_osdi_resistor_ladder(self, osdi_resistor_ladder_netlist, osdi_resistor_class):
        """Test OSDI resistor ladder with same topology and parameters."""
        models_map = {
            "osdi_resistor": osdi_resistor_class,
            "source_voltage": VoltageSource,
            "ground": lambda: 0,
        }

        circuit = compile_circuit(osdi_resistor_ladder_netlist, models_map)
        y_dc = circuit()

        # Extract node voltages
        v_n1 = float(circuit.get_port_field(y_dc, "R_S1,p2"))
        v_n2 = float(circuit.get_port_field(y_dc, "R_S2,p2"))
        v_n3 = float(circuit.get_port_field(y_dc, "R_S3,p2"))

        # Verify against expected values
        np.testing.assert_allclose(v_n1, 4.0, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(v_n2, 2.0, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(v_n3, 1.0, rtol=1e-6, atol=1e-10)

    def test_osdi_vs_python_resistor_ladder(
        self, resistor_ladder_netlist, osdi_resistor_ladder_netlist, osdi_resistor_class
    ):
        """Verify OSDI resistor ladder produces identical results to Python."""

        # Solve Python resistor ladder
        models_map_python = {
            "resistor": Resistor,
            "source_voltage": VoltageSource,
            "ground": lambda: 0,
        }
        circuit_python = compile_circuit(resistor_ladder_netlist, models_map_python)
        y_dc_python = circuit_python()

        v_n1_python = float(circuit_python.get_port_field(y_dc_python, "R_S1,p2"))
        v_n2_python = float(circuit_python.get_port_field(y_dc_python, "R_S2,p2"))
        v_n3_python = float(circuit_python.get_port_field(y_dc_python, "R_S3,p2"))

        # Solve OSDI resistor ladder
        models_map_osdi = {
            "osdi_resistor": osdi_resistor_class,
            "source_voltage": VoltageSource,
            "ground": lambda: 0,
        }
        circuit_osdi = compile_circuit(osdi_resistor_ladder_netlist, models_map_osdi)
        y_dc_osdi = circuit_osdi()

        v_n1_osdi = float(circuit_osdi.get_port_field(y_dc_osdi, "R_S1,p2"))
        v_n2_osdi = float(circuit_osdi.get_port_field(y_dc_osdi, "R_S2,p2"))
        v_n3_osdi = float(circuit_osdi.get_port_field(y_dc_osdi, "R_S3,p2"))

        # Compare results
        np.testing.assert_allclose(v_n1_osdi, v_n1_python, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(v_n2_osdi, v_n2_python, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(v_n3_osdi, v_n3_python, rtol=1e-10, atol=1e-12)

    def test_osdi_resistor_simple_dc(self, osdi_resistor_class):
        """Test basic OSDI resistor DC evaluation (Ohm's Law)."""
        models_map = {
            "resistor": osdi_resistor_class,
            "source_voltage": VoltageSource,
            "ground": lambda: 0,
        }

        # Simple circuit: V_source --[R=50Ω]-- GND
        # Expected: I = 1V / 50Ω = 0.02 A
        netlist = {
            "instances": {
                "GND": {"component": "ground"},
                "V_SRC": {"component": "source_voltage", "settings": {"V": 1.0}},
                "R": {"component": "resistor", "settings": {"R": 50.0, "m": 1.0}},
            },
            "connections": {
                "GND,p1": ("V_SRC,p2", "R,p2"),
                "V_SRC,p1": "R,p1",
            },
        }

        circuit = compile_circuit(netlist, models_map)
        y_dc = circuit()

        # Get voltage across resistor
        v_r = float(circuit.get_port_field(y_dc, "R,p1"))

        # Current through resistor: I = V / R = 1V / 50Ω = 0.02 A
        # By KCL at node: V_node = V_src - V_r
        np.testing.assert_allclose(v_r, 1.0, rtol=1e-6, atol=1e-10)


class TestOsdiComponentIntegration:
    """Test OSDI component integration with Circulax."""

    def test_osdi_resistor_instantiation(self, osdi_resistor_class):
        """Test that OSDI resistor can be instantiated with different parameters."""
        r1 = osdi_resistor_class(R=100.0, m=1.0)
        r2 = osdi_resistor_class(R=1000.0, m=2.0)

        assert float(r1.R) == 100.0
        assert float(r1.m) == 1.0
        assert float(r2.R) == 1000.0
        assert float(r2.m) == 2.0

    def test_osdi_resistor_ports_and_params(self, osdi_resistor_class):
        """Verify OSDI resistor has correct port and parameter metadata."""
        assert osdi_resistor_class.ports == ("p1", "p2")
        assert osdi_resistor_class.param_names == ("m", "R")
        assert osdi_resistor_class.states == ()

    def test_osdi_resistor_solver_call(self, osdi_resistor_class):
        """Test OSDI resistor solver_call() directly."""
        r = osdi_resistor_class(R=50.0, m=1.0)

        # State vector: [V_p1, V_p2]
        y = jnp.array([1.0, 0.0])

        # Call solver interface
        f_vec, q_vec = r.solver_call(0.0, y, r)

        # f_vec should contain terminal currents
        # I = (1.0 - 0.0) / 50 = 0.02 A
        # By KCL: I_p1 = 0.02, I_p2 = -0.02
        np.testing.assert_allclose(f_vec[0], 0.02, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(f_vec[1], -0.02, rtol=1e-6, atol=1e-10)

        # q_vec should be zero (resistor has no charge storage)
        np.testing.assert_allclose(q_vec, 0.0, atol=1e-12)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
