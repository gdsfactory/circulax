"""Basic unit tests for OSDI component class without solver integration.

Tests the OsdiComponent class and osdi_component() factory function
at the unit level, without requiring circuit compilation.

Note: These tests must be run in an environment where BODI's JAX version
matches Circulax's JAX version. For now, we test the class structure
and factory without instantiating the solver_call()."""

import pytest
from circulax.components.osdi_component import _BOSDI_AVAILABLE, osdi_component, OsdiComponent

pytestmark = pytest.mark.skipif(not _BOSDI_AVAILABLE, reason="bosdi package not available")


class TestOsdiComponentFactory:
    """Test the osdi_component() factory function."""

    def test_osdi_component_creation(self):
        """Test that osdi_component() factory creates a valid class."""
        OsdiResistor = osdi_component(
            "tests/resistor_va.osdi",
            port_names=("p1", "p2"),
            param_names=("R", "m"),
            param_defaults={"R": 1e3, "m": 1.0},
        )

        # Verify class attributes
        assert issubclass(OsdiResistor, OsdiComponent)
        assert OsdiResistor.ports == ("p1", "p2")
        assert OsdiResistor.param_names == ("R", "m")
        assert OsdiResistor.states == ()
        assert OsdiResistor.osdi_model is not None

    def test_osdi_component_metadata(self):
        """Test that OSDI model metadata is correctly extracted."""
        OsdiResistor = osdi_component(
            "tests/resistor_va.osdi",
            port_names=("p1", "p2"),
            param_names=("R", "m"),
        )

        model = OsdiResistor.osdi_model
        assert model.num_pins == 2
        assert model.num_params == 2
        assert model.num_states == 0
        assert model.osdi_version == "0.4"

    def test_osdi_component_instantiation(self):
        """Test that OSDI component can be instantiated with parameters."""
        OsdiResistor = osdi_component(
            "tests/resistor_va.osdi",
            port_names=("p1", "p2"),
            param_names=("R", "m"),
            param_defaults={"R": 1e3, "m": 1.0},
        )

        # Create instances with different parameters
        r1 = OsdiResistor(R=100.0, m=1.0)
        r2 = OsdiResistor(R=1000.0, m=2.0)

        assert float(r1.R) == 100.0
        assert float(r1.m) == 1.0
        assert float(r2.R) == 1000.0
        assert float(r2.m) == 2.0

    def test_osdi_component_default_port_names(self):
        """Test that default port names are generated correctly."""
        OsdiResistor = osdi_component(
            "tests/resistor_va.osdi",
            param_names=("R", "m"),
        )

        # Should default to p0, p1, ... for 2-terminal device
        assert OsdiResistor.ports == ("p0", "p1")

    def test_osdi_component_default_param_names(self):
        """Test that default parameter names are generated correctly."""
        OsdiResistor = osdi_component(
            "tests/resistor_va.osdi",
            port_names=("p1", "p2"),
        )

        # Should default to param_0, param_1, ... for 2-param device
        assert OsdiResistor.param_names == ("param_0", "param_1")

    def test_osdi_component_default_param_values(self):
        """Test that default parameter values default to 1.0."""
        OsdiResistor = osdi_component(
            "tests/resistor_va.osdi",
            port_names=("p1", "p2"),
        )

        r = OsdiResistor()
        assert float(r.param_0) == 1.0
        assert float(r.param_1) == 1.0

    def test_osdi_component_port_count_mismatch(self):
        """Test that port count mismatch raises ValueError."""
        with pytest.raises(ValueError, match="Port count mismatch"):
            osdi_component(
                "tests/resistor_va.osdi",
                port_names=("p1",),  # Only 1 port, but OSDI has 2
                param_names=("R", "m"),
            )

    def test_osdi_component_param_count_mismatch(self):
        """Test that parameter count mismatch raises ValueError."""
        with pytest.raises(ValueError, match="Parameter count mismatch"):
            osdi_component(
                "tests/resistor_va.osdi",
                port_names=("p1", "p2"),
                param_names=("R",),  # Only 1 param, but OSDI has 2
            )

    def test_osdi_component_stateful_not_supported(self):
        """Test that stateful models raise NotImplementedError (for now)."""
        # Create a fake OSDI model with num_states > 0
        # For now, we can only test this indirectly if such a binary exists
        # This is a placeholder for future stateful model support
        pass


class TestOsdiComponentIntegration:
    """Test OSDI component integration with Circulax system."""

    def test_osdi_component_is_circuitcomponent(self):
        """Test that OSDI component is a CircuitComponent."""
        from circulax.components.base_component import CircuitComponent

        OsdiResistor = osdi_component(
            "tests/resistor_va.osdi",
            port_names=("p1", "p2"),
            param_names=("R", "m"),
        )

        assert issubclass(OsdiResistor, CircuitComponent)

    def test_osdi_component_has_solver_call(self):
        """Test that OSDI component has solver_call classmethod."""
        OsdiResistor = osdi_component(
            "tests/resistor_va.osdi",
            port_names=("p1", "p2"),
            param_names=("R", "m"),
        )

        assert hasattr(OsdiResistor, "solver_call")
        assert callable(OsdiResistor.solver_call)

    def test_osdi_component_parameter_extraction(self):
        """Test that parameters can be extracted from component instances."""
        from circulax.components.osdi_component import _extract_param

        OsdiResistor = osdi_component(
            "tests/resistor_va.osdi",
            port_names=("p1", "p2"),
            param_names=("R", "m"),
            param_defaults={"R": 500.0, "m": 1.0},
        )

        r = OsdiResistor()

        # Test extraction from object
        assert _extract_param(r, "R") == 500.0
        assert _extract_param(r, "m") == 1.0

        # Test extraction from dict
        params_dict = {"R": 100.0, "m": 2.0}
        assert _extract_param(params_dict, "R") == 100.0
        assert _extract_param(params_dict, "m") == 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
