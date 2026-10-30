# Copyright (c) 2026 Circulax contributors
"""Independent observable checks of the full I/Q response interface."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax import Circuit, compile_circuit
from circulax.components.electronic import Resistor
from circulax.components.optical_passive import OpticalPhaseShifter
from circulax.components.photodetectors import IdealPhotodetector, OpticalCW
from circulax.components.photonic import OpticalSource
from circulax.optical_analysis import quadrature_response


def detector_circuit(phase: float = 0.0) -> Circuit:
    net = {
        "instances": {
            "G": {"component": "ground"},
            "S": {"component": "S", "settings": {"power": 0.002, "phase": phase}},
            "D": {"component": "D", "settings": {"reflectivity": 0.0}},
            "R": {"component": "R", "settings": {"R": 50.0}},
        },
        "connections": {"S,p1": "D,optical_in", "D,anode": "R,p1", "G,p1": ("S,p2", "D,cathode", "R,p2")},
        "ports": {"voltage": "R,p1", "optical": "D,optical_in"},
    }
    return compile_circuit(net, {"S": OpticalCW, "D": IdealPhotodetector, "R": Resistor}, is_complex=True)


@pytest.mark.parametrize("phase", [0.0, 0.73, 1.57])
def test_parameter_power_drive_phase_invariant_detector(phase: float) -> None:
    c = detector_circuit(phase)
    bias = c.dc()
    ac = quadrature_response(c, freqs=jnp.array([0.0, 1e8]), y_dc=bias, parameter_drives={"S.power": 1e-4 + 2e-5j})
    output = c.port_map["voltage"]
    np.testing.assert_allclose(ac["in_phase"][:, output], 0.8 * 50 * (1e-4 + 2e-5j), atol=1e-11)
    np.testing.assert_allclose(ac["quadrature"][:, output], 0, atol=1e-11)
    # Both sidebands reconstruct a real amplitude perturbation with carrier phase.
    optical = c.port_map["optical"]
    amplitude = (1e-4 + 2e-5j) / (2 * np.sqrt(0.002))
    np.testing.assert_allclose(ac["upper_sideband"][:, optical], amplitude * np.exp(1j * phase) / 2, atol=1e-11)
    np.testing.assert_allclose(ac["lower_sideband"][:, optical], np.conj(amplitude) * np.exp(1j * phase) / 2, atol=1e-11)


def test_independent_optical_iq_source_and_jitted_gradient() -> None:
    net = {
        "instances": {
            "G": {"component": "ground"},
            "S": {"component": "S"},
            "D": {"component": "D", "settings": {"phase": 0.4}},
            "R": {"component": "R", "settings": {"R": 1.0}},
        },
        "connections": {"S,p1": "D,p1", "D,p2": "R,p1", "G,p1": ("S,p2", "R,p2")},
        "ports": {"out": "D,p2"},
    }
    c = compile_circuit(net, {"S": OpticalSource, "D": OpticalPhaseShifter, "R": Resistor}, is_complex=True)
    dc = c.dc()

    def response(phase: jax.Array) -> jax.Array:
        return quadrature_response(
            c, freqs=jnp.array([2.0]), sources={"S": (1 + 0.2j, 0.3 - 0.1j)}, params={"D.phase": phase}, y_dc=dc
        )["in_phase"][0, c.port_map["out"]]

    expected = np.cos(0.4) * (1 + 0.2j) - np.sin(0.4) * (0.3 - 0.1j)
    np.testing.assert_allclose(jax.jit(response)(0.4), expected, atol=1e-11)
    derivative = -np.sin(0.4) * (1 + 0.2j) - np.cos(0.4) * (0.3 - 0.1j)
    np.testing.assert_allclose(jax.jacfwd(response)(0.4), derivative, atol=1e-11)


def test_invalid_source_and_shapes() -> None:
    c = detector_circuit()
    dc = c.dc()
    with pytest.raises(ValueError, match="excitation stamp"):
        quadrature_response(c, freqs=jnp.array([1.0]), sources={"S": 1.0}, y_dc=dc)
    with pytest.raises(ValueError, match="one-dimensional"):
        quadrature_response(c, freqs=jnp.array(1.0), y_dc=dc)


@pytest.mark.parametrize("reflection", [-0.1, 1.0, 1.1])
def test_detector_invalid_reflection_rejected(reflection: float) -> None:
    model = IdealPhotodetector(reflectivity=reflection, reflection_phase_rad=np.pi)
    with pytest.raises(Exception, match="Detector reflectivity"):
        model(optical_in=0.1)
