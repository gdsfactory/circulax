"""Driven small-signal AC retains physical response independently of scattering."""

from copy import deepcopy

import jax
import jax.numpy as jnp
import pytest

from circulax import Circuit, compile_circuit
from circulax.components.base_component import PhysicsReturn, Signals, component
from circulax.components.electronic import Capacitor, Resistor, WaveformCurrentSource, WaveformVoltageSource
from circulax.components.photonic import OpticalSource


def _rc() -> Circuit:
    """Build a source-driven RC circuit without external characterization ports.

    @tags circulax simulation
    """
    return compile_circuit(
        {
            "instances": {
                "V": {"component": "voltage", "settings": {"dc": 2.0}},
                "I": {"component": "current", "settings": {"dc": 0.0}},
                "R": {"component": "resistor", "settings": {"R": 1000.0}},
                "C": {"component": "capacitor", "settings": {"C": 1e-6}},
                "GND": {"component": "ground"},
            },
            "connections": {
                "V,p1": "R,p1",
                "R,p2": ("C,p1", "I,p1"),
                "GND,p1": ("V,p2", "C,p2", "I,p2"),
            },
            "ports": {},
        },
        {"voltage": WaveformVoltageSource, "current": WaveformCurrentSource, "resistor": Resistor, "capacitor": Capacitor},
    )


def test_rc_response_retains_physical_voltage_and_current() -> None:
    """A driven RC response follows its transfer function without matching loads.

    @tags circulax simulation
    """
    circuit = _rc()
    freqs = jnp.array([0.0, 100.0, 1000.0])
    amplitude = 2.0 + 1j
    result = circuit.ac_response(freqs=freqs, sources={"V": amplitude})
    expected = amplitude / (1 + 2j * jnp.pi * freqs * 1e-3)
    assert circuit.ports == ()
    assert jnp.allclose(result["solution"][:, circuit.port_map["R,p2"]], expected)
    assert jnp.allclose(result["solution"][:, circuit.port_map["V,i_src"]], -(amplitude - expected) / 1000)
    assert jnp.allclose(result["y_dc"][circuit.port_map["R,p2"]], 2.0)
    assert result["port_map"] == circuit.port_map
    assert jnp.array_equal(result["freqs"], freqs)


def test_dc_bias_and_ac_excitation_are_independent() -> None:
    """DC and AC overrides do not replace or scale one another.

    @tags circulax simulation
    """
    circuit = _rc()
    kwargs = {"freqs": jnp.array([100.0]), "sources": {"V": 1j}}
    zero = circuit.ac_response(**kwargs, params={"V.dc": 0.0})
    biased = circuit.ac_response(**kwargs, params={"V.dc": 3.0})
    assert jnp.allclose(zero["solution"], biased["solution"])
    assert not jnp.allclose(zero["y_dc"], biased["y_dc"])
    silent = circuit.ac_response(freqs=kwargs["freqs"], sources={})
    assert jnp.allclose(silent["solution"], 0)


def test_multiple_sources_superpose_with_current_orientation() -> None:
    """Independent current excitation retains the physical source sign.

    @tags circulax simulation
    """
    circuit = _rc()
    freqs = jnp.array([10.0, 1000.0])
    voltage = circuit.ac_response(freqs=freqs, sources={"V": 1.0})
    current = circuit.ac_response(freqs=freqs, sources={"I": 1e-3j})
    both = circuit.ac_response(freqs=freqs, sources={"V": 1.0, "I": 1e-3j})
    assert jnp.allclose(both["solution"], voltage["solution"] + current["solution"])
    expected = -1j / (1 + 2j * jnp.pi * freqs * 1e-3)
    assert jnp.allclose(current["solution"][:, circuit.port_map["R,p2"]], expected)


def test_batched_sources_keep_distinct_instance_excitations() -> None:
    """Sources sharing a compiled group keep separate physical RHS stamps.

    @tags circulax simulation
    """
    original = _rc()
    device = deepcopy(original.source_netlist)
    device["instances"]["I2"] = {"component": "current", "settings": {"dc": 0.0}}
    device["connections"]["R,p2"] += ("I2,p1",)
    device["connections"]["GND,p1"] += ("I2,p2",)
    circuit = compile_circuit(device, original.source_models)
    result = circuit.ac_response(freqs=jnp.array([0.0]), sources={"I": 1e-3, "I2": 2e-3})
    assert jnp.allclose(result["solution"][:, circuit.port_map["R,p2"]], -3.0)
    assert any(set(group.index_map or {}) == {"I", "I2"} for group in circuit.groups.values())


@pytest.mark.parametrize("target", ["missing", "R", "V,p1"])
def test_invalid_source_targets_raise(target: str) -> None:
    """Targets must identify source instances with explicit excitation stamps.

    @tags circulax simulation
    """
    with pytest.raises(ValueError, match="source"):
        _rc().ac_response(freqs=jnp.array([10.0]), sources={target: 1.0})


def test_response_supports_jit_and_source_amplitude_vmap() -> None:
    """Source excitation remains differentiable and independently batchable.

    @tags circulax simulation
    """
    circuit = _rc()
    freqs = jnp.array([10.0, 1000.0])
    y_dc = circuit.dc()

    def run(amplitude: jax.Array) -> jax.Array:
        """Return the physical response for a traced excitation.

        @tags circulax simulation
        """
        return circuit.ac_response(freqs=freqs, sources={"V": amplitude}, y_dc=y_dc)["solution"]

    amplitudes = jnp.array([1.0, 2.0j])
    actual = jax.jit(jax.vmap(run))(amplitudes)
    assert jnp.allclose(actual[1], 2j * actual[0])


def test_holomorphic_optical_source_has_independent_field_excitation() -> None:
    """Optical source AC amplitude is a field phasor, independent of DC power.

    @tags circulax simulation
    """
    circuit = compile_circuit(
        {
            "instances": {
                "laser": {"component": "laser", "settings": {"power": 1e-6}},
                "load": {"component": "load"},
                "GND": {"component": "ground"},
            },
            "connections": {"laser,p1": "load,p1", "GND,p1": ("laser,p2", "load,p2")},
            "ports": {},
        },
        {"laser": OpticalSource, "load": Resistor},
        is_complex=True,
    )
    result = circuit.ac_response(freqs=jnp.array([0.0, 1e9]), sources={"laser": 1.0 + 2j})
    assert jnp.allclose(result["solution"][:, circuit.port_map["laser,p1"]], 1 + 2j)


def test_nonholomorphic_optical_response_reports_quadrature_limitation() -> None:
    """Ambiguous complex optical quadratures are explicitly rejected.

    @tags circulax simulation
    """

    @component(ports=("p1", "p2"))
    def intensity_load(signals: Signals) -> PhysicsReturn:
        """Non-holomorphic test load requires quadrature observables.

        @tags circulax simulation
        """
        current = jnp.abs(signals.p1 - signals.p2) ** 2
        return {"p1": current, "p2": -current}, {}

    circuit = compile_circuit(
        {
            "instances": {"load": {"component": "load"}, "GND": {"component": "ground"}},
            "connections": {"load,p2": "GND,p1"},
            "ports": {"out": "load,p1"},
        },
        {"load": intensity_load},
        is_complex=True,
    )
    with pytest.raises(ValueError, match="quadrature/sideband"):
        circuit.ac_response(freqs=jnp.array([10.0]), sources={})
