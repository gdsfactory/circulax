"""Checked DC distinguishes solver failure from a finite final Newton estimate."""

import jax
import jax.numpy as jnp

from circulax import Circuit, compile_circuit
from circulax.components.electronic import Resistor, WaveformVoltageSource


def _circuit() -> Circuit:
    """Use one shared device batch under independent scalar or vector source bias.

    @tags circulax simulation
    """
    return compile_circuit(
        {
            "instances": {
                "V": {"component": "source"},
                "R": {"component": "resistor", "settings": {"R": 1000}},
                "GND": {"component": "ground"},
            },
            "connections": {"V,p1": "R,p1", "GND,p1": ["V,p2", "R,p2"]},
            "ports": {"out": "R,p1"},
        },
        {"source": WaveformVoltageSource, "resistor": Resistor},
        backend="dense",
    )


def test_checked_dc_reports_failed_finite_estimate_in_batch() -> None:
    """A deliberately exhausted iteration budget is failure even with finite output.

    @tags circulax simulation
    """
    circuit = _circuit()
    solution, converged = circuit.dc_checked(params={"V.dc": jnp.array([0.0, 1.0])}, max_steps=1)
    assert converged.tolist() == [True, False]
    assert jnp.isfinite(solution).all()
    assert solution.shape == (2, circuit.sys_size)


def test_checked_dc_is_jittable_and_matches_successful_dc() -> None:
    """Checked execution retains normal values, batching, and compiled use.

    @tags circulax simulation
    """
    circuit = _circuit()
    values = jnp.array([-2.0, 0.0, 3.0])
    solution, converged = jax.jit(lambda bias: circuit.dc_checked(params={"V.dc": bias}))(values)
    assert converged.tolist() == [True, True, True]
    assert jnp.allclose(solution, circuit.dc(params={"V.dc": values}))
