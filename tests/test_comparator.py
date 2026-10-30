# Copyright (c) 2026 Circulax contributors
"""Analytical DC/AC and circuit transient references for the comparator."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax import Circuit, compile_circuit
from circulax.components.comparator import Comparator
from circulax.components.electronic import PulseVoltageSource, Resistor, VoltageSource


def _circuit(settings: dict[str, Any], source: dict[str, float], *, pulse: bool = False) -> Circuit:
    return compile_circuit(
        {
            "instances": {
                "C": {"component": "comparator", "settings": settings},
                "V": {"component": "voltage", "settings": source},
                "R": {"component": "resistor", "settings": {"R": 1000.0}},
            },
            "connections": {
                "V,p1": "C,in_p",
                "V,p2": "GND,p1",
                "C,in_m": "GND,p1",
                "C,out_p": "R,p1",
                "C,out_m": "GND,p1",
                "R,p2": "GND,p1",
            },
            "ports": {"out": "C,out_p", "input": "C,in_p"},
        },
        {"comparator": Comparator, "voltage": PulseVoltageSource if pulse else VoltageSource, "resistor": Resistor},
        g_leak=0.0,
    )


@pytest.mark.parametrize("bias", [-0.001, 0.0, 0.001])
def test_no_hysteresis_analytical_dc_and_ac(bias: float) -> None:
    tau, width, rout = 2e-6, 0.002, 50.0
    circuit = _circuit({"Tau": tau, "Vsmooth": width, "Rout": rout}, {"V": bias})
    alpha = 1000.0 / (1000.0 + rout)
    expected = alpha * 2.5 * (1.0 + np.tanh(bias / width))
    np.testing.assert_allclose(circuit.port(circuit.dc(), "out"), expected, rtol=1e-9)
    frequencies = jnp.array([0.0, 1e3, 1e5, 1e6])
    result = circuit.ac_response(freqs=frequencies, sources={"V": 1.0})
    gain = alpha * 2.5 / width / np.cosh(bias / width) ** 2
    expected_ac = gain / (1.0 + 2j * jnp.pi * frequencies * tau)
    np.testing.assert_allclose(circuit.port(result["solution"], "out"), expected_ac, rtol=1e-8, atol=1e-10)


def test_finite_response_step_and_output_limits() -> None:
    tau, delay = 2e-6, 1e-5
    circuit = _circuit({"Tau": tau, "Vlow": -2.0, "Vhigh": 3.0, "Rout": 50.0}, {"V": 0.02, "delay": delay})
    times = jnp.linspace(0.0, 4e-5, 401)
    sol = circuit.transient(t0=0.0, t1=4e-5, dt0=2e-8, saveat=times, max_steps=2050)
    alpha = 1000.0 / 1050.0
    # Initial input zero is at the middle output; final input is fully high.
    expected = alpha * (0.5 + 2.5 * (1.0 - jnp.exp(-jnp.maximum(times - delay, 0.0) / tau)))
    output = np.asarray(circuit.port(sol.ys, "out"))
    np.testing.assert_allclose(output, expected, atol=0.025, rtol=0.01)
    assert output.min() >= alpha * 0.5 - 1e-9
    assert output.max() <= alpha * 3.0 + 1e-9


def test_hysteresis_thresholds_and_transient_memory() -> None:
    vh, width, tau = 0.1, 0.01, 1e-6
    circuit = _circuit(
        {"Vh": vh, "Vsmooth": width, "Tau": tau, "Vlow": -1.0, "Vhigh": 1.0, "Rout": 10.0},
        {"v1": -0.2, "v2": 0.2, "td": 2e-5, "tr": 2e-4, "tf": 2e-4, "pw": 2e-5, "per": 1e-3},
        pulse=True,
    )
    times = jnp.linspace(0.0, 4.6e-4, 2301)
    sol = circuit.transient(t0=0.0, t1=4.6e-4, dt0=2e-8, saveat=times, max_steps=23100)
    output = np.asarray(circuit.port(sol.ys, "out"))
    inputs = np.asarray(circuit.port(sol.ys, "input"))
    upward = np.flatnonzero((output[:-1] < 0.0) & (output[1:] >= 0.0))[0] + 1
    downward = np.flatnonzero((output[:-1] > 0.0) & (output[1:] <= 0.0))[0] + 1
    z = np.sqrt(1.0 - width / vh)
    threshold = vh * z - width * np.arctanh(z)
    assert inputs[upward] == pytest.approx(threshold, abs=0.012)
    assert inputs[downward] == pytest.approx(-threshold, abs=0.012)
    # The same zero differential input retains different states on each ramp.
    rising_zero = np.argmin(abs(np.asarray(times) - 1.2e-4))
    falling_zero = np.argmin(abs(np.asarray(times) - 3.4e-4))
    assert output[rising_zero] < -0.9
    assert output[falling_zero] > 0.9
    assert np.max(abs(output)) <= 1000.0 / 1010.0 + 1e-6


def test_select_initial_dc_memory_branch() -> None:
    circuit = _circuit({"Vh": 0.1, "Vsmooth": 0.002, "Vlow": -1.0, "Vhigh": 1.0}, {"V": 0.0})
    guess = jnp.zeros(circuit.sys_size)
    high = circuit.dc(guess.at[circuit.port_map["C,x"]].set(1.0))
    low = circuit.dc(guess.at[circuit.port_map["C,x"]].set(-1.0))
    assert circuit.port(high, "out") > 0.9
    assert circuit.port(low, "out") < -0.9


def test_jit_vmap_and_parameter_derivatives() -> None:
    bias = jnp.array([0.5, 0.0, 0.003, 0.0, 0.2])

    def evaluate(v: jax.Array, hysteresis: jax.Array) -> tuple[jax.Array, jax.Array]:
        return Comparator.solver_call(0.0, v, Comparator(Vh=hysteresis))

    f, q = jax.jit(evaluate)(bias, jnp.array(0.01))
    derivatives = jax.jit(jax.jacfwd(evaluate, argnums=(0, 1)))(bias, jnp.array(0.01))
    assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(derivatives))
    values = jax.jit(jax.vmap(lambda v: evaluate(v, jnp.array(0.01))))(jnp.stack([bias, -bias]))
    assert jnp.all(jnp.isfinite(values[0]))
    np.testing.assert_allclose(f[:2].sum(), 0.0, atol=1e-15)
    assert q[4] > 0.0
