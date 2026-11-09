"""Buffered filter voltage responses against independent SciPy analog designs."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import signal

from circulax import Circuit, compile_circuit
from circulax.components import Bessel3, Biquad, LowPass1
from circulax.components.electronic import Resistor, VoltageSource


def _circuit(model: type, settings: dict, *, resistance: float = 50, delay: float = 0) -> Circuit:
    return compile_circuit(
        {
            "instances": {
                "GND": {"component": "ground"},
                "V": {"component": "V", "settings": {"V": 1.0, "delay": delay}},
                "F": {"component": "filter", "settings": settings},
                "R": {"component": "R", "settings": {"R": resistance}},
            },
            "connections": {"V,p1": "F,inp", "V,p2": "GND,p1", "F,out": "R,p1", "R,p2": "GND,p1"},
            "ports": {"output": "F,out"},
        },
        {"V": VoltageSource, "filter": model, "R": Resistor},
        is_complex=False,
        g_leak=0,
    )


def _coefficients(name: str, f3db: float, q: float = 1 / np.sqrt(3)) -> tuple:
    if name == "lowpass":
        return signal.butter(1, 2 * np.pi * f3db, analog=True)
    if name == "bessel":
        return signal.bessel(3, 2 * np.pi * f3db, analog=True, norm="mag")
    # Independently solve |H(j*w3)|^2=1/2 for the natural frequency.
    u = np.roots([1, 1 / q**2 - 2, -1])
    w0 = 2 * np.pi * f3db / np.sqrt(u[u > 0][0])
    return [w0**2], [1, w0 / q, w0**2]


@pytest.mark.parametrize(
    ("model", "name", "settings"),
    [
        (LowPass1, "lowpass", {}),
        (Bessel3, "bessel", {}),
        (Biquad, "biquad", {"Q": 0.25}),
        (Biquad, "biquad", {"Q": 1 / np.sqrt(3)}),
        (Biquad, "biquad", {"Q": 1.2}),
    ],
)
def test_complex_response_cutoff_and_buffering(model: type, name: str, settings: dict) -> None:
    f3db = 35e9
    f = np.r_[0.0, np.geomspace(1e6, 200e9, 40), f3db]
    circuit = _circuit(model, settings | {"f3db": f3db})
    result = circuit.ac_response(freqs=jnp.asarray(f), sources={"V": 1.0})
    actual = np.asarray(circuit.port(result["solution"], "output"))
    b, a = _coefficients(name, f3db, settings.get("Q", 1 / np.sqrt(3)))
    expected = signal.freqs(b, a, worN=2 * np.pi * f)[1]
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(abs(actual[-1]), 1 / np.sqrt(2), rtol=1e-9)
    np.testing.assert_allclose(circuit.port(result["solution"], "V,i_src"), 0.0, atol=1e-10)
    assert abs(float(circuit.port(result["y_dc"], "output")) - 1.0) < 1e-10
    unloaded = _circuit(model, settings | {"f3db": f3db}, resistance=10000)
    another = unloaded.ac_response(freqs=jnp.array([f3db]), sources={"V": 1.0})
    np.testing.assert_allclose(unloaded.port(another["solution"], "output")[0], actual[-1], atol=1e-9)


@pytest.mark.parametrize(("model", "name"), [(LowPass1, "lowpass"), (Biquad, "biquad"), (Bessel3, "bessel")])
def test_causal_step_against_scipy(model: type, name: str) -> None:
    f3db, delay, dt = 1e9, 1e-9, 2e-12
    circuit = _circuit(model, {"f3db": f3db}, delay=delay)
    times = np.arange(0, 5e-9 + dt / 2, dt)
    solution = circuit.transient(t0=0.0, t1=float(times[-1]), dt0=dt, saveat=jnp.asarray(times), max_steps=len(times) + 20)
    actual = np.asarray(circuit.port(solution.ys, "output"))
    before = times < delay - dt
    np.testing.assert_allclose(actual[before], 0.0, atol=1e-10)
    b, a = _coefficients(name, 1 / (2 * np.pi))
    after = times >= delay
    expected = signal.step((b, a), T=(times[after] - delay) * (2 * np.pi * f3db))[1]
    np.testing.assert_allclose(actual[after], expected, atol=0.012, rtol=0.005)
    assert abs(actual[-1] - 1.0) < 1e-5
