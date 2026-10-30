# Copyright (c) 2026 Circulax contributors
"""Analytical and circuit references for the generic operational amplifier."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from circulax import Circuit, compile_circuit
from circulax.components.electronic import PulseVoltageSource, Resistor, VoltageSource
from circulax.components.opamp import OpAmp


def _follower(
    settings: dict[str, Any], voltage: float = 0.0, delay: float = 0.0, waveform: dict[str, float] | None = None
) -> Circuit:
    return compile_circuit(
        {
            "instances": {
                "A": {"component": "amp", "settings": settings},
                "V": {"component": "voltage", "settings": waveform if waveform is not None else {"V": voltage, "delay": delay}},
                "R": {"component": "resistor", "settings": {"R": 1000.0}},
            },
            "connections": {
                "V,p1": "A,in_p",
                "V,p2": "GND,p1",
                "A,in_m": "A,out_p",
                "A,out_m": "GND,p1",
                "A,out_p": "R,p1",
                "R,p2": "GND,p1",
            },
            "ports": {"out": "A,out_p"},
        },
        {"amp": OpAmp, "voltage": PulseVoltageSource if waveform is not None else VoltageSource, "resistor": Resistor},
        g_leak=0.0,
    )


def test_follower_dc_gain_and_ac_bandwidth() -> None:
    settings = {"A": 1000.0, "GBW": 1e5, "Rout": 50.0, "SR": 1e9}
    circuit = _follower(settings, voltage=0.1)
    attenuation = 1000.0 / 1050.0
    gain = attenuation * settings["A"] / (1.0 + attenuation * settings["A"])
    np.testing.assert_allclose(circuit.port(circuit.dc(), "out"), 0.1 * gain, rtol=1e-9)
    frequencies = jnp.array([0.0, 100.0, 1e4, 1e5, 1e6])
    result = circuit.ac_response(freqs=frequencies, sources={"V": 1.0})
    tau = settings["A"] / (2 * jnp.pi * settings["GBW"])
    expected = attenuation * settings["A"] / (1.0 + attenuation * settings["A"] + 2j * jnp.pi * frequencies * tau)
    np.testing.assert_allclose(circuit.port(result["solution"], "out"), expected, rtol=1e-7, atol=1e-10)


def test_follower_step_bandwidth() -> None:
    settings = {"A": 100.0, "GBW": 1e4, "Rout": 10.0, "SR": 1e9}
    delay, step = 1e-5, 2e-7
    circuit = _follower(settings, voltage=0.01, delay=delay)
    times = jnp.linspace(0.0, 1e-4, 501)
    sol = circuit.transient(t0=0.0, t1=1e-4, dt0=step, saveat=times, max_steps=550)
    alpha = 1000.0 / 1010.0
    gain = alpha * 100.0 / (1.0 + alpha * 100.0)
    tau = 100.0 / (2 * jnp.pi * 1e4) / (1.0 + alpha * 100.0)
    expected = 0.01 * gain * (1.0 - jnp.exp(-jnp.maximum(times - delay, 0.0) / tau))
    np.testing.assert_allclose(circuit.port(sol.ys, "out"), expected, atol=8e-5, rtol=0.01)


def test_slew_limited_follower_step() -> None:
    circuit = _follower({"A": 1000.0, "GBW": 1e6, "Rout": 10.0, "SR": 1e4}, voltage=2.0, delay=1e-5)
    times = jnp.linspace(0.0, 2e-4, 201)
    sol = circuit.transient(t0=0.0, t1=2e-4, dt0=1e-6, saveat=times, max_steps=230)
    output = circuit.port(sol.ys, "out")
    slope = np.diff(np.asarray(output)) / np.diff(np.asarray(times))
    np.testing.assert_allclose(slope[20:80], 1e4 * 1000.0 / 1010.0, rtol=0.01)
    assert np.max(slope) <= 1.01e4


def test_rail_rates_and_recovery_without_windup() -> None:
    model = OpAmp(A=100.0, GBW=1e5, Vmin=-1.0, Vmax=1.0, SR=1e6)

    def rate(vin: float, drive: float) -> jax.Array:
        f, _ = model(in_p=vin, in_m=0.0, v_drive=drive)
        return -f["v_drive"] / (100.0 / (2 * jnp.pi * 1e5))

    np.testing.assert_allclose(rate(1.0, 1.0), 0.0, atol=1e-15)
    assert rate(-1.0, 1.0) < -1e5
    assert rate(1.0, -1.0) > 1e5
    assert rate(1.0, 1.1) < 0.0
    assert rate(-1.0, -1.1) > 0.0
    circuit = _follower({"A": 100.0, "GBW": 1e4, "Vmin": -1.0, "Vmax": 1.0, "Rout": 10.0}, voltage=3.0)
    np.testing.assert_allclose(circuit.port(circuit.dc(), "out"), 1000.0 / 1010.0, rtol=1e-8)


def test_jit_batch_and_parameters() -> None:
    def f(v: jax.Array, gain: jax.Array) -> tuple[jax.Array, jax.Array]:
        return OpAmp.solver_call(0.0, v, OpAmp(A=gain))

    bias = jnp.array([0.0, 0.0, 1e-5, 0.0, 0.1])
    fval, q = jax.jit(f)(bias, jnp.array(1e5))
    jac = jax.jit(jax.jacfwd(f, argnums=(0, 1)))(bias, jnp.array(1e5))
    assert all(jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(jac))
    batch = jax.jit(jax.vmap(lambda v: f(v, jnp.array(1e5))))(jnp.stack([bias, -bias]))
    np.testing.assert_allclose(batch[0][0], -batch[0][1], atol=1e-12)
    np.testing.assert_allclose(fval[:2].sum(), 0.0, atol=1e-15)
    assert q[4] > 0.0


def test_transient_rail_saturation_and_recovery() -> None:
    circuit = _follower(
        {"A": 100.0, "GBW": 1e5, "Rout": 10.0, "Vmin": -1.0, "Vmax": 1.0, "SR": 1e6, "RailTau": 1e-7},
        waveform={"v1": 0.0, "v2": 3.0, "td": 1e-5, "tr": 1e-6, "tf": 1e-6, "pw": 8e-5, "per": 1e-3},
    )
    times = jnp.linspace(0.0, 1.2e-4, 241)
    sol = circuit.transient(t0=0.0, t1=1.2e-4, dt0=1e-7, saveat=times, max_steps=1250)
    output = np.asarray(circuit.port(sol.ys, "out"))
    assert output.max() <= 1000.0 / 1010.0 + 1e-8
    np.testing.assert_allclose(output[(times > 3e-5) & (times < 8e-5)], 1000.0 / 1010.0, atol=1e-8)
    assert output[np.argmin(abs(np.asarray(times) - 9.4e-5))] < 0.8
    assert output[-1] < 1e-5
