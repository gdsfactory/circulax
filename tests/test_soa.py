# Copyright (c) 2026 GDSFactory

"""Physical limits of the diode-driven integrated-gain SOA."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

from circulax import compile_circuit
from circulax.circuit import Circuit
from circulax.components.electronic import Resistor, VoltageSource
from circulax.components.photonic import OpticalSource
from circulax.components.soa import SemiconductorOpticalAmplifier, ase_field, soa_scales, unsaturated_gain


@pytest.fixture(scope="module")
def soa_circuit() -> Circuit:
    netlist = {
        "instances": {
            "SOA": {"component": "soa"},
            "BIAS": {"component": "bias"},
            "LIGHT": {"component": "light"},
            "LOAD": {"component": "load", "settings": {"R": 1.0}},
            "GND": {"component": "ground"},
        },
        "connections": {
            "GND,p1": ("BIAS,p2", "LIGHT,p2", "LOAD,p2", "SOA,cathode"),
            "BIAS,p1": "SOA,anode",
            "LIGHT,p1": "SOA,optical_in",
            "SOA,optical_out": "LOAD,p1",
        },
        "ports": {"out": "SOA,optical_out"},
    }
    return compile_circuit(
        netlist,
        {
            "soa": SemiconductorOpticalAmplifier,
            "bias": VoltageSource,
            "light": OpticalSource,
            "load": Resistor,
            "ground": lambda: 0,
        },
        is_complex=True,
        backend="dense",
        rtol=1e-10,
        atol=1e-12,
    )


def test_diode_and_saturated_cw_balance(soa_circuit: Circuit) -> None:
    circuit = soa_circuit
    current = 0.08
    bias = 1.8 * 0.02585 * np.log1p(current / 1e-12) + 5 * current
    powers = jnp.geomspace(1e-9, 1e-2, 16)
    state = circuit.dc(params={"BIAS.V": bias, "LIGHT.power": powers})
    np.testing.assert_allclose(-circuit.port(state, "BIAS,i_src"), current, rtol=1e-7)
    target = float(unsaturated_gain(current))
    energy = float(soa_scales()[2])
    psat = energy / 280e-12
    roots = [brentq(lambda h, p=p: h - target + p / psat * np.expm1(h), 0, target) for p in powers]
    gain = np.abs(circuit.port(state, "out")) ** 2 / np.asarray(powers)
    np.testing.assert_allclose(gain, np.exp(roots), rtol=1e-7)
    assert np.all(np.diff(gain) < 0)


def test_dark_recovery_and_phase(soa_circuit: Circuit) -> None:
    circuit = soa_circuit
    bias = 1.8 * 0.02585 * np.log1p(0.08 / 1e-12) + 0.4
    params = {"BIAS.V": bias, "LIGHT.power": 1e-12}
    initial = circuit.dc(params=params)
    target = float(np.real(circuit.port(initial, "SOA,h")))
    # Prepare a depleted reservoir, keeping all algebraic field constraints
    # consistent with that gain before release into the weak probe.
    depleted = target - 1.0
    hi = circuit.port_map["SOA,h"]
    oi = circuit.port_map["SOA,optical_out"]
    ii = circuit.port_map["SOA,i_out"]
    field = 1e-6 * np.exp((1 - 3.5j) * depleted / 2)
    initial = initial.at[hi].set(depleted)
    initial = initial.at[oi].set(field.real).at[oi + circuit.sys_size].set(field.imag)
    initial = initial.at[ii].set(-field.real).at[ii + circuit.sys_size].set(-field.imag)
    ts = jnp.linspace(0, 1.4e-9, 281)
    sol = circuit.transient(t0=0.0, t1=1.4e-9, dt0=2e-12, y0=initial, saveat=ts, params=params, max_steps=1000)
    gain_state = np.real(circuit.port(sol.ys, "SOA,h"))
    expected = target - np.exp(-np.asarray(ts) / 280e-12)
    np.testing.assert_allclose(gain_state, expected, atol=2e-5)
    output = circuit.port(sol.ys, "out")
    np.testing.assert_allclose(output, 1e-6 * np.exp((1 - 3.5j) * gain_state / 2), atol=1e-10)


def test_length_and_noise_limits() -> None:
    # Keeping injection per active volume fixed gives equal gain per length.
    short = unsaturated_gain(0.04, length_um=300)
    long = unsaturated_gain(0.08, length_um=600)
    np.testing.assert_allclose(long, 2 * short)
    np.testing.assert_allclose(soa_scales(length_um=300)[2], soa_scales(length_um=600)[2])
    gain = np.full(200_000, 40.0)
    noise, power = ase_field(gain, 1e12, rng=np.random.default_rng(4))
    np.testing.assert_allclose(np.mean(np.abs(noise) ** 2), power.mean(), rtol=0.01)
    np.testing.assert_allclose(np.mean(noise**2), 0, atol=power.mean() * 0.01)
    quiet, expected = ase_field(np.array([0.5, 1.0]), 1e12, rng=np.random.default_rng(4))
    np.testing.assert_array_equal(quiet, 0)
    np.testing.assert_array_equal(expected, 0)


def test_passive_loss_excludes_absorbed_photons_from_carrier_extraction(soa_circuit: Circuit) -> None:
    """Passive attenuation and stimulated generation are different power terms."""
    circuit = soa_circuit
    current = 0.08
    bias = 1.8 * 0.02585 * np.log1p(current / 1e-12) + 5 * current
    pin = 1e-3
    loss_db_per_cm = 24.0
    state = circuit.dc(params={"BIAS.V": bias, "LIGHT.power": pin, "SOA.loss_db_per_cm": loss_db_per_cm})
    h = float(np.real(circuit.port(state, "SOA,h")))
    target = float(unsaturated_gain(current))
    loss = loss_db_per_cm * 600 / 10000 * np.log(10) / 10
    pout = float(np.abs(circuit.port(state, "out")) ** 2)
    np.testing.assert_allclose(pout, pin * np.exp(h - loss), rtol=1e-7)
    # Independently integrate stimulated generation along the constant-gain
    # section; passive absorption is excluded from carrier consumption.
    position = np.linspace(0, 1, 10001)
    stimulated = np.trapezoid(h * pin * np.exp((h - loss) * position), position)
    energy = float(soa_scales()[2])
    np.testing.assert_allclose(target - h, 280e-12 * stimulated / energy, rtol=1e-7)
    assert stimulated > pout - pin


def test_zero_net_gain_has_finite_derivative() -> None:
    """Gain exactly balancing passive loss must remain differentiable."""
    device = SemiconductorOpticalAmplifier(loss_db_per_cm=10.0)
    h = 10.0 * 600 / 10000 * np.log(10) / 10
    pin = 1e-3

    def residual(gain: jax.Array) -> jax.Array:
        flow, _ = device(h=gain, optical_in=np.sqrt(pin))
        return flow["h"]

    derivative = jax.grad(residual)(h)
    expected = 1 + 280e-12 * pin / float(soa_scales()[2]) * (1 + h / 2)
    np.testing.assert_allclose(derivative, expected, rtol=1e-12)
