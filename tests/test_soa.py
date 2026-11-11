# Copyright (c) 2026 GDSFactory

"""Physical limits of the diode-driven integrated-gain SOA."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq

from circulax import compile_circuit
from circulax.circuit import Circuit
from circulax.components.base_component import PhysicsReturn, Signals, component
from circulax.components.electronic import Resistor, VoltageSource
from circulax.components.soa import SemiconductorOpticalAmplifier, ase_field, soa_scales, unsaturated_gain


@component(ports=("p1", "p2"), states=("i_src",), holomorphic=True, ac_source_stamp={"i_src": 2.0})
def matched_light(signals: Signals, power: float = 1.0, phase: float = 0.0) -> PhysicsReturn:
    """Unit-impedance source launching sqrt(power), absorbing returned waves."""
    field = 2 * jnp.sqrt(power) * jnp.exp(1j * phase)
    return {
        "p1": signals.i_src,
        "p2": -signals.i_src,
        "i_src": signals.p1 - signals.p2 - field - signals.i_src,
    }, {}


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
            "light": matched_light,
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
    incoming_right = circuit.port_map["SOA,wave_optical_out"]
    field = 1e-6 * np.exp((1 - 3.5j) * depleted / 2)
    initial = initial.at[hi].set(depleted)
    initial = initial.at[oi].set(field.real).at[oi + circuit.sys_size].set(field.imag)
    initial = initial.at[incoming_right].set(0).at[incoming_right + circuit.sys_size].set(0)
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
        flow, _ = device(h=gain, wave_optical_in=np.sqrt(pin))
        return flow["h"]

    derivative = jax.grad(residual)(h)
    expected = 1 + 280e-12 * pin / float(soa_scales()[2]) * (1 + h / 2)
    np.testing.assert_allclose(derivative, expected, rtol=1e-12)


def test_modulation_response_above_and_below_carrier_corner(soa_circuit: Circuit) -> None:
    """Input modulation is compressed slowly; pump modulation is low pass."""
    circuit = soa_circuit
    current = 0.08
    power = 2e-4
    lifetime = 280e-12
    voltage = 1.8 * 0.02585 * np.log1p(current / 1e-12) + 5 * current
    model = circuit.small_signal(params={"BIAS.V": voltage, "LIGHT.power": power})
    gain = float(np.abs(circuit.port(model.y_dc, "out")) ** 2 / power)
    energy = float(soa_scales()[2])
    rate = 1 / lifetime + power * gain / energy
    frequencies = jnp.array([0.0, 0.01, 1.0, 100.0]) * rate / (2 * jnp.pi)
    optical = model(freqs=frequencies, sources={"LIGHT": np.sqrt(power) * 1e-3 / 2})
    normalized = optical.power("out", field_unit="sqrt(W)") / (gain * optical.power("SOA,optical_in", field_unit="sqrt(W)"))
    expected = 1 - power * (gain - 1) / energy / (rate + 2j * jnp.pi * frequencies)
    np.testing.assert_allclose(normalized, expected, rtol=2e-7, atol=1e-9)
    assert abs(normalized[0]) < abs(normalized[-1])
    np.testing.assert_allclose(normalized[-1], 1, atol=0.01)

    pump = model(freqs=frequencies, sources={"BIAS": 1e-3})
    pump_power = pump.power("out", field_unit="sqrt(W)")
    diode_conductance = (current + 1e-12) / (1.8 * 0.02585)
    electrical_time = 5 * 150e-15 / (1 + 5 * diode_conductance)
    expected_pump = rate / (rate + 2j * jnp.pi * frequencies) / (1 + 2j * jnp.pi * frequencies * electrical_time)
    np.testing.assert_allclose(pump_power / pump_power[0], expected_pump, rtol=2e-7, atol=1e-9)


@pytest.fixture(scope="module")
def bidirectional_circuit() -> Circuit:
    """Independent matched light sources at both optical ends."""
    return compile_circuit(
        {
            "instances": {
                "SOA": {"component": "soa"},
                "BIAS": {"component": "bias"},
                "LEFT": {"component": "light"},
                "RIGHT": {"component": "light"},
                "GND": {"component": "ground"},
            },
            "connections": {
                "GND,p1": ("BIAS,p2", "LEFT,p2", "RIGHT,p2", "SOA,cathode"),
                "BIAS,p1": "SOA,anode",
                "LEFT,p1": "SOA,optical_in",
                "RIGHT,p1": "SOA,optical_out",
            },
            "ports": {},
        },
        {"soa": SemiconductorOpticalAmplifier, "bias": VoltageSource, "light": matched_light, "ground": lambda: 0},
        is_complex=True,
        backend="dense",
        rtol=1e-10,
        atol=1e-12,
    )


def test_both_directions_share_carrier_depletion(bidirectional_circuit: Circuit) -> None:
    """Unequal counterpropagating beams saturate one reservoir, independently of phase."""
    circuit = bidirectional_circuit
    current = 0.08
    bias = 1.8 * 0.02585 * np.log1p(current / 1e-12) + 5 * current
    powers = [(2e-4, 0.0), (0.0, 2e-4), (5e-5, 1.5e-4)]
    gains = []
    target = float(unsaturated_gain(current))
    energy = float(soa_scales()[2])
    for left, right in powers:
        params = {"BIAS.V": bias, "LEFT.power": left, "RIGHT.power": right, "RIGHT.phase": 0.73}
        state = circuit.dc(params=params)
        h = np.real(circuit.port(state, "SOA,h"))
        gains.append(h)
        expected_h = brentq(lambda x, total=left + right: x - target + total * 280e-12 / energy * np.expm1(x), 0, target)
        np.testing.assert_allclose(h, expected_h, rtol=1e-8)
        a_left, a_right = np.sqrt(left), np.sqrt(right) * np.exp(0.73j)
        transmission = np.exp((1 - 3.5j) * h / 2)
        np.testing.assert_allclose(circuit.port(state, "SOA,wave_optical_in"), a_left, atol=1e-10)
        np.testing.assert_allclose(circuit.port(state, "SOA,wave_optical_out"), a_right, atol=1e-10)
        np.testing.assert_allclose(circuit.port(state, "SOA,optical_in"), a_left + transmission * a_right, atol=1e-9)
        np.testing.assert_allclose(circuit.port(state, "SOA,optical_out"), a_right + transmission * a_left, atol=1e-9)
    np.testing.assert_allclose(gains, gains[0], rtol=1e-8)
    # A reverse beam changes the forward beam's gain rather than acquiring a
    # separate reservoir. Phase changes do not resolve a standing-wave grating.
    h_two = []
    for phase in [0.0, np.pi / 2, np.pi]:
        state = circuit.dc(params={"BIAS.V": bias, "LEFT.power": 2e-4, "RIGHT.power": 2e-4, "RIGHT.phase": phase})
        h_two.append(np.real(circuit.port(state, "SOA,h")))
    np.testing.assert_allclose(h_two, h_two[0], rtol=1e-8)
    assert h_two[0] < gains[0]


def test_external_reflector_returns_through_soa(soa_circuit: Circuit) -> None:
    """An external mismatched load produces a double-pass reflected field."""
    circuit = soa_circuit
    current = 0.08
    bias = 1.8 * 0.02585 * np.log1p(current / 1e-12) + 5 * current
    incident = 1e-6
    reflection = 0.25
    resistance = (1 + reflection) / (1 - reflection)
    params = {"BIAS.V": bias, "LIGHT.power": incident**2, "LOAD.R": resistance, "SOA.saturation": 0.0}
    state = circuit.dc(params=params)
    h = float(unsaturated_gain(current))
    transmission = np.exp((1 - 3.5j) * h / 2)
    returned = reflection * transmission * incident
    np.testing.assert_allclose(circuit.port(state, "SOA,wave_optical_out"), returned, rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(circuit.port(state, "SOA,optical_in") - incident, transmission * returned, rtol=1e-7, atol=1e-12)
    # With saturation enabled, the reflected power must consume carriers too.
    power = 2e-4
    state = circuit.dc(params={**params, "LIGHT.power": power, "SOA.saturation": 1.0})
    h = float(np.real(circuit.port(state, "SOA,h")))
    reverse_power = reflection**2 * power * np.exp(h)
    target = float(unsaturated_gain(current))
    np.testing.assert_allclose(target - h, 280e-12 * (power + reverse_power) * np.expm1(h) / float(soa_scales()[2]), rtol=1e-7)


def test_reverse_modulation_cross_saturates_forward_beam(bidirectional_circuit: Circuit) -> None:
    """A reverse intensity perturbation acts through the shared carrier pole."""
    circuit = bidirectional_circuit
    left, right = 1e-4, 2e-4
    bias = 1.8 * 0.02585 * np.log1p(0.08 / 1e-12) + 0.4
    model = circuit.small_signal(params={"BIAS.V": bias, "LEFT.power": left, "RIGHT.power": right, "SOA.alpha_h": 0.0})
    h = float(np.real(circuit.port(model.y_dc, "SOA,h")))
    energy = float(soa_scales()[2])
    rate = 1 / 280e-12 + (left + right) * np.exp(h) / energy
    frequencies = jnp.array([0, 1, 100]) * rate / (2 * jnp.pi)
    delta_power = right * 1e-3
    result = model(freqs=frequencies, sources={"RIGHT": delta_power / (2 * np.sqrt(right))})
    expected = -delta_power * np.expm1(h) / energy / (rate + 2j * jnp.pi * frequencies)
    np.testing.assert_allclose(result.voltage("SOA,h"), expected, rtol=1e-7, atol=1e-10)
    assert abs(expected[0]) > abs(expected[-1])


def test_bidirectional_passive_loss_balance(bidirectional_circuit: Circuit) -> None:
    """Both directional stimulated powers exclude passive absorption."""
    circuit = bidirectional_circuit
    left, right = 1e-4, 3e-4
    bias = 1.8 * 0.02585 * np.log1p(0.08 / 1e-12) + 0.4
    loss = 24 * 600 / 10000 * np.log(10) / 10
    state = circuit.dc(params={"BIAS.V": bias, "LEFT.power": left, "RIGHT.power": right, "SOA.loss_db_per_cm": 24.0})
    h = float(np.real(circuit.port(state, "SOA,h")))
    z = np.linspace(0, 1, 10001)
    # Integrate the counterpropagating powers from their respective ends.
    total_power = left * np.exp((h - loss) * z) + right * np.exp((h - loss) * (1 - z))
    stimulated = np.trapezoid(h * total_power, z)
    np.testing.assert_allclose(float(unsaturated_gain(0.08)) - h, 280e-12 * stimulated / float(soa_scales()[2]), rtol=1e-7)


def test_transparent_bidirectional_stamp_is_nonsingular(bidirectional_circuit: Circuit) -> None:
    """Unity transmission needs no singular S-to-Y conversion."""
    circuit = bidirectional_circuit
    volume = float(soa_scales()[0])
    current = 1.602176634e-19 * volume * 5.5e23 / (0.8 * 280e-12)
    bias = 1.8 * 0.02585 * np.log1p(current / 1e-12) + 5 * current
    left, right = 0.01, 0.02j
    state = circuit.dc(
        params={"BIAS.V": bias, "LEFT.power": abs(left) ** 2, "RIGHT.power": abs(right) ** 2, "RIGHT.phase": np.pi / 2}
    )
    np.testing.assert_allclose(circuit.port(state, "SOA,h"), 0, atol=1e-9)
    for port in ["SOA,optical_in", "SOA,optical_out"]:
        np.testing.assert_allclose(circuit.port(state, port), left + right, atol=1e-9)
