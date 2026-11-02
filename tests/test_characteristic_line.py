"""Uniform lines realized by characteristic admittance and de-embedded delay."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax import Circuit, compile_circuit
from circulax.components import characteristic_rlgc_transmission_line
from circulax.components.electronic import WAVE_PWL, WAVE_SIN, Resistor, WaveformVoltageSource
from circulax.fitting import CharacteristicLineModel, RationalResponse, fit_rlgc_characteristic, rlgc_sparameters

jax.config.update("jax_enable_x64", val=True)

C0 = 299_792_458.0
LENGTH, Z_LINE, N_RF = 5e-3, 45.0, 2.2
R_DC, R_SKIN = 300.0, 2 * Z_LINE * (50 * np.log(10) / 20) / np.sqrt(1e9)
FREQUENCIES = np.r_[0.0, np.geomspace(1e5, 250e9, 300)]


def r_skin(f: np.ndarray) -> np.ndarray:
    return R_DC + R_SKIN * np.sqrt(f)


def l_skin(f: np.ndarray) -> np.ndarray:
    internal = np.divide(R_SKIN * np.sqrt(f), 2 * np.pi * f, out=np.zeros_like(f), where=f > 0)
    return Z_LINE * N_RF / C0 + internal


def c_line(f: np.ndarray) -> np.ndarray:
    return np.full_like(f, N_RF / (Z_LINE * C0))


RLGC = {"R": r_skin, "L": l_skin, "G": 0.0, "C": c_line}


@pytest.fixture(scope="module")
def skin_model() -> CharacteristicLineModel:
    return fit_rlgc_characteristic(FREQUENCIES, **RLGC, length=LENGTH)


def bench(line: Circuit, drive: dict, *, is_complex: bool = False) -> Circuit:
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "VS": {"component": "drive", "settings": drive},
            "RS": {"component": "R", "settings": {"R": 50.0}},
            "RL": {"component": "R", "settings": {"R": 50.0}},
            "LINE": {"component": "line"},
        },
        "connections": {"VS,p1": "RS,p1", "RS,p2": "LINE,p1", "LINE,p2": "RL,p1", "VS,p2": "GND,p1", "RL,p2": "GND,p1"},
        "ports": {"v1": "LINE,p1", "v2": "LINE,p2"},
    }
    models = {"drive": WaveformVoltageSource, "R": Resistor, "line": line}
    return compile_circuit(net, models, is_complex=is_complex, rtol=1e-9, atol=1e-10)


def test_fit_matches_exact_line_and_passes_checks(skin_model: CharacteristicLineModel) -> None:
    check = np.r_[0.0, np.geomspace(1e3, 250e9, 701)]
    error = np.max(np.abs(skin_model.sparameters(check) - rlgc_sparameters(check, **RLGC, length=LENGTH)))
    assert error < 1e-3
    meta = skin_model.metadata
    assert meta["stable"]
    assert meta["passivity_grid_maximum_singular_value"] <= 1
    # The delay is de-embedded, so a 5 mm line to 250 GHz needs far fewer poles than an S fit.
    assert meta["order"] <= 20
    assert 0.9 * LENGTH * N_RF / C0 < skin_model.delay < LENGTH * np.sqrt(l_skin(FREQUENCIES)[-1] * c_line(FREQUENCIES)[-1])


def test_small_signal_response_matches_model(skin_model: CharacteristicLineModel) -> None:
    line = skin_model.circuit()
    freqs = np.geomspace(1e6, 200e9, 25)
    simulated = np.asarray(line.sp(ports=["p1", "p2"], freqs=jnp.asarray(freqs), z0=50.0))
    np.testing.assert_allclose(simulated, skin_model.sparameters(freqs), atol=1e-9)


def test_dc_resistance_is_exact_without_conductance(skin_model: CharacteristicLineModel) -> None:
    circuit = bench(skin_model.circuit(), {"dc": 1.0})
    y = circuit.dc()
    v1, v2 = (float(np.real(circuit.port(y, port))) for port in ("v1", "v2"))
    resistance = R_DC * LENGTH
    np.testing.assert_allclose([v1, v2], [(resistance + 50) / (resistance + 100), 50 / (resistance + 100)], atol=2e-5)


@pytest.mark.parametrize("is_complex", [False, True])
def test_transient_pulse_matches_frequency_domain(skin_model: CharacteristicLineModel, *, is_complex: bool) -> None:
    ui = 1 / 56e9
    edge = 0.2 * ui
    pwl_t = np.array([0.0, ui, ui + edge, 5 * ui, 5 * ui + edge, 20 * ui])
    pwl_v = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    circuit = bench(
        skin_model.circuit(), {"kind": WAVE_PWL, "pwl_t": jnp.asarray(pwl_t), "pwl_v": jnp.asarray(pwl_v)}, is_complex=is_complex
    )
    times = np.arange(0.0, 12 * ui, ui / 16)
    solution = circuit.transient(t0=0.0, t1=float(times[-1]), dt0=ui / 64, saveat=jnp.asarray(times), max_steps=2000, throw=True)
    v2 = np.real(np.asarray(circuit.port(solution.ys, "v2")))

    dt = ui / 64
    t = np.arange(0.0, 80 * ui, dt)
    f = np.fft.rfftfreq(len(t), dt)
    s21 = rlgc_sparameters(np.maximum(f, 1e3), **RLGC, length=LENGTH)[:, 0, 1]
    reference = np.fft.irfft(np.fft.rfft(np.interp(t, pwl_t, pwl_v)) * s21 / 2, len(t))
    error = v2 - np.interp(times, t, reference)
    assert np.sqrt(np.mean(error**2)) < 2e-3
    # Nothing arrives before the delay.
    assert np.max(np.abs(v2[times < ui + 0.95 * skin_model.delay])) < 1e-3


@pytest.mark.parametrize("is_complex", [False, True])
def test_harmonic_balance_fundamental_matches_exact_line(skin_model: CharacteristicLineModel, *, is_complex: bool) -> None:
    f0 = 20e9
    circuit = bench(skin_model.circuit(), {"kind": WAVE_SIN, "amplitude": 1.0, "freq": f0}, is_complex=is_complex)
    y_time, _ = circuit.hb(freq=f0, harmonics=3)
    v2 = np.real(np.asarray(circuit.port(y_time, "v2")))
    t = np.arange(len(v2)) / (len(v2) * f0)
    fundamental = 2 / len(v2) * np.sum(v2 * np.exp(-2j * np.pi * f0 * t))
    # sin(wt) = Re(-j exp(jwt)); a matched source and load give V2 = S21 * Vs / 2.
    expected = -0.5j * rlgc_sparameters(np.array([1.0, f0]), **RLGC, length=LENGTH)[1, 0, 1]
    assert abs(fundamental - expected) < 1e-4


def test_lossless_line_uses_resistance_floor() -> None:
    frequencies = np.r_[0.0, np.geomspace(1e5, 100e9, 200)]
    lossless = {"R": 0.0, "L": Z_LINE * N_RF / C0, "G": 0.0, "C": N_RF / (Z_LINE * C0)}
    model = fit_rlgc_characteristic(frequencies, **lossless, length=20e-3)
    assert model.resistance_floor > 0
    assert model.conductance_floor > 0
    exact = rlgc_sparameters(frequencies, **lossless, length=20e-3)
    assert np.max(np.abs(model.sparameters(frequencies) - exact)) < 1e-3


def test_factory_returns_two_port() -> None:
    line = characteristic_rlgc_transmission_line(FREQUENCIES, **RLGC, length=LENGTH)
    assert set(line.port_map) >= {"p1", "p2"}


@pytest.mark.parametrize(
    ("frequencies", "kwargs", "match"),
    [
        (np.geomspace(1e5, 250e9, 50), {}, "DC sample"),
        (np.r_[0.0, np.geomspace(1e7, 250e9, 50)], {}, "too high"),
        (FREQUENCIES, {"delay_fraction": 1.5}, "delay_fraction"),
        (FREQUENCIES, {"length": -1.0}, "length"),
        (FREQUENCIES, {"tolerance": 0}, "tolerance"),
        (FREQUENCIES, {"tolerance": np.nan}, "tolerance"),
        (FREQUENCIES, {"max_order": 2}, "max_order"),
        (FREQUENCIES, {"max_order": 4.5}, "max_order"),
        (FREQUENCIES, {"z0": -50}, "z0"),
        (FREQUENCIES, {"tolerance": 1e-4, "max_order": 6}, "no characteristic line fit"),
    ],
)
def test_invalid_inputs_raise(frequencies: np.ndarray, kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        fit_rlgc_characteristic(frequencies, **RLGC, **{"length": LENGTH, **kwargs})


@pytest.mark.parametrize("propagation", [1.0, -1.0])
def test_lossless_through_response_is_finite(propagation: float) -> None:
    empty = np.array([], dtype=complex)
    model = CharacteristicLineModel(
        length=1e-3,
        delay=0.0,
        resistance_floor=0.0,
        conductance_floor=0.0,
        admittance=RationalResponse(empty, empty, 0.02),
        propagation=RationalResponse(empty, empty, propagation),
        metadata={},
    )
    expected = np.array([[[0.0, propagation], [propagation, 0.0]]])
    np.testing.assert_allclose(model.sparameters([0.0]), expected, atol=1e-15)
