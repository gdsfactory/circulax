# Copyright 2026 Circulax contributors

"""Distributed-line reference, safe fitting and native reduced-line simulation."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm
from skrf.network import a2s

from circulax import compile_circuit
from circulax.components import reduced_rlgc_transmission_line
from circulax.components.electronic import WAVE_PWL, Resistor, WaveformVoltageSource
from circulax.fitting import ModelCoefficients, ModelFitOptions, circuit_from_coefficients, fit_rlgc, rlgc_sparameters


@pytest.mark.parametrize(
    ("R", "L", "G", "C"),
    [
        (100.0, 400e-9, 0.01, 160e-12),
        (0.0, 400e-9, 0.0, 160e-12),
        (100.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.01, 0.0),
        (0.0, 0.0, 0.0, 0.0),
    ],
)
def test_distributed_reference_matches_matrix_exponential(R: float, L: float, G: float, C: float) -> None:
    f = np.array([0.0, 1e6, 1e9, 15.625e9, 50e9])
    length = 0.002
    expected = np.stack([expm(length * np.array([[0, R + 2j * np.pi * x * L], [G + 2j * np.pi * x * C, 0]])) for x in f])
    actual = rlgc_sparameters(f, R=R, L=L, G=G, C=C, length=length)
    np.testing.assert_allclose(actual, a2s(expected, z0=50), atol=1e-13)


def test_long_lossy_line_does_not_overflow() -> None:
    s = rlgc_sparameters([0, 1e9], R=1e6, L=1e-6, G=1e3, C=1e-9, length=100.0)
    assert np.isfinite(s).all()
    np.testing.assert_allclose(s[:, 0, 1], 0, atol=1e-100)


@pytest.fixture(scope="module")
def line_fit() -> tuple[np.ndarray, dict, ModelCoefficients]:
    f = np.r_[0.0, np.geomspace(1e6, 40e9, 100)]
    # Causal square-root conductor loss, with its associated inductive dispersion.
    s = 2j * np.pi * f
    z = 100.0 + s * 400e-9 + 0.02 * np.sqrt(s)
    l = np.divide(z.imag, s.imag, out=np.full_like(f, 400e-9), where=f > 0)
    kwargs = {"R": z.real, "L": l, "G": 0.01, "C": 160e-12, "length": 0.002}
    return f, kwargs, fit_rlgc(f, **kwargs)


def test_skin_effect_reduction_is_checked_and_serializable(
    line_fit: tuple[np.ndarray, dict, ModelCoefficients], tmp_path: Path
) -> None:
    f, p, fit = line_fit
    assert fit.metadata["validation"]["status"] == "pass"
    assert fit.metadata["validation"]["validation_nrmse"] < 0.02
    assert fit.metadata["options"]["enforce_passivity"]
    assert fit.metadata["passivity_grid_maximum_singular_value"] <= 1 + 1e-9
    assert not fit.metadata["global_passivity_certified"]
    assert np.all(fit.poles.real < 0)
    path = tmp_path / "line.npz"
    fit.save(path)
    loaded = ModelCoefficients.load(path)
    assert loaded.metadata == fit.metadata
    np.testing.assert_allclose(loaded.evaluate(f), fit.evaluate(f))
    model = circuit_from_coefficients(loaded)
    actual = np.asarray(model.sp(ports=["p1", "p2"], freqs=jnp.asarray(f[::10])))
    np.testing.assert_allclose(actual, fit.evaluate(f[::10]), atol=5e-7)
    assert np.linalg.norm(actual - rlgc_sparameters(f, **p)[::10]) / np.linalg.norm(actual) < 0.02


def test_component_factory_connects_and_drives_a_native_step() -> None:
    f = np.r_[0.0, np.geomspace(1e6, 40e9, 80)]
    line = reduced_rlgc_transmission_line(f, R=100, L=400e-9, G=0.01, C=160e-12, length=0.002)
    assert set(line.ports) == {"p1", "p2"}
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "line": {"component": "line"},
            "VS": {
                "component": "drive",
                "settings": {
                    "kind": WAVE_PWL,
                    "pwl_t": jnp.array([0.0, 20e-12, 30e-12, 300e-12]),
                    "pwl_v": jnp.array([0.0, 0.0, 1.0, 1.0]),
                },
            },
            "RS": {"component": "R", "settings": {"R": 50.0}},
            "RL": {"component": "R", "settings": {"R": 50.0}},
        },
        "connections": {"VS,p1": "RS,p1", "VS,p2": "GND,p1", "RS,p2": "line,p1", "line,p2": "RL,p1", "RL,p2": "GND,p1"},
        "ports": {"out": "line,p2"},
    }
    circuit = compile_circuit(net, {"line": line, "R": Resistor, "drive": WaveformVoltageSource}, is_complex=True)
    solution = circuit.transient(t0=0.0, t1=300e-12, dt0=1e-12, saveat=jnp.linspace(0, 300e-12, 301), max_steps=310, throw=True)
    voltage = np.asarray(circuit.port(solution.ys, "out")).real
    assert np.isfinite(voltage).all()
    np.testing.assert_allclose(voltage[:21], 0, atol=1e-12)
    expected = rlgc_sparameters(f, R=100, L=400e-9, G=0.01, C=160e-12, length=0.002)[0, 1, 0] / 2
    np.testing.assert_allclose(voltage[-1], expected, atol=1e-4)


def test_native_delay_option_preserves_flight_time() -> None:
    f = np.linspace(0.0, 40e9, 101)
    tau = 0.002 * np.sqrt(400e-9 * 160e-12)
    fit = fit_rlgc(
        f,
        R=0,
        L=400e-9,
        G=0,
        C=160e-12,
        length=0.002,
        options=ModelFitOptions(delay_mode="supplied", port_delays=(tau / 2, tau / 2), enforce_passivity=True),
    )
    np.testing.assert_allclose(fit.port_delays, [tau / 2, tau / 2])
    np.testing.assert_allclose(fit.evaluate(f)[:, 1, 0], np.exp(-2j * np.pi * f * tau), atol=2e-6)
    assert np.all(fit.port_delays >= 0)


@pytest.mark.parametrize(
    "change",
    [
        {"R": -1},
        {"L": 1j},
        {"C": [1, 2, 3]},
        {"G": np.nan},
        {"length": 0},
        {"z0": -1},
        {"frequencies": [1, 0]},
        {"frequencies": [0, 1]},
        {"options": ModelFitOptions(reciprocal=False)},
        {"z0": 1j},
    ],
)
def test_bad_inputs_are_rejected(change: dict) -> None:
    p = {"frequencies": np.linspace(0.0, 40e9, 21), "R": 100, "L": 400e-9, "G": 0.01, "C": 160e-12, "length": 0.002}
    with pytest.raises(ValueError):
        fit_rlgc(**(p | change))


def test_callables_match_arrays() -> None:
    f = np.r_[0.0, np.geomspace(1e6, 10e9, 20)]
    a = rlgc_sparameters(f, R=lambda x: 100 + 0 * x, L=400e-9, G=0.0, C=160e-12, length=0.001)
    b = rlgc_sparameters(f, R=np.full_like(f, 100), L=400e-9, G=0.0, C=160e-12, length=0.001)
    np.testing.assert_array_equal(a, b)


def test_tight_accuracy_target_rejects_inadequate_order() -> None:
    with pytest.raises(ValueError, match=r"accuracy|limits"):
        fit_rlgc(
            np.linspace(0.0, 40e9, 61),
            R=100,
            L=400e-9,
            G=0.01,
            C=160e-12,
            length=0.02,
            options=ModelFitOptions(vector_fit_order=(1, 0), normalized_rmse=1e-6, max_absolute_error=1e-6, enforce_passivity=True),
        )
