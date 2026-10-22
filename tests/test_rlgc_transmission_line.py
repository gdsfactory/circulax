"""End-to-end tests for passive RLGC line synthesis."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm
from skrf.network import a2s

from circulax import compile_circuit
from circulax.components import rlgc_transmission_line
from circulax.components.electronic import Resistor, VoltageSource


def test_independent_dispersive_fits_and_native_circuit() -> None:
    """Different series/shunt relaxation rates must survive circuit realization."""
    f = np.r_[0.0, np.geomspace(1e6, 30e9, 80)]
    s = 2j * np.pi * f
    poles = np.array([2e8, 3e10])
    z = 800 + s * 400e-9 + 8000 * s / (s + poles[0])
    y = 0.02 + s * 120e-12 + 2 * s / (s + poles[1])
    l = np.divide(z.imag, s.imag, out=np.full_like(f, 400e-9 + 8000 / poles[0]), where=f > 0)
    c = np.divide(y.imag, s.imag, out=np.full_like(f, 120e-12 + 2 / poles[1]), where=f > 0)
    line = rlgc_transmission_line(f, R=z.real, L=l, G=y.real, C=c, poles=poles, tolerance=1e-8)
    np.testing.assert_allclose(line.series.evaluate(f), z, atol=1e-8)
    np.testing.assert_allclose(line.shunt.evaluate(f), y, atol=1e-11)
    length, sections = 60e-6, 2
    circuit = line(length=length, n_sections=sections)
    check = np.array([1e7, 5e9, 20e9])
    result = np.asarray(circuit.sp(ports=["p1", "p2"], freqs=jnp.asarray(check), z0=50.0))
    zz, yy = line.series.evaluate(check), line.shunt.evaluate(check)
    d = length / sections
    arm = np.tile(np.eye(2, dtype=complex), (len(check), 1, 1))
    shunt = arm.copy()
    arm[:, 0, 1], shunt[:, 1, 0] = zz * d / 2, yy * d
    expected = a2s(np.linalg.matrix_power(arm @ shunt @ arm, sections), z0=50.0)
    np.testing.assert_allclose(result, expected, atol=2e-8, rtol=1e-6)
    assert np.max(np.linalg.svd(result, compute_uv=False)) <= 1 + 1e-10
    # At this short cell length the ladder also matches the distributed line.
    exact = np.stack([expm(length * np.array([[0, zf], [yf, 0]])) for zf, yf in zip(zz, yy, strict=True)])
    np.testing.assert_allclose(result, a2s(exact, z0=50), atol=1e-4)
    assert set(line.netlist(length=length, n_sections=sections)["ports"]) == {"p1", "p2", "tap_0", "tap_1"}
    # A compiled section must also connect as a parent subcircuit.
    parent = compile_circuit(
        {
            "instances": {
                "line": {"component": "line"},
                "R": {"component": "R", "settings": {"R": 50}},
                "V": {"component": "V"},
                "GND": {"component": "ground"},
            },
            "connections": {"V,p1": "line,p1", "V,p2": "GND,p1", "R,p1": "line,p2", "R,p2": "GND,p1"},
            "ports": {"out": "line,p2"},
        },
        {"line": circuit, "R": Resistor, "V": VoltageSource},
    )
    ac = parent.ac_response(freqs=jnp.asarray(check[:1]), sources={"V": 1.0})
    assert np.isfinite(np.asarray(ac["solution"])).all()
    expected_h = 1 / (
        np.linalg.matrix_power(arm @ shunt @ arm, sections)[0, 0, 0]
        + np.linalg.matrix_power(arm @ shunt @ arm, sections)[0, 0, 1] / 50
    )
    np.testing.assert_allclose(ac["solution"][0, ac["port_map"]["out"]], expected_h, atol=1e-7)


def test_lossless_and_zero_terms_are_not_replaced_by_loss() -> None:
    f = np.r_[0.0, np.geomspace(1e6, 10e9, 40)]
    line = rlgc_transmission_line(f, R=0, L=400e-9, G=0, C=160e-12)
    net = line.netlist(length=100e-6)
    assert {v["component"] for v in net["instances"].values()} == {"ground", "L", "C"}
    for field in (line.series, line.shunt):
        assert field.constant == 0
        assert len(field.weights) == 0
    circuit = line(length=100e-6)
    scattering = np.asarray(circuit.sp(ports=["p1", "p2"], freqs=jnp.array([1e9]), z0=50.0))
    np.testing.assert_allclose(scattering.conj().swapaxes(1, 2) @ scattering, np.eye(2)[None], atol=1e-9)


def test_skin_effect_fit() -> None:
    f = np.geomspace(1e6, 100e9, 301)
    s = 2j * np.pi * f
    z = 1000 + s * 400e-9 + 0.02 * np.sqrt(s)
    line = rlgc_transmission_line(f, R=z.real, L=z.imag / s.imag, G=1e-3, C=150e-12, n_poles=20)
    assert line.series.normalized_rmse < 0.01
    assert line.shunt.normalized_rmse < 0.01
    assert np.all(line.series.weights >= 0)
    assert np.all(line.series.poles > 0)


@pytest.mark.parametrize(
    "change",
    [
        {"frequencies": [1e9, 1e8]},
        {"R": -1},
        {"C": [1, 2, 3]},
        {"L": 1j},
        {"R": np.nan},
        {"poles": [-1, 2]},
        {"poles": [2, 1]},
        {"n_poles": 0},
        {"tolerance": 0},
        {"R": 0, "L": 0},
    ],
)
def test_invalid_inputs(change: dict) -> None:
    kwargs = {"frequencies": np.geomspace(1e6, 1e9, 10), "R": 1000.0, "L": 400e-9, "G": 0.0, "C": 150e-12}
    with pytest.raises(ValueError):
        rlgc_transmission_line(**(kwargs | change))


def test_unrepresentable_data_raises_instead_of_silently_fitting() -> None:
    f = np.geomspace(1e6, 1e10, 100)
    with pytest.raises(ValueError, match="passive RLC fit NRMSE"):
        rlgc_transmission_line(f, R=np.linspace(1e5, 100, len(f)), L=1e-12, G=0, C=1e-12)
