"""Native EKV26 NMOS/PMOS DC, charge dynamics and independent OSDI parity."""

# ruff: noqa: ANN001, ANN201, ANN202, CPY001, FBT003, S603

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax.components.ekv import EKV26

jax.config.update("jax_enable_x64", True)

_NBIAS = jnp.array([[0.1, 0.1, 0, 0], [0.05, 1.2, 0, 0], [1.2, 1.2, 0, 0], [-0.4, 1.0, 0, 0], [0.6, 0.8, 0.2, -0.2], [0, 0, 0, 0]])
_CASES = [
    {},
    {"TYPE": -1, "VTO": -0.5},
    {"W": 3e-6, "L": 1e-6, "NS": 2, "M": 3, "DTEMP": 50.0, "_mfactor": 2.0},
    {"TEMP": 80.0, "TNOM": 25.0, "DTEMP": 30.0, "RSH": 100.0, "NRS": 2.0, "NRD": 3.0},
    {"HDIF": 1e-6, "RSH": 200.0, "cj": 1e-3, "cjsw": 1e-10, "cjswg": 2e-10},
    {
        "HDIF": 2e-6,
        "AS": 2e-12,
        "AD": 3e-12,
        "PS": 4e-6,
        "PD": 5e-6,
        "cj": 1e-3,
        "cjsw": 1e-10,
        "cjswg": 2e-10,
        "js": 1e-3,
        "jsw": 1e-8,
    },
    {"THETA": 0.2, "IBA": 8e8, "AVTO": 1.1e-6, "AKP": 1.01e-6, "AGAMMA": 1.01e-6},
    {"Noise": 0},
]


def _evaluate(model, y):
    return model.solver_call(0.0, y, model)


_batch = jax.jit(jax.vmap(_evaluate, in_axes=(None, 0)))
_jacobian = jax.jit(jax.vmap(jax.jacfwd(_evaluate, argnums=1), in_axes=(None, 0)))


def _bias(settings):
    return settings.get("TYPE", 1) * _NBIAS


@pytest.mark.parametrize("settings", _CASES)
def test_native_conservation_common_mode_jit_and_finite_ad(settings):
    model = EKV26(**settings)
    biases = _bias(settings)
    f, q = _batch(model, biases)
    assert np.isfinite(f).all()
    assert np.isfinite(q).all()
    np.testing.assert_allclose(f.sum(axis=-1), 0, atol=1e-16)
    np.testing.assert_allclose(q.sum(axis=-1), 0, atol=1e-27)
    g, c = _jacobian(model, biases)
    assert np.isfinite(g).all()
    assert np.isfinite(c).all()
    translated = _batch(model, biases + 0.7)
    np.testing.assert_allclose(translated[0], f, rtol=1e-10, atol=1e-16)
    np.testing.assert_allclose(translated[1], q, rtol=1e-10, atol=1e-25)
    eager = _evaluate(model, biases[2])
    np.testing.assert_allclose(eager[0], f[2], rtol=1e-10, atol=1e-16)
    np.testing.assert_allclose(eager[1], q[2], rtol=1e-10, atol=1e-25)


def test_dynamic_pmos_and_channel_polarity():
    caps = {"Cgso": 0.0, "Cgdo": 0.0, "Cgbo": 0.0}
    n = _batch(EKV26(**caps), _NBIAS)
    p = _batch(EKV26(TYPE=-1, VTO=-0.5, **caps), -_NBIAS)
    # EKV has polarity-specific mobility/charge coefficients; compare signs,
    # while the independent reference tests verify each polarity exactly.
    assert np.all(p[0][:, 0] * n[0][:, 0] <= 0)
    assert np.all(p[1][:, 1] * n[1][:, 1] <= 0)
    assert abs(p[1][2, 1]) > 0
    assert abs(n[0][2, 0]) > abs(n[0][0, 0])


def test_body_overlap_and_geometry_parameter_gradients():
    y = jnp.array([1.0, 1.0, 0.0, 0.0])
    default = _evaluate(EKV26(), y)
    body = _evaluate(EKV26(), y.at[3].set(-0.5))
    assert abs(body[0][0]) < abs(default[0][0])
    no_overlap = _evaluate(EKV26(Cgso=0.0, Cgdo=0.0, Cgbo=0.0), y)
    assert default[1][1] != no_overlap[1][1]

    def channel_current(width):
        return _evaluate(EKV26(W=width), y)[0][0]

    width, step = 3e-6, 1e-10
    grad = jax.jit(jax.grad(channel_current))(width)
    expected = (channel_current(width + step) - channel_current(width - step)) / (2 * step)
    assert np.isfinite(grad)
    assert grad != 0
    np.testing.assert_allclose(grad, expected, rtol=1e-6)


def test_param_given_temperature_and_diffusion_geometry():
    y = jnp.array([1.0, 1.0, 0.0, 0.0])
    shifted = _evaluate(EKV26(DTEMP=50.0), y)
    explicit = _evaluate(EKV26(TEMP=77.0, DTEMP=99.0), y)
    np.testing.assert_allclose(shifted[0], explicit[0], rtol=1e-12)
    np.testing.assert_allclose(shifted[1], explicit[1], rtol=1e-12)
    inferred = _evaluate(EKV26(HDIF=1e-6, cj=0.001), y)
    explicit_zero = _evaluate(EKV26(HDIF=1e-6, AD=0.0, AS=0.0, cj=0.001), y)
    assert not np.array_equal(inferred[1], explicit_zero[1])
    square_r = _evaluate(EKV26(RSH=100.0, NRS=2.0, NRD=3.0), y)[0][0]
    no_r = _evaluate(EKV26(), y)[0][0]
    assert square_r < no_r


@pytest.fixture(scope="module")
def osdi_reference(tmp_path_factory):
    pytest.importorskip("osdi_loader")
    pytest.importorskip("osdi_jax")
    from osdi_loader import load_osdi_model

    compiler = shutil.which("openvaf-r")
    if compiler is None:
        pytest.skip("openvaf-r required for compiled reference verification")
    source = Path(__file__).parent / "data/va_models/ekv26/ekv26.va"
    binary = tmp_path_factory.mktemp("ekv26_osdi") / "ekv26.osdi"
    subprocess.run([compiler, str(source), "-o", str(binary)], check=True, capture_output=True)
    return load_osdi_model(str(binary), temperature=300.15)


@pytest.mark.parametrize("settings", _CASES)
def test_osdi_f_q_g_c(osdi_reference, settings):
    from osdi_jax import osdi_eval

    ref = osdi_reference
    overrides = {"$mfactor" if k == "_mfactor" else k: v for k, v in settings.items()}
    p = jnp.array([[overrides.get(name, np.nan) for name in ref.param_names]])
    p = jnp.repeat(p, len(_NBIAS), axis=0)
    biases = _bias(settings)
    rf, rg, rq, rc, _ = osdi_eval(ref.id, biases, p, jnp.zeros((len(biases), ref.num_states)))
    model = EKV26(**settings)
    f, q = _batch(model, biases)
    g, c = _jacobian(model, biases)
    np.testing.assert_allclose(f, rf, rtol=2e-10, atol=1e-16)
    np.testing.assert_allclose(q, rq, rtol=2e-10, atol=1e-25)
    np.testing.assert_allclose(g, rg.reshape((-1, 4, 4)), rtol=2e-9, atol=1e-15)
    np.testing.assert_allclose(c, rc.reshape((-1, 4, 4)), rtol=2e-9, atol=1e-24)


def _circuit(model, polarity=1):
    from circulax import compile_circuit
    from circulax.components.electronic import Resistor, VoltageSource

    net = {
        "instances": {
            "M": {"component": "mos", "settings": {"TYPE": polarity, "VTO": 0.5 * polarity}},
            "R": {"component": "R", "settings": {"R": 1000.0}},
            "VD": {"component": "V", "settings": {"V": 1.2 * polarity}},
            "VG": {"component": "V", "settings": {"V": 0.8 * polarity}},
            "GND": {"component": "ground"},
        },
        "connections": {"VD,p1": "R,p1", "R,p2": "M,d", "VG,p1": "M,g", "GND,p1": ("M,s", "M,b", "VD,p2", "VG,p2")},
        "ports": {"out": "M,d"},
    }
    return compile_circuit(net, {"mos": model, "R": Resistor, "V": VoltageSource})


@pytest.mark.parametrize("polarity", [-1, 1])
def test_actual_circuit_dc_and_ac(polarity):
    circuit = _circuit(EKV26, polarity)
    y = circuit.dc()
    output = polarity * circuit.port(y, "M,d")
    assert 0 < output < 1.2
    ac = circuit.ac_response(freqs=jnp.array([1e3, 1e6]), sources={"VG": 1.0})
    assert np.isfinite(ac["solution"]).all()
    response = ac["solution"][:, ac["port_map"]["M,d"]]
    assert np.all(np.real(response) < 0)
    assert np.any(np.imag(response) != 0)


@pytest.mark.parametrize("polarity", [-1, 1])
def test_actual_circuit_osdi_dc_and_ac(osdi_reference, polarity):
    pytest.importorskip("bosdi.circulax")
    from bosdi.circulax import osdi_component

    descriptor = osdi_component(osdi_reference.path, ("d", "g", "s", "b"), temperature=300.15)
    n, r = _circuit(EKV26, polarity), _circuit(descriptor, polarity)
    yn, yr = n.dc(), r.dc()
    np.testing.assert_allclose(n.port(yn, "M,d"), r.port(yr, "M,d"), rtol=1e-9, atol=1e-12)
    an = n.ac_response(freqs=jnp.array([1e3, 1e6]), sources={"VG": 1.0})
    ar = r.ac_response(freqs=jnp.array([1e3, 1e6]), sources={"VG": 1.0})
    np.testing.assert_allclose(
        an["solution"][:, an["port_map"]["M,d"]], ar["solution"][:, ar["port_map"]["M,d"]], rtol=1e-8, atol=1e-12
    )
