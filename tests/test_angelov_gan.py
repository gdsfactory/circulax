"""Angelov GaN deterministic source modes, dynamic networks and OSDI parity."""

# ruff: noqa: ANN001, ANN201, ANN202, CPY001, FBT003, S603

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax.components.angelov_gan import AngelovGaN

jax.config.update("jax_enable_x64", True)

_BASE = {
    "Cgs0": 1e-12,
    "Cgd0": 0.3e-12,
    "Cgspi": 0.1e-12,
    "Cgdpi": 0.05e-12,
    "P2": 0.1,
    "P3": 0.03,
    "Lvg": 0.005,
    "P111": 0.1,
    "P222": 0.2,
    "P10pk": 1.2,
}
_CASES = [
    {},
    *[{**_BASE, "Idsmod": mode} for mode in range(5)],
    *[{**_BASE, "Capmod": mode} for mode in range(5)],
    {
        **_BASE,
        "Selft": 1,
        "Igmod": 1,
        "Ne": 1.3,
        "Trise": 25.0,
        "Tau": 1e-9,
        "Lg": 1e-9,
        "Ls": 0.5e-9,
        "Ld": 2e-9,
        "Crf": 2e-12,
        "Crfin": 1e-12,
        "Rd2": 0.1,
        "Kbgate": 0.1,
        "KRFDC": 0.4,
        "Idsmod": 4,
        "Rth": 10.0,
        "Cth": 1e-5,
    },
    {**_BASE, "Temp": 80.0, "Tnom": 25.0, "Pg": 12.0, "Igmod": 1, "_mfactor": 2.0},
]
_Y = jnp.array(
    [
        1.0,
        -0.2,
        0.0,
        10.0,
        0.95,
        0.98,
        0.99,
        -0.19,
        0.01,
        0.0,
        -0.2,
        -0.18,
        -0.19,
        -0.19,
        0.0,
        0.03,
        0.03,
        0.01,
        -0.01,
        0.03,
        1e-5,
        1e-6,
        1e-6,
        0.03,
        0.03,
        0.03,
        0.03,
    ]
)
_RAW_INDEX = jnp.array([*range(17), *range(19, 27)])
_Y = _Y[_RAW_INDEX]
_BIAS = jnp.stack([_Y, _Y.at[0].set(0.3).at[5].set(0.29).at[6].set(0.3), _Y.at[1].set(-0.8).at[7].set(-0.79).at[10:14].set(-0.79)])
_ELECTRICAL = jnp.array([0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])


def _evaluate(model, y):
    return model.solver_call(0.0, y, model)


_batch = jax.jit(jax.vmap(_evaluate, in_axes=(None, 0)))
_jacobian = jax.jit(jax.vmap(jax.jacfwd(_evaluate, argnums=1), in_axes=(None, 0)))


@pytest.mark.parametrize("settings", _CASES)
def test_finite_conservative_eager_jit_and_jacobians(settings):
    model = AngelovGaN(**settings)
    f, q = _batch(model, _BIAS)
    assert np.isfinite(f).all()
    assert np.isfinite(q).all()
    np.testing.assert_allclose(f[:, _ELECTRICAL].sum(axis=-1), 0, atol=2e-15)
    np.testing.assert_allclose(q[:, _ELECTRICAL].sum(axis=-1), 0, atol=1e-26)
    eager = _evaluate(model, _Y)
    np.testing.assert_allclose(eager[0], f[0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(eager[1], q[0], rtol=1e-10, atol=1e-25)
    g, c = _jacobian(model, _BIAS)
    assert np.isfinite(g).all()
    assert np.isfinite(c).all()


def test_physics_modes_are_active_and_parameters_differentiable():
    currents = [float(_evaluate(AngelovGaN(**{**_BASE, "Idsmod": mode}), _Y)[0][15]) for mode in range(5)]
    charges = [float(_evaluate(AngelovGaN(**{**_BASE, "Capmod": mode}), _Y)[1][11]) for mode in range(5)]
    assert len(set(currents)) == 5
    assert len(set(charges)) == 5

    def current(ipk):
        return _evaluate(AngelovGaN(Ipk0=ipk), _Y)[0][15]

    value, step = 0.05, 1e-6
    derivative = jax.jit(jax.grad(current))(value)
    expected = (current(value + step) - current(value - step)) / (2 * step)
    assert np.isfinite(derivative)
    assert derivative != 0
    np.testing.assert_allclose(derivative, expected, rtol=1e-8)


def test_delay_and_thermal_storage_and_presence_semantics():
    m = AngelovGaN(Selft=1, Tau=3e-9, Cth=2e-5, Lg=1e-9, Ls=2e-9, Ld=3e-9)
    _, q = _evaluate(m, _Y)
    assert q[3] == pytest.approx(2e-5 * _Y[3])
    assert q[15] == pytest.approx(3e-9 * _Y[15])
    assert q[17] == pytest.approx(1e-9 * _Y[17])
    assert q[20] == pytest.approx(1e-9 * _Y[20])
    assert q[22] == pytest.approx(2e-9 * _Y[22])
    assert q[24] == pytest.approx(3e-9 * _Y[24])
    for n, p in zip(_evaluate(AngelovGaN(Trise=50.0), _Y), _evaluate(AngelovGaN(Temp=77.0, Trise=99.0), _Y), strict=True):
        np.testing.assert_allclose(n, p, rtol=1e-12, atol=1e-25)
    for bad in ({"Rc": 0.0}, {"Rcin": 0.0}, {"Rdel": 0.0}, {"Rg": 0.0}, {"Capmod": 5}, {"Idsmod": 5}):
        with pytest.raises(ValueError):
            AngelovGaN(**bad)
    with pytest.raises(TypeError):
        AngelovGaN(Noise=1)


@pytest.fixture(scope="module")
def osdi_reference(tmp_path_factory):
    pytest.importorskip("osdi_loader")
    pytest.importorskip("osdi_jax")
    from osdi_loader import load_osdi_model

    compiler = shutil.which("openvaf-r")
    if compiler is None:
        pytest.skip("openvaf-r required for independent OSDI verification")
    source = Path(__file__).parent / "data/va_models/angelov_gan/angelov_gan.va"
    binary = tmp_path_factory.mktemp("angelov_gan") / "angelov_gan.osdi"
    subprocess.run([compiler, str(source), "-o", str(binary)], check=True, capture_output=True)
    return load_osdi_model(str(binary), temperature=300.15)


@pytest.mark.parametrize("settings", _CASES)
def test_all_source_f_q_g_c(osdi_reference, settings):
    from osdi_jax import osdi_eval

    ref = osdi_reference
    overrides = {"Noise": 0, "Noimod": 0, **{"$mfactor" if k == "_mfactor" else k: v for k, v in settings.items()}}
    params = jnp.array([[overrides.get(name, np.nan) for name in ref.param_names]])
    params = jnp.repeat(params, len(_BIAS), axis=0)
    raw_bias = jnp.zeros((len(_BIAS), 27)).at[:, _RAW_INDEX].set(_BIAS)
    rf, rg, rq, rc, _ = osdi_eval(ref.id, raw_bias, params, jnp.zeros((len(_BIAS), ref.num_states)))
    model = AngelovGaN(**settings)
    f, q = _batch(model, _BIAS)
    g, c = _jacobian(model, _BIAS)
    np.testing.assert_allclose(f, rf[:, _RAW_INDEX], rtol=2e-10, atol=1e-12)
    np.testing.assert_allclose(q, rq[:, _RAW_INDEX], rtol=2e-10, atol=1e-25)
    np.testing.assert_allclose(g, rg.reshape((-1, 27, 27))[:, _RAW_INDEX, :][:, :, _RAW_INDEX], rtol=2e-9, atol=1e-12)
    np.testing.assert_allclose(c, rc.reshape((-1, 27, 27))[:, _RAW_INDEX, :][:, :, _RAW_INDEX], rtol=2e-9, atol=1e-24)


def _circuit(model, heated):
    from circulax import compile_circuit
    from circulax.components.electronic import Resistor, VoltageSource

    settings = {
        **_BASE,
        "Selft": heated,
        "Rth": 10.0,
        "Cth": 1e-5,
        "Tau": 1e-9,
        "Lg": 1e-9,
        "Ls": 0.5e-9,
        "Ld": 2e-9,
        "Crf": 2e-12,
        "Crfin": 1e-12,
    }
    net = {
        "instances": {
            "M": {"component": "gan", "settings": settings},
            "R": {"component": "R", "settings": {"R": 10.0}},
            "VD": {"component": "V", "settings": {"V": 2.0}},
            "VG": {"component": "V", "settings": {"V": -0.5}},
            "GND": {"component": "ground"},
        },
        "connections": {"VD,p1": "R,p1", "R,p2": "M,d", "VG,p1": "M,g", "GND,p1": ("VD,p2", "VG,p2", "M,s")},
        "ports": {"out": "M,d", "thermal": "M,t", "rf": "M,rf"},
    }
    return compile_circuit(net, {"gan": model, "R": Resistor, "V": VoltageSource})


@pytest.mark.parametrize("heated", [0, 1])
def test_actual_dc_ac_circuit(heated):
    circuit = _circuit(AngelovGaN, heated)
    y = circuit.dc()
    assert np.isfinite(y).all()
    assert 0 < circuit.port(y, "M,d") < 2
    if heated:
        assert circuit.port(y, "M,t") > 0
    ac = circuit.ac_response(freqs=jnp.array([1e3, 1e6]), sources={"VG": 1.0})
    assert np.isfinite(ac["solution"]).all()
    assert np.any(abs(ac["solution"][:, ac["port_map"]["M,d"]]) > 0)


@pytest.mark.parametrize("heated", [0, 1])
def test_actual_osdi_dc_ac_circuit(osdi_reference, heated):
    pytest.importorskip("bosdi.circulax")
    from bosdi.circulax import osdi_component

    descriptor = osdi_component(
        osdi_reference.path, ("d", "g", "s", "t", "rf"), temperature=300.15, default_params={"Noise": 0, "Noimod": 0}
    )
    native, reference = _circuit(AngelovGaN, heated), _circuit(descriptor, heated)
    yn, yr = native.dc(), reference.dc()
    for port in ("M,d", "M,t", "M,rf"):
        np.testing.assert_allclose(native.port(yn, port), reference.port(yr, port), rtol=1e-7, atol=1e-9)
    an = native.ac_response(freqs=jnp.array([1e3, 1e6]), sources={"VG": 1.0})
    ar = reference.ac_response(freqs=jnp.array([1e3, 1e6]), sources={"VG": 1.0})
    for port in ("M,d", "M,t", "M,rf"):
        np.testing.assert_allclose(
            an["solution"][:, an["port_map"][port]], ar["solution"][:, ar["port_map"][port]], rtol=1e-7, atol=1e-10
        )


def test_time_domain_current_and_storage_rate_match_osdi(osdi_reference):
    from osdi_jax import osdi_eval

    settings = _CASES[11]
    reference = osdi_reference
    overrides = {"Noise": 0, "Noimod": 0, **settings}
    params = jnp.array([[overrides.get(name, np.nan) for name in reference.param_names]])
    params = jnp.repeat(params, len(_BIAS), axis=0)
    raw_bias = jnp.zeros((len(_BIAS), 27)).at[:, _RAW_INDEX].set(_BIAS)
    rf, _, _, rc, _ = osdi_eval(reference.id, raw_bias, params, jnp.zeros((len(_BIAS), reference.num_states)))
    ref_c = rc.reshape((-1, 27, 27))[:, _RAW_INDEX, :][:, :, _RAW_INDEX]
    model = AngelovGaN(**settings)
    f, _ = _batch(model, _BIAS)
    _, c = _jacobian(model, _BIAS)
    dydt = jnp.broadcast_to(jnp.linspace(-1e6, 1e6, 25), _BIAS.shape).at[:, 3].set(0.5)
    actual = f + jnp.einsum("nij,nj->ni", c, dydt)
    expected = rf[:, _RAW_INDEX] + jnp.einsum("nij,nj->ni", ref_c, dydt)
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=1e-10)


def test_parameter_presence_changes_after_pytree_update():
    import equinox as eqx

    for name, value in (("Temp", 80.0), ("Tnom", 25.0), ("Pg", 12.0), ("Ne", 1.3)):
        baseline = AngelovGaN()
        changed = eqx.tree_at(lambda model, name=name: getattr(model, name), baseline, value)
        explicit = AngelovGaN(**{name: value})
        for actual, expected in zip(_evaluate(changed, _Y), _evaluate(explicit, _Y), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-25)

    def current_at_temperature(temperature):
        changed = eqx.tree_at(lambda model: model.Temp, AngelovGaN(), temperature)
        return _evaluate(changed, _Y)[0][15]

    derivative = jax.jit(jax.grad(current_at_temperature))(50.0)
    assert np.isfinite(derivative)
    assert derivative != 0
