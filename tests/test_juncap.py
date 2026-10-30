"""Native JUNCAP200 current/charge and optional independent OSDI parity."""

# ruff: noqa: ANN001, ANN201, ANN202, CPY001, FBT003, S603

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax.components.juncap import JUNCAP200

jax.config.update("jax_enable_x64", True)

_BIAS = jnp.array([[-2.0, 0.0], [-0.5, 0.0], [0.0, 0.0], [0.3, 0.0], [0.7, 0.0]])
_CASES = [
    {},
    {"SWJUNEXP": 1.0},
    {"TYPE": -1},
    {"TRISE": 60.0, "AB": 2e-12, "LS": 4e-6, "LG": 3e-6, "MULT": 2.0},
]


def _evaluate(device, voltage):
    return device.solver_call(0.0, voltage, device)


@pytest.mark.parametrize("settings", _CASES)
def test_native_current_charge_conserve_and_differentiate(settings):
    device = JUNCAP200(**settings)
    f, q = jax.jit(jax.vmap(lambda v: _evaluate(device, v)))(_BIAS)
    assert np.isfinite(f).all()
    assert np.isfinite(q).all()
    np.testing.assert_array_equal(f.sum(axis=-1), 0.0)
    np.testing.assert_array_equal(q.sum(axis=-1), 0.0)
    assert np.any(np.asarray(f) != 0.0)
    assert np.any(np.asarray(q) != 0.0)
    g, c = jax.jacfwd(lambda v: _evaluate(device, v))(jnp.array([0.3, 0.0]))
    assert np.isfinite(g).all()
    assert np.isfinite(c).all()
    assert c[0, 0] > 0.0
    np.testing.assert_allclose(g.sum(axis=-1), 0.0, atol=1e-20)
    np.testing.assert_allclose(c.sum(axis=-1), 0.0, atol=1e-25)


def test_polarity_and_parallel_scaling():
    v = jnp.array([0.4, 0.0])
    positive = _evaluate(JUNCAP200(), v)
    negative = _evaluate(JUNCAP200(TYPE=-1), -v)
    parallel = _evaluate(JUNCAP200(MULT=3.0), v)
    for p, n, m in zip(positive, negative, parallel, strict=True):
        np.testing.assert_allclose(n, -p, rtol=1e-13)
        np.testing.assert_allclose(m, 3.0 * p, rtol=1e-13)


def test_area_parameter_gradient_includes_initialization():
    v = jnp.array([0.4, 0.0])

    def current(area):
        return _evaluate(JUNCAP200(AB=area), v)[0][0]

    area = 1e-12
    step = 1e-16
    derivative = jax.grad(current)(area)
    finite_difference = (current(area + step) - current(area - step)) / (2.0 * step)
    assert np.isfinite(derivative)
    assert derivative != 0.0
    np.testing.assert_allclose(derivative, finite_difference, rtol=1e-7)


def test_native_compiles_into_circulax_netlist():
    from circulax import compile_netlist

    net = {
        "instances": {"D": {"component": "junction"}, "GND": {"component": "ground"}},
        "connections": {"D,K": "GND,p1"},
        "ports": {"anode": "D,A"},
    }
    groups, size, port_map = compile_netlist(net, {"junction": JUNCAP200})
    assert len(groups) == 1
    assert size >= 1
    assert "D,A" in port_map


@pytest.fixture(scope="module")
def osdi_reference(tmp_path_factory):
    pytest.importorskip("bosdi")
    pytest.importorskip("osdi_jax")
    from bosdi.circulax import osdi_component

    compiler = shutil.which("openvaf-r")
    if compiler is None:
        pytest.skip("openvaf-r required for independent OSDI verification")
    source = Path(__file__).parent / "data/va_models/juncap/juncap200.va"
    binary = tmp_path_factory.mktemp("juncap") / "juncap200.osdi"
    subprocess.run([compiler, str(source), "-o", str(binary)], check=True, capture_output=True, text=True)
    return osdi_component(str(binary), ("A", "K"), temperature=300.15)


@pytest.mark.parametrize("settings", _CASES)
def test_osdi_current_charge_conductance_capacitance(osdi_reference, settings):
    from osdi_jax import osdi_eval

    descriptor = osdi_reference
    device = JUNCAP200(**settings)
    params = jnp.array([[settings.get(name, np.nan) for name in descriptor.param_names]])
    params = jnp.repeat(params, len(_BIAS), axis=0)
    f_ref, g_ref, q_ref, c_ref, _ = osdi_eval(
        descriptor.model.id,
        _BIAS,
        params,
        jnp.zeros((len(_BIAS), descriptor.model.num_states)),
    )
    f, q = jax.jit(jax.vmap(lambda v: _evaluate(device, v)))(_BIAS)
    g, c = jax.jit(jax.vmap(jax.jacfwd(lambda v: _evaluate(device, v))))(_BIAS)
    np.testing.assert_allclose(f, f_ref, rtol=2e-11, atol=1e-22)
    np.testing.assert_allclose(q, q_ref, rtol=2e-11, atol=1e-27)
    np.testing.assert_allclose(g, g_ref.reshape((-1, 2, 2)), rtol=2e-10, atol=1e-20)
    np.testing.assert_allclose(c, c_ref.reshape((-1, 2, 2)), rtol=2e-10, atol=1e-26)


def test_time_varying_terminal_current_matches_osdi(osdi_reference):
    """Compare F + dQ/dV dV/dt along a prescribed time-domain waveform."""
    from osdi_jax import osdi_eval

    descriptor = osdi_reference
    device = JUNCAP200()
    omega = 2.0 * jnp.pi * 1e6
    phase = jnp.linspace(0.0, 2.0 * jnp.pi, 13)
    volts = jnp.stack([0.2 + 0.3 * jnp.sin(phase), jnp.zeros_like(phase)], axis=-1)
    dvdt = jnp.stack([0.3 * omega * jnp.cos(phase), jnp.zeros_like(phase)], axis=-1)
    params = jnp.full((len(phase), len(descriptor.param_names)), jnp.nan)
    f_ref, _, _, c_ref, _ = osdi_eval(
        descriptor.model.id,
        volts,
        params,
        jnp.zeros((len(phase), descriptor.model.num_states)),
    )
    f, _ = jax.jit(jax.vmap(lambda v: _evaluate(device, v)))(volts)
    _, c = jax.jit(jax.vmap(jax.jacfwd(lambda v: _evaluate(device, v))))(volts)
    actual = f + jnp.einsum("nij,nj->ni", c, dvdt)
    expected = f_ref + jnp.einsum("nij,nj->ni", c_ref.reshape((-1, 2, 2)), dvdt)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=1e-21)


def test_eager_and_compiled_initialization_agree():
    voltage = jnp.array([0.4, 0.0])
    for settings in ({}, {"SWJUNEXP": 1.0}, {"TYPE": -1}):
        device = JUNCAP200(**settings)
        eager = _evaluate(device, voltage)
        traced = jax.jit(lambda v, device=device: _evaluate(device, v))(voltage)
        for expected, actual in zip(eager, traced, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-25)
