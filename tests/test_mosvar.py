"""Native MOSVAR finite-resistance charge dynamics and compiled OSDI parity."""

# ruff: noqa: ANN001, ANN201, ANN202, CPY001, FBT003, S603

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax.components.mosvar import MOSVAR

jax.config.update("jax_enable_x64", True)

_CASES = [
    {},
    {"TYPE": 1, "TYPEP": 1},
    {"TYPE": 1, "TYPEP": -1, "NPO": 1e25},
    {"QMC": 0.0, "TAU": 0.02},
    {
        "W": 3e-6,
        "L": 2e-6,
        "NGCON": 2,
        "DTA": 60.0,
        "CFRL": 1e-10,
        "CFRW": 2e-10,
        "RPV": 1e-12,
        "RSHS": 700.0,
        "REND": 2e-4,
        "_mfactor": 2.0,
    },
    {"DNSUBO": 0.2, "VNSUBO": 0.1, "MNSUBO": 1.2, "UACRED": 0.1},
    {"SWIGATE": 1, "IGINVLW": 0.1, "IGOVW": 1e-8, "IGCHVLW": 0.1, "IGOVHVW": 1e-8, "LOV": 1e-8, "GCOO": 1.0, "GCOHVO": 1.0},
]
_BIAS = jnp.array(
    [
        [-1.0, 0.0, 0.0, -0.99, -0.98, 0.02, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.01, 0.02, 0.01, -0.001, 0.01, 0.0],
        [0.5, 0.0, 0.0, 0.49, 0.48, 0.02, -0.005, 0.02, 0.0],
        [1.0, 0.0, -0.01, 0.99, 0.98, -0.02, -0.01, -0.01, 0.0],
    ]
)


def _evaluate(model, y):
    return model.solver_call(0.0, y, model)


_batch = jax.jit(jax.vmap(_evaluate, in_axes=(None, 0)))
_batch_jacobian = jax.jit(jax.vmap(jax.jacfwd(_evaluate, argnums=1), in_axes=(None, 0)))


@pytest.mark.parametrize("settings", _CASES)
def test_conservation_and_eager_jit_parity(settings):
    device = MOSVAR(**settings)
    f, q = _batch(device, _BIAS)
    assert np.isfinite(f).all()
    assert np.isfinite(q).all()
    # First six rows are electrical KCL; inversion-relaxation and current
    # observation equations are auxiliary constraints, not terminal flows.
    np.testing.assert_allclose(f[:, :6].sum(axis=-1), 0.0, atol=1e-13)
    np.testing.assert_allclose(q[:, :6].sum(axis=-1), 0.0, atol=1e-27)
    eager = _evaluate(device, _BIAS[2])
    np.testing.assert_allclose(eager[0], f[2], rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(eager[1], q[2], rtol=1e-10, atol=1e-25)
    g, c = _batch_jacobian(device, _BIAS)
    assert np.isfinite(g).all()
    assert np.isfinite(c).all()
    assert np.any(q != 0)


def test_internal_relaxation_and_geometry_parameter_gradients():
    y = _BIAS[2]
    _f, q = _evaluate(MOSVAR(TAU=0.02), y)
    assert q[6] == pytest.approx(0.02 * y[6])
    inversion_y = _BIAS[0]
    assert _evaluate(MOSVAR(), inversion_y)[0][6] != inversion_y[6]  # Static inversion source.

    def gate_charge(width):
        return _evaluate(MOSVAR(W=width), y)[1][4]

    width, step = 2e-6, 1e-10
    derivative = jax.jit(jax.grad(gate_charge))(width)
    finite_difference = (gate_charge(width + step) - gate_charge(width - step)) / (2 * step)
    assert np.isfinite(derivative)
    assert derivative != 0
    np.testing.assert_allclose(derivative, finite_difference, rtol=2e-7)


def test_selected_physics_switches_change_equations():
    y = _BIAS[2]
    default = _evaluate(MOSVAR(), y)
    quantum_off = _evaluate(MOSVAR(QMC=0.0), y)
    assert not np.isclose(default[1][4], quantum_off[1][4], rtol=1e-5, atol=1e-30)
    tunnel = {"SWIGATE": 1, "IGINVLW": 0.1, "IGOVW": 1e-8, "IGCHVLW": 0.1, "IGOVHVW": 1e-8, "LOV": 1e-8, "GCOO": 1.0, "GCOHVO": 1.0}
    tunnel_y = jnp.array([5.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0])
    enabled = _evaluate(MOSVAR(**tunnel), tunnel_y)
    disabled = _evaluate(MOSVAR(**{**tunnel, "SWIGATE": 0}), tunnel_y)
    assert not np.array_equal(enabled[0], disabled[0])
    with pytest.raises(TypeError):
        MOSVAR(SWRES=0)


@pytest.fixture(scope="module")
def osdi_reference(tmp_path_factory):
    pytest.importorskip("osdi_loader")
    pytest.importorskip("osdi_jax")
    from osdi_loader import load_osdi_model

    compiler = shutil.which("openvaf-r")
    if compiler is None:
        pytest.skip("openvaf-r required for independent reference verification")
    source = Path(__file__).parent / "data/va_models/mosvar/mosvar.va"
    binary = tmp_path_factory.mktemp("mosvar_osdi") / "mosvar.osdi"
    subprocess.run([compiler, str(source), "-o", str(binary)], check=True, capture_output=True)
    return load_osdi_model(str(binary), temperature=300.15)


@pytest.mark.parametrize("settings", _CASES)
def test_osdi_f_q_g_c_parity(osdi_reference, settings):
    from osdi_jax import osdi_eval

    model = osdi_reference
    mapping = {"_mfactor": "$mfactor", "_temperature": None}
    overrides = {mapping.get(k, k): v for k, v in settings.items()}
    overrides["SWRES"] = 1
    params = jnp.array([[overrides.get(name, np.nan) for name in model.param_names]])
    params = jnp.repeat(params, len(_BIAS), axis=0)
    rf, rg, rq, rc, _ = osdi_eval(model.id, _BIAS, params, jnp.zeros((len(_BIAS), model.num_states)))
    device = MOSVAR(**settings)
    f, q = _batch(device, _BIAS)
    g, c = _batch_jacobian(device, _BIAS)
    np.testing.assert_allclose(f, rf, rtol=2e-10, atol=1e-12)
    np.testing.assert_allclose(q, rq, rtol=2e-10, atol=1e-25)
    np.testing.assert_allclose(g, rg.reshape((-1, 9, 9)), rtol=2e-9, atol=1e-12)
    np.testing.assert_allclose(c, rc.reshape((-1, 9, 9)), rtol=2e-9, atol=1e-24)


def test_time_varying_charge_current_osdi_parity(osdi_reference):
    from osdi_jax import osdi_eval

    model = osdi_reference
    params = jnp.full((len(_BIAS), model.num_params), jnp.nan)
    rf, _, _, rc, _ = osdi_eval(model.id, _BIAS, params, jnp.zeros((len(_BIAS), model.num_states)))
    f, _ = _batch(MOSVAR(), _BIAS)
    _, c = _batch_jacobian(MOSVAR(), _BIAS)
    dydt = jnp.broadcast_to(jnp.array([1e6, 0, 0, 1e6, 1e6, 0, 0.01, 0, 0]), _BIAS.shape)
    current = f + jnp.einsum("nij,nj->ni", c, dydt)
    reference = rf + jnp.einsum("nij,nj->ni", rc.reshape((-1, 9, 9)), dydt)
    np.testing.assert_allclose(current, reference, rtol=2e-9, atol=1e-12)


def _biased_circuit(model):
    from circulax import compile_circuit
    from circulax.components.electronic import VoltageSource

    net = {
        "instances": {
            "M": {"component": "MOSVAR"},
            "V": {"component": "V", "settings": {"V": 0.5}},
            "GND": {"component": "ground"},
        },
        "connections": {"V,p1": "M,g", "GND,p1": ("V,p2", "M,bi", "M,b")},
        "ports": {"gate": "M,g"},
    }
    return compile_circuit(net, {"MOSVAR": model, "V": VoltageSource})


def test_circuit_dc_and_ac_with_internal_state_network():
    circuit = _biased_circuit(MOSVAR)
    solution = circuit.dc()
    assert np.isfinite(solution).all()
    assert circuit.port(solution, "M,g") == pytest.approx(0.5)
    assert circuit.port(solution, "M,v_gi") == pytest.approx(0.5, abs=1e-8)
    assert circuit.port(solution, "V,i_src") == pytest.approx(0, abs=1e-12)
    ac = circuit.ac_response(freqs=jnp.array([1e3, 1e6]), sources={"V": 1.0})
    assert np.isfinite(ac["solution"]).all()
    source_current = ac["solution"][:, ac["port_map"]["V,i_src"]]
    assert np.all(np.imag(source_current) < 0)  # Capacitive input current leaves the voltage source.
    assert abs(source_current[1]) > 100 * abs(source_current[0])


def test_circuit_dc_and_ac_osdi_parity(osdi_reference):
    pytest.importorskip("bosdi.circulax")
    from bosdi.circulax import osdi_component

    descriptor = osdi_component(osdi_reference.path, ("g", "bi", "b"), temperature=300.15, default_params={"SWRES": 1})
    native = _biased_circuit(MOSVAR)
    reference = _biased_circuit(descriptor)
    native_dc, reference_dc = native.dc(), reference.dc()
    for port in ("M,g", "V,i_src"):
        np.testing.assert_allclose(native.port(native_dc, port), reference.port(reference_dc, port), atol=1e-12)
    actual = native.ac_response(freqs=jnp.array([1e3, 1e6]), sources={"V": 1.0})
    expected = reference.ac_response(freqs=jnp.array([1e3, 1e6]), sources={"V": 1.0})
    ia = actual["solution"][:, actual["port_map"]["V,i_src"]]
    ie = expected["solution"][:, expected["port_map"]["V,i_src"]]
    np.testing.assert_allclose(ia, ie, rtol=1e-8, atol=1e-18)


def test_active_gate_leakage_matches_osdi(osdi_reference):
    from osdi_jax import osdi_eval

    settings = {
        "SWIGATE": 1,
        "IGINVLW": 0.1,
        "IGOVW": 1e-8,
        "IGCHVLW": 0.1,
        "IGOVHVW": 1e-8,
        "LOV": 1e-8,
        "GCOO": 1.0,
        "GCOHVO": 1.0,
    }
    y = jnp.array([5.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0])
    parameters = jnp.array([[settings.get(name, np.nan) for name in osdi_reference.param_names]])
    rf, _, _, _, _ = osdi_eval(osdi_reference.id, y[None, :], parameters, jnp.zeros((1, osdi_reference.num_states)))
    f, _ = _evaluate(MOSVAR(**settings), y)
    assert abs(f[1]) > 1e-19
    np.testing.assert_allclose(f, rf[0], rtol=1e-8, atol=1e-29)
