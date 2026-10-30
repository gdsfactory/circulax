"""Native Angelov deterministic model versus pinned independent OSDI data."""

# ruff: noqa: ANN001, ANN201, ANN202, CPY001, FBT003, S603
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax.components.angelov import Angelov

jax.config.update("jax_enable_x64", True)
_DATA = Path(__file__).parent / "data/va_models/angelov"
_CASES = json.loads((_DATA / "reference.json").read_text())["cases"]
_KEEP = list(range(14)) + list(range(16, 21))


def _evaluate(device, voltage):
    return device.solver_call(0.0, voltage, device)


_EVAL = jax.jit(jax.vmap(_evaluate, in_axes=(None, 0)))
_JAC = jax.jit(jax.vmap(jax.jacfwd(_evaluate, argnums=1), in_axes=(None, 0)))


@pytest.mark.parametrize("case", _CASES)
def test_all_current_and_capacitance_modes_match_osdi(case):
    device = Angelov(**case["settings"])
    bias = jnp.array(case["bias"])
    f, q = _EVAL(device, bias)
    g, c = _JAC(device, bias)
    for actual, key, tolerance in ((f, "f", 1e-13), (q, "q", 1e-25), (g, "g", 1e-12), (c, "c", 1e-24)):
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, case[key], rtol=2e-10, atol=tolerance)


def test_eager_jit_and_parameter_derivative():
    case = _CASES[-3]
    bias = jnp.array(case["bias"])[0]
    device = Angelov(**case["settings"])
    eager = _evaluate(device, bias)
    compiled = _EVAL(device, bias[None, :])
    for a, b in zip(eager, compiled, strict=True):
        np.testing.assert_allclose(a, b[0], rtol=1e-12, atol=1e-20)

    def current(ipk):
        settings = {**case["settings"], "Ipk0": ipk}
        return _evaluate(Angelov(**settings), bias)[0][12]

    gradient = jax.jit(jax.grad(current))(0.05)
    finite = (current(0.050001) - current(0.049999)) / 0.000002
    assert np.isfinite(gradient)
    assert gradient != 0.0
    np.testing.assert_allclose(gradient, finite, rtol=2e-9)


def test_optional_parameter_presence_survives_fitting_and_updates():
    import equinox as eqx

    default = Angelov()
    explicit = Angelov(Temp=25.0, Ne=1.2)
    assert np.isnan(default.Temp)
    assert np.isnan(default.Pg)
    assert explicit.Temp == 25.0
    assert explicit.Ne == 1.2
    bias = jnp.array(_CASES[-3]["bias"])[0]
    default_f = _evaluate(default, bias)[0]
    explicit_default_f = _evaluate(Angelov(Temp=25.0), bias)[0]
    assert abs(default_f[12] - explicit_default_f[12]) > 1e-7
    # Updating an omitted optional parameter makes it given dynamically.
    updated = eqx.tree_at(lambda m: m.Temp, default, 85.0)
    for actual, expected in zip(_evaluate(updated, bias), _evaluate(Angelov(Temp=85.0), bias), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-25)

    def temperature_current(temp):
        model = eqx.tree_at(lambda m: m.Temp, default, temp)
        return _evaluate(model, bias)[0][12]

    derivative = jax.jit(jax.grad(temperature_current))(85.0)
    finite = (temperature_current(85.001) - temperature_current(84.999)) / 0.002
    assert np.isfinite(derivative)
    assert derivative != 0.0
    np.testing.assert_allclose(derivative, finite, rtol=2e-8)


@pytest.mark.parametrize("field", ["Rg", "Rd", "Rs", "Ri", "Rgd", "Rc", "Rcin"])
def test_unsupported_collapsed_topologies_rejected(field):
    with pytest.raises(ValueError, match="zero-resistance collapsed topology"):
        Angelov(**{field: 0.0})
    with pytest.raises(ValueError, match="zero-resistance collapsed topology"):
        Angelov(**{field: jnp.array(0.0)})


def _circuit(model, settings):
    from circulax import compile_circuit
    from circulax.components.electronic import Resistor, VoltageSource

    net = {
        "instances": {
            "M": {"component": "fet", "settings": settings},
            "VD": {"component": "v", "settings": {"V": 5.0}},
            "VG": {"component": "v", "settings": {"V": -0.3}},
            "RD": {"component": "r", "settings": {"R": 50.0}},
            "RG": {"component": "r", "settings": {"R": 1000.0}},
        },
        "connections": {
            "VD,p1": "RD,p1",
            "RD,p2": "M,d",
            "VG,p1": "RG,p1",
            "RG,p2": "M,g",
            "M,s": "GND,p1",
            "VD,p2": "GND,p1",
            "VG,p2": "GND,p1",
        },
        "ports": {"drain": "M,d", "gate": "M,g"},
    }
    return compile_circuit(net, {"fet": model, "v": VoltageSource, "r": Resistor})


def test_dc_and_frequency_response_with_thermal_dispersion_and_delay():
    circuit = _circuit(Angelov, _CASES[-3]["settings"])
    dc = circuit.dc()
    assert np.isfinite(dc).all()
    assert circuit.get_port_field(dc, "M,v_t") > 0.0
    assert circuit.get_port_field(dc, "VD,i_src") < 0.0
    response = circuit.sp(ports=["gate", "drain"], freqs=jnp.array([1e6, 1e8, 1e9]), y_dc=dc)
    assert np.isfinite(response).all()
    assert np.max(np.abs(response[0] - response[-1])) > 0.1
    # Nontrivial gain, rather than a source-clamped reflection-only test.
    assert abs(response[0, 1, 0]) > 1.0


def test_recovery_and_parasitic_time_current_matches_osdi():
    case = _CASES[-3]
    voltage = jnp.array(case["bias"])
    device = Angelov(**case["settings"])
    f, _ = _EVAL(device, voltage)
    _, c = _JAC(device, voltage)
    derivative = jnp.array(
        [[1e6, -2e5, 0.0, 9e5, -2e5, 1e3, 0.0, -2e5, -2e5, 1e3, 1e3, 1.0, 1e3, 1e3, 1e3, 1e2, 1e3, 1e3, -1e3]] * len(voltage)
    )
    actual = f + jnp.einsum("nij,nj->ni", c, derivative)
    expected = jnp.array(case["f"]) + jnp.einsum("nij,nj->ni", jnp.array(case["c"]), derivative)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=1e-13)


@pytest.fixture(scope="module")
def osdi_model(tmp_path_factory):
    pytest.importorskip("bosdi")
    pytest.importorskip("osdi_jax")
    from bosdi.circulax import osdi_component

    compiler = shutil.which("openvaf-r")
    if compiler is None:
        pytest.skip("OpenVAF-r required for fresh OSDI parity")
    binary = tmp_path_factory.mktemp("angelov") / "angelov.osdi"
    subprocess.run([compiler, str(_DATA / "angelov.va"), "-o", str(binary)], check=True, capture_output=True)
    return osdi_component(str(binary), ("d", "g", "s"), default_params={"Noise": 0.0}, temperature=300.15)


def test_fresh_osdi_matches_stored_fixtures(osdi_model):
    from osdi_jax import osdi_eval

    for case in _CASES:
        params = jnp.array(
            [[{"Noise": 0.0, **case["settings"]}.get(k, np.nan) for k in osdi_model.param_names]] * len(case["bias"])
        )
        voltage = jnp.array(case["raw_bias"])
        f, g, q, c, _ = osdi_eval(osdi_model.model.id, voltage, params, jnp.zeros((len(voltage), osdi_model.model.num_states)))
        for actual, key in (
            (f[:, _KEEP], "f"),
            (q[:, _KEEP], "q"),
            (g.reshape((-1, 21, 21))[:, _KEEP, :][:, :, _KEEP], "g"),
            (c.reshape((-1, 21, 21))[:, _KEEP, :][:, :, _KEEP], "c"),
        ):
            np.testing.assert_allclose(actual, case[key], rtol=2e-12, atol=1e-25)


def test_actual_dc_and_ac_circuit_matches_osdi(osdi_model):
    settings = _CASES[-3]["settings"]
    native = _circuit(Angelov, settings)
    reference = _circuit(osdi_model, settings)
    yn = native.dc()
    yr = reference.dc()
    np.testing.assert_allclose(native.get_port_field(yn, "VD,i_src"), reference.get_port_field(yr, "VD,i_src"), rtol=1e-9)
    frequencies = jnp.array([1e6, 1e8, 1e9])
    sn = native.sp(ports=["gate", "drain"], freqs=frequencies, y_dc=yn)
    sr = reference.sp(ports=["gate", "drain"], freqs=frequencies, y_dc=yr)
    np.testing.assert_allclose(sn, sr, rtol=2e-8, atol=1e-10)


def test_circuit_temperature_update_changes_optional_presence():
    circuit = _circuit(Angelov, {})
    updated = circuit.dc(params={"M.Temp": 85.0})
    explicit = _circuit(Angelov, {"Temp": 85.0})
    expected = explicit.dc()
    np.testing.assert_allclose(
        circuit.get_port_field(updated, "VD,i_src"), explicit.get_port_field(expected, "VD,i_src"), rtol=1e-9
    )
