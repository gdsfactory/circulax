"""DIODE_CMC native F/Q/G/C checks against pinned compiled OSDI fixtures."""

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

from circulax.components.diode_cmc import DIODE_CMC

jax.config.update("jax_enable_x64", True)
_DATA = Path(__file__).parent / "data/va_models/diode_cmc"
_CASES = json.loads((_DATA / "reference.json").read_text())["cases"]


def _evaluate(device, voltage):
    return device.solver_call(0.0, voltage, device)


_EVALUATE = jax.jit(jax.vmap(_evaluate, in_axes=(None, 0)))
_JACOBIAN = jax.jit(jax.vmap(jax.jacfwd(_evaluate, argnums=1), in_axes=(None, 0)))


@pytest.mark.parametrize("case", _CASES)
def test_current_charge_and_jacobians_match_osdi_fixture(case):
    device = DIODE_CMC(**case["settings"])
    bias = jnp.array(case["bias"])
    projection = jnp.array(case["projection"])
    active = np.argmax(np.asarray(projection), axis=0)
    f, q = _EVALUATE(device, bias)
    g, c = _JACOBIAN(device, bias)
    for actual, name in ((f, "f"), (q, "q")):
        expected = jnp.array(case[name])
        np.testing.assert_allclose(actual[:, active], expected[:, active], rtol=2e-10, atol=1e-22 if name == "f" else 1e-25)
    for actual, name in ((g, "g"), (c, "c")):
        expected = jnp.array(case[name])
        # OSDI exposes collapsed slots as voltage constraints too. Drop
        # constraint rows and substitute the tied voltages in both Jacobians.
        actual = jnp.einsum("nkj,jl->nkl", actual[:, active, :], projection)
        expected = jnp.einsum("nkj,jl->nkl", expected[:, active, :], projection)
        np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=1e-20 if name == "g" else 1e-22)
    assert np.isfinite(f).all()
    assert np.isfinite(q).all()
    assert np.isfinite(g).all()
    assert np.isfinite(c).all()


def test_disabled_internal_states_have_zero_constraints():
    device = DIODE_CMC()
    voltage = jnp.array([0.3, 0.1, 0.2, 1.0, 2.0, 3.0])
    f, q = _evaluate(device, voltage)
    np.testing.assert_allclose(f[2:], [0.1, 1.0, 2.0, 3.0], atol=1e-15)
    np.testing.assert_array_equal(q[2:], 0.0)
    np.testing.assert_allclose(f[0] + f[1], 0.0, atol=1e-25)
    np.testing.assert_allclose(q[0] + q[1], 0.0, atol=1e-25)


def test_eager_and_jit_have_same_semantics():
    for case in (_CASES[0], _CASES[3], _CASES[4]):
        device = DIODE_CMC(**case["settings"])
        bias = jnp.array(case["bias"])[1]
        eager = _evaluate(device, bias)
        compiled = _EVALUATE(device, bias[None, :])
        for e, c in zip(eager, compiled, strict=True):
            np.testing.assert_allclose(e, c[0], rtol=2e-12, atol=1e-22)


def test_area_gradient_includes_initialization():
    voltage = jnp.array([0.4, 0.0, 0.01, 2e-10, 3e-10, 1e-6])

    def current(area):
        return _evaluate(DIODE_CMC(AB=area, RSCOM=1.0, CORECOVERY=1.0, DEPNQS=1e-8), voltage)[0][0]

    area = 1e-12
    step = 1e-16
    derivative = jax.jit(jax.grad(current))(area)
    finite_difference = (current(area + step) - current(area - step)) / (2 * step)
    assert np.isfinite(derivative)
    assert derivative != 0.0
    np.testing.assert_allclose(derivative, finite_difference, rtol=1e-7)


@pytest.mark.parametrize("settings", [{}, {"RSCOM": 1.0, "CORECOVERY": 1.0, "DEPNQS": 1e-8}])
def test_compiled_circuit_dc_solves_internal_states(settings):
    from circulax import compile_circuit
    from circulax.components.electronic import VoltageSource

    net = {
        "instances": {"D": {"component": "diode", "settings": settings}, "V": {"component": "v", "settings": {"V": 0.3}}},
        "connections": {"V,p1": "D,A", "V,p2": "GND,p1", "D,K": "GND,p1"},
        "ports": {},
    }
    circuit = compile_circuit(net, {"diode": DIODE_CMC, "v": VoltageSource})
    solution = circuit.dc()
    assert np.isfinite(solution).all()
    current = circuit.get_port_field(solution, "V,i_src")
    # Independently compiled OSDI current at 0.3 V (source current is negative).
    np.testing.assert_allclose(current, -2.8195450087513913e-11, rtol=1e-7, atol=1e-18)
    if settings:
        assert circuit.get_port_field(solution, "D,v_depl_A") > 0.0


def test_recovery_total_time_current_matches_osdi_fixture():
    case = _CASES[3]
    device = DIODE_CMC(**case["settings"])
    voltage = jnp.array(case["bias"])
    f, _ = _EVALUATE(device, voltage)
    _, c = _JACOBIAN(device, voltage)
    # Prescribed electrical and recovery-state derivatives, retaining SI scaling.
    dvdt = jnp.array([[1e6, 0.0, 1e4, 1e-2, 1e-3, 1.0]] * len(voltage))
    actual = f + jnp.einsum("nij,nj->ni", c, dvdt)
    reference = jnp.array(case["f"]) + jnp.einsum("nij,nj->ni", jnp.array(case["c"]), dvdt)
    np.testing.assert_allclose(actual, reference, rtol=2e-9, atol=1e-20)


def test_recompiled_osdi_matches_stored_reference(tmp_path):
    pytest.importorskip("bosdi")
    pytest.importorskip("osdi_jax")
    from bosdi.circulax import osdi_component
    from osdi_jax import osdi_eval

    compiler = shutil.which("openvaf-r")
    if compiler is None:
        pytest.skip("OpenVAF-r needed to refresh independent OSDI verification")
    binary = tmp_path / "diode.osdi"
    subprocess.run([compiler, str(_DATA / "diode_cmc.va"), "-o", str(binary)], check=True, capture_output=True)
    descriptor = osdi_component(str(binary), ("A", "K"), temperature=300.15)
    for case in _CASES:
        voltage = jnp.array(case["bias"])
        params = jnp.array([[case["settings"].get(k, np.nan) for k in descriptor.param_names]] * len(voltage))
        f, g, q, c, _ = osdi_eval(descriptor.model.id, voltage, params, jnp.zeros((len(voltage), descriptor.model.num_states)))
        for actual, key in ((f, "f"), (q, "q"), (g.reshape((-1, 6, 6)), "g"), (c.reshape((-1, 6, 6)), "c")):
            np.testing.assert_allclose(actual, case[key], rtol=2e-12, atol=1e-25)
