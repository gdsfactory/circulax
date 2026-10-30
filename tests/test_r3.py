# Copyright (c) 2026 Circulax contributors
"""Native R3_CMC residuals and derivatives against pinned compiled OSDI data."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax import compile_circuit
from circulax.components.electronic import VoltageSource
from circulax.components.r3 import R3_CMC

_DATA = Path(__file__).parent / "data/va_models/r3_cmc"
_CASES = json.loads((_DATA / "reference.json").read_text())


@pytest.mark.parametrize("case", _CASES, ids=[f"{case['name']}-{i % 2}" for i, case in enumerate(_CASES)])
def test_compiled_osdi_reference(case: dict[str, Any]) -> None:
    model = R3_CMC(**case["settings"], _temperature=case["temperature"])
    bias = jnp.array(case["bias"])
    evaluate = jax.jit(lambda v: R3_CMC.solver_call(0.0, v, model))
    f, q = evaluate(bias)
    g, c = jax.jit(jax.jacfwd(evaluate))(bias)
    if "g_fd" in case:
        # OpenVAF analytic G for full pinch-off disagrees with its own F.
        np.testing.assert_allclose(g, case["g_fd"], rtol=2e-6, atol=1e-10)
        assert np.max(np.abs(np.array(case["g"]) - np.array(case["g_fd"]))) > 0.01
        case = {**case, "g": np.asarray(g)}
    for actual, key, atol in [(f, "f", 1e-12), (q, "q", 1e-24), (g, "g", 1e-11), (c, "c", 1e-23)]:
        assert jnp.all(jnp.isfinite(actual))
        np.testing.assert_allclose(actual, case[key], rtol=2e-9, atol=atol)
    # Electrical currents/charges include internal nodes, not thermal or constraints.
    electrical = jnp.array([0, 1, 2, 4, 5])
    np.testing.assert_allclose(f[electrical].sum(), 0.0, atol=1e-12)
    np.testing.assert_allclose(q[electrical].sum(), 0.0, atol=1e-24)


def test_live_osdi_reference(tmp_path: Path) -> None:
    pytest.importorskip("osdi_jax")
    from bosdi.circulax.osdi_component import osdi_component
    from osdi_jax import osdi_eval

    compiler = shutil.which("openvaf-r")
    if compiler is None:
        pytest.skip("OpenVAF compiler unavailable; stored OSDI references still run")
    target = tmp_path / "r3.osdi"
    subprocess.run([compiler, str(_DATA / "r3_cmc.va"), "-o", str(target)], check=True, capture_output=True)  # noqa: S603
    case = _CASES[4]
    desc = osdi_component(str(target), ports=R3_CMC.ports, temperature=case["temperature"])
    instance = desc.make_instance(case["settings"])
    params = jnp.array([[instance[k] for k in desc.param_names]])
    f, g, q, c, _ = osdi_eval(desc.model.id, jnp.array([case["bias"]]), params, jnp.empty((1, 0)))
    for actual, key in [(f[0], "f"), (q[0], "q"), (g.reshape(8, 8), "g"), (c.reshape(8, 8), "c")]:
        np.testing.assert_allclose(actual, case[key], rtol=1e-12, atol=1e-25)


def test_jit_vmap_and_parameter_gradient() -> None:
    bias = jnp.array(_CASES[0]["bias"])

    def residual(rsh: jax.Array) -> jax.Array:
        return R3_CMC.solver_call(0.0, bias, R3_CMC(rsh=rsh))[0][4]

    values = jax.jit(jax.vmap(residual))(jnp.array([80.0, 100.0, 120.0]))
    assert jnp.all(jnp.isfinite(values))
    derivative = jax.jit(jax.grad(residual))(100.0)
    finite = (residual(100.001) - residual(99.999)) / 0.002
    np.testing.assert_allclose(derivative, finite, rtol=1e-8)


def test_linear_resistor_circuit() -> None:
    net = {
        "instances": {
            "R": {"component": "r3", "settings": {"sw_lin": 1, "rsh": 100.0}},
            "V": {"component": "v", "settings": {"V": 0.1}},
        },
        "connections": {"R,n1": "V,p1", "R,n2": "GND,p1", "R,nc": "GND,p1", "R,dt": "GND,p1", "V,p2": "GND,p1"},
        "ports": {},
    }
    circuit = compile_circuit(net, {"r3": R3_CMC, "v": VoltageSource})
    y = circuit.dc()
    assert jnp.all(jnp.isfinite(y))
    np.testing.assert_allclose(circuit.get_port_field(y, "V,i_src"), -0.001, rtol=1e-7, atol=1e-12)


def test_contacts_and_self_heating_circuit() -> None:
    settings = {"rc": 2.0, "c1": 1, "c2": 2, "gth0": 0.01, "cth0": 1e-6, "tc1": 0.002}
    net = {
        "instances": {"R": {"component": "r3", "settings": settings}, "V": {"component": "v", "settings": {"V": 0.1}}},
        "connections": {"R,n1": "V,p1", "R,n2": "GND,p1", "R,nc": "GND,p1", "R,dt": "R,dt", "V,p2": "GND,p1"},
        "ports": {},
    }
    circuit = compile_circuit(net, {"r3": R3_CMC, "v": VoltageSource})
    y = circuit.dc()
    assert jnp.all(jnp.isfinite(y))
    local = jnp.array([circuit.get_port_field(y, f"R,{key}") for key in R3_CMC.ports + R3_CMC.states])
    f, q = R3_CMC.solver_call(0.0, local, R3_CMC(**settings))
    np.testing.assert_allclose(f[3:], 0.0, atol=2e-9)
    assert 0.005 < float(local[3]) < 0.02
    assert q[3] > 0.0
    # AC admittance at the solved bias includes thermal storage and contact coupling.
    g, c = jax.jacfwd(lambda v: R3_CMC.solver_call(0.0, v, R3_CMC(**settings)))(local)
    assert jnp.all(jnp.isfinite(g + 2j * jnp.pi * 1e3 * c))
    np.testing.assert_allclose(c[3, 3], 1e-6, rtol=1e-12)
