# Copyright (c) 2026 Circulax contributors.
"""R2_CMC deterministic equations and optional compiled upstream OSDI parity."""

import shutil
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax.components.base_component import CircuitComponent
from circulax.components.r2 import R2CMC, R2ETCMC

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


def _evaluate(model: CircuitComponent, voltage: jax.Array) -> tuple[jax.Array, jax.Array]:
    f, q = model(y=voltage)
    return jnp.array([f.get(port, 0.0) for port in model.ports]), jnp.array([q.get(port, 0.0) for port in model.ports])


@pytest.mark.parametrize(
    ("settings", "resistance"),
    [
        ({}, 100.0),
        ({"r": 500.0}, 500.0),
        ({"w": 2e-6, "l": 4e-6, "rsh": 70.0}, 140.0),
        ({"w": 2e-6, "l": 4e-6, "r": 500.0, "rsh": 70.0}, 140.0),
        ({"r": 500.0, "l": 4e-6}, 500.0),
    ],
)
def test_geometry_precedence(settings: dict[str, float], resistance: float) -> None:
    f, q = R2CMC(**settings)(n1=2.0, n2=-1.0)
    assert float(f["n1"]) == pytest.approx(3.0 / resistance)
    assert f["n1"] + f["n2"] == 0
    assert q == {}


def test_field_and_contact_temperature_equations() -> None:
    model = R2CMC(
        l=4e-6,
        w=2e-6,
        xl=0.4,
        xw=0.1,
        c1=1,
        c2=0,
        tc1=1e-3,
        tc1l=2e-3,
        tc1w=3e-3,
        temperature=330.15,
        p2=0.2,
        q2=0.3,
        p3=0.1,
        q3=0.5,
        m=3,
    )
    effective_l, effective_w = 4.2, 2.1
    r0 = 100 * effective_l / effective_w
    tc1 = 1e-3 + 0.5 * 2e-3 / effective_l + 3e-3 / effective_w
    field = 3 / 4
    nonlinear = 0.7 + 0.2 * np.sqrt(1 + (0.3 * field) ** 2) + 0.1 * (1 + (0.5 * field) ** 3) ** (1 / 3)
    assert model(n1=3)[0]["n1"] == pytest.approx(9 / (r0 * (1 + 30 * tc1) * nonlinear))
    assert model(n1=-3)[0]["n1"] == -model(n1=3)[0]["n1"]


def test_electrothermal_energy_and_jacobians() -> None:
    model = R2ETCMC(l=4e-6, w=2e-6, c1=1, c2=0, tc1=0.004, m=2, gth0=0.01, gthp=1e-3, gtha=2e-4, cth0=1e-3, cthp=1e-4, ctha=2e-5)
    y = jnp.array([2.0, 0.0, 10.0])
    f, q = _evaluate(model, y)
    perimeter, area = 10.0, 8.0
    conductance = 0.01 + 0.001 * perimeter + 0.0002 * area
    capacitance = 0.001 + 0.0001 * perimeter + 0.00002 * area
    assert f[0] + f[1] == 0
    assert f[2] == pytest.approx(2 * 10 * conductance - 2 * f[0])
    assert q[2] == pytest.approx(2 * 10 * capacitance)
    jac = jax.jacfwd(lambda v: _evaluate(model, v)[0])(y)
    assert jac[0, 2] < 0  # Positive TC reduces current with self-heating.
    assert jax.jacfwd(lambda v: _evaluate(model, v)[1])(y)[2, 2] == pytest.approx(2 * capacitance)


@pytest.mark.parametrize("temperature", [100.0, 173.2, 300.15, 772.8, 1200.0])
def test_temperature_clipping_finite_ad(temperature: float) -> None:
    model = R2CMC(temperature=temperature, tc1=-0.02, tc2=1e-4, q2=0.5, p2=0.2)
    evaluate = jax.jit(jax.vmap(lambda v: model(n1=v)[0]["n1"]))
    voltages = jnp.array([-3.0, 0.0, 2.0])
    assert np.isfinite(evaluate(voltages)).all()
    assert np.isfinite(jax.jacfwd(evaluate)(voltages)).all()


def test_parameter_gradient_and_thermal_relaxation() -> None:
    # Implicit self-heating equilibrium: g*dt = V²/[R0*(1+tc*dt)].
    voltage, resistance, tc, g = 2.0, 100.0, 0.004, 0.002
    rise = (-1 + np.sqrt(1 + 4 * tc * voltage**2 / (g * resistance))) / (2 * tc)
    model = R2ETCMC(r=resistance, tc1=tc, gth0=g, cth0=0.1)
    assert model(n1=voltage, dt=rise)[0]["dt"] == pytest.approx(0, abs=1e-15)
    assert model(n1=0, dt=rise)[0]["dt"] > 0  # Positive residual cools through dQ/dt=-F.
    derivative = jax.grad(lambda r: R2CMC(r=r)(n1=voltage)[0]["n1"])(resistance)
    assert derivative == pytest.approx(-voltage / resistance**2)


@pytest.fixture(scope="module")
def osdi_binaries(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    pytest.importorskip("osdi_loader")
    pytest.importorskip("osdi_jax")
    compiler = shutil.which("openvaf-r")
    if compiler is None:
        pytest.skip("openvaf-r unavailable")
    source = Path(__file__).parent / "data" / "va_models" / "r2_cmc"
    target = tmp_path_factory.mktemp("r2_osdi")
    paths = {}
    for name in ("r2_cmc", "r2_et_cmc"):
        paths[name] = target / f"{name}.osdi"
        subprocess.run([compiler, str(source / f"{name}.va"), "-o", str(paths[name])], check=True, capture_output=True)  # noqa: S603
    return paths


@pytest.mark.parametrize("electrothermal", [False, True])
@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"r": 230.0, "m": 2.0},
        {"l": 4e-6, "r": 230.0, "xw": 0.1, "xl": 0.2, "c1": 0.0},
        {"w": 2e-6, "r": 150.0, "xw": 0.1, "xl": 0.2, "c2": 0.0},
        {
            "w": 2e-6,
            "l": 4e-6,
            "r": 300.0,
            "shrink": 10.0,
            "scale": 0.8,
            "p2": 0.2,
            "q2": 0.3,
            "p3": 0.1,
            "q3": 0.4,
            "dxle": 0.1,
            "sw_efgeo": 1.0,
            "tc1": 0.001,
            "tc2": 1e-5,
            "tc1l": 0.002,
            "tc2l": 1e-6,
            "tc1w": 0.003,
            "tc2w": 2e-6,
            "trise": 5.0,
        },
        {"tc1": -0.02, "temperature": 430.0, "tmaxclip": 120.0},
        {"temperature": 130.0, "tminclip": -100.0, "tc1": 0.001},
    ],
)
def test_osdi_residual_charge_and_ac_parity(
    osdi_binaries: dict[str, Path],
    electrothermal: bool,  # noqa: FBT001
    settings: dict[str, float],
) -> None:
    from osdi_jax import osdi_eval
    from osdi_loader import load_osdi_model

    settings = dict(settings)
    if electrothermal:
        settings.update(gth0=0.002, gthp=1e-4, gtha=2e-5, cth0=0.01, cthp=1e-3, ctha=2e-4)
    model = (R2ETCMC if electrothermal else R2CMC)(**settings)
    name = "r2_et_cmc" if electrothermal else "r2_cmc"
    reference = load_osdi_model(str(osdi_binaries[name]), temperature=settings.get("temperature", 300.15))
    params = jnp.full((1, reference.num_params), jnp.nan)
    settings["$mfactor"] = settings.pop("m", 1.0)
    settings["isnoisy"] = 0
    for key, value in settings.items():
        if key != "temperature":
            params = params.at[0, reference.param_names.index(key)].set(value)
    for v in (-3.0, 0.0, 2.0):
        y = jnp.array([v, 0.1, 15.0] if electrothermal else [v, 0.1])
        f, q = _evaluate(model, y)
        df, dq = jax.jacfwd(lambda y: _evaluate(model, y))(y)
        rf, rdf, rq, rdq, _ = osdi_eval(reference.id, y[None, :], params, jnp.empty((1, reference.num_states)))
        np.testing.assert_allclose(f, rf[0], rtol=2e-12, atol=1e-13)
        np.testing.assert_allclose(q, rq[0], rtol=2e-12, atol=1e-13)
        # osdi_eval packs Jacobians in the union of resistive/reactive pairs.
        np.testing.assert_allclose(df.ravel(), rdf[0], rtol=2e-11, atol=1e-13)
        np.testing.assert_allclose(dq.ravel(), rdq[0], rtol=2e-11, atol=1e-13)


def test_thermal_subcircuit_matches_implicit_equilibrium() -> None:
    from circulax import compile_circuit
    from circulax.components.electronic import Capacitor, Resistor, VoltageSource
    from circulax.components.r2 import r2_thermal_subcircuit

    netlist = {
        "top": {
            "instances": {
                "R": {"component": "thermal_resistor"},
                "V": {"component": "VoltageSource", "settings": {"V": 2.0}},
                "GND": {"component": "ground"},
            },
            "connections": {"V,p1": "R,n1", "GND,p1": ("V,p2", "R,n2")},
        },
        "thermal_resistor": r2_thermal_subcircuit(
            resistor_settings={"r": 100.0, "tc1": 0.004, "m": 2.0}, thermal_resistance=500, thermal_capacitance=0.1
        ),
    }
    circuit = compile_circuit(
        netlist, {"R2ETCMC": R2ETCMC, "Resistor": Resistor, "Capacitor": Capacitor, "VoltageSource": VoltageSource}
    )
    solution = circuit.dc()
    expected_rise = (-1 + np.sqrt(1 + 4 * 0.004 * 4 / (0.002 * 100))) / (0.008)
    assert circuit.port(solution, "R~core,dt") == pytest.approx(expected_rise, rel=1e-6)
