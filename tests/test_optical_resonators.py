# Copyright (c) 2026 Circulax contributors.
# ruff: noqa: ANN001, ANN201, ANN202 -- numerical verification fixtures.
"""Independent CMT energy identities, scattering and compiled optical circuits."""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax import compile_circuit
from circulax.components.electronic import Resistor
from circulax.components.optical_filters import OpticalFilter
from circulax.components.optical_passive import OpticalTermination
from circulax.components.photonic import OpticalSource
from circulax.components.resonators import (
    AddDropRing,
    AllPassRing,
    AllPassRingEO,
    RingModSetup,
    RingModulator,
    RingModulatorEO,
    loaded_ring,
)


def bench(model, settings, input_port="p1"):
    instances = {
        "GND": {"component": "ground"},
        "S": {"component": "source", "settings": {"power": 4.0}},
        "R": {"component": "R", "settings": {"R": 1.0}},
        "D": {"component": "D", "settings": settings},
    }
    connections = {"S,p2": "GND,p1", "S,p1": "R,p1", "R,p2": f"D,{input_port}"}
    for port in model.ports:
        if port != input_port:
            instances["L" + port] = {"component": "load"}
            connections[f"D,{port}"] = "L" + port + ",p1"
    return compile_circuit(
        {"instances": instances, "connections": connections, "ports": {p: f"D,{p}" for p in model.ports}},
        {"ground": lambda: 0, "source": OpticalSource, "R": Resistor, "D": model, "load": OpticalTermination},
        is_complex=True,
        backend="dense",
        g_leak=0,
    )


@pytest.mark.parametrize(("gi", "ge"), [(0.0, 2e10), (2e10, 2e10), (5e10, 2e10), (1e10, 0.0)])
@pytest.mark.parametrize("delta", [-3e10, 0.0, 3e10])
def test_allpass_compiled_dc_ac_against_cmt(gi, ge, delta):
    circuit = bench(AllPassRing, {"gamma_i": gi, "gamma_e": ge, "detuning": delta})
    expected = 1 - 2 * ge / (gi + ge - 1j * delta)
    np.testing.assert_allclose(circuit.port(circuit.dc(), "p2"), expected, atol=2e-9)
    f = jnp.array([0.0, 1e9, 1e10])
    ac = circuit.ac_response(freqs=f, sources={"S": 2.0})
    expected_ac = 1 - 2 * ge / (gi + ge + 2j * np.pi * np.asarray(f) - 1j * delta)
    np.testing.assert_allclose(ac["solution"][:, circuit.port_map["p2"]], expected_ac, atol=2e-9)
    assert np.max(np.abs(expected_ac)) <= 1 + 1e-14


@pytest.mark.parametrize("gi", [0.0, 0.7e10])
def test_adddrop_four_input_scattering_and_energy(gi):
    g1, g2, delta = 2e10, 3e10, 1e10
    coupling = np.array([[np.sqrt(2 * g1), 0, np.sqrt(2 * g2), 0], [0, np.sqrt(2 * g1), 0, np.sqrt(2 * g2)]])
    direct = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    # Independently ordered output couplings from the two propagation modes.
    out = np.array([[0, np.sqrt(2 * g1)], [np.sqrt(2 * g1), 0], [0, np.sqrt(2 * g2)], [np.sqrt(2 * g2), 0]])
    expected = direct - out @ coupling / (gi + g1 + g2 - 1j * delta)
    columns = []
    for port in AddDropRing.ports:
        circuit = bench(AddDropRing, {"gamma_i": gi, "gamma_1": g1, "gamma_2": g2, "detuning": delta}, port)
        y = circuit.dc()
        columns.append([complex(circuit.port(y, p)) - (p == port) for p in AddDropRing.ports])
    actual = np.array(columns).T
    np.testing.assert_allclose(actual, expected, atol=2e-9)
    np.testing.assert_allclose(actual.T, actual, atol=2e-9)
    assert np.max(np.linalg.svd(actual, compute_uv=False)) <= 1 + 1e-9
    if gi == 0:
        np.testing.assert_allclose(actual.conj().T @ actual, np.eye(4), atol=3e-9)
    else:
        drive = np.array([0.3 + 0.4j, 0.7j, -0.2 + 0.1j, 0.5])
        cavity = coupling @ drive / (gi + g1 + g2 - 1j * delta)
        np.testing.assert_allclose(
            np.vdot(drive, drive).real - np.vdot(actual @ drive, actual @ drive).real,
            2 * gi * np.vdot(cavity, cavity).real,
            atol=3e-9,
        )


def test_cavity_instantaneous_energy_identity_both_directions():
    gi, g1, g2, delta = 2.0, 3.0, 4.0, 5.0
    a = np.array([0.2 + 0.1j, 0.5j, -0.3 + 0.7j, 0.4 - 0.2j])
    c = np.array([0.1 + 0.2j, -0.3 + 0.4j])
    model = AddDropRing(gamma_i=gi, gamma_1=g1, gamma_2=g2, detuning=delta)
    f, _ = model(y=jnp.concatenate([jnp.zeros(4), a, c]))
    inward = np.array([f[p] for p in model.ports])
    b = a - inward
    dcdt = -np.array([f["c_forward"], f["c_reverse"]])
    np.testing.assert_allclose(
        np.vdot(a, a).real - np.vdot(b, b).real, 2 * np.vdot(c, dcdt).real + 2 * gi * np.vdot(c, c).real, atol=1e-13
    )


def test_ring_ringdown_amplitude_vs_energy_and_jit_ad():
    model = AllPassRing(gamma_i=2.0, gamma_e=3.0, detuning=4.0)
    y = jnp.array([0j, 0j, 0j, 0j, 1 + 0.3j, 0j])
    f, q = model(y=y)
    np.testing.assert_allclose(-f["c_forward"] / q["c_forward"], -5 + 4j)
    # The real rate of |c|² is -10, twice the amplitude rate.
    np.testing.assert_allclose(2 * np.real(np.conj(q["c_forward"]) * -f["c_forward"]), -10 * abs(1 + 0.3j) ** 2)

    def loss(p):
        f, _ = AllPassRing.solver_call(0.0, y, AllPassRing(gamma_i=p[0], gamma_e=p[1], detuning=p[2]))
        return jnp.real(jnp.vdot(f, f))

    p = jnp.array([2.0, 3.0, 4.0])
    np.testing.assert_allclose(jax.jacfwd(loss)(p), jax.jacrev(loss)(p))
    assert np.isfinite(np.asarray(jax.jit(jax.vmap(loss))(jnp.stack([p, p * 2])))).all()


def test_promoted_models_match_original_notebook_definitions():
    # Original extracted code is not a runtime or test dependency. Independently
    # restate representative legacy equations instead of importing their helpers.
    p = {"ng": 3.8, "L": 3.14159265e-5, "gamma": 0.976, "alpha0": 0.969, "voltage": -2.0, "v_to_wr": 1e10}
    m = RingModulator(**p)
    c = 2.998e8
    ge = (1 - p["gamma"] ** 2) * c / (2 * p["ng"] * p["L"])
    gi = (1 - p["alpha0"] ** 2) * c / (2 * p["ng"] * p["L"])
    delta = 2 * np.pi * (2.2904e14 - 2.2901e14) + p["v_to_wr"] * p["voltage"]
    k = np.sqrt(2 * ge)
    y = jnp.array([0.1 + 0.2j, 0.2j, 1e-7 + 2e-7j, 0.01j])
    f, q = m(y=y)
    np.testing.assert_allclose(f["a"], (ge + gi - 1j * delta) * y[2] + 1j * k * y[0])
    np.testing.assert_allclose(f["i_out"], y[1] - y[0] + 1j * k * y[2])
    eo = RingModulatorEO(**{k: v for k, v in p.items() if k != "voltage"})
    ef, eq = eo(y=jnp.array([y[0], y[1], p["voltage"], y[2], y[3]]))
    for key in ["p1", "p2", "a", "i_out"]:
        np.testing.assert_allclose(ef[key], f[key])
    np.testing.assert_allclose(eq["a"], q["a"])
    z = RingModSetup(kappa=0.3, neff=2.4, alpha=1e-3)
    t = np.sqrt(1 - 0.3**2)
    a = np.exp(-1e-3 * (2 * np.pi * 5e-6) / 2)
    phi = 2 * np.pi * 2.4 * (2 * np.pi * 5e-6)
    h = (t - a * np.exp(1j * phi)) / (1 - t * a * np.exp(1j * phi))
    np.testing.assert_allclose(z(in_=1.0, thru=0.0)[0]["thru"], -h)
    derivative = jax.grad(lambda k: jnp.angle(RingModSetup(kappa=k)(in_=1.0, thru=0.0)[0]["thru"]))(0.3)
    assert np.isfinite(derivative)


def test_eo_imaginary_restore_and_loaded_factory():
    f, _ = AllPassRingEO()(p1=0.0, p2=0.0, v_e=1.0 + 2j)
    np.testing.assert_allclose(f["v_e"], 2j)
    circuit = loaded_ring()
    assert {"p1", "p2", "drive", "gnd", "junction"} <= set(circuit.port_map)
    with pytest.raises(ValueError, match="positive"):
        loaded_ring(resistance=0.0)


@pytest.mark.parametrize("delta", [0.0, 2e9, -3e9])
def test_filter_compiled_transfer_pole_and_bandwidth(delta):
    bandwidth = 1e9
    rate = 2 * np.pi * bandwidth
    circuit = bench(OpticalFilter, {"bandwidth_hz": bandwidth, "detuning": delta, "transmission": 0.8})
    f = jnp.array([-3e9, 0.0, delta / (2 * np.pi), 1e9, 4e9])
    actual = circuit.ac_response(freqs=f, sources={"S": 2.0})["solution"][:, circuit.port_map["p2"]]
    expected = 0.8 * rate / (rate + 2j * np.pi * np.asarray(f) - 1j * delta)
    np.testing.assert_allclose(actual, expected, atol=2e-9)
    if delta == 0:
        np.testing.assert_allclose(abs(actual[3]) / 0.8, 1 / np.sqrt(2), atol=2e-9)


def test_filter_loaded_step_converges_to_independent_ode():
    # Start with zero cavity under a constant incident source. Algebraic wave
    # constraints are initialized consistently; the exact step is 1-exp(-kt).
    rate = 2 * np.pi * 1e9
    circuit = bench(OpticalFilter, {"bandwidth_hz": 1e9})
    y = circuit.dc()
    y = y.at[circuit.port_map["D,c1"]].set(0.0)
    y = y.at[circuit.port_map["p2"]].set(0.0)
    # Incoming wave remains one, no reflections. State aliases are explicit.
    ts = jnp.linspace(0.0, 1e-9, 101)
    exact = 1 - np.exp(-rate * np.asarray(ts))
    errors = []
    for dt in [4e-12, 2e-12]:
        sol = circuit.transient(
            t0=0.0,
            t1=1e-9,
            dt0=dt,
            y0=y,
            saveat=diffrax.SaveAt(ts=ts),
            stepsize_controller=diffrax.ConstantStepSize(),
            max_steps=1000,
        )
        errors.append(np.max(np.abs(np.asarray(circuit.port(sol.ys, "p2")) - exact)))
    assert errors[1] < 3e-4
    assert errors[1] < errors[0] * 0.8


def test_loaded_ring_eo_both_sidebands_against_independent_cmt_rc():
    from circulax.components.electronic import VoltageSource
    from circulax.optical_analysis import quadrature_response

    gi, ge, delta, tuning = 1e10, 2e10, 3e10, 1.3e10
    resistance, capacitance = 50.0, 100e-15
    ring = loaded_ring(resistance=resistance, capacitance=capacitance, gamma_i=gi, gamma_e=ge, detuning=delta, tuning=tuning)
    circuit = compile_circuit(
        {
            "instances": {
                "GND": {"component": "ground"},
                "S": {"component": "S", "settings": {"power": 4.0}},
                "R": {"component": "R", "settings": {"R": 1.0}},
                "D": {"component": "ring"},
                "L": {"component": "load"},
                "V": {"component": "V", "settings": {"V": 0.0}},
            },
            "connections": {
                "S,p2": "GND,p1",
                "S,p1": "R,p1",
                "R,p2": "D,p1",
                "D,p2": "L,p1",
                "D,gnd": "GND,p1",
                "V,p2": "GND,p1",
                "V,p1": "D,drive",
            },
            "ports": {"out": "D,p2", "electrical": "D,junction"},
        },
        {"ground": lambda: 0, "S": OpticalSource, "R": Resistor, "ring": ring, "load": OpticalTermination, "V": VoltageSource},
        is_complex=True,
        backend="dense",
        g_leak=0,
    )
    freqs = jnp.array([1e8, 1e9, 10e9])
    omega = 2 * np.pi * np.asarray(freqs)
    result = quadrature_response(circuit, freqs=freqs, sources={"V": 1.0})
    index = circuit.port_map["out"]
    eindex = circuit.port_map["electrical"]
    rc = 1 / (1 + 1j * omega * resistance * capacitance)
    steady = np.sqrt(2 * ge) / (gi + ge - 1j * delta)
    upper = -1j * np.sqrt(2 * ge) * tuning * steady * rc / (gi + ge + 1j * omega - 1j * delta) / 2
    lower = -1j * np.sqrt(2 * ge) * tuning * steady * np.conj(rc) / (gi + ge - 1j * omega - 1j * delta) / 2
    np.testing.assert_allclose(result["upper_sideband"][:, index], upper, rtol=2e-8, atol=1e-10)
    np.testing.assert_allclose(result["lower_sideband"][:, index], lower, rtol=2e-8, atol=1e-10)
    np.testing.assert_allclose(result["in_phase"][:, eindex], rc, rtol=2e-8, atol=1e-10)
    np.testing.assert_allclose(result["quadrature"][:, eindex], 0.0, atol=1e-12)


def test_actual_cavity_ringdown_transient_convergence():
    gi, ge, delta = 1e10, 2e10, 1.7e10
    circuit = bench(AllPassRing, {"gamma_i": gi, "gamma_e": ge, "detuning": delta})
    params = {"S.power": 0.0}
    y = circuit.dc(params=params)
    initial = 1e-6 * (1 + 0.3j)
    n = circuit.sys_size
    ci = circuit.port_map["D,c_forward"]
    out = circuit.port_map["p2"]
    y = y.at[ci].set(initial.real).at[ci + n].set(initial.imag)
    field = -np.sqrt(2 * ge) * initial
    y = y.at[out].set(field.real).at[out + n].set(field.imag)
    times = jnp.linspace(0.0, 150e-12, 101)
    expected = initial * np.exp((-(gi + ge) + 1j * delta) * np.asarray(times))
    errors = []
    for dt in [1e-12, 0.5e-12]:
        sol = circuit.transient(
            t0=0.0,
            t1=150e-12,
            dt0=dt,
            y0=y,
            params=params,
            saveat=diffrax.SaveAt(ts=times),
            stepsize_controller=diffrax.ConstantStepSize(),
            max_steps=1000,
        )
        actual = np.asarray(sol.ys[:, ci] + 1j * sol.ys[:, ci + n])
        errors.append(np.max(np.abs(actual - expected)) / abs(initial))
    assert errors[1] < 2e-4
    assert errors[1] < errors[0] * 0.8


def test_filter_reciprocity_storage_and_parameter_ad():
    model = OpticalFilter(bandwidth_hz=1.5, detuning=2.0, transmission=0.7)
    a = np.array([0.3 + 0.2j, -0.4 + 0.7j])
    c = np.array([0.1 + 0.3j, -0.2 + 0.1j])
    y = jnp.concatenate([jnp.zeros(2), a, c])
    f, _ = model(y=y)
    b = a - np.array([f["p1"], f["p2"]])
    dcdt = -np.array([f["c1"], f["c2"]])
    net = np.vdot(a, a).real - np.vdot(b, b).real - 2 * np.vdot(c, dcdt).real
    rate = 2 * np.pi * 1.5
    hidden = np.sum(abs(a - np.sqrt(rate) * c) ** 2) + (1 - 0.7**2) * rate * np.vdot(c, c).real
    np.testing.assert_allclose(net, hidden, atol=1e-13)

    def loss(p):
        f, q = OpticalFilter.solver_call(0.0, y, OpticalFilter(bandwidth_hz=p[0], detuning=p[1], transmission=p[2]))
        return jnp.real(jnp.vdot(f, f) + jnp.vdot(q, q))

    p = jnp.array([1.5, 2.0, 0.7])
    np.testing.assert_allclose(jax.jacfwd(loss)(p), jax.jacrev(loss)(p), rtol=1e-12)
    assert np.isfinite(np.asarray(jax.jit(jax.vmap(loss))(jnp.stack([p, p])))).all()
    reverse = bench(OpticalFilter, {"bandwidth_hz": 1e9, "detuning": 2e9}, input_port="p2")
    np.testing.assert_allclose(reverse.port(reverse.dc(), "p1"), 2 * np.pi * 1e9 / (2 * np.pi * 1e9 - 2e9j), atol=2e-9)


@pytest.mark.parametrize(
    ("model", "settings"), [(AllPassRing, {"detuning": 2e10}), (OpticalFilter, {"bandwidth_hz": 1e9, "detuning": 2e9})]
)
def test_linear_optical_hb_matches_both_independent_sidebands(model, settings):
    import copy

    from circulax.components.electronic import VoltageSourceAC

    original = bench(model, settings)
    net = copy.deepcopy(original.source_netlist)
    net["instances"]["S"]["settings"] = {"V": 2.0, "freq": 1e9}
    models = dict(original.source_models)
    models["source"] = VoltageSourceAC
    circuit = compile_circuit(net, models, is_complex=True, backend="dense", g_leak=0)
    _, coefficients = circuit.hb(freq=1e9, harmonics=3)
    index = circuit.port_map["p2"]
    n = circuit.sys_size
    positive = coefficients[1, index] + 1j * coefficients[1, index + n]
    negative = np.conj(coefficients[1, index]) + 1j * np.conj(coefficients[1, index + n])
    omega = 2 * np.pi * 1e9
    if model is AllPassRing:
        hp = 1 - 2e10 / (2e10 + 1j * omega - 2e10j)
        hm = 1 - 2e10 / (2e10 - 1j * omega - 2e10j)
    else:
        hp = omega / (omega + 1j * omega - 2e9j)
        hm = omega / (omega - 1j * omega - 2e9j)
    np.testing.assert_allclose(positive, -0.5j * hp, atol=2e-9)
    np.testing.assert_allclose(negative, 0.5j * hm, atol=2e-9)
    np.testing.assert_allclose(coefficients[2:], 0.0, atol=1e-10)


@pytest.mark.parametrize(
    ("model", "settings"),
    [
        (AllPassRing, {"gamma_i": -1.0}),
        (AllPassRing, {"gamma_i": 0.0, "gamma_e": 0.0}),
        (AddDropRing, {"gamma_1": -1.0}),
        (AllPassRing, {"z0": 0.0}),
        (OpticalFilter, {"bandwidth_hz": 0.0}),
        (OpticalFilter, {"bandwidth_hz": -1.0}),
        (OpticalFilter, {"transmission": 1.1}),
        (OpticalFilter, {"transmission": -0.1}),
        (OpticalFilter, {"z0": 0.0}),
    ],
)
def test_invalid_physical_domains_rejected_under_jit(model, settings):
    import equinox as eqx

    instance = model(**settings)
    y = jnp.zeros(len(instance.ports) + len(instance.states), dtype=complex)
    with pytest.raises(eqx.EquinoxRuntimeError, match="requires"):
        eqx.filter_jit(lambda y: model.solver_call(0.0, y, instance)[0])(y).block_until_ready()


def test_zero_adddrop_coupling_valid_exact_endpoint():
    circuit = bench(AddDropRing, {"gamma_i": 1e10, "gamma_1": 0.0, "gamma_2": 0.0})
    y = circuit.dc()
    np.testing.assert_allclose(circuit.port(y, "p2"), 1.0, atol=2e-9)
    np.testing.assert_allclose(circuit.port(y, "p3"), 0.0, atol=2e-9)
    np.testing.assert_allclose(circuit.port(y, "p4"), 0.0, atol=2e-9)
