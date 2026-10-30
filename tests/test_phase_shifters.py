# Copyright (c) 2026 Circulax contributors.
# ruff: noqa: ANN001, ANN201, ANN202
"""Promoted PN phase and charge laws, real-block AD and loaded control."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax import compile_circuit
from circulax.components.electronic import Resistor, WaveformVoltageSource
from circulax.components.phase_shifters import ElectricalPhaseShifter, PNJunctionPhaseShifter, PNPhaseShifter, loaded_phase_shifter
from circulax.components.photonic import OpticalSource

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


def real_block_ac(circuit, dc, frequencies, source="V"):
    """Independent dense solve of compiled F/Q quadrature Jacobians."""
    from circulax.solvers.assembly import assemble_residual_only_complex

    def residual(y):
        return assemble_residual_only_complex(y, circuit.groups, 0.0, 1.0)

    jf, jq = jax.jacfwd(residual)(dc)
    jf = np.array(jf)
    jq = np.array(jq)
    for index in (0, circuit.sys_size):
        jf[index, :] = 0
        jf[index, index] = 1
        jq[index, :] = 0
    rhs = np.zeros(len(dc))
    rhs[circuit.port_map[f"{source},i_src"]] = 1
    solutions = np.stack([np.linalg.solve(np.asarray(jf) + 2j * np.pi * f * np.asarray(jq), rhs) for f in frequencies])
    return solutions


@pytest.mark.parametrize(
    ("model", "settings"),
    [(PNPhaseShifter, {"length_um": 200, "capacitance": 64e-15}), (PNJunctionPhaseShifter, {"length": 200, "Cj0_per_um_fF": 0.32})],
)
def test_promoted_charge_and_quadratic_phase(model, settings):
    ps = model(**settings, a1=1.2, a2=0.15, bias=-1)
    voltage = -0.5
    phi = 0.2 * (1.2 * (voltage + 1) + 0.15 * (voltage**2 - 1))
    half = 10 ** (-1 * 200 / 10000 / 40) * np.exp(-1j * np.pi * 2.4 * 200 / 1.55)
    f, q = ps(anode=voltage + 0.1j, cathode=0, a_forward=1, m_forward=half)
    np.testing.assert_allclose(f["o2"], -(half**2) * np.exp(1j * phi), atol=1e-12)
    assert q["anode"] == pytest.approx(64e-15 * (voltage + 0.1j))
    assert q["anode"] + q["cathode"] == 0


def test_ideal_control_real_block_ad_jit_and_parameter_gradient():
    def evaluate(x, efficiency):
        f, _ = ElectricalPhaseShifter(efficiency=efficiency)(anode=x[0] + 1j * x[1], a1=x[2] + 1j * x[3])
        z = f["o2"]
        return jnp.array([z.real, z.imag])

    point = jnp.array([0.4, 0.1, 0.8, -0.3])
    forward = jax.jit(jax.jacfwd(evaluate))(point, 0.7)
    reverse = jax.jacrev(evaluate)(point, 0.7)
    np.testing.assert_allclose(forward, reverse, atol=1e-14)
    step = 1e-6
    finite = np.stack(
        [
            (np.asarray(evaluate(point + jnp.eye(4)[k] * step, 0.7)) - np.asarray(evaluate(point - jnp.eye(4)[k] * step, 0.7)))
            / (2 * step)
            for k in range(4)
        ],
        axis=1,
    )
    np.testing.assert_allclose(forward, finite, atol=1e-10)

    np.testing.assert_allclose(forward[:, 1], 0, atol=1e-14)
    assert np.isfinite(jax.jacfwd(evaluate, argnums=1)(point, 0.7)).all()
    assert np.isfinite(jax.jit(jax.vmap(evaluate, in_axes=(0, None)))(jnp.stack([point, point]), 0.7)).all()


def test_loaded_phase_shifter_actual_dc_ac_and_transient():
    R, C, eff = 100.0, 1e-5, 0.3
    device = loaded_phase_shifter(resistance=R, capacitance=C, efficiency=eff)
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "D": {"component": "D"},
            "V": {"component": "V", "settings": {"dc": 0.4}},
            "S": {"component": "S", "settings": {"power": 1}},
            "L": {"component": "R", "settings": {"R": 1}},
        },
        "connections": {"V,p1": "D,anode", "GND,p1": ("V,p2", "D,cathode", "S,p2", "L,p2"), "S,p1": "D,o1", "D,o2": "L,p1"},
        "ports": {"junction": "D,junction", "out": "D,o2"},
    }
    c = compile_circuit(net, {"D": device, "V": WaveformVoltageSource, "S": OpticalSource, "R": Resistor}, is_complex=True)
    dc = c.dc()
    np.testing.assert_allclose(c.port(dc, "out"), np.exp(0.12j), atol=1e-9)
    f = jnp.array([10.0, 100.0, 1000.0])
    ac = {"solution": real_block_ac(c, dc, f)}
    np.testing.assert_allclose(ac["solution"][:, c.port_map["junction"]], 1 / (1 + 2j * np.pi * np.asarray(f) * R * C), atol=1e-8)
    # Relax an initially zero junction from a constant source: independent RC step.
    initial = jnp.zeros_like(dc)
    # Circuit transient initialization accepts y0 for an explicit charged-state test.
    t = jnp.linspace(0, 0.005, 101)
    sol = c.transient(t0=0, t1=0.005, dt0=2.5e-5, saveat=t, y0=initial, max_steps=220)
    actual = np.asarray(c.port(sol.ys, "junction")).real
    np.testing.assert_allclose(actual, 0.4 * (1 - np.exp(-np.asarray(t) / (R * C))), atol=0.002)
    coarse = c.transient(t0=0, t1=0.005, dt0=1e-4, saveat=t, y0=initial, max_steps=70)
    exact = 0.4 * (1 - np.exp(-np.asarray(t) / (R * C)))
    assert np.max(abs(actual - exact)) < np.max(abs(np.asarray(c.port(coarse.ys, "junction")).real - exact))
    np.testing.assert_allclose(np.asarray(c.port(sol.ys, "junction")).imag, 0, atol=5e-10)


@pytest.mark.parametrize(
    ("model", "settings"),
    [(PNPhaseShifter, {"length_um": 200, "capacitance": 64e-15}), (PNJunctionPhaseShifter, {"length": 200, "Cj0_per_um_fF": 0.32})],
)
def test_promoted_phase_shifter_pulse_delay_compiled(model, settings):
    from circulax.components.electronic import WAVE_PWL

    length, group, neff, wavelength = 200.0, 4.05, 2.4, 1.55
    delay = length * group / 2.99792458e14
    times = jnp.linspace(0, 7e-12, 281)
    knots = jnp.array([0, 1e-12, 1.5e-12, 7e-12])
    envelope = jnp.array([0, 0, 1, 1])
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "PS": {"component": "PS", "settings": settings},
            "S": {"component": "V", "settings": {"kind": WAVE_PWL, "pwl_t": knots, "pwl_v": envelope}},
            "B": {"component": "V", "settings": {"dc": -1}},
            "L": {"component": "R", "settings": {"R": 1}},
        },
        "connections": {"S,p1": "PS,o1", "PS,o2": "L,p1", "B,p1": "PS,anode", "GND,p1": ("S,p2", "B,p2", "PS,cathode", "L,p2")},
        "ports": {"out": "PS,o2"},
    }
    c = compile_circuit(net, {"PS": model, "V": WaveformVoltageSource, "R": Resistor}, is_complex=True)
    sol = c.transient(t0=0, t1=7e-12, dt0=2.5e-14, saveat=times, max_steps=300)
    transmission = 10 ** (-length / 10000 / 20) * np.exp(-2j * np.pi * neff * length / wavelength)
    expected = transmission * np.interp(np.asarray(times) - delay, np.asarray(knots), np.asarray(envelope), left=0)
    np.testing.assert_allclose(c.port(sol.ys, "out"), expected, atol=0.014)
    # This checks actual envelope flight, not a wavelength sweep or cosmetic pole.
    assert abs(c.port(sol.ys, "out")[int((1e-12 + delay - 1e-13) / 2.5e-14)]) < 1e-6


def test_ideal_nonunit_reference_impedance_pseudovoltage_scaling():
    z0 = 3.2
    a1 = 0.8 + 0.4j
    a2 = -0.2 + 0.5j
    voltage = 0.7
    t = 10 ** (-(0.3 + 0.2 * voltage**2) / 20) * np.exp(1j * (0.1 + 0.4 * voltage))
    v1 = a1 + t * a2
    v2 = a2 + t * a1
    model = ElectricalPhaseShifter(z0=z0, phase=0.1, efficiency=0.4, loss_dB=0.3, loss_per_v2=0.2)
    f, _ = model(anode=voltage, a1=a1, a2=a2, o1=v1, o2=v2)
    np.testing.assert_allclose([f["a1"], f["a2"]], 0, atol=1e-14)
    incident = np.array([(v1 + z0 * f["o1"]) / (2 * np.sqrt(z0)), (v2 + z0 * f["o2"]) / (2 * np.sqrt(z0))])
    outgoing = np.array([(v1 - z0 * f["o1"]) / (2 * np.sqrt(z0)), (v2 - z0 * f["o2"]) / (2 * np.sqrt(z0))])
    np.testing.assert_allclose(incident, np.array([a1, a2]) / np.sqrt(z0))
    np.testing.assert_allclose(outgoing, t * incident[::-1])
    assert np.sum(abs(outgoing) ** 2) <= np.sum(abs(incident) ** 2)
