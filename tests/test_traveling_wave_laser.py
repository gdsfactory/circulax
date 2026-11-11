# Copyright (c) 2026 GDSFactory
"""Native Circulax laser: propagation, facets, shared depletion and convergence."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from diffrax import Solution
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

from circulax import Circuit, compile_circuit
from circulax.components.base_component import CircuitComponent
from circulax.components.electronic import CurrentSource, Resistor
from circulax.components.photonic import OpticalWaveguide
from circulax.components.soa import ELECTRON_CHARGE, LIGHT_SPEED, PLANCK, stimulated_power
from circulax.components.traveling_wave_laser import (
    CoherentSeed,
    LaserParameters,
    TravelingWaveLaser,
    carrier_density,
    material_scales,
    resonant_detuning,
    seeded_initial_state,
    threshold,
    traveling_wave_laser,
)
from circulax.solvers import BDF2VectorizedTransientSolver, TrapVectorizedTransientSolver


def cavity(p: LaserParameters, segments: int = 4) -> Circuit:
    return compile_circuit(
        {
            "instances": {
                "LASER": {"component": "laser", "settings": p._asdict()},
                "PUMP": {"component": "pump"},
                "SEED": {"component": "seed", "settings": {"power_w": 0.0}},
                "GND": {"component": "ground"},
            },
            "connections": {"GND,p1": ("LASER,cathode", "PUMP,p1"), "PUMP,p2": "LASER,anode", "SEED,out": "LASER,seed"},
            "ports": {"left": "LASER,left", "right": "LASER,right"},
        },
        {"laser": traveling_wave_laser(segments), "pump": CurrentSource, "seed": CoherentSeed, "ground": lambda: 0},
        backend="klu",
        is_complex=True,
        g_leak=0,
        rtol=1e-10,
        atol=1e-12,
    )


def run(c: Circuit, y0: jax.Array, current: jax.Array, end: float, dt: float, ts: jax.Array, **params: float) -> Solution:
    solver = TrapVectorizedTransientSolver(linear_solver=c.solver, newton_rtol=1e-9, newton_atol=1e-12)
    return c.transient(
        t0=0.0,
        t1=end,
        dt0=dt,
        y0=y0,
        saveat=ts,
        params={"PUMP.I": current, **params},
        transient_solver=solver,
        max_steps=int(np.ceil(end / dt)) + 4,
        throw=True,
    )


def resonant(p: LaserParameters) -> LaserParameters:
    return p._replace(detuning_rad_s=float(resonant_detuning(p)))


def test_native_component_contract_and_shared_reservoir() -> None:
    assert issubclass(TravelingWaveLaser, CircuitComponent)
    assert TravelingWaveLaser.ports == ("anode", "cathode", "seed", "left", "right")
    cls = traveling_wave_laser(1)
    cell = cls(alpha_h=0, saturation=1, r_left=1, r_right=1)
    common = {"h0": 0.4, "v_j": 1.1}
    first, _ = cell(f0=0.01, b0=0.02, **common)
    swapped, _ = cell(f0=0.02, b0=0.01, **common)
    dark, _ = cell(f0=0.0, b0=0.0, **common)
    energy = float(material_scales(LaserParameters())[2])
    expected = 280e-12 * float(stimulated_power(0.4, 0.6, 0.01**2 + 0.02**2)) / energy
    np.testing.assert_allclose(first["h0"] - dark["h0"], expected, rtol=1e-12)
    np.testing.assert_allclose(first["h0"], swapped["h0"])


def test_passive_roundtrip_timing_decay_and_complex_reflectors() -> None:
    p = LaserParameters(transparency_m3=0, injection_efficiency=0, saturation=0, r_left=0.9j, r_right=0.7 * np.exp(0.3j))
    n = 4
    c = cavity(p, n)
    flight = p.length_m * p.group_index / LIGHT_SPEED
    dt = flight / (n * 8)
    ts = jnp.arange(5 * n * 8 + 1) * dt
    y0 = c.dc()
    sol = run(
        c,
        y0,
        0.0,
        float(ts[-1]),
        dt,
        ts,
        **{
            "SEED.power_w": 1e-3,
            "SEED.start_s": flight / 4,
            "SEED.duration_s": flight / 4,
        },
    )
    field = np.asarray(c.port(sol.ys, "right"))
    # The sin² field pulse peaks at start+duration/2, then travels L/vg.
    first = 44  # (1 + 1/4 + 1/8)*flight/dt
    second = first + 2 * n * 8
    assert np.max(abs(field[:40])) < 1e-10
    expected = np.sqrt(1 - abs(p.r_right) ** 2) * np.sqrt(1e-3) * np.exp(-p.loss_m1 * p.length_m / 2)
    np.testing.assert_allclose(field[first], expected, rtol=2e-6)
    ratio = p.r_left * p.r_right * np.exp(-p.loss_m1 * p.length_m)
    np.testing.assert_allclose(field[second] / field[first], ratio, rtol=3e-6)
    assert abs(float(ts[second] - ts[first]) - 2 * flight) < 1e-25


@pytest.mark.parametrize("offset", [-100.0, 0.0, 100.0])
def test_native_small_signal_threshold_and_henry_phase(offset: float) -> None:
    p = resonant(LaserParameters(saturation=0))
    n = 4
    c = cavity(p, n)
    gain, nth, _ = threshold(p)
    density = nth + offset / (p.confinement * p.differential_gain_m2)
    current = density * ELECTRON_CHARGE * material_scales(p)[0] / (p.injection_efficiency * p.lifetime_s)
    dc = c.dc(params={"PUMP.I": current})
    initial = seeded_initial_state(c, dc, p, n)
    for j in range(n):
        initial = initial.at[c.port_map[f"LASER,h{j}"]].set((gain + offset) * p.length_m / n)
    trt = 2 * p.length_m * p.group_index / LIGHT_SPEED
    dt = trt / (2 * n * 8)
    # Off-threshold constant prehistory has an algebraic jump at release.
    # BDF2's backward-Euler startup projects it without trapezoidal ringing.
    sol = c.transient(
        t0=0.0,
        t1=trt,
        dt0=dt,
        y0=initial,
        saveat=jnp.array([0, trt]),
        params={"PUMP.I": current},
        max_steps=68,
        throw=True,
        transient_solver=BDF2VectorizedTransientSolver(linear_solver=c.solver, newton_rtol=1e-9, newton_atol=1e-12),
    )
    ratio = c.port(sol.ys, "right")[-1] / c.port(initial, "right")
    expected = (
        p.r_left
        * p.r_right
        * np.exp(((1 - 1j * p.alpha_h) * (gain + offset) - p.loss_m1) * p.length_m + 1j * p.detuning_rad_s * trt)
    )
    np.testing.assert_allclose(ratio, expected, rtol=2e-6)
    if offset == 0:
        np.testing.assert_allclose(abs(ratio) ** 2, 1, rtol=2e-6)


def startup(n: int, divisor: float = 2, save_samples: int = 281) -> tuple:
    p = resonant(LaserParameters())
    c = cavity(p, n)
    current = 1.5 * threshold(p)[2]
    y0 = seeded_initial_state(c, c.dc(params={"PUMP.I": current}), p, n)
    flight = p.length_m * p.group_index / LIGHT_SPEED
    end = flight * 280
    ts = jnp.linspace(0, end, save_samples)
    sol = run(c, y0, current, end, flight / (n * divisor), ts)
    return p, c, sol


def test_startup_and_spatial_temporal_convergence() -> None:
    runs = [startup(n) for n in (4, 8, 16)]
    traces = [np.asarray(abs(c.port(sol.ys, "right")) ** 2) for _, c, sol in runs]
    errors = [np.linalg.norm(y - traces[-1]) for y in traces[:-1]]
    assert errors[1] < errors[0] * 0.75
    p, c, refined = startup(8, 4)
    fine = np.asarray(abs(c.port(refined.ys, "right")) ** 2)
    np.testing.assert_allclose(fine[-30:].mean(), traces[1][-30:].mean(), rtol=0.005)
    assert 0.0058 < fine[-30:].mean() < 0.0063
    gain = jnp.stack([c.port(refined.ys, f"LASER,h{j}") for j in range(8)], axis=-1)
    density = carrier_density(p, gain, 8)
    np.testing.assert_allclose(density[-30:].mean(), threshold(p)[1], rtol=0.002)


def test_high_q_lumped_rate_equation_limit() -> None:
    p = resonant(LaserParameters(r_left=0.999 + 0j, r_right=0.999 + 0j, loss_m1=10, alpha_h=0))
    n = 4
    c = cavity(p, n)
    gain, nth, ith = threshold(p)
    current = 1.002 * ith
    initial = seeded_initial_state(c, c.dc(params={"PUMP.I": current}), p, n, seed_power_w=1e-5)
    velocity = LIGHT_SPEED / p.group_index
    flight = p.length_m / velocity
    ts = jnp.linspace(0, 150 * flight, 151)
    sol = run(c, initial, current, float(ts[-1]), flight / (n * 2), ts)
    power = jnp.stack([abs(c.port(sol.ys, f"LASER,{kind}{j}")) ** 2 for kind in ("f", "b") for j in range(n)], axis=-1)
    energy = np.asarray(power.sum(axis=-1) * flight / n)
    hv = PLANCK * LIGHT_SPEED / p.wavelength_m
    volume = float(material_scales(p)[0])

    def rhs(t: float, y: np.ndarray) -> tuple:  # noqa: ARG001
        density = y[0] * float(nth)
        g = p.confinement * p.differential_gain_m2 * (density - p.transparency_m3)
        dn = (
            p.injection_efficiency * float(current) / (ELECTRON_CHARGE * volume)
            - density / p.lifetime_s
            - velocity * g * y[1] * energy[0] / (hv * volume)
        )
        return dn / float(nth), velocity * (g - float(gain)) * y[1]

    reference = solve_ivp(rhs, (0, float(ts[-1])), (1, 1), t_eval=np.asarray(ts), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(energy / energy[0], reference.y[1], rtol=0.015)


def test_batched_native_transient() -> None:
    p = resonant(LaserParameters())
    c = cavity(p, 4)
    currents = threshold(p)[2] * jnp.array([1.3, 1.5])
    initial = c.dc(params={"PUMP.I": currents})
    ts = jnp.linspace(0, 1e-9, 51)

    def one(current: jax.Array, dc: jax.Array) -> jax.Array:
        y0 = seeded_initial_state(c, dc, p, 4)
        sol = run(c, y0, current, 1e-9, 1e-12, ts)
        return abs(c.port(sol.ys, "right")) ** 2

    batch = jax.jit(jax.vmap(one))(currents, initial)
    np.testing.assert_allclose(batch[0], one(currents[0], initial[0]), rtol=1e-6, atol=1e-10)
    assert np.isfinite(batch).all()


def test_native_transient_gradient_matches_finite_difference() -> None:
    p = resonant(LaserParameters())
    c = cavity(p, 4)
    ith = threshold(p)[2]
    ts = jnp.linspace(0, 1.5e-9, 151)

    def objective(u: jax.Array) -> jax.Array:
        current = u * ith
        dc = c.dc(params={"PUMP.I": current})
        initial = seeded_initial_state(c, dc, p, 4)
        sol = run(c, initial, current, 1.5e-9, 1e-12, ts)
        return (abs(c.port(sol.ys, "right")) ** 2)[-20:].mean()

    value, derivative = jax.jit(jax.value_and_grad(objective))(jnp.array(1.5))
    fd = (objective(1.5001) - objective(1.4999)) / 0.0002
    assert value > 0
    np.testing.assert_allclose(derivative, fd, rtol=1e-4)


def test_unseeded_native_cavity_stays_dark() -> None:
    p = resonant(LaserParameters())
    c = cavity(p, 4)
    _, nth, ith = threshold(p)
    current = 1.5 * ith
    initial = seeded_initial_state(c, c.dc(params={"PUMP.I": current}), p, 4, seed_power_w=0.0)
    sol = run(c, initial, current, 0.5e-9, 1e-12, jnp.array([0.0, 0.5e-9]))
    np.testing.assert_array_equal(c.port(sol.ys, "right"), 0)
    density = carrier_density(p, c.port(sol.ys[-1], "LASER,h0"), 4)
    assert density > nth


def test_off_grid_time_resolution_of_startup_and_relaxation() -> None:
    traces = []
    for divisor in (2.7, 5.4, 10.8):
        _, c, sol = startup(4, divisor, save_samples=2241)
        traces.append(np.asarray(abs(c.port(sol.ys, "right")) ** 2))
    errors = [np.linalg.norm(y - traces[-1]) for y in traces[:-1]]
    assert errors[1] < 0.6 * errors[0]
    np.testing.assert_allclose(traces[1].max(), traces[-1].max(), rtol=0.003)
    ts = np.asarray(sol.ts)
    assert abs(ts[np.argmax(traces[1])] - ts[np.argmax(traces[-1])]) < 2e-12
    periods = []
    for trace in traces:
        peaks = find_peaks(trace, prominence=2e-5)[0][:4]
        assert peaks.size >= 3
        periods.append(np.diff(ts[peaks]).mean())
    np.testing.assert_allclose(periods[1], periods[-1], rtol=0.02)


def test_passive_off_grid_delay_convergence() -> None:
    p = LaserParameters(transparency_m3=0, injection_efficiency=0, saturation=0, r_left=0.9j, r_right=0.7 * np.exp(0.3j))
    c = cavity(p, 4)
    flight = p.length_m * p.group_index / LIGHT_SPEED
    ts = jnp.linspace(0, 5 * flight, 1601)
    initial = c.dc()

    # Continuous analytic pulse train at the right tap (three reflected passes).
    def pulse(t: np.ndarray) -> np.ndarray:
        phase = np.clip((t - flight / 4) / (flight / 4), 0, 1)
        return np.where((t > flight / 4) & (t < flight / 2), np.sin(np.pi * phase) ** 2, 0)

    expected = sum(
        (p.r_left * p.r_right * np.exp(-p.loss_m1 * p.length_m)) ** k * pulse(np.asarray(ts) - (2 * k + 1) * flight)
        for k in range(3)
    )
    expected *= np.sqrt(1 - abs(p.r_right) ** 2) * np.sqrt(1e-3) * np.exp(-p.loss_m1 * p.length_m / 2)
    errors = []
    traces = []
    for divisor in (7.3, 14.6, 29.2):
        sol = run(
            c,
            initial,
            0.0,
            float(ts[-1]),
            flight / (4 * divisor),
            ts,
            **{"SEED.power_w": 1e-3, "SEED.start_s": flight / 4, "SEED.duration_s": flight / 4},
        )
        field = np.asarray(c.port(sol.ys, "right"))
        traces.append(field)
        errors.append(np.linalg.norm(field - expected) / np.linalg.norm(expected))
    assert errors[2] < 0.5 * errors[1] < 0.25 * errors[0]
    first = (np.asarray(ts) > flight) & (np.asarray(ts) < 2 * flight)
    second = (np.asarray(ts) > 3 * flight) & (np.asarray(ts) < 4 * flight)
    t1 = np.asarray(ts)[first][np.argmax(abs(traces[-1][first]))]
    t2 = np.asarray(ts)[second][np.argmax(abs(traces[-1][second]))]
    assert abs((t2 - t1) - 2 * flight) < flight / 100


def test_matched_waveguide_composition() -> None:
    p = resonant(LaserParameters())
    n = 4
    net = {
        "instances": {
            "LASER": {"component": "laser", "settings": p._asdict()},
            "PUMP": {"component": "pump"},
            "SEED": {"component": "seed", "settings": {"power_w": 0}},
            "WG": {"component": "waveguide", "settings": {"length_um": 500.0, "loss_dB_cm": 10.0, "wavelength_nm": 1540.0}},
            "LOAD": {"component": "load", "settings": {"R": 1.0}},
            "GND": {"component": "ground"},
        },
        "connections": {
            "GND,p1": ("LASER,cathode", "PUMP,p1", "LOAD,p2"),
            "PUMP,p2": "LASER,anode",
            "SEED,out": "LASER,seed",
            "LASER,right": "WG,p1",
            "WG,p2": "LOAD,p1",
        },
        "ports": {"left": "LASER,left", "right": "LASER,right", "received": "WG,p2"},
    }
    c = compile_circuit(
        net,
        {
            "laser": traveling_wave_laser(n),
            "pump": CurrentSource,
            "seed": CoherentSeed,
            "waveguide": OpticalWaveguide,
            "load": Resistor,
            "ground": lambda: 0,
        },
        backend="klu",
        is_complex=True,
        g_leak=0,
        rtol=1e-10,
        atol=1e-12,
    )
    current = 1.5 * threshold(p)[2]
    initial = seeded_initial_state(c, c.dc(params={"PUMP.I": current}), p, n)
    ts = jnp.linspace(0, 1.6e-9, 161)
    sol = c.transient(
        t0=0.0,
        t1=1.6e-9,
        dt0=0.8e-12,
        y0=initial,
        saveat=ts,
        params={"PUMP.I": current},
        max_steps=2004,
        throw=True,
        transient_solver=BDF2VectorizedTransientSolver(linear_solver=c.solver, newton_rtol=1e-9, newton_atol=1e-12),
    )
    emitted = c.port(sol.ys, "right")[-40:]
    received = c.port(sol.ys, "received")[-40:]
    np.testing.assert_allclose(abs(received) ** 2 / abs(emitted) ** 2, 10 ** (-0.5 / 10), rtol=1e-8)
    # For this matched unit-impedance input, the nodal field equals the emitted wave.
    np.testing.assert_allclose(c.port(sol.ys, "LASER,i_right")[-40:], -emitted, rtol=1e-8, atol=1e-12)
