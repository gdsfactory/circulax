# Copyright (c) 2026 GDSFactory
"""Thermal/spectral tuning qualifies native component equations, not SMSR."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize

from circulax import Circuit, QuadratureDrive, compile_circuit
from circulax.components.electronic import CurrentSource, VoltageSource
from circulax.components.laser_mirrors import (
    MirrorParameters,
    ThermalReflector,
    mirror_detuning,
    mirror_power_balance,
    reflection_response,
)
from circulax.components.laser_modes import find_laser_modes, spectral_threshold
from circulax.components.photonic import OpticalSource
from circulax.components.soa import LIGHT_SPEED
from circulax.components.traveling_wave_laser import (
    CoherentSeed,
    LaserParameters,
    resonant_detuning,
    seeded_initial_state,
    threshold,
    traveling_wave_laser,
)
from circulax.solvers import BDF2VectorizedTransientSolver, TrapVectorizedTransientSolver


def reflector_circuit() -> Circuit:
    return compile_circuit(
        {
            "instances": {
                "M": {"component": "mirror"},
                "LIGHT": {"component": "light"},
                "HEAT": {"component": "voltage"},
                "GND": {"component": "ground"},
            },
            "connections": {"GND,p1": ("M,heater_return", "LIGHT,p2", "HEAT,p2"), "LIGHT,p1": "M,incident", "HEAT,p1": "M,heater"},
            "ports": {"reflected": "M,reflected", "tap": "M,tap"},
        },
        {"mirror": ThermalReflector, "light": OpticalSource, "voltage": VoltageSource, "ground": lambda: 0},
        is_complex=True,
        backend="klu",
        g_leak=0,
        rtol=1e-10,
        atol=1e-12,
    )


def tunable_circuit(p: LaserParameters, m: MirrorParameters, segments: int = 4) -> Circuit:
    return compile_circuit(
        {
            "instances": {
                "LASER": {"component": "laser", "settings": {**p._asdict(), **m._asdict()}},
                "PUMP": {"component": "pump"},
                "SEED": {"component": "seed", "settings": {"power_w": 0}},
                "HP": {"component": "voltage"},
                "HF": {"component": "voltage"},
                "GND": {"component": "ground"},
            },
            "connections": {
                "GND,p1": ("LASER,cathode", "LASER,heater_return", "PUMP,p1", "HP,p2", "HF,p2"),
                "PUMP,p2": "LASER,anode",
                "SEED,out": "LASER,seed",
                "HP,p1": "LASER,heater_phase",
                "HF,p1": "LASER,heater_filter",
            },
            "ports": {"right": "LASER,right", "left": "LASER,left"},
        },
        {
            "laser": traveling_wave_laser(segments, tunable_mirrors=True),
            "pump": CurrentSource,
            "seed": CoherentSeed,
            "voltage": VoltageSource,
            "ground": lambda: 0,
        },
        backend="klu",
        is_complex=True,
        g_leak=0,
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize("frame", [0.0, -2 * np.pi * 9e9])
def test_native_reflector_frequency_response_and_heater_loading(frame: float) -> None:
    c = reflector_circuit()
    model = c.small_signal(params={"HEAT.V": 0.5, "M.frame_offset_rad_s": frame})
    temperature = float(jnp.real(c.port(model.y_dc, "M,temperature")))
    np.testing.assert_allclose(temperature, 0.5)
    np.testing.assert_allclose(c.port(model.y_dc, "HEAT,i_src"), -0.005)
    frequency = jnp.array([-30e9, -10e9, 0, 10e9, 30e9])
    response = model(freqs=frequency, sources={"LIGHT": QuadratureDrive(x=1.0, y=-1j)})
    center = mirror_detuning(1540e-9, 0.08e-9, temperature, 1540e-9)
    expected = reflection_response(0.7, 12e9, 2 * jnp.pi * frequency + frame, center)
    np.testing.assert_allclose(response.sidebands("reflected")[0], expected, rtol=1e-8, atol=1e-10)
    assert np.all(abs(expected) <= 0.7)


def test_filter_passivity_including_stored_energy() -> None:
    rng = np.random.default_rng(27)
    field = rng.normal(size=100) + 1j * rng.normal(size=100)
    reflected = rng.normal(size=100) + 1j * rng.normal(size=100)
    gamma = 2 * np.pi * 12e9
    peak = 0.7 * np.exp(0.3j)
    derivative = -(gamma - 1j * 2e10) * reflected + gamma * peak * field
    tap, dump, stored, rate = mirror_power_balance(field, reflected, derivative, peak, gamma)
    np.testing.assert_allclose(abs(field) ** 2 - abs(reflected) ** 2 - tap - dump, rate, atol=1e-14)
    assert np.all(stored >= 0)
    assert np.all(dump >= 0)


def test_native_mirror_ringdown_energy_and_thermal_step() -> None:
    c = reflector_circuit()
    initial = c.dc(params={"LIGHT.power": 0.0})
    for name in ("M,m", "M,reflected"):
        initial = initial.at[c.port_map[name]].set(0.7)
    gamma = 2 * np.pi * 12e9
    times = jnp.linspace(0, 8 / gamma, 321)
    sol = c.transient(
        t0=0.0,
        t1=float(times[-1]),
        dt0=float(times[1]),
        y0=initial,
        saveat=times,
        params={"LIGHT.power": 0.0},
        max_steps=324,
        throw=True,
        transient_solver=TrapVectorizedTransientSolver(linear_solver=c.solver, newton_rtol=1e-10, newton_atol=1e-12),
    )
    field = np.asarray(c.port(sol.ys, "M,m"))
    np.testing.assert_allclose(field, 0.7 * np.exp(-gamma * np.asarray(times)), rtol=5e-4, atol=1e-8)
    # With no input, reflected and dump powers are each |m|².
    released = np.trapezoid(2 * abs(field) ** 2, np.asarray(times))
    np.testing.assert_allclose(released + abs(field[-1]) ** 2 / gamma, 0.7**2 / gamma, rtol=3e-4)
    initial = c.dc(params={"LIGHT.power": 0.0, "HEAT.V": 0.0})
    ts = jnp.linspace(0, 3e-6, 241)
    heated = c.transient(
        t0=0.0,
        t1=3e-6,
        dt0=3e-6 / 240,
        y0=initial,
        saveat=ts,
        params={"LIGHT.power": 0.0, "HEAT.V": 1.0},
        max_steps=244,
        throw=True,
        transient_solver=BDF2VectorizedTransientSolver(linear_solver=c.solver, newton_rtol=1e-10, newton_atol=1e-12),
    )
    temperature = jnp.real(c.port(heated.ys, "M,temperature"))
    np.testing.assert_allclose(temperature, 2 * (1 - np.exp(-np.asarray(ts) / 1e-6)), atol=4e-4)


def test_wavelength_design_through_native_heater_circuit() -> None:
    p = LaserParameters()
    p = p._replace(detuning_rad_s=float(resonant_detuning(p)))
    m = MirrorParameters()
    c = tunable_circuit(p, m)
    target_m = 1540.08e-9
    omega = 2 * np.pi * LIGHT_SPEED * (1 / target_m - 1 / p.wavelength_m)
    pump = 1.3 * float(threshold(p)[2])

    def conditions(v: jax.Array) -> tuple:
        dc = c.dc(params={"HP.V": v[0], "HF.V": v[1], "PUMP.I": pump})
        tp = jnp.real(c.port(dc, "LASER,temperature_phase"))
        tf = jnp.real(c.port(dc, "LASER,temperature_filter"))
        return spectral_threshold(p, m, omega, tp, tf), tp, tf

    objective = jax.jit(lambda v: conditions(v)[0].current_a / threshold(p)[2] + 1e-4 * jnp.sum(v * v))
    derivative = jax.jit(jax.grad(objective))
    phase = jax.jit(lambda v: conditions(v)[0].phase_rad)
    phase_gradient = jax.jit(jax.grad(phase))
    trial = np.array([0.8, 0.8])
    eps = 1e-4
    fd = np.array([(objective(trial + eps * np.eye(2)[j]) - objective(trial - eps * np.eye(2)[j])) / (2 * eps) for j in range(2)])
    np.testing.assert_allclose(derivative(trial), fd, rtol=2e-5, atol=1e-8)
    result = minimize(
        objective,
        trial,
        jac=derivative,
        method="SLSQP",
        bounds=[(0, 2), (0, 2)],
        constraints={"type": "eq", "fun": phase, "jac": phase_gradient},
        options={"ftol": 1e-12, "maxiter": 40},
    )
    assert result.success, result.message
    condition, tp, tf = conditions(result.x)
    assert abs(condition.phase_rad) < 1e-7
    modes = find_laser_modes(p, m, float(tp), float(tf))
    refined = find_laser_modes(p, m, float(tp), float(tf), frequency_bounds_hz=(-350e9, 350e9), samples=11201)
    selected = np.argmin(abs(modes.frequency_hz - omega / (2 * np.pi)))
    assert abs(modes.frequency_hz[selected] - omega / (2 * np.pi)) < 1e4
    assert selected == np.argmin(modes.threshold_a)
    gap = np.min(np.delete(modes.threshold_a, selected)) - modes.threshold_a[selected]
    assert gap > 0.005
    edge = spectral_threshold(p, m, 2 * jnp.pi * jnp.array([-350e9, 350e9]), tp, tf).current_a
    # The one-pole reflection magnitude decreases monotonically away from its
    # center, so all frequencies outside these bounds require still more gain.
    assert jnp.min(edge) > modes.threshold_a[selected] + gap
    assert abs(np.min(refined.threshold_a) - modes.threshold_a[selected]) < 1e-8
    updates = {
        "HP.V": result.x[0],
        "HF.V": result.x[1],
        "PUMP.I": 1.3 * float(condition.current_a),
        "LASER.frame_offset_rad_s": omega,
    }
    dc = c.dc(params=updates)
    initial = seeded_initial_state(c, dc, p, 4, mirrors=m, frame_offset_rad_s=omega)
    flight = p.length_m * p.group_index / LIGHT_SPEED
    ts = jnp.linspace(0, 420 * flight, 841)
    sol = c.transient(
        t0=0.0,
        t1=float(ts[-1]),
        dt0=flight / 8,
        y0=initial,
        saveat=ts,
        params=updates,
        max_steps=3364,
        throw=True,
        transient_solver=TrapVectorizedTransientSolver(linear_solver=c.solver, newton_rtol=1e-9, newton_atol=1e-12),
    )
    field = np.asarray(c.port(sol.ys, "right"))
    residual_hz = np.polyfit(np.asarray(ts[-200:]), np.unwrap(np.angle(field[-200:])), 1)[0] / (2 * np.pi)
    assert abs(residual_hz) < 2e7
    assert np.mean(abs(field[-100:]) ** 2) > 1e-3
