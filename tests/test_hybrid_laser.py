# Copyright (c) 2026 GDSFactory
"""Composition, scattering, timing and Vernier checks through public Circulax."""

import jax
import jax.numpy as jnp
import numpy as np
from diffrax import Solution

from circulax import Circuit, QuadratureDrive, compile_circuit
from circulax.components.electronic import CurrentSource, Resistor, VoltageSource
from circulax.components.hybrid_laser import (
    HybridSettings,
    grating_settings,
    hybrid_laser,
    hybrid_laser_netlist,
    hybrid_modes,
    hybrid_threshold,
    ring_reflector,
)
from circulax.components.laser_sections import (
    MatchedOpticalPulse,
    sampled_bragg_grating,
    sampled_grating_response,
)
from circulax.components.soa import LIGHT_SPEED
from circulax.components.traveling_wave_laser import LaserParameters
from circulax.solvers import BDF2VectorizedTransientSolver


def design() -> tuple[LaserParameters, HybridSettings]:
    p = LaserParameters()
    s = HybridSettings()
    g = float(hybrid_threshold(p, s, 0, (0, 0, 0)).gain_m1)
    p = p._replace(detuning_rad_s=LIGHT_SPEED / p.group_index * (p.alpha_h * g / 2 - np.pi / (2 * p.length_m)))
    return p, s


def cavity(
    p: LaserParameters,
    s: HybridSettings,
    segments: int = 4,
    *,
    right_reflector: object = None,
    left_reflector: object = None,
    hierarchical: bool = False,
) -> Circuit:
    if hierarchical:
        laser = hybrid_laser(p, s, segments=segments, left_reflector=left_reflector, right_reflector=right_reflector)
        net = {
            "instances": {"LASER": {"component": "laser"}},
            "connections": {},
            "ports": {"left": "LASER,p1", "right": "LASER,p2"},
        }
        models = {"laser": laser}
        ports = {
            name: f"LASER,{name}"
            for name in ("p1", "p2", "anode", "cathode", "heater_left", "heater_right", "heater_phase", "heater_return")
        }
    else:
        net, models = hybrid_laser_netlist(p, s, segments=segments, left_reflector=left_reflector, right_reflector=right_reflector)
        ports = net["ports"].copy()
        net["ports"] = {"left": ports["p1"], "right": ports["p2"]}
    net["instances"].update(
        {
            "PUMP": {"component": "pump"},
            "SEED": {"component": "seed"},
            "LOAD": {"component": "load", "settings": {"R": 1.0}},
            "HL": {"component": "voltage"},
            "HR": {"component": "voltage"},
            "HP": {"component": "voltage"},
            "GND": {"component": "ground"},
        }
    )
    net["connections"].update(
        {
            "GND,p1": (ports["cathode"], ports["heater_return"], "PUMP,p1", "SEED,p2", "LOAD,p2", "HL,p2", "HR,p2", "HP,p2"),
            "PUMP,p2": ports["anode"],
            "SEED,p1": ports["p1"],
            "LOAD,p1": ports["p2"],
            "HL,p1": ports["heater_left"],
            "HR,p1": ports["heater_right"],
            "HP,p1": ports["heater_phase"],
        }
    )
    models.update(pump=CurrentSource, seed=MatchedOpticalPulse, load=Resistor, voltage=VoltageSource, ground=lambda: 0)
    return compile_circuit(net, models, is_complex=True, backend="klu", g_leak=0, rtol=1e-10, atol=1e-12)


def passive_bench(component: object, settings: dict | None = None) -> Circuit:
    net = {
        "instances": {
            "D": {"component": "device", "settings": settings or {}},
            "S": {"component": "source"},
            "LOAD": {"component": "load", "settings": {"R": 1.0}},
            "H": {"component": "voltage"},
            "GND": {"component": "ground"},
        },
        "connections": {
            "S,p1": "D,p1",
            "D,p2": "LOAD,p1",
            "H,p1": "D,heater",
            "GND,p1": ("S,p2", "LOAD,p2", "H,p2", "D,heater_return"),
        },
        "ports": {"out": "D,p2", "in": "D,p1"},
    }
    return compile_circuit(
        net,
        {"device": component, "source": MatchedOpticalPulse, "load": Resistor, "voltage": VoltageSource, "ground": lambda: 0},
        is_complex=True,
        backend="klu",
        g_leak=0,
        rtol=1e-10,
        atol=1e-12,
    )


def test_native_sampled_grating_scattering_and_thermal_tuning() -> None:
    p, s = design()
    settings = grating_settings(p, s, "left")
    c = passive_bench(sampled_bragg_grating(s.samples), settings)
    frequencies = jnp.linspace(-450e9, 450e9, 601)
    for voltage in (0.0, 1.0):
        model = c.small_signal(params={"H.V": voltage, "S.power_w": 0})
        response = model(freqs=frequencies, sources={"S": QuadratureDrive(x=1, y=-1j)})
        r, t = sampled_grating_response(2 * jnp.pi * frequencies, 2 * voltage**2, samples=s.samples, **settings)
        np.testing.assert_allclose(response.sidebands("D,b0")[0], r, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(response.sidebands("out")[0], t, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(abs(r) ** 2 + abs(t) ** 2, 1, atol=3e-14)
        np.testing.assert_allclose(c.port(model.y_dc, "H,i_src"), -voltage / 100, atol=1e-12)
    r, _ = sampled_grating_response(2 * jnp.pi * jnp.array([0, s.left_fsr_hz]), samples=s.samples, **settings)
    np.testing.assert_allclose(abs(r) ** 2, np.tanh(s.grating_strength) ** 2, atol=1e-12)


def run(c: Circuit, params: dict, end: float = 1.5e-9, dt: float = 0.3e-12, samples: int = 501) -> Solution:
    initial = c.dc(params=params)
    return c.transient(
        t0=0.0,
        t1=end,
        dt0=dt,
        y0=initial,
        saveat=jnp.linspace(0, end, samples),
        params=params,
        max_steps=int(np.ceil(end / dt)) + 4,
        throw=True,
        transient_solver=BDF2VectorizedTransientSolver(linear_solver=c.solver, newton_rtol=1e-9, newton_atol=1e-12),
    )


def test_hybrid_threshold_native_startup_and_segmentation() -> None:
    p, s = design()
    condition = hybrid_threshold(p, s, 0, (0, 0, 0))
    np.testing.assert_allclose(condition.roundtrip, 1, atol=1e-12)
    powers = []
    for segments in (2, 4, 8):
        c = cavity(p, s, segments)
        sol = run(c, {"PUMP.I": 1.3 * condition.current_a, "SEED.power_w": 1e-12})
        power = abs(c.port(sol.ys, "right")) ** 2
        assert float(power[-30:].mean()) > 1e-4
        h = jnp.stack([jnp.real(c.port(sol.ys, f"GAIN{j},h")) * segments / p.length_m for j in range(segments)])
        np.testing.assert_allclose(h[:, -30:].mean(), condition.gain_m1, rtol=0.025)
        powers.append(np.asarray(power))
    errors = [np.linalg.norm(y - powers[-1]) / np.linalg.norm(powers[-1]) for y in powers[:-1]]
    assert errors[1] < errors[0]


def test_flat_and_hierarchical_assembly_agree() -> None:
    p, s = design()
    a = cavity(p, s, 2)
    b = cavity(p, s, 2, hierarchical=True)
    # Zero optical drive gives the same electrical DC and all optical fields zero.
    params = {"PUMP.I": 0.05, "HL.V": 0.3, "HR.V": 0.5, "HP.V": 0.7, "SEED.power_w": 0}
    ya, yb = a.dc(params=params), b.dc(params=params)
    np.testing.assert_allclose(a.port(ya, "GAIN0,v_j"), b.port(yb, "LASER~GAIN0,v_j"), rtol=1e-10)
    frequency = jnp.array([-20e9, 0.0, 20e9])
    ra = a.small_signal(params=params)(freqs=frequency, sources={"SEED": QuadratureDrive(x=1, y=-1j)})
    rb = b.small_signal(params=params)(freqs=frequency, sources={"SEED": QuadratureDrive(x=1, y=-1j)})
    np.testing.assert_allclose(ra.sidebands("right")[0], rb.sidebands("right")[0], rtol=1e-9, atol=1e-11)


def test_vernier_comb_overlap_and_target_phase_gradient() -> None:
    p, s = design()
    # Adjacent comb orders align at -2.2 THz (the least common spacing).
    beat = s.left_fsr_hz * s.right_fsr_hz / abs(s.right_fsr_hz - s.left_fsr_hz)
    frequencies = jnp.array([0, beat])
    rl = sampled_grating_response(2 * jnp.pi * frequencies, samples=s.samples, **grating_settings(p, s, "left"))[0]
    rr = sampled_grating_response(2 * jnp.pi * frequencies, samples=s.samples, **grating_settings(p, s, "right"))[0]
    np.testing.assert_allclose(abs(rl * rr) ** 2, np.tanh(s.grating_strength) ** 4, atol=1e-12)
    target = -200e9
    center_target = LIGHT_SPEED / (LIGHT_SPEED / p.wavelength_m + target)
    tr = (center_target - p.wavelength_m) / s.shift_m_per_k
    modes, thresholds = hybrid_modes(p, s, (0, tr, 0), bounds_hz=(-250e9, 250e9), points=10001)
    assert len(modes) > 10
    assert abs(modes[np.argmin(thresholds)] - target) < 40e9

    def objective(v: jax.Array) -> jax.Array:
        return hybrid_threshold(p, s, 2 * jnp.pi * target, (2 * v[0] ** 2, 2 * v[1] ** 2, 2 * v[2] ** 2)).current_a

    trial = jnp.array([0.1, np.sqrt(tr / 2), 0.6])
    derivative = jax.grad(objective)(trial)
    eps = 1e-4
    fd = np.array([(objective(trial + eps * jnp.eye(3)[i]) - objective(trial - eps * jnp.eye(3)[i])) / (2 * eps) for i in range(3)])
    np.testing.assert_allclose(derivative, fd, rtol=1e-4, atol=1e-9)


def test_ring_reflector_is_replaceable_and_passive() -> None:
    reflector = ring_reflector()
    c = passive_bench(reflector)
    frequencies = jnp.linspace(-150e9, 150e9, 301)
    response = c.small_signal(params={"S.power_w": 0})(freqs=frequencies, sources={"S": QuadratureDrive(x=1, y=-1j)})
    reflected = response.sidebands("in")[0] - 1
    transmitted = response.sidebands("out")[0]
    assert float(jnp.max(abs(reflected) ** 2)) > 0.4
    assert float(jnp.min(abs(reflected) ** 2)) < 0.01
    assert np.all(np.asarray(abs(reflected) ** 2 + abs(transmitted) ** 2) <= 1 + 1e-10)
    p, s = design()
    laser = cavity(p, s, 2, right_reflector=reflector)
    dc = laser.dc(params={"PUMP.I": 0.03, "SEED.power_w": 0, "HR.V": 0.4})
    assert abs(laser.port(dc, "right")) < 1e-12


def test_target_mode_native_frame_and_time_resolution() -> None:
    from circulax.components.hybrid_laser import hybrid_frame_parameters

    p, s = design()
    target = 200e9
    tr = (LIGHT_SPEED / (LIGHT_SPEED / p.wavelength_m + target - s.right_fsr_hz) - p.wavelength_m) / s.shift_m_per_k
    condition = hybrid_threshold(p, s, 2 * np.pi * target, (0, tr, 0))
    tp = (-np.angle(condition.roundtrip)) % (2 * np.pi) / (2 * s.phase_per_k)
    c = cavity(p, s, 4)
    params = {
        "PUMP.I": 1.15 * condition.current_a,
        "HR.V": np.sqrt(tr / 2),
        "HP.V": np.sqrt(tp / 2),
        "SEED.power_w": 1e-12,
        **hybrid_frame_parameters(2 * np.pi * target, 4),
    }
    fields = []
    for dt in (0.6e-12, 0.3e-12, 0.15e-12):
        sol = run(c, params, end=1.5e-9, dt=dt, samples=501)
        fields.append(np.asarray(c.port(sol.ys, "right")))
    errors = [np.linalg.norm(abs(y) ** 2 - abs(fields[-1]) ** 2) / np.linalg.norm(abs(fields[-1]) ** 2) for y in fields[:-1]]
    assert errors[1] < 0.8 * errors[0]
    assert errors[1] < 0.02
    settled = run(c, params, end=5e-9, dt=0.2e-12, samples=1001)
    field = np.asarray(c.port(settled.ys, "right"))
    residual = np.polyfit(np.asarray(settled.ts[-150:]), np.unwrap(np.angle(field[-150:])), 1)[0] / (2 * np.pi)
    assert abs(residual) < 1e6
    assert np.mean(abs(field[-100:]) ** 2) > 1e-4


def test_bidirectional_waveguide_pulse_timing_and_loss() -> None:
    from circulax.components.laser_sections import BidirectionalWaveguide

    length = 200e-6
    tau = length * 4 / LIGHT_SPEED
    c = compile_circuit(
        {
            "instances": {
                "W": {"component": "waveguide", "settings": {"length_m": length, "loss_m1": 500}},
                "L": {"component": "source"},
                "R": {"component": "source"},
                "GND": {"component": "ground"},
            },
            "connections": {"L,p1": "W,p1", "R,p1": "W,p2", "GND,p1": ("L,p2", "R,p2")},
            "ports": {"left": "W,p1", "right": "W,p2"},
        },
        {"waveguide": BidirectionalWaveguide, "source": MatchedOpticalPulse, "ground": lambda: 0},
        is_complex=True,
        backend="klu",
        g_leak=0,
    )
    dt = tau / 69.2
    ts = jnp.linspace(0, 15e-12, 601)
    params = {
        "L.power_w": 1e-3,
        "R.power_w": 2e-3,
        "L.start_s": 1e-12,
        "R.start_s": 1e-12,
        "L.duration_s": 3e-12,
        "R.duration_s": 3e-12,
    }
    sol = c.transient(
        t0=0.0,
        t1=float(ts[-1]),
        dt0=dt,
        y0=c.dc(params=params),
        saveat=ts,
        params=params,
        max_steps=int(np.ceil(float(ts[-1]) / dt)) + 4,
        throw=True,
        transient_solver=BDF2VectorizedTransientSolver(linear_solver=c.solver, newton_rtol=1e-9, newton_atol=1e-12),
    )

    def pulse(t: np.ndarray) -> np.ndarray:
        x = np.clip((t - 1e-12) / 3e-12, 0, 1)
        return np.sin(np.pi * x) ** 2

    for end, power in (("left", 2e-3), ("right", 1e-3)):
        own = c.port(sol.ys, "W,a1" if end == "left" else "W,a2")
        returned = np.asarray(c.port(sol.ys, end) - own)
        expected = np.sqrt(power) * np.exp(-500 * length / 2) * pulse(np.asarray(ts) - tau)
        np.testing.assert_allclose(returned, expected, atol=1e-4)
        assert np.max(abs(returned[np.asarray(ts) < tau + 0.8e-12])) < 1e-8


def test_gain_cell_shares_local_carriers_between_directions() -> None:
    from circulax.components.laser_sections import TravelingWaveGainCell

    cell = TravelingWaveGainCell(length_m=75e-6)
    first, _ = cell(a1=0.01, a2=0.02, h=0.4, v_j=1.1)
    reverse, _ = cell(a1=0.02, a2=0.01, h=0.4, v_j=1.1)
    dark, _ = cell(a1=0.0, a2=0.0, h=0.4, v_j=1.1)
    np.testing.assert_allclose(first["h"], reverse["h"], rtol=1e-12)
    assert first["h"] > dark["h"]


def test_ring_replacement_passive_transient_feedback() -> None:
    p, s = design()
    p = p._replace(transparency_m3=0.0, injection_efficiency=0.0, saturation=0.0, alpha_h=0.0, detuning_rad_s=0.0)
    c = cavity(p, s, 2, right_reflector=ring_reflector())
    sol = run(c, {"PUMP.I": 0.0, "SEED.power_w": 1e-3}, end=0.8e-9, dt=0.15e-12, samples=1601)
    power = np.asarray(abs(c.port(sol.ys, "right")) ** 2)
    assert power.max() > 1e-7
    assert power[-100:].max() < power.max() * 1e-5


def test_two_distinct_ring_reflectors_in_hierarchical_cavity() -> None:
    p, s = design()
    c = cavity(
        p, s, 2, left_reflector=ring_reflector(length_m=400e-6), right_reflector=ring_reflector(length_m=450e-6), hierarchical=True
    )
    params = {"PUMP.I": 0.03, "SEED.power_w": 0.0, "HL.V": 0.2, "HR.V": 0.4, "HP.V": 0.1}
    response = c.small_signal(params=params)(freqs=jnp.linspace(-80e9, 80e9, 81), sources={"SEED": QuadratureDrive(x=1, y=-1j)})
    transmission = np.asarray(abs(response.sidebands("right")[0]) ** 2)
    assert transmission.max() > 1e-8
    assert transmission.max() < 1.0


def test_relevant_vernier_roots_converge_with_frequency_grid() -> None:
    p, s = design()
    tr = (LIGHT_SPEED / (LIGHT_SPEED / p.wavelength_m - 20e9) - p.wavelength_m) / s.shift_m_per_k
    phase = np.angle(hybrid_threshold(p, s, 2 * np.pi * 200e9, (0, tr, 0)).roundtrip)
    temps = (0, tr, (-phase) % (2 * np.pi) / (2 * s.phase_per_k))
    ith = float(hybrid_threshold(p, s, 0, (0, 0, 0)).current_a)
    frequencies, currents = hybrid_modes(p, s, temps, max_current_a=1.3 * ith)
    refined, refined_currents = hybrid_modes(p, s, temps, max_current_a=1.3 * ith, points=44001)
    np.testing.assert_allclose(frequencies, refined, atol=1.0)
    np.testing.assert_allclose(currents, refined_currents, rtol=1e-9)
    selected = np.argsort(currents)
    assert abs(frequencies[selected[0]] - 200e9) < 1e6
    assert currents[selected[1]] - currents[selected[0]] > 1e-3
