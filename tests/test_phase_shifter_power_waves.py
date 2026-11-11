# Copyright 2026 Circulax contributors
"""Mixed phase-shifter acceptance using explicit boundaries and analytic laws."""

import importlib
from collections.abc import Callable
from typing import NoReturn

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax import Circuit, QuadratureDrive, compile_circuit, sax_component
from circulax.components import junction_model, optical_wave_source, phase_shifter_factory, pn_diode
from circulax.components.base_component import PhysicsReturn, Signals, source

C = 2.99792458e14
TAU = 0.13e-9
LENGTH = C * TAU / 4
WL = 1550.0
A, B = 0.6 + 0.8j, -0.3j
BIAS = -0.4
SLOPE = -1e-5
JUNCTION = junction_model(current=lambda v, g=0.02: g * v + 0.003 * v**3, charge=lambda v, cap=3e-10: cap * (v + 0.1 * v**3))


def dn(v: jax.Array, slope: float = SLOPE) -> jax.Array:
    return slope * v + 1e-6 * v**2


def loss(v: jax.Array, loss0: float = 2.0, loss1: float = -0.5) -> jax.Array:
    return loss0 + loss1 * v


def transmission(
    v: jax.Array, *, phase: float = 0.7, length: float = LENGTH, slope: float = SLOPE, loss0: float = 2.0
) -> jax.Array:
    # Independent analytic reference, including the preserved negative carrier phase.
    return 10 ** (-(loss0 - 0.5 * v) * length / 10000 / 20) * jnp.exp(
        -1j * (phase * length / LENGTH + 2 * jnp.pi * (slope * v + 1e-6 * v**2) * length * 1000 / WL)
    )


def transmission_slope(v: float) -> complex:
    return complex(transmission(v)) * (0.5 * np.log(10) * LENGTH / 200000 - 2j * np.pi * (SLOPE + 2e-6 * v) * LENGTH * 1000 / WL)


@pytest.fixture(autouse=True)
def forbid_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "Mixed phase-shifter realization called s_to_y"
        raise AssertionError(msg)

    for module in ("circulax.s_transforms", "circulax.components.photonic"):
        monkeypatch.setattr(importlib.import_module(module), "s_to_y", forbidden)


def bench(
    *,
    r0: float = 1.0,
    gamma: complex | None = None,
    waveform: Callable | None = None,
    voltage: Callable | None = None,
    loaded: bool = True,
    backend: str = "dense",
) -> Circuit:
    @source(ports=("p", "n"), states=("i",), ac_source_stamp={"i": 1.0}, holomorphic=True)
    def drive(signals: Signals, t: float, bias: float = BIAS) -> PhysicsReturn:
        v = bias if voltage is None else voltage(t)
        return {"p": signals.i, "n": -signals.i, "i": signals.p - signals.n - v}, {}

    def reflector() -> dict:
        return {("p", "p"): gamma}

    return compile_circuit(
        {
            "instances": {
                "PS": {
                    "component": "ps",
                    "settings": {
                        "length_um": LENGTH,
                        "n_group": 4.0,
                        "neff": 0.7 * WL / (2 * np.pi * LENGTH * 1000),
                        "wavelength_nm": WL,
                    },
                },
                "S": {"component": "s", "settings": {"amplitude": A}},
                "R": {"component": "r", "settings": {"amplitude": B} if gamma is None else {}},
                "V": {"component": "v"},
                "GND": {"component": "ground"},
            },
            "connections": {"S,p": "PS,o1", "PS,o2": "R,p", "V,p": "PS,anode", "GND,p1": ("V,n", "PS,cathode")},
        },
        {
            "ps": phase_shifter_factory(delta_neff=dn, loss_dB_cm=loss, junction=JUNCTION if loaded else None, r0=r0),
            "s": optical_wave_source(r0=r0, waveform=waveform),
            "r": sax_component(reflector, r0=r0) if gamma is not None else optical_wave_source(r0=r0, waveform=waveform),
            "v": drive,
        },
        is_complex=True,
        backend=backend,
        g_leak=0.0,
    )


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
@pytest.mark.parametrize("phase", [0.0, np.pi / 2, np.pi])
def test_dc_reciprocal_coherent_and_reference_invariant(r0: float, phase: float) -> None:
    circuit = bench(r0=r0)
    for length, loss0 in ((0.0, 0.0), (LENGTH, 0.0), (LENGTH, 1e-12), (LENGTH, 2.0)):
        # Disable voltage loss slope to exercise exact and near-zero optical loss.
        updates = {
            "PS.neff": phase * WL / (2 * np.pi * LENGTH * 1000),
            "PS.length_um": length,
            "PS.loss0": loss0,
            "PS.loss1": 0.0,
            "V.bias": 0.0,
        }
        y, ok = circuit.dc_checked(params=updates, atol=1e-12, rtol=1e-12)
        assert ok
        expected = 10 ** (-loss0 * length / 200000) * np.exp(-1j * phase * length / LENGTH)
        waves = [circuit.port_waves(y, f"PS,{p}", params=updates) for p in ("o1", "o2")]
        np.testing.assert_allclose([w.incident for w in waves], [A, B], atol=2e-11)
        np.testing.assert_allclose([w.outgoing for w in waves], expected * np.array([B, A]), atol=2e-11)
        np.testing.assert_allclose(
            sum(w.net_power for w in waves), (1 - abs(expected) ** 2) * (abs(A) ** 2 + abs(B) ** 2), atol=2e-11
        )
    y = circuit.dc()
    np.testing.assert_allclose(circuit.port(y, "PS,anode"), BIAS, atol=1e-12)
    np.testing.assert_allclose(circuit.port(y, "V,i"), -LENGTH * 1e-6 * (0.02 * BIAS + 0.003 * BIAS**3), atol=1e-12)


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
@pytest.mark.parametrize("gamma", [1.0, -1.0, 0.4j])
def test_reflection_boundary_power_and_electrical_rejection(r0: float, gamma: complex) -> None:
    circuit = bench(r0=r0, gamma=gamma)
    y = circuit.dc()
    t = transmission(BIAS)
    left, right = [circuit.port_waves(y, f"PS,{p}") for p in ("o1", "o2")]
    np.testing.assert_allclose(
        [left.incident, left.outgoing, right.incident, right.outgoing], [A, gamma * t * t * A, gamma * t * A, t * A], atol=1e-10
    )
    np.testing.assert_allclose(circuit.port_waves(y, "S,p").net_power + left.net_power, 0, atol=1e-11)
    np.testing.assert_allclose(right.net_power, -(1 - abs(gamma) ** 2) * abs(t * A) ** 2, atol=1e-11)
    assert not np.isclose(abs(circuit.port(y, "PS,o1")) ** 2 / r0, float(left.incident_power))
    for terminal in ("anode", "cathode"):
        with pytest.raises(ValueError, match="optical terminal"):
            circuit.port_waves(y, f"PS,{terminal}")
    response = circuit.small_signal(y_dc=y)(freqs=jnp.array([0.0]), sources={"V": 1.0})
    for terminal in ("o1", "o2", "anode", "cathode"):
        with pytest.raises(ValueError, match="terminal potential"):
            response.power(f"PS,{terminal}", field_unit="sqrt(W)")
    with pytest.raises(ValueError, match="not AC"):
        circuit.port_waves(y, "PS,o1", analysis="ac")
    with pytest.raises(ValueError, match="quadrature"):
        circuit.ac_response(freqs=jnp.array([1e9]), sources={"V": 1.0})


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
def test_small_signal_optical_full_transit_voltage_half_transit_and_junction(r0: float) -> None:
    circuit = bench(r0=r0)
    freqs = jnp.array([0.0, 1e9, 5e9])
    full = np.exp(-2j * np.pi * freqs * TAU)
    half = np.exp(-2j * np.pi * freqs * TAU / 2)
    model = circuit.small_signal()
    optical = jax.jit(model)(freqs=freqs, sources={"S": QuadratureDrive(x=0.7 + 0.2j, y=-0.4j)})
    tx, ty = complex(transmission(BIAS)).real, complex(transmission(BIAS)).imag
    dx, dy = optical.quadratures("PS,o2")
    np.testing.assert_allclose(dx, np.sqrt(r0) * (tx * (0.7 + 0.2j) - ty * (-0.4j)) * full, atol=1e-10)
    np.testing.assert_allclose(dy, np.sqrt(r0) * (ty * (0.7 + 0.2j) + tx * (-0.4j)) * full, atol=1e-10)
    modulation = model(freqs=freqs, sources={"V": 1.0})
    for port, incident in (("o1", B), ("o2", A)):
        dx, dy = modulation.quadratures(f"PS,{port}")
        slope = transmission_slope(BIAS) * incident
        np.testing.assert_allclose(dx, np.sqrt(r0) * slope.real * half, atol=1e-10)
        np.testing.assert_allclose(dy, np.sqrt(r0) * slope.imag * half, atol=1e-10)
    expected_g = LENGTH * 1e-6 * (0.02 + 0.009 * BIAS**2)
    expected_c = LENGTH * 1e-6 * 3e-10 * (1 + 0.3 * BIAS**2)
    np.testing.assert_allclose(modulation.voltage("V,i"), -(expected_g + 2j * np.pi * freqs * expected_c), atol=1e-12)


def pulse(t: jax.Array) -> jax.Array:
    return 0.2 + 0.8 * jnp.exp(-(((t - 0.35e-9) / 0.07e-9) ** 2))


def voltage_step(t: jax.Array) -> jax.Array:
    return BIAS - 0.4 * jax.nn.sigmoid((t - 0.55e-9) / 0.03e-9)


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
def test_transient_pulse_and_midpoint_voltage_timing_with_refinement(r0: float) -> None:
    circuit = bench(r0=r0, waveform=pulse, voltage=voltage_step)
    times = jnp.linspace(0, 1e-9, 51)
    # Constant prehistory is the t=0 DC state, including all directional states.
    delayed_source = pulse(jnp.maximum(times - TAU, 0))
    delayed_v = voltage_step(jnp.maximum(times - TAU / 2, 0))
    errors = []
    for dt in (0.1e-9, 0.02e-9, 0.005e-9):
        sol = circuit.transient(t0=0, t1=1e-9, dt0=dt, saveat=times, max_steps=250, throw=True)
        waves = [circuit.port_waves(sol.ys, f"PS,{p}", analysis="tran", t=times) for p in ("o1", "o2")]
        expected = transmission(delayed_v) * delayed_source
        error = max(float(jnp.max(abs(w.outgoing - a * expected))) for w, a in zip(waves, (B, A), strict=True))
        errors.append(error)
        np.testing.assert_allclose(waves[0].outgoing / B, waves[1].outgoing / A, atol=1e-9)
        np.testing.assert_allclose(
            circuit.port(sol.ys, "PS,anode"),
            np.interp(times, np.arange(round(1e-9 / dt) + 1) * dt, voltage_step(np.arange(round(1e-9 / dt) + 1) * dt)),
            atol=1e-10,
        )
        np.testing.assert_allclose(waves[1].outgoing[0], transmission(voltage_step(0)) * pulse(0) * A, atol=1e-10)
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 0.003


def test_reflective_pulses_two_midpoint_interactions() -> None:
    gamma = 0.4j
    circuit = bench(gamma=gamma, waveform=pulse, voltage=voltage_step)
    times = jnp.linspace(0, 1e-9, 51)
    sol = circuit.transient(t0=0, t1=1e-9, dt0=0.005e-9, saveat=times, max_steps=250)
    # Return at t crossed the midpoint at t-tau/2 and t-3*tau/2.
    expected = (
        gamma
        * A
        * pulse(jnp.maximum(times - 2 * TAU, 0))
        * transmission(voltage_step(jnp.maximum(times - TAU / 2, 0)))
        * transmission(voltage_step(jnp.maximum(times - 3 * TAU / 2, 0)))
    )
    np.testing.assert_allclose(circuit.port_waves(sol.ys, "PS,o1", analysis="tran", t=times).outgoing, expected, atol=0.003)


@pytest.mark.parametrize("phase", [0.0, np.pi / 2, np.pi])
def test_zero_delay_instantaneous_control(phase: float) -> None:
    circuit = bench(waveform=pulse, voltage=voltage_step)
    updates = {"PS.n_group": 0.0, "PS.neff": phase * WL / (2 * np.pi * LENGTH * 1000)}
    times = jnp.arange(21) * 0.05e-9
    sol = circuit.transient(t0=0, t1=1e-9, dt0=0.05e-9, saveat=times, params=updates, max_steps=30)
    np.testing.assert_allclose(
        circuit.port_waves(sol.ys, "PS,o2", analysis="tran", t=times, params=updates).outgoing,
        transmission(voltage_step(times), phase=phase) * A * pulse(times),
        atol=1e-9,
    )


def test_hb_periodic_voltage_modulation_and_both_envelope_harmonics() -> None:
    freq = 1e9

    def optical(t: jax.Array) -> jax.Array:
        return 0.4 + 0.2 * jnp.exp(2j * jnp.pi * freq * t) + 0.1j * jnp.exp(-2j * jnp.pi * freq * t)

    def voltage(t: jax.Array) -> jax.Array:
        return BIAS + 0.2 * jnp.cos(2 * jnp.pi * freq * t)

    circuit = bench(waveform=optical, voltage=voltage)
    harmonics = 8
    y, _ = circuit.hb(freq=freq, harmonics=harmonics)
    times = jnp.arange(2 * harmonics + 1) / ((2 * harmonics + 1) * freq)
    expected = transmission(voltage(times - TAU / 2)) * optical(times - TAU)
    for port, amplitude in (("o1", B), ("o2", A)):
        np.testing.assert_allclose(
            circuit.port_waves(y, f"PS,{port}", analysis="hb_time", t=times).outgoing, amplitude * expected, atol=1e-8
        )


@pytest.mark.parametrize("loaded", [False, True])
def test_jit_vmap_updates_and_voltage_parameter_derivatives(*, loaded: bool) -> None:
    circuit = bench(loaded=loaded)

    def solve(v: float, slope: float) -> jax.Array:
        params = {"V.bias": v, "PS.slope": slope}
        y = circuit.dc(params=params)
        return circuit.port_waves(y, "PS,o2", params=params).outgoing

    voltages = jnp.array([-1.0, BIAS, 0.1])
    np.testing.assert_allclose(jax.jit(jax.vmap(solve, in_axes=(0, None)))(voltages, SLOPE), transmission(voltages) * A, atol=1e-10)
    np.testing.assert_allclose(jax.jit(jax.jacfwd(solve, argnums=0))(BIAS, SLOPE), transmission_slope(BIAS) * A, atol=1e-10)
    expected = -2j * np.pi * BIAS * LENGTH * 1000 / WL * transmission(BIAS) * A
    np.testing.assert_allclose(jax.jit(jax.jacfwd(solve, argnums=1))(BIAS, SLOPE), expected, atol=1e-8)


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
def test_local_fq_conservation_complete_real_derivatives_and_history(r0: float) -> None:
    cls = phase_shifter_factory(delta_neff=dn, loss_dB_cm=loss, junction=pn_diode(), r0=r0)
    current = jnp.linspace(-0.6, 0.1, 16)
    history = jnp.linspace(0.7, -0.4, 16)

    def residual(x: jax.Array, h: jax.Array, length: float) -> jax.Array:
        f, q = cls.solver_call(0.0, x[:8] + 1j * x[8:], cls(length_um=length, neff=0.0), hist=h[:8] + 1j * h[8:])
        return jnp.concatenate((f.real, f.imag, q.real, q.imag))

    for arg, point in ((0, current), (1, history)):
        actual = jax.jit(jax.jacfwd(residual, argnums=arg))(current, history, 300.0)

        def evaluate(value: jax.Array, arg: int = arg) -> jax.Array:
            return residual(value, history, 300.0) if arg == 0 else residual(current, value, 300.0)

        eps = 1e-6
        finite = np.stack([(evaluate(point + eps * e) - evaluate(point - eps * e)) / (2 * eps) for e in jnp.eye(16)], axis=1)
        np.testing.assert_allclose(actual, finite, rtol=2e-6, atol=2e-9)
    hist_jac = jax.jacfwd(residual, argnums=1)(current, history, 300.0)
    np.testing.assert_array_equal(hist_jac[jnp.array([2, 3, 10, 11])], 0)
    x = current[:8] + 1j * current[8:]
    f, q = cls.solver_call(0.0, x, cls(length_um=300), hist=history[:8] + 1j * history[8:])
    v = float((x[0] - x[1]).real)
    diode = pn_diode()
    np.testing.assert_allclose([f[0], q[0]], [300e-6 * diode.current(v), 300e-6 * diode.charge(v)], rtol=1e-12)
    np.testing.assert_array_equal([f[0] + f[1], q[0] + q[1]], 0)
    df = jax.jacfwd(residual, argnums=0)(current, history, 300.0)
    np.testing.assert_allclose(df[0, :2], 300e-6 * 1e-8 * np.exp(v / 0.025852) / 0.025852 * np.array([1, -1]), rtol=1e-12)
    np.testing.assert_allclose(df[16, :2], 300e-6 * 3e-10 / (1 - v / 0.9) ** 0.5 * np.array([1, -1]), rtol=1e-12)
    np.testing.assert_array_equal(df[[0, 16], 8:], 0)
    derivative = jax.jacfwd(residual, argnums=2)(current, history, 300.0)
    np.testing.assert_allclose(
        derivative, (residual(current, history, 300.001) - residual(current, history, 299.999)) / 0.002, rtol=1e-6, atol=1e-10
    )


@pytest.mark.parametrize("reference", [0.0, -1.0, np.inf, np.nan, 1j])
def test_reference_validation(reference: float) -> None:
    with pytest.raises(ValueError, match="finite, positive, and real"):
        phase_shifter_factory(delta_neff=0.0, r0=reference)


@pytest.mark.parametrize("reverse", [False, True])
def test_single_direction_illumination(*, reverse: bool) -> None:
    circuit = bench()
    params = {"S.amplitude": 0.0 if reverse else A, "R.amplitude": A if reverse else 0.0}
    y = circuit.dc(params=params)
    near, far = ("o2", "o1") if reverse else ("o1", "o2")
    np.testing.assert_allclose(circuit.port_waves(y, f"PS,{near}", params=params).outgoing, 0, atol=1e-11)
    np.testing.assert_allclose(circuit.port_waves(y, f"PS,{far}", params=params).outgoing, A * transmission(BIAS), atol=1e-11)


@pytest.mark.parametrize("dt", [0.1e-9, 0.025e-9])
def test_transient_parameter_gradient_between_interpolation_knots(dt: float) -> None:
    circuit = bench(waveform=lambda t: 0.3 + t / 1e-9, voltage=lambda t: BIAS + 0.2 * t / 1e-9, loaded=False)

    def objective(length: float) -> jax.Array:
        updates = {"PS.length_um": length}
        sol = circuit.transient(t0=0, t1=0.3e-9, dt0=dt, saveat=jnp.array([0.3e-9]), params=updates, max_steps=30)
        return circuit.port_waves(sol.ys[-1], "PS,o2", analysis="tran", t=0.3e-9, params=updates).outgoing.real

    derivative = jax.jit(jax.grad(objective))(LENGTH)
    eps = 0.1
    finite = (objective(LENGTH + eps) - objective(LENGTH - eps)) / (2 * eps)
    np.testing.assert_allclose(derivative, finite, rtol=2e-4, atol=1e-9)


@pytest.mark.parametrize("backend", ["sparse", "klu"])
def test_mixed_wave_backend_regression(backend: str) -> None:
    circuit = bench(backend=backend)
    y = circuit.dc()
    np.testing.assert_allclose(circuit.port_waves(y, "PS,o2").outgoing, A * transmission(BIAS), atol=1e-8)


def test_small_signal_power_guard_covers_raw_sax_port_names() -> None:
    def reflection() -> dict:
        return {("optical:in", "optical:in"): 0.4j}

    circuit = compile_circuit(
        {"instances": {"S": {"component": "s"}, "D": {"component": "d"}}, "connections": {"S,p": "D,optical:in"}},
        {"s": optical_wave_source(), "d": sax_component(reflection)},
        backend="dense",
        g_leak=0.0,
    )
    result = circuit.small_signal()(freqs=jnp.array([0.0]), sources={"S": 1.0})
    for name in ("D,optical:in", "D,optical_in", "S,p"):
        with pytest.raises(ValueError, match="terminal potential"):
            result.power(name, field_unit="sqrt(W)")


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
def test_reflected_small_signal_midpoint_interactions_and_power_semantics(r0: float) -> None:
    gamma = 0.4j
    circuit = bench(r0=r0, gamma=gamma)
    freqs = jnp.array([0.0, 1e9, 5e9])
    half = np.exp(-2j * np.pi * freqs * TAU / 2)
    triple_half = np.exp(-2j * np.pi * freqs * 3 * TAU / 2)
    y = circuit.dc()
    response = circuit.small_signal(y_dc=y)(freqs=freqs, sources={"V": 1.0})
    t = complex(transmission(BIAS))
    coefficient = gamma * A * t * transmission_slope(BIAS)
    dx, dy = response.quadratures("PS,o1")
    np.testing.assert_allclose(dx, np.sqrt(r0) * coefficient.real * (half + triple_half), atol=1e-10)
    np.testing.assert_allclose(dy, np.sqrt(r0) * coefficient.imag * (half + triple_half), atol=1e-10)
    # The launched wave is fixed by S, so only the returning wave varies here.
    # This independent boundary fact allows converting these field phasors;
    # arbitrary terminal potentials would not supply a general AC power API.
    returning = complex(circuit.port_waves(y, "PS,o1").outgoing)
    wave_power_modulation = 2 * (returning.real * dx + returning.imag * dy) / np.sqrt(r0)
    expected = 2 * np.real(np.conj(gamma * A * t * t) * coefficient) * (half + triple_half)
    np.testing.assert_allclose(wave_power_modulation, expected, atol=1e-10)
    potential = complex(circuit.port(y, "PS,o1"))
    potential_intensity_modulation = 2 * (potential.real * dx + potential.imag * dy) / r0
    assert not np.allclose(potential_intensity_modulation, expected)
    dx, dy = response.quadratures("PS,o2")
    far_coefficient = (1 + gamma) * A * transmission_slope(BIAS)
    np.testing.assert_allclose(dx, np.sqrt(r0) * far_coefficient.real * half, atol=1e-10)
    np.testing.assert_allclose(dy, np.sqrt(r0) * far_coefficient.imag * half, atol=1e-10)
