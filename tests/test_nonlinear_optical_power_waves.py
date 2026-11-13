# Copyright 2026 Circulax contributors
"""Delayed directional TPA against independent public-analysis references."""

import importlib
from collections.abc import Callable
from typing import NoReturn

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax import Circuit, QuadratureDrive, compile_circuit, sax_component
from circulax.components import nonlinear_optical_waveguide, optical_wave_source

C = 2.99792458e14
LENGTH = 5000.0
TAU = LENGTH * 4 / C
A = 0.03 + 0.04j
B = -0.02j
SETTINGS = {"length_um": LENGTH, "loss_dB_cm": 3.0, "tpa_coeff": 4000.0, "neff": 0.7 * 1310 / (2 * np.pi * LENGTH * 1000)}


@pytest.fixture(autouse=True)
def forbid_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "Nonlinear realization called s_to_y"
        raise AssertionError(msg)

    for module in ("circulax.s_transforms", "circulax.components.photonic"):
        monkeypatch.setattr(importlib.import_module(module), "s_to_y", forbidden)


def propagate(
    u: np.ndarray | complex, *, length: float = LENGTH, loss: float = 3.0, tpa: float = 4000.0, phase: float = 0.7
) -> np.ndarray:
    """Independent integrated dP/dz=-alpha*P-beta*P^2, then carrier phase."""
    distance = length / 10000
    alpha = loss * np.log(10) / 10
    leff = (1 - np.exp(-alpha * distance)) / alpha if alpha > 1e-12 else distance
    power = np.abs(u) ** 2 * np.exp(-alpha * distance) / (1 + tpa * np.abs(u) ** 2 * leff)
    return np.sqrt(power) * np.exp(1j * (np.angle(u) - phase * length / LENGTH))


def driven(*, r0: float = 1.0, gamma: complex | None = None, waveform: Callable | None = None, backend: str = "dense") -> Circuit:
    def reflector() -> dict:
        return {("p", "p"): gamma}

    return compile_circuit(
        {
            "instances": {
                "S": {"component": "s", "settings": {"amplitude": A}},
                "D": {"component": "d", "settings": SETTINGS},
                "R": {"component": "r", "settings": {"amplitude": B} if gamma is None else {}},
            },
            "connections": {"S,p": "D,p1", "D,p2": "R,p"},
        },
        {
            "s": optical_wave_source(r0=r0, waveform=waveform),
            "d": nonlinear_optical_waveguide(r0=r0),
            "r": optical_wave_source(r0=r0, waveform=waveform) if gamma is None else sax_component(reflector, r0=r0),
        },
        backend=backend,
        g_leak=0.0,
    )


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
@pytest.mark.parametrize("phase", [0.0, np.pi / 2, np.pi])
def test_dc_bidirectional_phase_reference_and_passive_balance(r0: float, phase: float) -> None:
    circuit = driven(r0=r0)
    for length, loss, tpa in ((0.0, 0.0, 4000.0), (LENGTH, 0.0, 0.0), (LENGTH, 0.0, 4000.0), (LENGTH, 3.0, 4000.0)):
        params = {
            "D.length_um": length,
            "D.loss_dB_cm": loss,
            "D.tpa_coeff": tpa,
            "S.amplitude": A * np.exp(1j * phase),
            "D.neff": phase * 1310 / (2 * np.pi * LENGTH * 1000),
        }
        y, ok = circuit.dc_checked(params=params, atol=1e-12, rtol=1e-12)
        assert ok
        left, right = [circuit.port_waves(y, f"D,p{i}", params=params) for i in (1, 2)]
        incoming = np.array([A * np.exp(1j * phase), B])
        outgoing = propagate(incoming[::-1], length=length, loss=loss, tpa=tpa, phase=phase)
        np.testing.assert_allclose([left.incident, right.incident], incoming, atol=1e-11)
        np.testing.assert_allclose([left.outgoing, right.outgoing], outgoing, atol=1e-11)
        expected_loss = np.sum(abs(incoming) ** 2 - abs(outgoing) ** 2)
        np.testing.assert_allclose(left.net_power + right.net_power, expected_loss, atol=1e-12)
        assert expected_loss >= -1e-16


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
@pytest.mark.parametrize("gamma", [0.0, 0.4j, -1.0])
def test_reflecting_dc_and_potential_interference(r0: float, gamma: complex) -> None:
    circuit = driven(r0=r0, gamma=gamma)
    y = circuit.dc()
    forward = propagate(A)
    returned = propagate(gamma * forward)
    left, right = [circuit.port_waves(y, f"D,p{i}") for i in (1, 2)]
    np.testing.assert_allclose(
        [left.incident, left.outgoing, right.incident, right.outgoing], [A, returned, gamma * forward, forward], atol=1e-11
    )
    np.testing.assert_allclose(circuit.port(y, "D,p1"), np.sqrt(r0) * (A + returned), atol=1e-11)
    np.testing.assert_allclose(
        circuit.port_waves(y, "S,p").net_power + left.net_power + right.net_power + circuit.port_waves(y, "R,p").net_power,
        0,
        atol=1e-12,
    )


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
@pytest.mark.parametrize("n_group", [0.0, 4.0])
def test_small_signal_full_intensity_jacobian_and_rf_phase(r0: float, n_group: float) -> None:
    circuit = driven(r0=r0)
    params = {"D.n_group": n_group}
    op = circuit.dc(params=params)
    freq = jnp.array([-3e9, 0.0, 7e9])
    tau = TAU * n_group / 4
    drives = {"S": QuadratureDrive(x=0.003 + 0.004j, y=-0.002 + 0.001j), "R": QuadratureDrive(x=-0.001j, y=0.002 - 0.003j)}
    response = circuit.small_signal(y_dc=op, params=params)(freqs=freq, sources=drives)
    alpha = 3 * np.log(10) / 10
    k = 4000 * (1 - np.exp(-alpha * LENGTH / 10000)) / alpha
    attenuation = np.exp(-alpha * LENGTH / 10000)
    for terminal, u, name in (("D,p2", A, "S"), ("D,p1", B, "R")):
        dx, dy = drives[name].x, drives[name].y
        dp = 2 * (u.real * dx + u.imag * dy)
        expected = attenuation / (1 + k * abs(u) ** 2) ** 2 * dp * np.exp(-2j * np.pi * freq * tau)
        result = jax.jit(lambda r, terminal=terminal: circuit.port_power_response(r, terminal, params=params).outgoing)(response)
        np.testing.assert_allclose(result, expected, atol=1e-12)
        # Both envelope quadratures: G'(u)du = T/sqrt(d)*(du-k*u*Re(conj(u)*du)/d).
        d = 1 + k * abs(u) ** 2
        t = np.sqrt(attenuation) * np.exp(-0.7j)
        jac = np.array([[t / np.sqrt(d) * (1 - k * u * u.real / d), t / np.sqrt(d) * (1j - k * u * u.imag / d)]])
        qx, qy = response.quadratures(terminal)
        expected_x = (jac.real @ np.array([dx, dy]))[0] * np.exp(-2j * np.pi * freq * tau)
        expected_y = (jac.imag @ np.array([dx, dy]))[0] * np.exp(-2j * np.pi * freq * tau)
        other = drives["R" if name == "S" else "S"]
        np.testing.assert_allclose(qx, np.sqrt(r0) * (expected_x + other.x), atol=1e-11)
        np.testing.assert_allclose(qy, np.sqrt(r0) * (expected_y + other.y), atol=1e-11)
    for flag in ("auto", True, False):
        with pytest.raises(ValueError, match="Non-holomorphic"):
            circuit.ac_response(freqs=freq, sources={"S": 1.0}, holomorphic=flag)
    with pytest.raises(ValueError, match="not AC"):
        circuit.port_waves(op, "D,p1", analysis="ac")


def test_state_history_and_parameter_derivatives() -> None:
    cls = nonlinear_optical_waveguide(r0=50)
    current = jnp.linspace(-0.04, 0.09, 8)
    history = jnp.linspace(0.07, -0.02, 8)

    def residual(x: jax.Array, h: jax.Array, beta: float) -> jax.Array:
        f, q = cls.solver_call(0.0, x[:4] + 1j * x[4:], cls(**{**SETTINGS, "tpa_coeff": beta}), hist=h[:4] + 1j * h[4:])
        np.testing.assert_allclose(q, 0)
        return jnp.concatenate((f.real, f.imag))

    for arg, point in ((0, current), (1, history)):
        derivative = jax.jacfwd(residual, argnums=arg)(current, history, 4000.0)

        def evaluate(value: jax.Array, arg: int = arg) -> jax.Array:
            return residual(value, history, 4000.0) if arg == 0 else residual(current, value, 4000.0)

        eps = 1e-6
        finite = np.stack([(evaluate(point + eps * e) - evaluate(point - eps * e)) / (2 * eps) for e in jnp.eye(8)], axis=1)
        np.testing.assert_allclose(derivative, finite, atol=1e-8)
    np.testing.assert_allclose(jax.jacfwd(residual, argnums=1)(current, history, 4000.0)[jnp.array([0, 1, 4, 5])], 0)
    circuit = driven()

    def objective(beta: jax.Array) -> jax.Array:
        params = {"D.tpa_coeff": beta}
        return circuit.port_waves(circuit.dc(params=params), "D,p2", params=params).outgoing_power

    betas = jnp.array([0.0, 1000.0, 4000.0])
    actual = jax.jit(jax.vmap(objective))(betas)
    expected = np.array([abs(propagate(A, tpa=b)) ** 2 for b in betas])
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    grad = jax.jit(jax.grad(objective))(4000.0)
    eps = 0.1
    np.testing.assert_allclose(grad, (objective(4000.0 + eps) - objective(4000.0 - eps)) / (2 * eps), rtol=1e-8)
    # Zero loss/zero length must retain finite derivatives (no inactive 0/0).
    for loss, length in ((0.0, LENGTH), (3.0, 0.0)):

        def power(value: jax.Array, loss: float = loss, length: float = length) -> jax.Array:
            params = {"D.loss_dB_cm": loss, "D.length_um": length, "D.tpa_coeff": value}
            return circuit.port_waves(circuit.dc(params=params), "D,p2", params=params).outgoing_power

        assert np.isfinite(jax.grad(power)(4000.0))


def pulse(t: jax.Array) -> jax.Array:
    return 0.2 + 0.8 * jax.nn.sigmoid((t - 0.3e-9) / 0.05e-9)


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
@pytest.mark.parametrize("gamma", [None, -0.6j])
def test_transient_prehistory_reverse_reflection_and_refinement(r0: float, gamma: complex | None) -> None:
    circuit = driven(r0=r0, gamma=gamma, waveform=pulse)
    times = jnp.linspace(0, 0.8e-9, 81)
    forward = propagate(A * np.asarray(pulse(jnp.maximum(times - TAU, 0))))
    if gamma is None:
        returned = propagate(B * np.asarray(pulse(jnp.maximum(times - TAU, 0))))
    else:
        returned = propagate(gamma * propagate(A * np.asarray(pulse(jnp.maximum(times - 2 * TAU, 0)))))
    errors = []
    for dt in (0.04e-9, 0.01e-9, 0.0025e-9):
        sol = circuit.transient(t0=0.0, t1=float(times[-1]), dt0=dt, saveat=times, max_steps=400)
        left, right = [circuit.port_waves(sol.ys, f"D,p{i}", analysis="tran", t=times) for i in (1, 2)]
        errors.append(max(float(jnp.max(abs(right.outgoing - forward))), float(jnp.max(abs(left.outgoing - returned)))))
        np.testing.assert_allclose(right.outgoing[0], forward[0], atol=1e-11)
        np.testing.assert_allclose(left.outgoing[0], returned[0], atol=1e-11)
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 2e-5


def test_hb_nonlinear_total_samples_and_harmonic_refinement() -> None:
    freq = 1e9

    def waveform(t: jax.Array) -> jax.Array:
        return 0.7 + 0.15 * jnp.exp(2j * jnp.pi * freq * t) + 0.1j * jnp.exp(-2j * jnp.pi * freq * t)

    gamma = -0.6j
    circuit = driven(waveform=waveform, gamma=gamma)
    errors = []
    for harmonics in (2, 4, 8):
        y, _ = circuit.hb(freq=freq, harmonics=harmonics)
        times = jnp.arange(2 * harmonics + 1) / ((2 * harmonics + 1) * freq)
        expected = propagate(A * np.asarray(waveform(times - TAU)))
        right = circuit.port_waves(y, "D,p2", analysis="hb_time", t=times)
        left = circuit.port_waves(y, "D,p1", analysis="hb_time", t=times)
        expected_return = propagate(gamma * propagate(A * np.asarray(waveform(times - 2 * TAU))))
        errors.append(float(jnp.max(abs(left.outgoing - expected_return))))
        np.testing.assert_allclose(right.outgoing, expected, atol=1e-10)
        np.testing.assert_allclose(right.outgoing_power, abs(expected) ** 2, atol=1e-12)
    # The reflected nonlinear field acquires harmonics before its second delay.
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 1e-9


@pytest.mark.parametrize("gamma", [0.4j, -1.0])
def test_reflected_small_signal_round_trip_and_dc_finite_difference(gamma: complex) -> None:
    circuit = driven(gamma=gamma)
    op = circuit.dc()
    freq = jnp.array([0.0, 3e9, 9e9])
    drive = QuadratureDrive(x=0.003 + 0.002j, y=-0.001 + 0.004j)
    response = circuit.small_signal(y_dc=op)(freqs=freq, sources={"S": drive})
    measured = circuit.port_power_response(response, "D,p1").outgoing
    alpha = 3 * np.log(10) / 10
    attenuation = np.exp(-alpha * LENGTH / 10000)
    k = 4000 * (1 - attenuation) / alpha
    first_power = abs(propagate(A)) ** 2
    first_slope = attenuation / (1 + k * abs(A) ** 2) ** 2
    second_slope = attenuation / (1 + k * abs(gamma) ** 2 * first_power) ** 2
    dp = 2 * (A.real * drive.x + A.imag * drive.y)
    expected = second_slope * abs(gamma) ** 2 * first_slope * dp * np.exp(-4j * np.pi * freq * TAU)
    np.testing.assert_allclose(measured, expected, atol=1e-12)
    for dx, dy, part in ((drive.x.real, drive.y.real, np.real), (drive.x.imag, drive.y.imag, np.imag)):
        eps = 1e-4
        values = []
        for sign in (-1, 1):
            params = {"S.amplitude": A + sign * eps * (dx + 1j * dy)}
            values.append(circuit.port_waves(circuit.dc(params=params), "D,p1", params=params).outgoing_power)
        np.testing.assert_allclose((values[1] - values[0]) / (2 * eps), part(measured[0]), atol=1e-12)


def test_public_complex_parameter_jacobian_and_phase_invariant_intensity() -> None:
    circuit = driven()

    # Parameter derivatives cover physical length/effective length, attenuation,
    # carrier phase, and both incident intensity coordinates in one public solve.
    def output(x: jax.Array) -> jax.Array:
        params = {"D.length_um": x[0], "D.loss_dB_cm": x[1], "D.tpa_coeff": x[2], "S.amplitude": x[3] + 1j * x[4]}
        return circuit.port_waves(circuit.dc(params=params), "D,p2", params=params).outgoing

    point = jnp.array([LENGTH, 3.0, 4000.0, A.real, A.imag])
    derivative = jax.jit(jax.jacfwd(output))(point)
    eps = np.array([0.01, 1e-4, 0.1, 1e-6, 1e-6])
    reference = []
    for index in range(5):
        xp, xm = np.array(point), np.array(point)
        xp[index] += eps[index]
        xm[index] -= eps[index]

        def independent(x: np.ndarray) -> np.ndarray:
            return propagate(x[3] + 1j * x[4], length=x[0], loss=x[1], tpa=x[2])

        reference.append((independent(xp) - independent(xm)) / (2 * eps[index]))
    np.testing.assert_allclose(derivative, reference, rtol=2e-7, atol=1e-11)

    def power(phase: jax.Array) -> jax.Array:
        params = {"S.amplitude": A * jnp.exp(1j * phase)}
        return circuit.port_waves(circuit.dc(params=params), "D,p2", params=params).outgoing_power

    phases = jnp.array([0.0, np.pi / 2, np.pi])
    np.testing.assert_allclose(jax.jit(jax.vmap(power))(phases), abs(propagate(A)) ** 2, atol=1e-12)
    np.testing.assert_allclose(jax.jit(jax.vmap(jax.grad(power)))(phases), 0, atol=1e-12)


@pytest.mark.parametrize("backend", ["sparse", "klu"])
def test_sparse_nonlinear_dc_and_transient(backend: str) -> None:
    circuit = driven(backend=backend, gamma=0.4j)
    y = circuit.dc()
    np.testing.assert_allclose(circuit.port_waves(y, "D,p1").outgoing, propagate(0.4j * propagate(A)), atol=1e-11)
    sol = circuit.transient(t0=0.0, t1=3 * TAU, dt0=TAU / 4, saveat=jnp.array([0.0, 3 * TAU]), max_steps=20)
    np.testing.assert_allclose(circuit.port_waves(sol.ys, "D,p2", analysis="tran").outgoing, propagate(A), atol=1e-11)


@pytest.mark.parametrize("analysis", ["tran", "hb"])
def test_nonlinear_objective_gradients_through_public_dynamic_analysis(analysis: str) -> None:
    freq = 1e9

    def periodic(t: jax.Array) -> jax.Array:
        return 0.7 + 0.15 * jnp.exp(2j * jnp.pi * freq * t) + 0.1j * jnp.exp(-2j * jnp.pi * freq * t)

    waveform = pulse if analysis == "tran" else periodic
    # A second matched source with zero emission is an absorbing boundary.
    circuit = driven(waveform=waveform)
    times = jnp.array([0.5e-9]) if analysis == "tran" else jnp.arange(9) / (9 * freq)

    def objective(beta: jax.Array) -> jax.Array:
        params = {"D.tpa_coeff": beta, "R.amplitude": 0j}
        if analysis == "tran":
            y = circuit.transient(t0=0.0, t1=0.5e-9, dt0=0.01e-9, saveat=times, params=params, max_steps=60).ys
        else:
            y, _ = circuit.hb(freq=freq, harmonics=4, params=params)
        return jnp.mean(
            circuit.port_waves(
                y, "D,p2", analysis="tran" if analysis == "tran" else "hb_time", t=times, params=params
            ).outgoing_power
        )

    actual = jax.grad(objective)(4000.0)
    eps = 0.1
    finite = (objective(4000.0 + eps) - objective(4000.0 - eps)) / (2 * eps)
    np.testing.assert_allclose(actual, finite, rtol=1e-6, atol=1e-12)
    alpha = 3 * np.log(10) / 10
    attenuation = np.exp(-alpha * LENGTH / 10000)
    leff = (1 - attenuation) / alpha
    incident_power = abs(A * np.asarray(waveform(times - TAU))) ** 2
    analytic = np.mean(-attenuation * leff * incident_power**2 / (1 + 4000 * leff * incident_power) ** 2)
    np.testing.assert_allclose(actual, analytic, rtol=1e-3, atol=1e-12)
