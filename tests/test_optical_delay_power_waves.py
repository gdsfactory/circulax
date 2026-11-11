# Copyright 2026 Circulax contributors
"""Bidirectional delays through public analyses and independent network laws."""

import importlib
from collections.abc import Callable
from typing import NoReturn

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax import Circuit, compile_circuit, sax_component
from circulax.components import (
    directional_coupler,
    optical_delay_line,
    optical_delay_line_fdomain,
    optical_wave_source,
    optical_wave_termination,
)

C = 2.99792458e14
TAU = 0.13e-9
LENGTH = C * TAU / 4


@pytest.fixture(autouse=True)
def forbid_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "Delay realization called s_to_y"
        raise AssertionError(msg)

    for module in ("circulax.s_transforms", "circulax.components.photonic"):
        monkeypatch.setattr(importlib.import_module(module), "s_to_y", forbidden)


def settings(*, frequency: bool = False, phase: float = 0.7, length: float = LENGTH, loss: float = 0.0) -> dict:
    result = {"length_um": length, "loss_dB_cm": loss, "n_group": 4.0}
    if not frequency:
        result.update(neff=phase * 1310 / (2 * np.pi * LENGTH * 1000), wavelength_nm=1310.0)
    return result


def driven(
    *,
    frequency: bool = False,
    r0: float = 1.0,
    waveform: Callable | None = None,
    gamma: complex | None = None,
    backend: str = "dense",
) -> Circuit:
    def reflector() -> dict:
        return {("p", "p"): gamma}

    return compile_circuit(
        {
            "instances": {
                "D": {"component": "d", "settings": settings(frequency=frequency)},
                "S": {"component": "s", "settings": {"amplitude": 0.6 + 0.8j}},
                "R": {"component": "r", "settings": {} if gamma is not None else {"amplitude": -0.3j}},
            },
            "connections": {"S,p": "D,p1", "D,p2": "R,p"},
        },
        {
            "d": (optical_delay_line_fdomain if frequency else optical_delay_line)(r0=r0),
            "s": optical_wave_source(r0=r0, waveform=waveform),
            "r": sax_component(reflector, r0=r0) if gamma is not None else optical_wave_source(r0=r0, waveform=waveform),
        },
        backend=backend,
        g_leak=0,
    )


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
@pytest.mark.parametrize("phase", [0.0, np.pi / 2, np.pi])
@pytest.mark.parametrize("frequency", [False, True])
def test_dc_bidirectional_coherent_loss_zero_delay(r0: float, phase: float, *, frequency: bool) -> None:
    circuit = driven(frequency=frequency, r0=r0)
    incident = np.array([0.6 + 0.8j, -0.3j])
    for length, loss in ((0.0, 0.0), (LENGTH, 0.0), (LENGTH, 1e-12), (LENGTH, 2.0)):
        updates = {f"D.{key}": value for key, value in settings(frequency=frequency, phase=phase, length=length, loss=loss).items()}
        y, ok = circuit.dc_checked(params=updates, atol=1e-12, rtol=1e-12)
        assert ok
        transmission = 10 ** (-loss * length / 10000 / 20) * np.exp(-1j * (0 if frequency else phase * length / LENGTH))
        waves = [circuit.port_waves(y, f"D,p{i}", params=updates) for i in (1, 2)]
        np.testing.assert_allclose([w.incident for w in waves], incident, atol=2e-11)
        np.testing.assert_allclose([w.outgoing for w in waves], transmission * incident[::-1], atol=2e-11)
        np.testing.assert_allclose(
            sum(w.net_power for w in waves), (1 - abs(transmission) ** 2) * np.sum(abs(incident) ** 2), atol=2e-11
        )


@pytest.mark.parametrize("frequency", [False, True])
@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
def test_ac_scattering_modulation_and_carrier_separate(r0: float, *, frequency: bool) -> None:
    factory = optical_delay_line_fdomain if frequency else optical_delay_line
    circuit = compile_circuit(
        {"instances": {"D": {"component": "d", "settings": settings(frequency=frequency)}}, "ports": {"a": "D,p1", "b": "D,p2"}},
        {"d": factory(r0=r0)},
        backend="dense",
        g_leak=0,
    )
    freqs = jnp.array([-3e9, 0, 1e9, 3e9])
    expected = np.exp(-2j * np.pi * freqs * TAU - 1j * (0 if frequency else 0.7))
    for holomorphic in (False, True):
        actual = circuit.sp(ports=circuit.ports, freqs=freqs, z0=r0, y_dc=jnp.zeros(2 * circuit.sys_size), holomorphic=holomorphic)
        np.testing.assert_allclose(actual[:, 1, 0], expected, atol=1e-11)
        np.testing.assert_allclose(actual[:, 0, 1], expected, atol=1e-11)
        np.testing.assert_allclose(actual[:, [0, 1], [0, 1]], 0, atol=1e-11)

    def transmitted(length: float) -> jax.Array:
        return circuit.sp(
            ports=circuit.ports, freqs=jnp.array([1e9]), z0=r0, y_dc=jnp.zeros(2 * circuit.sys_size), length_um=length
        )[0, 1, 0].real

    phase_slope = (0 if frequency else 0.7 / LENGTH) + 2 * np.pi * 1e9 * 4 / C
    expected_slope = (-1j * phase_slope * expected[2]).real
    np.testing.assert_allclose(jax.jit(jax.grad(transmitted))(LENGTH), expected_slope, atol=1e-12)


@pytest.mark.parametrize("frequency", [False, True])
def test_hb_periodic_delay_and_observable_limits(*, frequency: bool) -> None:
    freq = 1e9

    def waveform(t: jax.Array) -> jax.Array:
        return 0.4 + jnp.exp(2j * jnp.pi * freq * t) + 0.2j * jnp.exp(-2j * jnp.pi * freq * t)

    circuit = driven(frequency=frequency, waveform=waveform)
    y, _ = circuit.hb(freq=freq, harmonics=3)
    times = jnp.arange(7) / (7 * freq)
    transmission = np.exp(-1j * (0 if frequency else 0.7))
    # Boundary current includes both waves and is fully reconstructible in HB.
    np.testing.assert_allclose(
        circuit.port_waves(y, "R,p", analysis="hb_time", t=times).incident,
        transmission * (0.6 + 0.8j) * waveform(times - TAU),
        atol=1e-8,
    )
    if frequency:
        with pytest.raises(ValueError, match="DC only"):
            circuit.port_waves(y, "D,p1", analysis="hb_time", t=times)
        with pytest.raises(RuntimeError, match=r"[Ff]requency|fdomain|F-domain"):
            circuit.transient(t0=0, t1=1e-9, dt0=1e-11)
    else:
        np.testing.assert_allclose(
            circuit.port_waves(y, "D,p1", analysis="hb_time", t=times).outgoing,
            transmission * -0.3j * waveform(times - TAU),
            atol=1e-8,
        )
    with pytest.raises(ValueError, match="not AC"):
        circuit.port_waves(y[0], "D,p1", analysis="ac")


def pulse(t: jax.Array) -> jax.Array:
    return 0.2 + 0.8 * jax.nn.sigmoid((t - 0.35e-9) / 0.06e-9)


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
def test_pulses_prehistory_and_timestep_refinement(r0: float) -> None:
    circuit = driven(r0=r0, waveform=pulse)
    times = jnp.linspace(0, 1e-9, 41)
    delayed = pulse(jnp.maximum(times - TAU, 0))
    errors = []
    # First step size exceeds tau; latter two resolve it. Prehistory is nonzero.
    for dt in (0.2e-9, 0.02e-9, 0.005e-9):
        sol = circuit.transient(t0=0, t1=1e-9, dt0=dt, saveat=times, max_steps=250)
        left = circuit.port_waves(sol.ys, "D,p1", analysis="tran", t=times)
        right = circuit.port_waves(sol.ys, "D,p2", analysis="tran", t=times)
        mesh = jnp.arange(round(1e-9 / dt) + 1) * dt
        saved_drive = np.interp(times, mesh, pulse(mesh))
        np.testing.assert_allclose(left.incident, (0.6 + 0.8j) * saved_drive, atol=1e-9)
        expected = np.exp(-0.7j) * (0.6 + 0.8j) * delayed
        errors.append(float(jnp.max(abs(right.outgoing - expected))))
        np.testing.assert_allclose(left.outgoing, np.exp(-0.7j) * -0.3j * delayed, atol=errors[-1] * 0.31 + 1e-8)
        np.testing.assert_allclose(right.incident, -0.3j * saved_drive, atol=1e-9)
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 0.001


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
def test_reflective_delay_round_trip(r0: float) -> None:
    gamma = 0.4j
    circuit = driven(r0=r0, waveform=pulse, gamma=gamma)
    y = circuit.dc()
    returned = gamma * np.exp(-1.4j) * (0.6 + 0.8j) * pulse(0)
    np.testing.assert_allclose(circuit.port_waves(y, "S,p").incident, returned, atol=1e-10)
    times = jnp.linspace(0, 1e-9, 41)
    sol = circuit.transient(t0=0, t1=1e-9, dt0=0.005e-9, saveat=times, max_steps=250)
    expected = gamma * np.exp(-1.4j) * (0.6 + 0.8j) * pulse(jnp.maximum(times - 2 * TAU, 0))
    np.testing.assert_allclose(circuit.port_waves(sol.ys, "S,p", analysis="tran", t=times).incident, expected, atol=0.001)
    # The public device accessor needs no history even under reflection.
    np.testing.assert_allclose(circuit.port_waves(sol.ys, "D,p1", analysis="tran", t=times).outgoing, expected, atol=0.001)


@pytest.mark.parametrize("frequency", [False, True])
def test_jit_vmap_parameter_updates_and_derivatives(*, frequency: bool) -> None:
    circuit = driven(frequency=frequency)

    def transmitted(length: float) -> jax.Array:
        params = {"D.length_um": length, "D.loss_dB_cm": 0.7}
        return circuit.port_waves(circuit.dc(params=params), "D,p2", params=params).outgoing

    lengths = jnp.array([0.0, LENGTH / 2, LENGTH])
    phase_slope = 0 if frequency else 0.7 / LENGTH
    slope = -0.7 * np.log(10) / 200000 - 1j * phase_slope
    expected = (0.6 + 0.8j) * np.exp(slope * lengths)
    np.testing.assert_allclose(jax.jit(jax.vmap(transmitted))(lengths), expected, atol=1e-10)
    np.testing.assert_allclose(jax.jit(jax.jacfwd(transmitted))(LENGTH), slope * expected[-1], atol=1e-12)


def test_current_and_history_jacobians_and_parameter_gradient() -> None:
    cls = optical_delay_line(r0=50)
    current = jnp.linspace(-0.4, 0.9, 8)
    history = jnp.linspace(0.7, -0.2, 8)

    def residual(x: jax.Array, h: jax.Array, length: float) -> jax.Array:
        z, delayed = x[:4] + 1j * x[4:], h[:4] + 1j * h[4:]
        f, q = cls.solver_call(0.0, z, cls(**settings(length=length)), hist=delayed)
        np.testing.assert_equal(q.shape, (4,))
        return jnp.concatenate((f.real, f.imag))

    for arg, point in ((0, current), (1, history)):
        derivative = jax.jit(jax.jacfwd(residual, argnums=arg))(current, history, LENGTH)

        def evaluate(value: jax.Array, arg: int = arg) -> jax.Array:
            return residual(value, history, LENGTH) if arg == 0 else residual(current, value, LENGTH)

        eps = 1e-6
        finite = np.stack([(evaluate(point + eps * e) - evaluate(point - eps * e)) / (2 * eps) for e in jnp.eye(8)], axis=1)
        np.testing.assert_allclose(derivative, finite, atol=1e-9)
    derivative = jax.jacfwd(residual, argnums=2)(current, history, LENGTH)
    eps = 1e-2
    np.testing.assert_allclose(
        derivative, (residual(current, history, LENGTH + eps) - residual(current, history, LENGTH - eps)) / (2 * eps), atol=1e-10
    )
    # Only constraints depend on history; terminal flows are complete/algebraic.
    np.testing.assert_allclose(jax.jacfwd(residual, argnums=1)(current, history, LENGTH)[[0, 1, 4, 5], :], 0)


def test_delayed_all_pass_ring_and_physical_singularity() -> None:
    circuit = compile_circuit(
        {
            "instances": {
                "C": {"component": "c"},
                "D": {"component": "d", "settings": settings()},
                "S": {"component": "s"},
                "L": {"component": "l"},
            },
            "connections": {"S,p": "C,p1", "C,p3": "L,p", "C,p4": "D,p1", "D,p2": "C,p2"},
        },
        {
            "c": directional_coupler(),
            "d": optical_delay_line(),
            "s": optical_wave_source(waveform=lambda t: jnp.exp(2j * jnp.pi * 1e9 * t)),
            "l": optical_wave_termination(),
        },
        backend="dense",
        g_leak=0,
    )
    y = circuit.dc()
    z, t = np.exp(-0.7j), np.sqrt(0.5)
    np.testing.assert_allclose(circuit.port_waves(y, "L,p").incident, (t - z) / (1 - t * z), atol=1e-9)
    y, _ = circuit.hb(freq=1e9, harmonics=2)
    times = jnp.arange(5) / 5e9
    z = np.exp(-0.7j - 2j * np.pi * 1e9 * TAU)
    np.testing.assert_allclose(
        circuit.port_waves(y, "L,p", analysis="hb_time", t=times).incident,
        (t - z) / (1 - t * z) * np.exp(2j * np.pi * 1e9 * times),
        atol=1e-8,
    )
    # Decoupled lossless resonant loop: whole-system Jacobian has a nullspace.
    from circulax.solvers.assembly import assemble_gc_complex
    from circulax.solvers.linear import _build_index_arrays

    groups = circuit._with_param_values({"C.coupling": 0.0, "D.neff": 0.0})  # noqa: SLF001 -- nullspace audit
    values, _ = assemble_gc_complex(jnp.zeros(2 * circuit.sys_size), groups)
    rows, cols, _, _ = _build_index_arrays(groups, circuit.sys_size, is_complex=False)
    matrix = jnp.zeros((circuit.sys_size, circuit.sys_size), dtype=complex).at[rows, cols].add(values)
    # Ground is not a physical optical port; remove its structurally empty row.
    assert np.linalg.matrix_rank(np.asarray(matrix)[1:, 1:]) < circuit.sys_size - 1


@pytest.mark.parametrize("factory", [optical_delay_line, optical_delay_line_fdomain])
@pytest.mark.parametrize("reference", [0, -1, np.inf, np.nan, 1j])
def test_invalid_reference(factory: Callable, reference: float) -> None:
    with pytest.raises(ValueError, match="finite, positive, and real"):
        factory(r0=reference)


@pytest.mark.parametrize("phase", [0.0, np.pi / 2, np.pi])
def test_zero_group_delay_transient_reduces_to_algebraic_waves(phase: float) -> None:
    circuit = driven(waveform=pulse)
    updates = {"D.n_group": 0.0, "D.neff": phase * 1310 / (2 * np.pi * LENGTH * 1000)}
    times = jnp.arange(21) * 0.05e-9
    sol = circuit.transient(t0=0, t1=1e-9, dt0=0.05e-9, saveat=times, params=updates, max_steps=30)
    for port, incident, other in (("p1", 0.6 + 0.8j, -0.3j), ("p2", -0.3j, 0.6 + 0.8j)):
        waves = circuit.port_waves(sol.ys, f"D,{port}", analysis="tran", t=times, params=updates)
        np.testing.assert_allclose(waves.incident, incident * pulse(times), atol=1e-9)
        np.testing.assert_allclose(waves.outgoing, np.exp(-1j * phase) * other * pulse(times), atol=1e-9)


@pytest.mark.parametrize("tau", [0.0, 0.4, 1.4, 3.0])
def test_delay_stamp_interpolation_trial_history_and_tau_derivatives(tau: float) -> None:
    from circulax.solvers.assembly import _interp_delayed

    cls = optical_delay_line(r0=0.01)
    trial = jnp.linspace(-0.3, 0.8, 8)
    data = jnp.stack([trial * 0.5, trial * 0.8])

    def residual(x: jax.Array, hist: jax.Array, delay: float) -> jax.Array:
        delayed = _interp_delayed(jnp.array([0.0, 1.0]), hist, jnp.array([delay]), jnp.arange(8)[None, :], 2.0, x[None, :])[0]
        # Delay is in normalised seconds here; the carrier phase is disabled.
        f, _ = cls.solver_call(
            2.0,
            x[:4] + 1j * x[4:],
            cls(length_um=C / 4, n_group=4 * delay, neff=0, loss_dB_cm=0),
            hist=delayed[:4] + 1j * delayed[4:],
        )
        return jnp.concatenate((f.real, f.imag))

    # Away from knots, compare all current/history/delay derivatives to FD.
    eps = 1e-6
    for arg, value in ((0, trial), (1, data)):
        jac = jax.jit(jax.jacfwd(residual, argnums=arg))(trial, data, tau)
        directions = jnp.eye(value.size).reshape((value.size, *value.shape))

        def evaluate(v: jax.Array, arg: int = arg) -> jax.Array:
            return residual(v, data, tau) if arg == 0 else residual(trial, v, tau)

        fd = jnp.stack([(evaluate(value + eps * e) - evaluate(value - eps * e)) / (2 * eps) for e in directions], axis=-1)
        np.testing.assert_allclose(jac.reshape(8, -1), fd, atol=1e-8)
    if tau > 0:
        np.testing.assert_allclose(
            jax.jacfwd(residual, argnums=2)(trial, data, tau),
            (residual(trial, data, tau + eps) - residual(trial, data, tau - eps)) / (2 * eps),
            atol=1e-8,
        )


@pytest.mark.parametrize("backend", ["sparse", "klu"])
def test_delay_solver_backends_with_substep_delay(backend: str) -> None:
    def ramp(t: jax.Array) -> jax.Array:
        return 0.2 + 1e9 * t

    circuit = driven(backend=backend, waveform=ramp)
    times = jnp.arange(6) * 0.2e-9
    sol = circuit.transient(t0=0, t1=1e-9, dt0=0.2e-9, saveat=times, max_steps=10)
    for port, opposite in (("p1", -0.3j), ("p2", 0.6 + 0.8j)):
        waves = circuit.port_waves(sol.ys, f"D,{port}", analysis="tran", t=times)
        expected = opposite * np.exp(-0.7j) * ramp(jnp.maximum(times - TAU, 0))
        np.testing.assert_allclose(waves.outgoing, expected, atol=1e-7)
