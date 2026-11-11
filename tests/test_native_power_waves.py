# Copyright 2026 Circulax contributors
"""Native passive waves checked against independent scattering/network laws."""

import importlib
from collections.abc import Callable
from typing import NoReturn

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax import Circuit, compile_circuit, sax_component
from circulax.components import optical_wave_source, optical_wave_termination
from circulax.components.photonic import (
    directional_coupler,
    grating,
    optical_waveguide,
    splitter,
    tunable_beam_splitter,
)


@pytest.fixture(autouse=True)
def forbid_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "Native passive realization called s_to_y"
        raise AssertionError(msg)

    for module in ("circulax.s_transforms", "circulax.components.photonic"):
        monkeypatch.setattr(importlib.import_module(module), "s_to_y", forbidden)


def bare(factory: Callable, ports: tuple[str, ...], r0: float) -> Circuit:
    return compile_circuit(
        {"instances": {"D": {"component": "d"}}, "ports": {p: f"D,{p}" for p in ports}},
        {"d": factory(r0=r0)},
        backend="dense",
        g_leak=0,
    )


def scattering(circuit: Circuit, r0: float = 1, **params: float) -> jax.Array:
    return circuit.sp(ports=circuit.ports, freqs=jnp.array([0.0, 1e9]), z0=r0, y_dc=jnp.zeros(2 * circuit.sys_size), **params)[0]


def coupler_reference(coupling: float) -> np.ndarray:
    t, k = np.sqrt(1 - coupling), 1j * np.sqrt(coupling)
    return np.array([[0, 0, t, k], [0, 0, k, t], [t, k, 0, 0], [k, t, 0, 0]])


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
@pytest.mark.parametrize("phase", [0.0, np.pi / 2, np.pi])
@pytest.mark.parametrize("loss", [0.0, 1e-9, 2.0])
def test_waveguide_loss_phase(r0: float, phase: float, loss: float) -> None:
    c = bare(optical_waveguide, ("p1", "p2"), r0)
    # neff=1 and wavelength=1000 nm give phi=2*pi*length_um.
    length = 1 + phase / (2 * np.pi)
    actual = scattering(c, r0, length_um=length, neff=1.0, n_group=1.0, wavelength_nm=1000.0, loss_dB_cm=loss)
    t = 10 ** (-loss * length / 10000 / 20) * np.exp(-1j * phase)
    np.testing.assert_allclose(actual, [[0, t], [t, 0]], atol=1e-11)


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
def test_grating_spectrum_and_peak(r0: float) -> None:
    c = bare(grating, ("grating", "waveguide"), r0)
    for loss in (0.0, 1e-10, 3.0):
        for wavelength in (1300.0, 1310.0, 1320.0):
            actual = scattering(c, r0, peak_loss_dB=loss, wavelength_nm=wavelength)
            t = 10 ** (-(loss + ((wavelength - 1310) / 10) ** 2) / 20)
            np.testing.assert_allclose(actual, [[0, t], [t, 0]], atol=1e-11)


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
@pytest.mark.parametrize("factory", [directional_coupler, tunable_beam_splitter, splitter])
def test_full_multiport_scattering(factory: Callable, r0: float) -> None:
    ports = ("p1", "p2", "p3") if factory is splitter else ("p1", "p2", "p3", "p4")
    c = bare(factory, ports, r0)
    for ratio in (0.0, 0.1, 0.5, 0.9, 1.0):
        if factory is splitter:
            r, k = np.sqrt(ratio), 1j * np.sqrt(1 - ratio)
            expected = np.array([[0, r, k], [r, 0, 0], [k, 0, 0]])
            params = {"split_ratio": ratio}
        else:
            expected = coupler_reference(ratio)
            params = {"coupling": ratio} if factory is directional_coupler else {"theta": np.arcsin(np.sqrt(ratio))}
        actual = scattering(c, r0, **params)
        np.testing.assert_allclose(actual, expected, atol=1e-11)
        np.testing.assert_allclose(actual.T, actual, atol=1e-11)
        assert np.linalg.svd(actual, compute_uv=False).max() <= 1 + 1e-11
        if factory is not splitter:
            np.testing.assert_allclose(actual.conj().T @ actual, np.eye(4), atol=1e-11)
        else:
            np.testing.assert_allclose(np.linalg.svd(actual, compute_uv=False), [1, 1, 0], atol=1e-11)
            loss_row = np.array([0, 1j * np.sqrt(1 - ratio), np.sqrt(ratio)])
            np.testing.assert_allclose(actual.conj().T @ actual + np.outer(loss_row.conj(), loss_row), np.eye(3), atol=1e-11)


def illuminated(factory: Callable, ports: tuple[str, ...], r0: float, waveform: Callable | None = None) -> Circuit:
    instances = {"D": {"component": "d"}}
    connections = {}
    for i, p in enumerate(ports):
        instances[f"S{i}"] = {"component": "s", "settings": {"amplitude": 0j}}
        connections[f"S{i},p"] = f"D,{p}"
    return compile_circuit(
        {"instances": instances, "connections": connections},
        {"d": factory(r0=r0), "s": optical_wave_source(r0=r0, waveform=waveform)},
        backend="dense",
        g_leak=0,
    )


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
@pytest.mark.parametrize("factory", [optical_waveguide, grating, splitter, directional_coupler, tunable_beam_splitter])
def test_coherent_public_waves(factory: Callable, r0: float) -> None:
    ports = (
        ("grating", "waveguide")
        if factory is grating
        else tuple(
            f"p{i + 1}"
            for i in range(3 if factory is splitter else 4 if factory in (directional_coupler, tunable_beam_splitter) else 2)
        )
    )
    c = illuminated(factory, ports, r0)
    a = np.array([0.2 + 0.3j, -0.4j, 0.7, -0.2 + 0.1j])[: len(ports)]
    params = {f"S{i}.amplitude": v for i, v in enumerate(a)}
    y, ok = c.dc_checked(params=params, atol=1e-12, rtol=1e-12)
    assert ok
    waves = [c.port_waves(y, f"D,{p}", params=params) for p in ports]
    if factory is optical_waveguide:
        t = 10 ** (-0.01 / 20) * np.exp(-2j * np.pi * 2.4 * 100 * 1000 / 1310)
        expected = np.array([[0, t], [t, 0]])
    elif factory is grating:
        expected = np.array([[0, 1], [1, 0]])
    elif factory is splitter:
        expected = np.array([[0, 1 / np.sqrt(2), 1j / np.sqrt(2)], [1 / np.sqrt(2), 0, 0], [1j / np.sqrt(2), 0, 0]])
    else:
        expected = coupler_reference(0.5)
    np.testing.assert_allclose([w.incident for w in waves], a, atol=1e-11)
    np.testing.assert_allclose([w.outgoing for w in waves], expected @ a, atol=1e-11)
    dissipation = np.linalg.norm(a) ** 2 - np.linalg.norm(expected @ a) ** 2
    np.testing.assert_allclose(sum(w.net_power for w in waves), dissipation, atol=1e-11)
    if factory is splitter:
        np.testing.assert_allclose(dissipation, abs(1j * a[1] / np.sqrt(2) + a[2] / np.sqrt(2)) ** 2, atol=1e-11)


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
def test_mzi_and_reflective_ring(r0: float) -> None:
    instances = {
        "C1": {"component": "c"},
        "C2": {"component": "c"},
        "A": {"component": "w", "settings": {"length_um": 0.0}},
        "B": {
            "component": "w",
            "settings": {"length_um": 0.25, "neff": 1.0, "n_group": 1.0, "wavelength_nm": 1000.0, "loss_dB_cm": 0.0},
        },
        "S": {"component": "s"},
        "L0": {"component": "l"},
        "L1": {"component": "l"},
        "L2": {"component": "l"},
    }
    connections = {
        "S,p": "C1,p1",
        "L0,p": "C1,p2",
        "C1,p3": "A,p1",
        "A,p2": "C2,p1",
        "C1,p4": "B,p1",
        "B,p2": "C2,p2",
        "C2,p3": "L1,p",
        "C2,p4": "L2,p",
    }
    models = {
        "c": directional_coupler(r0=r0),
        "w": optical_waveguide(r0=r0),
        "s": optical_wave_source(r0=r0),
        "l": optical_wave_termination(r0=r0),
    }
    c = compile_circuit({"instances": instances, "connections": connections}, models, backend="dense", g_leak=0)
    y = c.dc()
    block = coupler_reference(0.5)[2:, :2]
    expected = block @ np.diag([1, -1j]) @ block @ np.array([1, 0])
    np.testing.assert_allclose([c.port_waves(y, f"L{i},p").incident for i in (1, 2)], expected, atol=1e-11)
    np.testing.assert_allclose(sum(c.port_waves(y, f"L{i},p").incident_power for i in (1, 2)), 1, atol=1e-11)
    # A coupler with p4 -> phase arm -> p2 feedback is an all-pass ring.
    ring = compile_circuit(
        {
            "instances": {"C": {"component": "c"}, "W": instances["B"], "S": {"component": "s"}, "L": {"component": "l"}},
            "connections": {"S,p": "C,p1", "C,p3": "L,p", "C,p4": "W,p1", "W,p2": "C,p2"},
        },
        models,
        backend="dense",
        g_leak=0,
    )
    for coupling, length in ((0.0, 0.25), (0.5, 0.25), (1.0, 0.25), (0.01, 0.0), (0.5, 0.0), (1.0, 0.0)):
        params = {"C.coupling": coupling, "W.length_um": length}
        y, success = ring.dc_checked(params=params, atol=1e-12, rtol=1e-12)
        assert success
        t, k, z = np.sqrt(1 - coupling), np.sqrt(coupling), np.exp(-2j * np.pi * length)
        expected_ring = t - k * k * z / (1 - t * z)
        wave = ring.port_waves(y, "L,p", params=params)
        np.testing.assert_allclose(wave.incident, expected_ring, atol=1e-10)
        np.testing.assert_allclose(wave.incident_power, 1, atol=1e-10)


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
def test_reflective_load(r0: float) -> None:
    def reflector(gamma: complex = -1.0 + 0j) -> dict:
        return {("p", "p"): gamma}

    c = compile_circuit(
        {
            "instances": {
                "S": {"component": "s"},
                "D": {"component": "d", "settings": {"length_um": 0.0}},
                "L": {"component": "l"},
            },
            "connections": {"S,p": "D,p1", "D,p2": "L,p"},
        },
        {"s": optical_wave_source(r0=r0), "d": optical_waveguide(r0=r0), "l": sax_component(reflector, r0=r0)},
        backend="dense",
        g_leak=0,
    )
    for gamma in (-1.0, 1.0, 0.3j):
        params = {"L.gamma": gamma}
        y = c.dc(params=params)
        w = c.port_waves(y, "S,p", params=params)
        np.testing.assert_allclose([w.incident, w.outgoing], [gamma, 1], atol=1e-11)


@pytest.mark.parametrize("factory", [optical_waveguide, grating, splitter, directional_coupler, tunable_beam_splitter])
def test_reference_validation(factory: Callable) -> None:
    for reference in (0, -1, np.nan, np.inf, 1 + 0j):
        with pytest.raises(ValueError, match="finite, positive, and real"):
            factory(r0=reference)


def time_envelope(t: jax.Array) -> jax.Array:
    return 0.8 + 0.2 * jnp.exp(2j * jnp.pi * 1e9 * t)


@pytest.mark.parametrize("factory", [optical_waveguide, grating, splitter, directional_coupler, tunable_beam_splitter])
@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
def test_analysis_paths(factory: Callable, r0: float) -> None:
    ports = factory().ports
    c = illuminated(factory, ports, r0, waveform=time_envelope)
    # Unit excitation at p1 (grating for Grating). Independent column of S.
    params = {"S0.amplitude": 1.0 + 0j}
    if factory is optical_waveguide:
        transfer = np.array([0, 10 ** (-0.01 / 20) * np.exp(-2j * np.pi * 2.4 * 100000 / 1310)])
    elif factory is grating:
        transfer = np.array([0, 1])
    elif factory is splitter:
        transfer = np.array([0, np.sqrt(0.5), 1j * np.sqrt(0.5)])
    else:
        transfer = coupler_reference(0.5)[:, 0]
    times = jnp.linspace(0, 1e-9, 11)
    sol = c.transient(t0=0, t1=1e-9, dt0=1e-11, saveat=times, params=params, max_steps=200)
    hb_time, _ = c.hb(freq=1e9, harmonics=2, params=params)
    hb_times = jnp.arange(5) / 5e9
    for i, port in enumerate(ports):
        tran = c.port_waves(sol.ys, f"D,{port}", analysis="tran", t=sol.ts, params=params)
        hb = c.port_waves(hb_time, f"D,{port}", analysis="hb_time", t=hb_times, params=params)
        np.testing.assert_allclose(tran.outgoing, transfer[i] * time_envelope(times), atol=1e-8)
        np.testing.assert_allclose(hb.outgoing, transfer[i] * time_envelope(hb_times), atol=1e-8)
        np.testing.assert_allclose(hb.outgoing_power, abs(transfer[i] * time_envelope(hb_times)) ** 2, atol=1e-8)
    delta = 0.3 - 0.4j
    ac = c.ac_response(freqs=jnp.array([0.0, 1e9]), sources={"S0": delta}, params=params)
    # Potentials here are perturbations. Waves/powers require total time samples.
    for i, port in enumerate(ports):
        expected = np.sqrt(r0) * (transfer[i] + (i == 0)) * delta
        np.testing.assert_allclose(ac["solution"][:, c.port_map[f"D,{port}"]], expected, atol=1e-10)
    with pytest.raises(ValueError, match="not AC perturbations"):
        c.port_waves(ac["solution"], f"D,{ports[0]}", analysis="ac")


@pytest.mark.parametrize("factory", [optical_waveguide, grating, splitter, directional_coupler, tunable_beam_splitter])
def test_unknown_and_parameter_derivatives(factory: Callable) -> None:
    cls = factory(r0=50.0)
    names = (*cls.ports, *cls.states)
    n = len(cls.ports)
    params = {
        optical_waveguide: ("length_um", 0.23),
        grating: ("peak_loss_dB", 0.3),
        splitter: ("split_ratio", 0.3),
        directional_coupler: ("coupling", 0.3),
        tunable_beam_splitter: ("theta", 0.3),
    }
    key, value = params[factory]
    x = jnp.linspace(-0.7, 1.1, 4 * n)

    def residual(x: jax.Array, parameter: float) -> jax.Array:
        z = x[: 2 * n] + 1j * x[2 * n :]
        f, q = cls(**{key: parameter})(**dict(zip(names, z, strict=True)))
        assert not q
        v = jnp.array([f[name] for name in names])
        return jnp.concatenate((v.real, v.imag))

    actual = jax.jit(jax.jacfwd(residual))(x, value)
    eps = 1e-6
    finite = np.stack([(residual(x + eps * e, value) - residual(x - eps * e, value)) / (2 * eps) for e in jnp.eye(4 * n)], axis=1)
    np.testing.assert_allclose(actual, finite, atol=1e-8)
    actual_p = jax.jit(jax.jacfwd(residual, argnums=1))(x, value)
    finite_p = (residual(x, value + eps) - residual(x, value - eps)) / (2 * eps)
    np.testing.assert_allclose(actual_p, finite_p, atol=1e-7)


def test_jit_vmap_updates_and_smooth_angle_endpoints() -> None:
    c = illuminated(tunable_beam_splitter, ("p1", "p2", "p3", "p4"), 1.0)

    def transmitted(theta: float) -> jax.Array:
        params = {"D.theta": theta, "S0.amplitude": 1.0 + 0j}
        return c.port_waves(c.dc(params=params), "D,p4", params=params).outgoing

    angles = jnp.array([0.0, 0.2, np.pi / 4, np.pi / 2, np.pi])
    np.testing.assert_allclose(jax.jit(jax.vmap(transmitted))(angles), 1j * np.sin(angles), atol=1e-11)
    np.testing.assert_allclose(jax.jit(jax.vmap(jax.grad(lambda x: transmitted(x).imag)))(angles), np.cos(angles), atol=1e-10)


def test_splitter_bright_dark_reverse_inputs() -> None:
    c = illuminated(splitter, ("p1", "p2", "p3"), 1.0)
    for phase, output_power in ((1j, 0.0), (-1j, 1.0)):
        params = {"S1.amplitude": 1 / np.sqrt(2) + 0j, "S2.amplitude": phase / np.sqrt(2)}
        y = c.dc(params=params)
        waves = [c.port_waves(y, f"D,p{i}", params=params) for i in (1, 2, 3)]
        np.testing.assert_allclose(sum(w.outgoing_power for w in waves), output_power, atol=1e-11)
        np.testing.assert_allclose(sum(w.net_power for w in waves), 1 - output_power, atol=1e-11)


def test_zero_phase_nonzero_length_and_dispersion_updates() -> None:
    c = bare(optical_waveguide, ("p1", "p2"), 1.0)
    for length in (0.0, 1.0, 10.0, 100.0):
        for wavelength in (1000.0, 1300.0, 1500.0):
            actual = scattering(
                c, length_um=length, wavelength_nm=wavelength, loss_dB_cm=0.0, neff=1.0, n_group=4.0, center_wavelength_nm=1000.0
            )
            index = 1 - 3 * (wavelength - 1000) / 1000
            expected = np.exp(-2j * np.pi * index * length * 1000 / wavelength)
            np.testing.assert_allclose(actual, [[0, expected], [expected, 0]], atol=1e-11)


def test_isolated_resonant_loop_is_nonunique() -> None:
    # Public component equations certify a family of exact solutions for an
    # isolated zero-length lossless loop. This is a physical nullspace, not
    # an admittance conversion singularity. No assertion about solver success
    # can establish uniqueness when a zero initial guess already has F=0.
    waveguide = optical_waveguide()(length_um=0.0)
    for amplitude in (0j, 1.0 + 0j, 0.2 - 0.7j):
        f, q = waveguide(p1=amplitude, p2=amplitude, wave_p1=amplitude, wave_p2=0j)
        assert not q
        np.testing.assert_allclose([f["wave_p1"], f["wave_p2"], f["p1"] + f["p2"]], 0, atol=1e-12)


@pytest.mark.parametrize(
    ("factory", "parameter", "value"),
    [
        (optical_waveguide, "length_um", -1.0),
        (optical_waveguide, "loss_dB_cm", -1.0),
        (optical_waveguide, "wavelength_nm", 0.0),
        (grating, "bandwidth_1dB", 0.0),
        (grating, "peak_loss_dB", -1.0),
        (splitter, "split_ratio", -0.1),
        (splitter, "split_ratio", 1.1),
        (directional_coupler, "coupling", -0.1),
        (directional_coupler, "coupling", 1.1),
        (directional_coupler, "coupling", np.nan),
    ],
)
def test_physical_parameter_validation(factory: Callable, parameter: str, value: float) -> None:
    cls = factory()
    model = cls(**{parameter: value})
    fields = dict.fromkeys((*cls.ports, *cls.states), 0j)
    with pytest.raises(Exception, match="must be finite"):
        model(**fields)


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
@pytest.mark.parametrize("backend", ["sparse", "klu"])
def test_native_cascade_constraint_rows(r0: float, backend: str) -> None:
    # The internal connection has zero potential diagonal: it is determined
    # by wave constraints, not leakage. No optical terminal touches ground.
    c = compile_circuit(
        {
            "instances": {
                "S": {"component": "s"},
                "A": {"component": "w", "settings": {"length_um": 0.0}},
                "B": {
                    "component": "w",
                    "settings": {"length_um": 0.25, "neff": 1.0, "n_group": 1.0, "wavelength_nm": 1000.0, "loss_dB_cm": 0.0},
                },
                "L": {"component": "l"},
            },
            "connections": {"S,p": "A,p1", "A,p2": "B,p1", "B,p2": "L,p"},
        },
        {
            "s": optical_wave_source(r0=r0, waveform=time_envelope),
            "w": optical_waveguide(r0=r0),
            "l": optical_wave_termination(r0=r0),
        },
        backend=backend,
        g_leak=0,
    )
    y, success = c.dc_checked()
    assert success
    np.testing.assert_allclose(c.port_waves(y, "L,p").incident, -1j, atol=1e-9)
    times = jnp.linspace(0, 1e-9, 11)
    sol = c.transient(t0=0, t1=1e-9, dt0=1e-11, saveat=times, max_steps=200)
    np.testing.assert_allclose(
        c.port_waves(sol.ys, "L,p", analysis="tran", t=sol.ts).incident, -1j * time_envelope(times), atol=1e-8
    )
