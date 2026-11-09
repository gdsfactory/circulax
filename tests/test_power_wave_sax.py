# Copyright 2026 Circulax contributors

"""Public SAX power-wave regressions against independent scattering laws."""

import functools
import importlib
from collections.abc import Callable, Sequence
from typing import NoReturn

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sax

from circulax import Circuit, compile_circuit, fdomain_model
from circulax.components.base_component import PhysicsReturn, Signals, component, source
from circulax.s_transforms import sax_component


@pytest.fixture(autouse=True)
def forbid_s_to_y(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "SAX realization must not convert S to Y"
        raise AssertionError(msg)

    monkeypatch.setattr(importlib.import_module("circulax.s_transforms"), "s_to_y", forbidden)


def through(f: float = 0.0, phase: float = 0.0, loss: float = 0.0, delay: float = 0.0) -> dict:
    transmission = (1 - loss) * jnp.exp(1j * phase - 2j * jnp.pi * f * delay)
    return {("2", "1"): transmission, ("1", "2"): transmission}


def scattering_circuit(model: Callable, *, frequency: bool = False, r0: float = 1.0, backend: str = "dense") -> Circuit:
    wrapped = sax_component(model, frequency_param="f" if frequency else None, r0=r0)
    net = {
        "instances": {"D": {"component": "device"}},
        "connections": {},
        "ports": {p: f"D,{p}" for p in wrapped._raw_to_sanitized_ports},  # noqa: SLF001
    }
    return compile_circuit(net, {"device": wrapped}, backend=backend)


def response(
    circuit: Circuit, frequencies: Sequence[float] | jax.Array = (0.0,), *, r0: float = 1.0, **kwargs: object
) -> jax.Array:
    # The exact unbiased operating point of this source-free linear network.
    # Supply it explicitly: an unterminated lossless network may have a
    # nonunique operating point although its matched scattering solve is unique.
    return circuit.sp(
        ports=circuit.ports,
        freqs=jnp.asarray(frequencies),
        z0=r0,
        y_dc=jnp.zeros(2 * circuit.sys_size),
        **kwargs,
    )


@pytest.mark.parametrize("frequency", [False, True])
@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
@pytest.mark.parametrize("phase", [0.0, jnp.pi / 2, jnp.pi])
def test_exact_lossless_through(*, frequency: bool, r0: float, phase: float) -> None:
    circuit = scattering_circuit(functools.partial(through, phase=phase), frequency=frequency, r0=r0)
    assert circuit.solver.is_complex
    actual = response(circuit, r0=r0)[0]
    transmission = np.exp(1j * phase)
    np.testing.assert_allclose(actual, [[0, transmission], [transmission, 0]], atol=1e-11)
    np.testing.assert_allclose(actual.conj().T @ actual, np.eye(2), atol=1e-11)


@pytest.mark.parametrize("frequency", [False, True])
def test_reflective_multiport_and_port_order(*, frequency: bool) -> None:
    # Asymmetric, reflective contractive matrix, inserted in unsorted order.
    # SAX keys are (input, output); the operator rows are outputs.
    matrix = np.array([[0.2j, 0.3, -0.1j], [0.4, -0.1, 0.2j], [0.15j, 0.1, 0.3]])
    ports = ("z", "1", "a")

    def multiport(f: float = 0.0, scale: float = 1.0) -> dict:  # noqa: ARG001
        return {(q, p): scale * matrix[i, j] for i, p in enumerate(ports) for j, q in enumerate(ports)}

    circuit = scattering_circuit(multiport, frequency=frequency)
    order = [ports.index(p) for p in circuit.ports]
    expected = matrix[np.ix_(order, order)]
    actual = response(circuit)[0]
    np.testing.assert_allclose(actual, expected, atol=1e-11)
    # All coherent simultaneous inputs, including reflected components.
    incident = np.array([0.2 + 0.3j, 1 - 0.1j, -0.4j])
    assert np.linalg.norm(actual @ incident) ** 2 <= np.linalg.norm(incident) ** 2


@pytest.mark.parametrize("frequency", [False, True])
def test_ideal_coupler_and_splitter(*, frequency: bool) -> None:
    from sax.models import mmi1x2_ideal

    def coupler(f: float = 0.0, coupling: float = 0.5) -> dict:  # noqa: ARG001
        t, k = jnp.sqrt(1 - coupling), 1j * jnp.sqrt(coupling)
        return {
            ("p1", "p3"): t,
            ("p3", "p1"): t,
            ("p1", "p4"): k,
            ("p4", "p1"): k,
            ("p2", "p3"): k,
            ("p3", "p2"): k,
            ("p2", "p4"): t,
            ("p4", "p2"): t,
        }

    circuit = scattering_circuit(coupler, frequency=frequency)
    for coupling in (0.0, 0.5, 1.0):
        t, k = np.sqrt(1 - coupling), 1j * np.sqrt(coupling)
        expected = np.array([[0, 0, t, k], [0, 0, k, t], [t, k, 0, 0], [k, t, 0, 0]])
        np.testing.assert_allclose(response(circuit, coupling=coupling)[0], expected, atol=1e-11)

    def splitter(f: float = 0.0) -> dict:  # noqa: ARG001
        return mmi1x2_ideal()

    actual = response(scattering_circuit(splitter, frequency=frequency))[0]
    np.testing.assert_allclose(abs(actual[1:, 0]) ** 2, [0.5, 0.5], atol=1e-11)


@pytest.mark.parametrize("frequency", [False, True])
def test_sweep_jit_vmap_and_parameter_derivatives(*, frequency: bool) -> None:
    # Lossy defaults previously selected Y stamps, then optimization could
    # cross I+S singularities. Representation must stay fixed throughout.
    circuit = scattering_circuit(functools.partial(through, loss=0.2), frequency=frequency)

    def transmission(phase: float, loss: float) -> jax.Array:
        return response(circuit, phase=phase, loss=loss)[0, 1, 0]

    phases = jnp.array([0.0, jnp.pi / 2, jnp.pi, 2 * jnp.pi])
    for loss in (0.2, 1e-12, 0.0):
        actual = jax.jit(jax.vmap(lambda phase, loss=loss: transmission(phase, loss)))(phases)
        np.testing.assert_allclose(actual, (1 - loss) * np.exp(1j * phases), atol=1e-11)
    derivative = jax.jit(jax.grad(lambda phase: transmission(phase, 0.0).imag))(0.4)
    np.testing.assert_allclose(derivative, np.cos(0.4), atol=1e-10)
    loss_derivative = jax.grad(lambda loss: transmission(0.4, loss).real)(0.1)
    np.testing.assert_allclose(loss_derivative, -np.cos(0.4), atol=1e-10)


def test_frequency_delay_quadratures_and_gradients() -> None:
    circuit = scattering_circuit(through, frequency=True)
    frequencies = jnp.array([-3e9, 0.0, 1e9, 3e9])
    phase, delay = 0.7, 0.13e-9
    expected = np.exp(1j * phase - 2j * np.pi * frequencies * delay)
    for holomorphic in (False, True):
        actual = response(circuit, frequencies, phase=phase, delay=delay, holomorphic=holomorphic)
        np.testing.assert_allclose(actual[:, 1, 0], expected, atol=1e-11)
    derivative = jax.grad(lambda tau: response(circuit, [1e9], delay=tau)[0, 1, 0].real)(delay)
    np.testing.assert_allclose(derivative, -2 * np.pi * 1e9 * np.sin(2 * np.pi * 1e9 * delay), rtol=1e-11)
    assert not circuit.check_sax_compatibility()
    with pytest.raises(RuntimeError, match=r"[Ff]requency|[Ff]-domain|fdomain"):
        circuit.transient(t0=0.0, t1=1e-9, dt0=1e-10, y0=jnp.zeros(2 * circuit.sys_size))


def test_frequency_admittance_round_trip_retains_waves() -> None:
    def admittance(f: float, conductance: float = 0.2) -> jax.Array:  # noqa: ARG001
        return jnp.array([[conductance, -conductance], [-conductance, conductance]])

    circuit = scattering_circuit(fdomain_model(admittance, ports=("p1", "p2")), frequency=True)
    actual = response(circuit, [0, 1e9])[0]
    # Analytic scattering matrix of a series 5-ohm resistor at 1-ohm reference.
    np.testing.assert_allclose(actual, [[5 / 7, 2 / 7], [2 / 7, 5 / 7]], atol=1e-11)
    assert circuit.sys_size == 5  # ground + two potentials + two incident waves


@source(ports=("p",), holomorphic=True)
def MatchedDrive(signals: Signals, t: float, freq: float = 1e9, r0: float = 1.0) -> PhysicsReturn:  # noqa: N802
    drive = jnp.exp(2j * jnp.pi * freq * t)
    return {"p": (signals.p - 2 * jnp.sqrt(r0) * drive) / r0}, {}


@component(ports=("p",), holomorphic=True)
def MatchedLoad(signals: Signals, r0: float = 1.0) -> PhysicsReturn:  # noqa: N802
    return {"p": signals.p / r0}, {}


def driven_circuit(*, frequency: bool = False, backend: str = "dense", r0: float = 1.0) -> Circuit:
    device = sax_component(through, frequency_param="f" if frequency else None, r0=r0)
    net = {
        "instances": {
            "D": {"component": "device", "settings": {"phase": 0.7, "delay": 0.13e-9}},
            "S": {"component": "drive", "settings": {"r0": r0}},
            "L": {"component": "load", "settings": {"r0": r0}},
        },
        "connections": {"S,p": "D,1", "D,2": "L,p"},
    }
    return compile_circuit(net, {"device": device, "drive": MatchedDrive, "load": MatchedLoad}, backend=backend)


@pytest.mark.parametrize("frequency", [False, True])
@pytest.mark.parametrize("backend", ["dense", "klu"])
def test_driven_dc_and_hb(*, frequency: bool, backend: str) -> None:
    circuit = driven_circuit(frequency=frequency, backend=backend)
    dc, success = circuit.dc_checked()
    assert success
    np.testing.assert_allclose(circuit.port(dc, "D,2"), np.exp(0.7j), atol=1e-8)
    time_values, _ = circuit.hb(freq=1e9, harmonics=3)
    times = np.arange(7) / (7e9)
    delay = 0.13e-9 if frequency else 0.0
    expected = np.exp(0.7j + 2j * np.pi * 1e9 * (times - delay))
    np.testing.assert_allclose(circuit.port(time_values, "D,2"), expected, atol=1e-7)


def test_static_transient_keeps_both_quadratures() -> None:
    circuit = driven_circuit()
    times = jnp.linspace(0, 1e-9, 11)
    solution = circuit.transient(t0=0, t1=1e-9, dt0=1e-11, saveat=times, max_steps=200)
    np.testing.assert_allclose(circuit.port(solution.ys, "D,2"), np.exp(0.7j + 2j * np.pi * 1e9 * times), atol=1e-7)


@pytest.mark.parametrize("r0", [0.0, -1.0, float("inf"), float("nan"), 1 + 1j, np.complex128(1 + 1j)])
def test_invalid_reference_is_rejected(r0: complex) -> None:
    with pytest.raises(ValueError, match="finite, positive, and real"):
        sax_component(through, r0=r0)


@pytest.mark.parametrize("frequency", [False, True])
@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
def test_driven_physical_waves_are_reference_invariant(*, frequency: bool, r0: float) -> None:
    circuit = driven_circuit(frequency=frequency, r0=r0)
    dc, success = circuit.dc_checked(rtol=1e-12, atol=1e-12)
    assert success
    incoming = circuit.port(dc, "D,wave_p1")
    reverse = circuit.port(dc, "D,wave_p2")
    outgoing = circuit.port(dc, "D,2") / np.sqrt(r0) - reverse
    np.testing.assert_allclose(incoming, 1.0, atol=1e-8)
    np.testing.assert_allclose(reverse, 0.0, atol=1e-8)
    np.testing.assert_allclose(outgoing, np.exp(0.7j), atol=1e-8)
    np.testing.assert_allclose(abs(incoming) ** 2 - abs(outgoing) ** 2, 0.0, atol=1e-8)


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
def test_static_sax_export_preserves_reference(r0: float) -> None:
    circuit = scattering_circuit(functools.partial(through, phase=0.7), r0=r0)
    exported = circuit.to_sax_circuit()()
    np.testing.assert_allclose(exported[("1", "2")], np.exp(0.7j), atol=1e-11)
    np.testing.assert_allclose(exported[("1", "1")], 0.0, atol=1e-11)


@pytest.mark.parametrize("frequency", [False, True])
def test_nonreciprocal_admittance_round_trip(*, frequency: bool) -> None:
    def admittance(f: float, conductance: float = 0.2, transfer: float = 0.1) -> jax.Array:  # noqa: ARG001
        return jnp.array([[conductance, 0.0], [-transfer, conductance]])

    circuit = scattering_circuit(fdomain_model(admittance, ports=("p1", "p2")), frequency=frequency)
    actual = response(circuit)[0]
    expected = [[(1 - 0.2) / (1 + 0.2), 0.0], [2 * 0.1 / (1 + 0.2) ** 2, (1 - 0.2) / (1 + 0.2)]]
    np.testing.assert_allclose(actual, expected, atol=1e-11)


@pytest.mark.parametrize("frequency", [False, True])
@pytest.mark.parametrize("reflection", [-1.0, 1.0])
def test_exact_single_port_reflector(*, frequency: bool, reflection: float) -> None:
    def reflector(f: float = 0.0, reflection: float = reflection) -> dict:  # noqa: ARG001
        return {("p", "p"): reflection}

    circuit = scattering_circuit(reflector, frequency=frequency)
    np.testing.assert_allclose(response(circuit)[0], [[reflection]], atol=1e-11)


@pytest.mark.parametrize("mode", ["automatic", "explicit", "frequency"])
def test_default_retains_all_waves_with_reflecting_load(mode: str) -> None:
    def device(f: float = 0.0, reflection: complex = 0j) -> sax.SDict:
        transmission = 0.6 * jnp.exp(-2j * jnp.pi * f * 0.13e-9)
        return {
            ("p1", "p1"): reflection,
            ("p2", "p2"): reflection,
            ("p1", "p2"): transmission,
            ("p2", "p1"): transmission,
        }

    def load(reflection: complex = 0.4j) -> sax.SDict:
        return {("p", "p"): reflection}

    models = {"device": device, "load": load}
    if mode != "automatic":
        models["device"] = sax_component(device, frequency_param="f" if mode == "frequency" else None)
        models["load"] = sax_component(load)
    net = {
        "instances": {"D": {"component": "device"}, "L": {"component": "load"}},
        "connections": {"D,p2": "L,p"},
        "ports": {"in": "D,p1"},
    }
    circuit = compile_circuit(net, models, backend="dense")
    for name in ("D,wave_p1", "D,wave_p2", "L,wave_p"):
        assert name in circuit.port_map
    frequency = 1e9
    transmission = 0.6 * np.exp(-2j * np.pi * frequency * 0.13e-9) if mode == "frequency" else 0.6
    # Multiple reflection reference: b2 = t*a1 + r*a2, a2 = gamma*b2.
    # Change the device's own reflection from its zero default after compilation.
    for reflection in (0.0, 0.2 + 0.1j):
        actual = response(circuit, [frequency], params={"D.reflection": reflection})[0, 0, 0]
        expected = reflection + transmission**2 * 0.4j / (1 - reflection * 0.4j)
        np.testing.assert_allclose(actual, expected, atol=1e-11)
