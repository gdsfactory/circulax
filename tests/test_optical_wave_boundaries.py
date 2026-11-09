# Copyright 2026 Circulax contributors

"""Explicit wave boundaries and observables against independent network laws."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax import Circuit, compile_circuit, sax_component
from circulax.components import OpticalWaveSource, OpticalWaveTermination, optical_wave_source, optical_wave_termination
from circulax.components.photonic import OpticalSource, OpticalSourcePulse
from circulax.power_waves import reconstruct_waves


def device(phase: float = 0.4, loss: float = 0.2, reflection: complex = 0j) -> dict:
    transmission = (1 - loss) * jnp.exp(1j * phase)
    return {("in", "in"): reflection, ("out", "out"): reflection, ("in", "out"): transmission, ("out", "in"): transmission}


def network(
    *,
    r0: float = 1.0,
    gamma: complex | None = None,
    reverse: complex | None = None,
    waveform: Callable | None = None,
    backend: str = "dense",
) -> Circuit:
    models = {"device": sax_component(device, r0=r0), "source": optical_wave_source(r0=r0, waveform=waveform)}
    if gamma is not None:

        def reflector(gamma: complex = gamma) -> dict:
            return {("p", "p"): gamma}

        models["load"] = sax_component(reflector, r0=r0)
    elif reverse is not None:
        models["load"] = optical_wave_source(r0=r0)
    else:
        models["load"] = optical_wave_termination(r0=r0)
    instances = {"D": {"component": "device"}, "S": {"component": "source"}, "L": {"component": "load"}}
    if reverse is not None:
        instances["L"]["settings"] = {"amplitude": reverse}
    return compile_circuit(
        {"instances": instances, "connections": {"S,p": "D,in", "D,out": "L,p"}},
        models,
        backend=backend,
        g_leak=0.0,
    )


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
@pytest.mark.parametrize("gamma", [None, 0.5j, -1.0, np.exp(0.3j)])
@pytest.mark.parametrize("backend", ["dense", "klu"])
def test_dc_balance(r0: float, gamma: complex | None, backend: str) -> None:
    circuit = network(r0=r0, gamma=gamma, backend=backend)
    emitted = 0.7 - 0.6j
    # Device reflections are varied after construction; matching is never
    # inferred from its reflection-free defaults.
    reflection = 0.1 + 0.05j
    updates = {"S.amplitude": emitted, "D.reflection": reflection}
    y, success = circuit.dc_checked(params=updates, atol=1e-12, rtol=1e-12)
    assert success
    transmission = 0.8 * np.exp(0.4j)
    g = 0 if gamma is None else gamma
    received = transmission * emitted / (1 - reflection * g)
    returned = reflection * emitted + transmission * g * received
    source = circuit.port_waves(y, "S,p", params=updates)
    left = circuit.port_waves(y, "D,in", params=updates)
    right = circuit.port_waves(y, "D,out", params=updates)
    load = circuit.port_waves(y, "L,p", params=updates)
    np.testing.assert_allclose(source.outgoing, emitted, atol=1e-11)
    np.testing.assert_allclose(source.incident, returned, atol=1e-11)
    np.testing.assert_allclose([left.incident, left.outgoing], [emitted, returned], atol=1e-11)
    np.testing.assert_allclose([right.incident, right.outgoing], [g * received, received], atol=1e-11)
    np.testing.assert_allclose([load.incident, load.outgoing], [received, g * received], atol=1e-11)
    np.testing.assert_allclose(circuit.port(y, "D,in"), np.sqrt(r0) * (emitted + returned), atol=1e-11)
    np.testing.assert_allclose(source.incident_power, abs(returned) ** 2, atol=1e-11)
    np.testing.assert_allclose(source.outgoing_power, abs(emitted) ** 2, atol=1e-11)
    np.testing.assert_allclose(source.net_power, abs(returned) ** 2 - abs(emitted) ** 2, atol=1e-11)
    np.testing.assert_allclose(load.net_power, (1 - abs(g) ** 2) * abs(received) ** 2, atol=1e-11)
    np.testing.assert_allclose(source.net_power + left.net_power + right.net_power + load.net_power, 0, atol=1e-11)
    # No physical terminal touches the solver's reserved ground node.
    assert all(circuit.port_map[p] != 0 for p in ("S,p", "D,in", "D,out", "L,p"))


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
@pytest.mark.parametrize(("forward", "reverse"), [(0j, 0.2 - 0.7j), (0.6 + 0.3j, -0.2j)])
def test_reverse_and_coherent_superposition(r0: float, forward: complex, reverse: complex) -> None:
    circuit = network(r0=r0, reverse=reverse)
    # A unitary reflective two-port, with common carrier phase. Both sources
    # absorb waves emitted by the other, including coherent local reflections.
    r = 0.15j * np.exp(0.4j)
    updates = {"S.amplitude": forward, "D.reflection": r, "D.loss": 1 - np.sqrt(1 - 0.15**2)}
    y = circuit.dc(params=updates)
    left, right = [circuit.port_waves(y, p, params=updates) for p in ("D,in", "D,out")]
    transmission = np.sqrt(1 - 0.15**2) * np.exp(0.4j)
    np.testing.assert_allclose([left.incident, right.incident], [forward, reverse], atol=1e-10)
    np.testing.assert_allclose(
        [left.outgoing, right.outgoing], [r * forward + transmission * reverse, transmission * forward + r * reverse], atol=1e-10
    )
    np.testing.assert_allclose(left.net_power + right.net_power, 0, atol=1e-10)


def envelope(t: jax.Array) -> jax.Array:
    return 0.8 + 0.2 * jnp.exp(2j * jnp.pi * 1e9 * t)


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
def test_transient_and_hb_total_samples(r0: float) -> None:
    circuit = network(r0=r0, gamma=0.4j, waveform=envelope)
    times = jnp.linspace(0, 1e-9, 11)
    solution = circuit.transient(t0=0, t1=1e-9, dt0=1e-11, saveat=times, max_steps=200)
    source = circuit.port_waves(solution.ys, "S,p", analysis="tran", t=solution.ts)
    load = circuit.port_waves(solution.ys, "L,p", analysis="tran", t=solution.ts)
    transmission = 0.8 * np.exp(0.4j)
    np.testing.assert_allclose(source.outgoing, envelope(times), atol=1e-8)
    np.testing.assert_allclose(load.incident, transmission * envelope(times), atol=1e-8)
    np.testing.assert_allclose(source.incident, 0.4j * transmission**2 * envelope(times), atol=1e-8)
    y_time, _ = circuit.hb(freq=1e9, harmonics=2)
    hb_times = jnp.arange(5) / 5e9
    waves = circuit.port_waves(y_time, "L,p", analysis="hb_time", t=hb_times)
    np.testing.assert_allclose(waves.incident, transmission * envelope(hb_times), atol=1e-8)
    np.testing.assert_allclose(waves.incident_power, abs(transmission * envelope(hb_times)) ** 2, atol=1e-8)
    dc = circuit.port_waves(circuit.dc(), "S,p")
    np.testing.assert_allclose(dc.outgoing, envelope(0.0), atol=1e-10)


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
def test_ac_boundary_drive_is_emitted_wave_perturbation(r0: float) -> None:
    circuit = network(r0=r0, gamma=0.5j, waveform=lambda _t: 0.5j)
    delta = 0.2 - 0.3j
    result = circuit.ac_response(freqs=jnp.array([0.0, 1e9]), sources={"S": delta}, params={"S.amplitude": 2 + 3j})
    transmission = 0.8 * np.exp(0.4j)
    expected = np.sqrt(r0) * (1 + 0.5j * transmission**2) * delta
    # AC's public solution stores complex envelope perturbations directly,
    # unlike DC/transient real blocks. It is not a total field/power.
    np.testing.assert_allclose(result["solution"][:, circuit.port_map["S,p"]], expected, atol=1e-10)
    for analysis in ("ac", "hb", "hb_coefficients"):
        with pytest.raises(ValueError, match="not AC perturbations"):
            circuit.port_waves(result["solution"], "S,p", analysis=analysis)


def test_jit_vmap_and_parameter_gradients() -> None:
    circuit = network(gamma=0.5j)

    def received(phase: float, amplitude: complex) -> jax.Array:
        updates = {"D.phase": phase, "S.amplitude": amplitude}
        y = circuit.dc(params=updates)
        return circuit.port_waves(y, "L,p", params=updates).incident

    phases = jnp.array([0, 0.4, 1.7])
    actual = jax.jit(jax.vmap(received))(phases, jnp.array([1, 2j, 0.5 - 0.2j]))
    np.testing.assert_allclose(actual, 0.8 * np.exp(1j * phases) * np.array([1, 2j, 0.5 - 0.2j]), atol=1e-10)
    derivative = jax.jit(jax.grad(lambda phase: received(phase, 1).real))(0.4)
    np.testing.assert_allclose(derivative, -0.8 * np.sin(0.4), atol=1e-10)
    derivative = jax.grad(lambda amplitude: abs(received(0.4, amplitude)) ** 2)(1.3)
    np.testing.assert_allclose(derivative, 2 * 0.8**2 * 1.3, atol=1e-10)


@pytest.mark.parametrize("r0", [0.0, -1.0, np.inf, np.nan, 1 + 0j])
def test_invalid_references(r0: float) -> None:
    for factory in (optical_wave_source, optical_wave_termination):
        with pytest.raises(ValueError, match="finite, positive, and real"):
            factory(r0=r0)
    with pytest.raises(ValueError, match="finite, positive, and real"):
        reconstruct_waves(1, 1, r0=r0)


def test_standalone_reconstruction_and_default_boundaries() -> None:
    a, b = 0.3 - 0.4j, 0.8 + 0.1j
    waves = jax.jit(lambda v, i: reconstruct_waves(v, i, r0=50))((a + b) * np.sqrt(50), (a - b) / np.sqrt(50))
    np.testing.assert_allclose([waves.incident, waves.outgoing, waves.net_power], [a, b, abs(a) ** 2 - abs(b) ** 2], atol=1e-12)
    f, q = OpticalWaveSource(amplitude=b)(p=(a + b))
    np.testing.assert_allclose(f["p"], a - b)
    assert not q
    f, q = OpticalWaveTermination()(p=a)
    np.testing.assert_allclose(f["p"], a)
    assert not q


def test_legacy_sources_preserve_potential_constraint() -> None:
    potential = np.sqrt(2.0) * np.exp(0.7j)
    for model, kwargs in ((OpticalSource, {}), (OpticalSourcePulse, {"delay": 0.0, "rise": 1.0})):
        source = model(power=2.0, phase=0.7, **kwargs)
        f, _ = source(t=0, p1=potential, p2=0, i_src=0.3j)
        expected = 0 if model is OpticalSource else potential / 2
        np.testing.assert_allclose(f["i_src"], expected, atol=1e-12)
        np.testing.assert_allclose([f["p1"], f["p2"]], [0.3j, -0.3j])


def test_unsupported_ports_and_frequency_models() -> None:
    circuit = network()
    y = circuit.dc()
    for port in ("missing,p", "D,wave_in", "D,nonsense"):
        with pytest.raises(KeyError):
            circuit.port_waves(y, port)
    with pytest.raises(ValueError, match="component port"):
        circuit.port_waves(y, "in")
    with pytest.raises(ValueError, match="real-block"):
        circuit.port_waves(y[:-1], "D,in")

    def fdevice(f: float = 0.0) -> dict:
        return device(phase=f)

    freq_circuit = compile_circuit(
        {
            "instances": {"D": {"component": "d"}, "S": {"component": "s"}, "L": {"component": "l"}},
            "connections": {"S,p": "D,in", "D,out": "L,p"},
        },
        {"d": sax_component(fdevice, frequency_param="f"), "s": OpticalWaveSource, "l": OpticalWaveTermination},
        backend="dense",
        g_leak=0,
    )
    y_dc = freq_circuit.dc()
    waves = freq_circuit.port_waves(y_dc, "D,in")
    np.testing.assert_allclose([waves.incident, waves.outgoing], [1.0, 0.0], atol=1e-10)
    with pytest.raises(ValueError, match="supports DC only"):
        freq_circuit.port_waves(y_dc, "D,in", analysis="tran")
    with pytest.raises(RuntimeError, match=r"[Ff]requency|[Ff]-domain|fdomain"):
        freq_circuit.transient(t0=0, t1=1e-9, dt0=1e-10, y0=jnp.zeros(2 * freq_circuit.sys_size))


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
@pytest.mark.parametrize("phase", [0.0, np.pi / 2, np.pi])
def test_matched_scattering_law(r0: float, phase: float) -> None:
    circuit = network(r0=r0)
    updates = {"D.loss": 0.0, "D.phase": phase, "S.amplitude": 0.3 + 0.4j}
    y = circuit.dc(params=updates)
    source = circuit.port_waves(y, "S,p", params=updates)
    received = circuit.port_waves(y, "L,p", params=updates)
    np.testing.assert_allclose(source.incident, 0, atol=1e-11)
    np.testing.assert_allclose(received.incident, (0.3 + 0.4j) * np.exp(1j * phase), atol=1e-11)
    np.testing.assert_allclose([source.outgoing_power, received.incident_power], [0.25, 0.25], atol=1e-11)
    np.testing.assert_allclose(source.net_power + received.net_power, 0, atol=1e-11)


def test_quadrature_small_signal_emitted_wave_metadata() -> None:
    from circulax import QuadratureDrive

    circuit = network(r0=50.0)
    drive = QuadratureDrive(x=0.2 + 0.3j, y=-0.1 + 0.4j)
    response = circuit.small_signal().response(freqs=jnp.array([1e9]), sources={"S": drive})
    x, y = response.quadratures("D,in")
    np.testing.assert_allclose(x, np.sqrt(50) * drive.x, atol=1e-10)
    np.testing.assert_allclose(y, np.sqrt(50) * drive.y, atol=1e-10)


def test_accessor_rejects_legacy_source() -> None:
    # Even without invoking a ground solve, the observable must reject a
    # potential-difference source as lacking the declared wave-port contract.
    circuit = compile_circuit(
        {"instances": {"S": {"component": "s"}}, "ports": {"p": "S,p1", "n": "S,p2"}},
        {"s": OpticalSource},
        backend="dense",
    )
    with pytest.raises(ValueError, match="algebraic power-wave"):
        circuit.port_waves(jnp.zeros(2 * circuit.sys_size), "S,p1")
