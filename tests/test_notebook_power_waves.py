# Copyright 2026 Circulax contributors
"""Traveling-power measurements through public APIs and independent references."""

import importlib
import json
from pathlib import Path
from typing import Any, NoReturn

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax import QuadratureDrive, compile_circuit, sax_component
from circulax.components import ideal_photodetector, optical_wave_source, optical_wave_termination, phase_shifter_factory
from circulax.components.electronic import Resistor, VoltageSource
from circulax.components.photonic import directional_coupler, grating, optical_waveguide, splitter


@pytest.fixture(autouse=True)
def forbid_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "S-to-Y called by migrated optical realization"
        raise AssertionError(msg)

    for module in ("circulax.s_transforms", "circulax.components.photonic"):
        monkeypatch.setattr(importlib.import_module(module), "s_to_y", forbidden)


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
@pytest.mark.parametrize("rho", [0.0, 0.36, 1.0])
@pytest.mark.parametrize("phase", [0.0, np.pi / 2, np.pi])
def test_reflecting_detector_power_phasors(r0: float, rho: float, phase: float) -> None:
    amplitude = 0.01 * np.exp(1j * phase)
    drive = (0.003 + 0.004j) * np.array([np.cos(phase), np.sin(phase)])
    circuit = compile_circuit(
        {
            "instances": {
                "S": {"component": "source", "settings": {"amplitude": amplitude}},
                "PD": {"component": "pd", "settings": {"reflectivity": rho, "reflection_phase_rad": np.pi}},
                "R": {"component": "resistor", "settings": {"R": 50.0}},
                "GND": {"component": "ground"},
            },
            "connections": {"S,p": "PD,optical_in", "PD,anode": "R,p1", "GND,p1": ("R,p2", "PD,cathode")},
        },
        {"source": optical_wave_source(r0=r0), "pd": ideal_photodetector(r0=r0), "resistor": Resistor},
        is_complex=True,
        backend="dense",
        g_leak=0.0,
    )
    updates = {"PD.responsivity": 0.7}
    op = circuit.dc(params=updates)
    waves = circuit.port_waves(op, "PD,optical_in", params=updates)
    np.testing.assert_allclose(waves.incident_power, 1e-4, atol=1e-14)
    np.testing.assert_allclose(waves.net_power, (1 - rho) * 1e-4, atol=1e-14)
    np.testing.assert_allclose(circuit.port(op, "PD,anode"), 50 * 0.7 * (1 - rho) * 1e-4, atol=1e-12)
    response = circuit.small_signal(y_dc=op, params=updates)(
        freqs=jnp.array([0.0, 1e9]), sources={"S": QuadratureDrive(x=drive[0], y=drive[1])}
    )
    result = jax.jit(lambda r: circuit.port_power_response(r, "PD,optical_in", params=updates))(response)
    expected = 0.02 * (0.003 + 0.004j)
    np.testing.assert_allclose(result.incident, expected, atol=1e-13)
    np.testing.assert_allclose(result.outgoing, rho * expected, atol=1e-13)
    np.testing.assert_allclose(result.net, (1 - rho) * expected, atol=1e-13)
    np.testing.assert_allclose(response.voltage("PD,anode") / 50, 0.7 * result.net, atol=1e-13)
    # Independent DC finite difference, preserving complex RF phase separately.
    step = 1e-4
    for direction, part in ((drive.real, np.real), (drive.imag, np.imag)):
        delta = direction[0] + 1j * direction[1]
        ys = circuit.dc(params={**updates, "S.amplitude": jnp.array([amplitude - step * delta, amplitude + step * delta])})
        powers = circuit.port_waves(ys, "PD,optical_in", params=updates).net_power
        np.testing.assert_allclose((powers[1] - powers[0]) / (2 * step), part(result.net[0]), atol=1e-12)
    with pytest.raises(ValueError, match="independent source forcing"):
        circuit.port_power_response(response, "S,p")
    with pytest.raises(ValueError, match="optical terminal"):
        circuit.port_power_response(response, "PD,anode")


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
def test_delayed_power_derivative(r0: float) -> None:
    length, tau, loss_slope, bias = 1000.0, 20e-12, 2.0, -0.4
    phase = phase_shifter_factory(delta_neff=lambda v: 1e-4 * v, loss_dB_cm=lambda v: 4.0 + loss_slope * v, r0=r0)
    circuit = compile_circuit(
        {
            "instances": {
                "S": {"component": "source", "settings": {"amplitude": 0.02j}},
                "PS": {"component": "phase", "settings": {"length_um": length, "n_group": tau * 2.99792458e14 / length}},
                "V": {"component": "voltage", "settings": {"V": bias}},
                "L": {"component": "load"},
                "GND": {"component": "ground"},
            },
            "connections": {"S,p": "PS,o1", "PS,o2": "L,p", "V,p1": "PS,anode", "GND,p1": ("V,p2", "PS,cathode")},
        },
        {"source": optical_wave_source(r0=r0), "phase": phase, "voltage": VoltageSource, "load": optical_wave_termination(r0=r0)},
        is_complex=True,
        backend="dense",
        g_leak=0.0,
    )
    frequencies = jnp.array([0.0, 7e9, 23e9])
    drive = 0.3 + 0.4j
    op = circuit.dc()
    response = circuit.small_signal(y_dc=op)(freqs=frequencies, sources={"V": drive})
    result = circuit.port_power_response(response, "PS,o2")
    power = 0.02**2 * 10 ** (-(4 + loss_slope * bias) * length / 100000)
    expected = -np.log(10) * loss_slope * length / 100000 * power * drive * np.exp(-2j * np.pi * np.asarray(frequencies) * tau / 2)
    np.testing.assert_allclose(result.outgoing, expected, atol=1e-13)
    np.testing.assert_allclose(result.incident, 0, atol=1e-13)
    np.testing.assert_allclose(circuit.port_power_response(response, "L,p").net, expected, atol=1e-13)
    # RF source scaling and external vmap do not collapse the phasor quadratures.
    model = circuit.small_signal(y_dc=op)
    batched = jax.jit(
        jax.vmap(lambda scale: circuit.port_power_response(model(freqs=frequencies, sources={"V": scale * drive}), "L,p").net)
    )(jnp.array([1.0, 2.0]))
    np.testing.assert_allclose(batched, np.array([1, 2])[:, None] * expected, atol=1e-13)


def notebook_namespace(path: str, indices: list[int]) -> dict[str, Any]:
    cells = json.loads((Path(__file__).parents[1] / path).read_text())["cells"]
    namespace = {}
    for i in indices:
        exec(compile("".join(cells[i]["source"]), f"{path}:cell{i}", "exec"), namespace)  # noqa: S102 -- repository notebook
    return namespace


def cmz_reference(params: jax.Array, wavelengths: jax.Array) -> jax.Array:
    """Independent forward 2x2 propagation; the backward block is unexcited."""

    def coupler(k: jax.Array) -> jax.Array:
        return jnp.array([[jnp.sqrt(1 - k), 1j * jnp.sqrt(k)], [1j * jnp.sqrt(k), jnp.sqrt(1 - k)]])

    def arms(long: jax.Array, wl: jax.Array) -> jax.Array:
        index = 2.4 + (2.4 - 4.0) * (wl - 1310) / 1310
        return jnp.exp(-2j * jnp.pi * index * jnp.array([long, 250.0]) * 1000 / wl)

    def solve(wl: jax.Array) -> jax.Array:
        k1, k2, k3, ka1, ka2, kb1, kb2, l1, l2, la, lb = params
        root = coupler(k3) @ (arms(l2, wl) * (coupler(k2) @ (arms(l1, wl) * (coupler(k1) @ jnp.array([1.0, 0.0])))))
        a = coupler(ka2) @ (arms(la, wl) * (coupler(ka1) @ jnp.array([root[0], 0.0])))
        b = coupler(kb2) @ (arms(lb, wl) * (coupler(kb1) @ jnp.array([root[1], 0.0])))
        return jnp.abs(jnp.array([a[0], b[0], a[1], b[1]])) ** 2

    return jax.vmap(solve)(wavelengths)


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
def test_cmz_notebook_objective_and_gradients(r0: float) -> None:
    ns = notebook_namespace("examples/inverse_design/03_photonic_cmz_demux.ipynb", [3])
    ns["models"].update(
        source=optical_wave_source(r0=r0),
        monitor=optical_wave_termination(r0=r0),
        dc=directional_coupler(r0=r0),
        waveguide=optical_waveguide(r0=r0),
        short=sax_component(ns["optical_short"], r0=r0),
    )
    # Execute the actual authored netlist and objective; omit multi-start training here.
    cells = json.loads((Path(__file__).parents[1] / "examples/inverse_design/03_photonic_cmz_demux.ipynb").read_text())["cells"]
    ns["circuit"] = compile_circuit(ns["net_dict"], ns["models"], is_complex=True, backend="dense", g_leak=0.0)
    ns["Y_GUESS"] = jnp.ones(ns["circuit"].sys_size * 2)
    exec("".join(cells[5]["source"]), ns)  # noqa: S102 -- execute repository notebook acceptance code
    exec("".join(cells[8]["source"]), ns)  # noqa: S102 -- execute repository notebook acceptance code
    p, wavelengths = ns["PAPER_PARAMS"], ns["LOSS_WLS"]
    reference = cmz_reference(p, wavelengths)
    np.testing.assert_allclose(reference.sum(axis=1), 1, atol=2e-12)
    measured = ns["get_power_matrix"](p, wavelengths)
    np.testing.assert_allclose(measured, reference / (reference.sum(axis=1, keepdims=True) + 1e-12), atol=1e-9)

    def raw_power(wl: jax.Array) -> jax.Array:
        updates = ns["_parameter_updates"](p, wl)
        y = ns["circuit"].dc(params=updates, y_guess=ns["Y_GUESS"])
        return ns["_monitor_powers"](y, updates)

    np.testing.assert_allclose(jax.jit(jax.vmap(raw_power))(wavelengths), reference, atol=1e-9)
    rotated = ns["circuit"].dc(params={"Laser.amplitude": 0.6 + 0.8j, "wavelength_nm": 1300.0})
    unrotated = ns["circuit"].dc(wavelength_nm=1300.0)
    np.testing.assert_allclose(ns["_monitor_powers"](rotated, {}), ns["_monitor_powers"](unrotated, {}), atol=1e-10)
    # Every trainable parameter is checked against both an independent analytic
    # gradient and finite differences of the actual compiled objective.
    targets = ns["LOSS_TARGETS"]

    def objective(p: jax.Array) -> jax.Array:
        power = cmz_reference(p, wavelengths)
        power = power / (power.sum(axis=1, keepdims=True) + 1e-12)
        return jnp.mean((1 - power[jnp.arange(len(wavelengths)), targets]) ** 2)

    p = p + jnp.array([0.03, -0.02, 0.01, 0.04, -0.03, 0.02, -0.01, 0.03, -0.02, 0.04, -0.03])
    gradient = jax.jit(jax.grad(ns["loss_fn"]))(p)
    np.testing.assert_allclose(gradient, jax.grad(objective)(p), rtol=1e-7, atol=2e-9)
    step = 1e-5
    offsets = jnp.eye(11) * step
    loss = jax.jit(jax.vmap(ns["loss_fn"]))
    finite = (loss(p + offsets) - loss(p - offsets)) / (2 * step)
    np.testing.assert_allclose(gradient, finite, rtol=1e-5, atol=1e-7)


def test_frequency_flow_rejection_and_boundary_observation() -> None:
    from circulax.components import optical_delay_line_fdomain

    circuit = compile_circuit(
        {
            "instances": {
                "S": {"component": "source", "settings": {"amplitude": 0.6 + 0.8j}},
                "D": {"component": "delay", "settings": {"length_um": 1000.0, "loss_dB_cm": 0.0}},
                "L": {"component": "load"},
            },
            "connections": {"S,p": "D,p1", "D,p2": "L,p"},
        },
        {"source": optical_wave_source(), "delay": optical_delay_line_fdomain(), "load": optical_wave_termination()},
        is_complex=True,
        backend="dense",
        g_leak=0.0,
    )
    frequencies = jnp.array([0.0, 17e9])
    dx, dy = 0.2 + 0.1j, -0.3 + 0.4j
    response = circuit.small_signal()(freqs=frequencies, sources={"S": QuadratureDrive(x=dx, y=dy)})
    with pytest.raises(ValueError, match="instantaneous terminal flow"):
        circuit.port_power_response(response, "D,p2")
    # The same frequency-dependent network can be observed at its instantaneous
    # matched boundary; the solver already includes the delay in both coordinates.
    expected = 2 * (0.6 * dx + 0.8 * dy) * np.exp(-2j * np.pi * np.asarray(frequencies) * 1000 * 4 / 2.99792458e14)
    np.testing.assert_allclose(circuit.port_power_response(response, "L,p").incident, expected, atol=1e-12)


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
def test_mzm_notebook_monitor_reference(r0: float) -> None:
    ns = notebook_namespace("examples/photonic/mzm.ipynb", [2, 3])
    circuit = compile_circuit(
        ns["net_dict"],
        {
            "source": optical_wave_source(r0=r0),
            "monitor": optical_wave_termination(r0=r0),
            "grating": grating(r0=r0),
            "splitter": splitter(r0=r0),
            "waveguide": optical_waveguide(r0=r0),
        },
        is_complex=True,
        backend="dense",
        g_leak=0.0,
    )
    wavelengths = jnp.array([1285.0, 1310.0, 1330.0])

    def measure(wl: jax.Array) -> jax.Array:
        updates = {"wavelength_nm": wl}
        y = circuit.dc(params=updates)
        return circuit.port_waves(y, "Detector,p", params=updates).incident_power

    actual = jax.jit(jax.vmap(measure))(wavelengths)

    # Two gratings, two common 50-um paths, and coherent arm recombination.
    def reference(wl: jax.Array) -> jax.Array:
        index = 2.4 + (2.4 - 4.0) * (wl - 1310) / 1310

        def path(length: float) -> jax.Array:
            return 10 ** (-length / 200000) * jnp.exp(-2j * jnp.pi * index * length * 1000 / wl)

        gc = 10 ** (-(1 + ((wl - 1310) / 20) ** 2) / 20)
        return jnp.abs(gc**2 * path(50) ** 2 * (path(150) - path(100)) / 2) ** 2

    np.testing.assert_allclose(actual, jax.vmap(reference)(wavelengths), rtol=1e-9, atol=1e-12)
