# Copyright 2026 Circulax contributors

"""Source-driven linearization preserves RF phase and physical optical power."""

import jax
import jax.numpy as jnp
import pytest

from circulax import Circuit, QuadratureDrive, SmallSignalModel, SmallSignalResponse, compile_circuit
from circulax.components.base_component import PhysicsReturn, Signals, component
from circulax.components.electronic import Capacitor, Resistor, WaveformCurrentSource, WaveformVoltageSource
from circulax.components.photonic import OpticalSource


def _rc() -> Circuit:
    """Build a source-driven RC circuit without external characterization ports."""
    return compile_circuit(
        {
            "instances": {
                "V": {"component": "voltage", "settings": {"dc": 2.0}},
                "I": {"component": "current", "settings": {"dc": 0.0}},
                "R": {"component": "resistor", "settings": {"R": 1000.0}},
                "C": {"component": "capacitor", "settings": {"C": 1e-6}},
                "GND": {"component": "ground"},
            },
            "connections": {
                "V,p1": "R,p1",
                "R,p2": ("C,p1", "I,p1"),
                "GND,p1": ("V,p2", "C,p2", "I,p2"),
            },
            "ports": {},
        },
        {"voltage": WaveformVoltageSource, "current": WaveformCurrentSource, "resistor": Resistor, "capacitor": Capacitor},
    )


@component(ports=("ein", "out", "v1", "v2"), states=("u", "iout"), holomorphic=False)
def _probe(signals: Signals, tau: float = 1e-10) -> PhysicsReturn:
    target = signals.ein * ((1 + 1j) + (2 + 3j) * jnp.real(signals.u))
    return {
        "ein": 0j,
        "v1": 0j,
        "v2": 0j,
        "out": signals.iout,
        "iout": signals.out - target,
        "u": signals.u - signals.v1 - signals.v2,
    }, {"u": tau * signals.u}


def _optical() -> Circuit:
    return compile_circuit(
        {
            "instances": {
                "laser": {"component": "laser", "settings": {"power": 1e-9}},
                "V1": {"component": "voltage"},
                "V2": {"component": "voltage"},
                "probe": {"component": "probe"},
                "load": {"component": "load", "settings": {"R": 1.0}},
                "GND": {"component": "ground"},
            },
            "connections": {
                "laser,p1": "probe,ein",
                "V1,p1": "probe,v1",
                "V2,p1": "probe,v2",
                "probe,out": "load,p1",
                "GND,p1": ("laser,p2", "V1,p2", "V2,p2", "load,p2"),
            },
            "ports": {"out": "probe,out"},
        },
        {"laser": OpticalSource, "voltage": WaveformVoltageSource, "probe": _probe, "load": Resistor},
        is_complex=True,
    )


def _frequencies() -> jax.Array:
    return jnp.array([0.0, 1e8, 1 / (2 * jnp.pi * 1e-10), 1e10, 1e11])


def test_analytical_quadratures_power_and_sidebands() -> None:
    circuit = _optical()
    model = circuit.small_signal()
    freqs = _frequencies()
    result = model(freqs=freqs, sources={"V1": 1.0})
    assert isinstance(model, SmallSignalModel)
    assert isinstance(result, SmallSignalResponse)
    h = 1 / (1 + 2j * jnp.pi * freqs * 1e-10)
    dx, dy = result.quadratures("out")
    assert jnp.allclose(dx, 2 * jnp.sqrt(1e-9) * h, rtol=1e-7, atol=1e-16)
    assert jnp.allclose(dy, 3 * jnp.sqrt(1e-9) * h, rtol=1e-7, atol=1e-16)
    power = result.power("out", field_unit="sqrt(W)")
    assert jnp.allclose(power, 1e-8 * h, rtol=1e-7, atol=1e-20)
    assert jnp.allclose(jnp.angle(power[2]), -jnp.pi / 4)
    assert result.port_map == circuit.port_map
    assert result.solution.shape == (len(freqs), 2 * circuit.sys_size)
    upper, lower = result.sidebands("out")
    assert jnp.allclose(upper + jnp.conj(lower), dx)
    assert jnp.allclose((upper - jnp.conj(lower)) / 1j, dy)
    index = circuit.port_map["out"]
    carrier = model.y_dc[index] + 1j * model.y_dc[index + circuit.sys_size]
    assert jnp.allclose(power, 2 * (jnp.conj(carrier) * upper + carrier * jnp.conj(lower)), rtol=1e-10, atol=1e-20)


def test_real_response_alias_and_voltage_difference() -> None:
    circuit = _rc()
    model = circuit.small_signal()
    freqs = jnp.array([0.0, 10.0, 1000.0])
    sources = {"V": 1 + 2j, "I": -2e-3j}
    actual = model(freqs=freqs, sources=sources)
    expected = circuit.ac_response(freqs=freqs, sources=sources)
    assert jnp.allclose(actual.solution, expected["solution"])
    assert jnp.array_equal(actual.solution, model.response(freqs=freqs, sources=sources).solution)
    assert jnp.allclose(jax.jit(model)(freqs=freqs, sources=sources).solution, actual.solution)
    assert jnp.allclose(jax.jit(model.response)(freqs=freqs, sources=sources).solution, actual.solution)
    assert jnp.allclose(actual.voltage("R,p1", "R,p2"), actual.voltage("R,p1") - actual.voltage("R,p2"))
    _, imaginary_coordinate = actual.quadratures("R,p2")
    assert jnp.all(imaginary_coordinate == 0)
    assert jnp.all(model(freqs=freqs, sources={}).solution == 0)


def test_carrier_rotation_changes_field_but_preserves_power() -> None:
    circuit = _optical()
    freqs = _frequencies()
    base = circuit.small_signal()(freqs=freqs, sources={"V1": 1.0})
    phase = 0.73
    rotated = circuit.small_signal(params={"laser.phase": phase})(freqs=freqs, sources={"V1": 1.0})
    dx, dy = base.quadratures("out")
    rx, ry = rotated.quadratures("out")
    assert jnp.allclose(rx, jnp.cos(phase) * dx - jnp.sin(phase) * dy, rtol=1e-7, atol=1e-16)
    assert jnp.allclose(ry, jnp.sin(phase) * dx + jnp.cos(phase) * dy, rtol=1e-7, atol=1e-16)
    assert jnp.allclose(rotated.power("out", field_unit="sqrt(W)"), base.power("out", field_unit="sqrt(W)"), rtol=1e-7, atol=1e-20)


def test_jit_vmap_sources_and_full_result() -> None:
    model = _optical().small_signal()
    freqs = _frequencies()
    amplitudes = jnp.array([[1 + 2j, 0], [0, -0.4j], [1 + 2j, -0.4j]])

    def run(amplitudes: jax.Array) -> SmallSignalResponse:
        return model(freqs=freqs, sources={"V1": amplitudes[0], "V2": amplitudes[1]})

    results = jax.jit(jax.vmap(run))(amplitudes)
    power = results.power("out", field_unit="sqrt(W)")
    assert power.shape == (3, len(freqs))
    assert jnp.allclose(power[2], power[0] + power[1], rtol=1e-10, atol=1e-20)
    baseline = model(freqs=freqs, sources={"V1": 1.0})
    assert jnp.allclose(power[0], (1 + 2j) * baseline.power("out", field_unit="sqrt(W)"), rtol=1e-10, atol=1e-20)
    assert jnp.array_equal(results.freqs, jnp.broadcast_to(freqs, (3, len(freqs))))
    assert jnp.all(results.y_dc == model.y_dc)
    # Passing the whole model into jit must work, not only capturing a closure.
    compiled = jax.jit(lambda m: m(freqs=freqs, sources={"V1": 1.0}))(model)
    assert jnp.allclose(compiled.solution, baseline.solution)


def test_vmap_parameter_linearizations_and_frequency_alignment() -> None:
    circuit = _optical()
    bias = circuit.dc()
    taus = jnp.array([1e-10, 2e-10])
    grids = jnp.array([[0.0, 1e9], [2e9, 3e9]])

    def run(tau: jax.Array, freqs: jax.Array) -> SmallSignalResponse:
        model = circuit.small_signal(y_dc=bias, params={"probe.tau": tau})
        return model(freqs=freqs, sources={"V1": 1.0})

    results = jax.jit(jax.vmap(run))(taus, grids)
    expected = 1e-8 / (1 + 2j * jnp.pi * grids * taus[:, None])
    assert jnp.array_equal(results.freqs, grids)
    assert jnp.allclose(results.power("out", field_unit="sqrt(W)"), expected, rtol=1e-7, atol=1e-20)


def test_optical_drive_keeps_rf_phase_separate_from_field_coordinates() -> None:
    model = _optical().small_signal()
    freqs = _frequencies()
    scalar = model(freqs=freqs, sources={"laser": 1j})
    x, y = scalar.quadratures("laser,p1")
    assert jnp.allclose(x, 1j)
    assert jnp.allclose(y, 0)
    explicit = jax.jit(lambda drive: model(freqs=freqs, sources={"laser": drive}))(QuadratureDrive(x=1j, y=2 - 3j))
    x, y = explicit.quadratures("laser,p1")
    assert jnp.allclose(x, 1j)
    assert jnp.allclose(y, 2 - 3j)


def test_dc_derivative_and_fixed_snapshot() -> None:
    circuit = _optical()
    model = circuit.small_signal()
    result = model(freqs=jnp.array([0.0]), sources={"V1": 1.0})
    epsilon = 1e-5
    index = circuit.port_map["out"]
    n = circuit.sys_size

    def dc_power(voltage: float) -> jax.Array:
        bias = circuit.dc(params={"V1.dc": voltage})
        return bias[index] ** 2 + bias[index + n] ** 2

    derivative = (dc_power(epsilon) - dc_power(-epsilon)) / (2 * epsilon)
    assert jnp.allclose(result.power("out", field_unit="sqrt(W)")[0], derivative, rtol=1e-7, atol=1e-20)
    circuit.small_signal(params={"V1.dc": 0.5})
    assert jnp.array_equal(model.y_dc, result.y_dc)
    assert jnp.array_equal(model(freqs=result.freqs, sources={"V1": 1.0}).solution, result.solution)


@pytest.mark.parametrize("sources", [{"unknown": 1}, {"load": 1}, {"V1": jnp.ones(2)}, {"laser": QuadratureDrive(y=jnp.ones(2))}])
def test_invalid_sources(sources: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="source"):
        _optical().small_signal()(freqs=jnp.array([1.0]), sources=sources)


@pytest.mark.parametrize("freqs", [1.0, [[1.0]], [1j]])
def test_invalid_frequencies(freqs: object) -> None:
    with pytest.raises(ValueError, match="frequencies"):
        _optical().small_signal()(freqs=freqs, sources={})


def test_invalid_bias_observables_and_parameter_batches() -> None:
    circuit = _optical()
    with pytest.raises(ValueError, match="y_dc"):
        circuit.small_signal(y_dc=jnp.zeros(circuit.sys_size))
    with pytest.raises(ValueError, match="does not batch"):
        circuit.small_signal(params={"probe.tau": jnp.ones(2)})
    result = circuit.small_signal()(freqs=jnp.array([0.0]), sources={})
    with pytest.raises(KeyError, match="Unknown"):
        result.voltage("missing")
    with pytest.raises(TypeError, match="field_unit"):
        result.power("out")
    with pytest.raises(ValueError, match="sqrt"):
        result.power("out", field_unit="V")
    real = _rc().small_signal()
    with pytest.raises(ValueError, match="complex-mode"):
        real(freqs=jnp.array([0.0]), sources={"V": QuadratureDrive(x=1.0)})
    with pytest.raises(NotImplementedError, match="optical"):
        real(freqs=jnp.array([0.0]), sources={}).power("R,p2", field_unit="sqrt(W)")


def test_preparation_does_not_solve(monkeypatch: pytest.MonkeyPatch) -> None:
    circuit = _optical()

    def no_dc(*_args: object, **_kwargs: object) -> None:
        pytest.fail("preparation must not solve")

    monkeypatch.setattr(type(circuit), "dc", no_dc)
    assert circuit.prepare_analysis("small_signal") is circuit


@component(ports=("input", "output"), states=("branch",), holomorphic=False)
def _delayed_probe(signals: Signals, tau: float = 2e-10) -> PhysicsReturn:
    value = signals.at_delay(tau).input
    return {
        "input": 0j,
        "output": signals.branch,
        "branch": signals.output - (2 + 3j) * jnp.real(value),
    }, {}


@pytest.mark.parametrize("complex_mode", [False, True])
def test_delay_preserves_rf_phase_with_reusable_model(*, complex_mode: bool) -> None:
    @component(ports=("input", "output"), states=("branch",), holomorphic=True)
    def delay(signals: Signals, tau: float = 2e-10) -> PhysicsReturn:
        return {
            "input": 0.0,
            "output": signals.branch,
            "branch": signals.output - signals.at_delay(tau).input,
        }, {}

    circuit = compile_circuit(
        {
            "instances": {
                "V": {"component": "voltage"},
                "delay": {"component": "delay"},
                "R": {"component": "load"},
                "GND": {"component": "ground"},
            },
            "connections": {
                "V,p1": "delay,input",
                "delay,output": "R,p1",
                "GND,p1": ("V,p2", "R,p2"),
            },
            "ports": {},
        },
        {"voltage": WaveformVoltageSource, "delay": _delayed_probe if complex_mode else delay, "load": Resistor},
        is_complex=complex_mode,
    )
    model = circuit.small_signal()
    freqs = jnp.array([1e8, 1e9, 3e9])
    result = jax.jit(lambda m, f: m(freqs=f, sources={"V": 1j}))(model, freqs)
    expected = 1j * jnp.exp(-2j * jnp.pi * freqs * 2e-10)
    dx, dy = result.quadratures("delay,output")
    assert jnp.allclose(dx, (2 if complex_mode else 1) * expected, rtol=1e-7, atol=1e-12)
    assert jnp.allclose(dy, 3 * expected if complex_mode else 0, rtol=1e-7, atol=1e-12)


@pytest.mark.parametrize("complex_mode", [False, True])
def test_frequency_dependent_admittance_matches_analytic_divider(*, complex_mode: bool) -> None:
    from circulax.s_transforms import fdomain_component

    @fdomain_component(ports=("p1", "p2"))
    def skin_resistor(f: jax.Array, resistance: float = 100.0) -> jax.Array:
        admittance = 1 / (resistance + jnp.sqrt(jnp.abs(f)))
        return admittance * jnp.array([[1.0, -1.0], [-1.0, 1.0]])

    circuit = compile_circuit(
        {
            "instances": {
                "V": {"component": "voltage"},
                "skin": {"component": "skin"},
                "R": {"component": "load", "settings": {"R": 1000.0}},
                "GND": {"component": "ground"},
            },
            "connections": {"V,p1": "skin,p1", "skin,p2": "R,p1", "GND,p1": ("V,p2", "R,p2")},
            "ports": {},
        },
        {"voltage": WaveformVoltageSource, "skin": skin_resistor, "load": Resistor},
        is_complex=complex_mode,
    )
    freqs = jnp.array([1.0, 1e3, 1e6])
    model = circuit.small_signal()
    drive = QuadratureDrive(x=1j, y=2 - 3j) if complex_mode else 1j
    result = jax.jit(lambda m: m(freqs=freqs, sources={"V": drive}))(model)
    expected = 1000 / (1100 + jnp.sqrt(freqs))
    dx, dy = result.quadratures("R,p1")
    assert jnp.allclose(dx, 1j * expected, rtol=1e-7)
    assert jnp.allclose(dy, (2 - 3j) * expected if complex_mode else 0, rtol=1e-7)


def test_source_gradient_preserves_complex_power_response() -> None:
    model = _optical().small_signal()

    def power_gain(amplitude: jax.Array) -> jax.Array:
        response = model(freqs=jnp.array([1e9]), sources={"V1": amplitude})
        return jnp.real(response.power("out", field_unit="sqrt(W)")[0])

    expected = jnp.real(1e-8 / (1 + 2j * jnp.pi * 1e9 * 1e-10))
    assert jnp.allclose(jax.jit(jax.grad(power_gain))(1.0), expected, rtol=1e-7, atol=1e-20)


@pytest.mark.parametrize("tau", [0.0, 1e-10])
def test_complex_frequency_domain_field_coupling_preserves_rf_phase(tau: float) -> None:
    from circulax.s_transforms import fdomain_component

    @fdomain_component(ports=("p1",))
    def optical_load(f: jax.Array, admittance: complex = 2 + 3j, time_constant: float = tau) -> jax.Array:
        return jnp.array([[admittance * (1 + 2j * jnp.pi * f * time_constant)]])

    circuit = compile_circuit(
        {
            "instances": {"I": {"component": "current"}, "load": {"component": "load"}, "GND": {"component": "ground"}},
            "connections": {"I,p1": "load,p1", "I,p2": "GND,p1"},
            "ports": {},
        },
        {"current": WaveformCurrentSource, "load": optical_load},
        is_complex=True,
    )
    amplitude = 1 + 2j
    freqs = jnp.array([1.0, 1e9])
    result = circuit.small_signal()(freqs=freqs, sources={"I": -amplitude})
    dx, dy = result.quadratures("load,p1")
    temporal_response = 1 / (1 + 2j * jnp.pi * freqs * tau)
    assert jnp.allclose(dx, amplitude * jnp.real(1 / (2 + 3j)) * temporal_response, rtol=1e-8)
    assert jnp.allclose(dy, amplitude * jnp.imag(1 / (2 + 3j)) * temporal_response, rtol=1e-8)


@pytest.mark.parametrize("complex_mode", [False, True])
def test_frequency_domain_capacitor_preserves_electrical_rf_phase(*, complex_mode: bool) -> None:
    from circulax.s_transforms import fdomain_component

    @fdomain_component(ports=("p1", "p2"))
    def capacitor(f: jax.Array, C: float = 1e-6) -> jax.Array:
        return 2j * jnp.pi * f * C * jnp.array([[1.0, -1.0], [-1.0, 1.0]])

    primitive = _rc()
    circuit = compile_circuit(
        primitive.source_netlist,
        {"voltage": WaveformVoltageSource, "current": WaveformCurrentSource, "resistor": Resistor, "capacitor": capacitor},
        is_complex=complex_mode,
    )
    freqs = jnp.array([0.0, 10.0, 1000.0])
    result = circuit.small_signal()(freqs=freqs, sources={"V": 1 + 2j})
    expected = (1 + 2j) / (1 + 2j * jnp.pi * freqs * 1e-3)
    dx, dy = result.quadratures("R,p2")
    assert jnp.allclose(dx, expected, rtol=1e-7)
    assert jnp.allclose(dy, 0.0, atol=1e-12)


def test_jit_vmap_includes_automatic_dc_bias() -> None:
    circuit = _rc()
    freqs = jnp.array([10.0, 100.0])

    def run(bias: jax.Array) -> SmallSignalResponse:
        return circuit.small_signal(params={"V.dc": bias})(freqs=freqs, sources={"V": 1j})

    biases = jnp.array([0.0, 3.0])
    result = jax.jit(jax.vmap(run))(biases)
    assert jnp.allclose(result.y_dc[:, circuit.port_map["R,p2"]], biases)
    assert jnp.allclose(result.voltage("R,p2"), 1j / (1 + 2j * jnp.pi * freqs * 1e-3))
