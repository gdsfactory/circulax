"""Physical one-way delay artifacts and composed solver regression tests."""

import json

import jax.numpy as jnp
import numpy as np
import pytest

from circulax import compile_circuit
from circulax.fitting import ModelCoefficients, ModelFitOptions, component_from_coefficients, fit_model


def static(D, delays):
    return ModelCoefficients(np.empty(0), np.empty((*np.shape(D), 0)), D, port_delays=delays)


@pytest.mark.parametrize("delays", [[-1, 0], [np.nan, 0], [np.inf, 0], [1], [[1, 2]]])
def test_invalid_delays(delays):
    with pytest.raises(ValueError, match="port_delays"):
        static(np.eye(2), delays)


def test_convention_and_persistence(tmp_path):
    c = static([[0.1, 0.3], [0.3, 0.2]], [0.03, 0.07])
    f = np.linspace(0, 10, 51)
    expected = c.D * np.exp(-2j * np.pi * f[:, None, None] * np.array([[0.06, 0.1], [0.1, 0.14]]))
    np.testing.assert_allclose(c.evaluate(f), expected)
    path = tmp_path / "model.npz"
    c.save(path)
    np.testing.assert_allclose(ModelCoefficients.load(path).evaluate(f), expected)
    with np.load(path, allow_pickle=False) as archive:
        header = json.loads(str(archive["header"]))
        header["version"] = 1
        np.savez(tmp_path / "old.npz", poles=c.poles, residues=c.residues, D=c.D, header=json.dumps(header))
    np.testing.assert_array_equal(ModelCoefficients.load(tmp_path / "old.npz").port_delays, [0, 0])


@pytest.mark.parametrize("amplitude", [1.0, 0.7])
def test_supplied_static_and_ac(amplitude):
    c = static([[0, amplitude], [amplitude, 0]], [0.03, 0.07])
    f = np.linspace(0, 10, 101)
    fitted = fit_model(c.evaluate(f), f, options=ModelFitOptions(delay_mode="supplied", port_delays=(0.03, 0.07)))
    assert len(fitted.poles) == 0
    np.testing.assert_allclose(fitted.evaluate(f), c.evaluate(f), atol=1e-14)
    model = component_from_coefficients(fitted)
    result = model.sp(
        ports=["p1", "p2"], freqs=jnp.asarray(f), y_dc=jnp.zeros(model.sys_size * (2 if model.solver.is_complex else 1))
    )
    np.testing.assert_allclose(result, c.evaluate(f), atol=1e-8)


def test_reflections_nested_and_distinct_models():
    a = static([[0.1, 0.3], [0.3, 0.2]], [0.03, 0.07])
    b = static([[0.2, 0.1], [0.1, 0.1]], [0.0, 0.02])
    models = {"a": component_from_coefficients(a), "b": component_from_coefficients(b)}
    net = {
        "instances": {"a": {"component": "a"}, "b": {"component": "b"}, "again": {"component": "a"}},
        "connections": {"b,p1": "again,p1", "b,p2": "again,p2"},
        "ports": {"p1": "a,p1", "p2": "a,p2"},
    }
    parent = compile_circuit(net, models, g_leak=0)
    nested = compile_circuit(
        {"instances": {"outer": {"component": "parent"}}, "connections": {}, "ports": {"p1": "outer,p1", "p2": "outer,p2"}},
        {"parent": parent},
        g_leak=0,
    )
    f = np.linspace(0, 10, 20)
    np.testing.assert_allclose(nested.sp(ports=["p1", "p2"], freqs=jnp.asarray(f)), a.evaluate(f), atol=1e-8)


def test_active_nonreciprocal_known_delays():
    c = static([[0.1, 0.1], [1.5, 0.2]], [0.01, 0.02])
    f = np.linspace(0, 10, 80)
    fitted = fit_model(c.evaluate(f), f, options=ModelFitOptions(delay_mode="supplied", port_delays=(0.01, 0.02), reciprocal=False))
    np.testing.assert_allclose(fitted.evaluate(f), c.evaluate(f), atol=1e-12)
    assert "enforcement" not in fitted.metadata


def test_auto_selection_and_sampling_decline():
    c = static([[0, 0.7], [0.7, 0]], [0.03, 0.03])
    f = np.linspace(0, 10, 101)
    fitted = fit_model(c.evaluate(f), f, options=ModelFitOptions(delay_mode="auto", auto_max_delay=0.1))
    assert len(fitted.poles) == 0
    np.testing.assert_allclose(fitted.port_delays, c.port_delays, atol=1e-12)
    assert fitted.metadata["auto_selection"]["selected"] == "delayed"
    baseline = fit_model(np.full((101, 1, 1), 0.2), f, options=ModelFitOptions(delay_mode="auto"))
    assert baseline.metadata["auto_selection"]["declined_reason"]
    with pytest.raises(ValueError, match="supplied poles"):
        fit_model(c.evaluate(f), f, initial_poles=np.array([-1]), options=ModelFitOptions(delay_mode="auto"))


@pytest.mark.parametrize("amplitude", [1.0, 0.7])
@pytest.mark.parametrize(
    ("delays", "step", "adaptive", "complex_state"),
    [
        ((0.03, 0.07), 0.002, False, False),
        ((0.001, 0.002), 0.005, False, True),
        ((0.001, 0.002), 0.005, True, False),
    ],
)
def test_loaded_dc_hb_transient(amplitude, delays, step, adaptive, complex_state):
    import copy

    import diffrax

    from circulax.components.electronic import Resistor, VoltageSource, VoltageSourceAC

    c = static([[0, amplitude], [amplitude, 0]], delays)
    total_delay = sum(delays)
    model = component_from_coefficients(c)
    net = {
        "instances": {
            "dut": {"component": "dut"},
            "source": {"component": "source", "settings": {"V": 1.0, "freq": 1.0}},
            "rs": {"component": "r", "settings": {"R": 50.0}},
            "rl": {"component": "r", "settings": {"R": 50.0}},
            "g": {"component": "ground"},
        },
        "connections": {"g,p1": ("source,p2", "rl,p2"), "source,p1": "rs,p1", "rs,p2": "dut,p1", "dut,p2": "rl,p1"},
    }
    circuit = compile_circuit(
        net,
        {"dut": model, "source": VoltageSourceAC, "r": Resistor, "ground": lambda: 0},
        g_leak=0,
        is_complex=True if complex_state else "auto",
    )
    np.testing.assert_allclose(circuit.dc(), 0, atol=1e-12)
    _, spectrum = circuit.hb(freq=1.0, harmonics=3)
    ratio = circuit.port(spectrum[1], "rl,p1") / circuit.port(spectrum[1], "rs,p2")
    np.testing.assert_allclose(ratio, amplitude * np.exp(-2j * np.pi * total_delay), atol=1e-7)
    t = jnp.linspace(0, 1.5, 301)
    controller = diffrax.PIDController(rtol=1e-5, atol=1e-7) if adaptive else diffrax.ConstantStepSize()
    solution = circuit.transient(t0=0, t1=1.5, dt0=step, saveat=t, max_steps=4000, throw=True, stepsize_controller=controller)
    actual = circuit.port(solution.ys, "rl,p1")
    expected = 0.5 * amplitude * np.where(t >= total_delay, np.sin(2 * np.pi * (t - total_delay)), 0)
    # A sub-step turn-on falls inside the first interpolation interval.
    # Bound its startup error separately from subsequent propagation.
    startup = np.asarray(t) <= step
    np.testing.assert_allclose(actual[startup], expected[startup], atol=1e-3)
    np.testing.assert_allclose(actual[~startup], expected[~startup], atol=5e-4)
    np.testing.assert_allclose(actual[np.asarray(t) < total_delay - step], 0, atol=1e-7)
    dc_net = copy.deepcopy(net)
    dc_net["instances"]["source"]["settings"] = {"V": 1.0}
    dc_circuit = compile_circuit(
        dc_net,
        {"dut": model, "source": VoltageSource, "r": Resistor, "ground": lambda: 0},
        g_leak=0,
        is_complex=True if complex_state else "auto",
    )
    np.testing.assert_allclose(dc_circuit.port(dc_circuit.dc(), "rl,p1"), 0.5 * amplitude, atol=1e-7)


def test_dynamic_known_delay_order_reduction():
    f = np.linspace(0, 10, 151)
    c = ModelCoefficients(
        np.array([-5.0]), np.array([[[0.0], [1.0]], [[1.0], [0.0]]]), np.array([[0.0, 0.2], [0.2, 0.0]]), port_delays=[0.03, 0.03]
    )
    limits = {"normalized_rmse": 1e-4, "max_absolute_error": 1e-3}
    baseline = fit_model(c.evaluate(f), f, options=ModelFitOptions(**limits))
    fitted = fit_model(c.evaluate(f), f, options=ModelFitOptions(delay_mode="supplied", port_delays=(0.03, 0.03), **limits))
    assert len(fitted.poles) < len(baseline.poles)
    np.testing.assert_allclose(fitted.evaluate_core(f), c.evaluate_core(f), atol=1e-5)


@pytest.mark.parametrize("case", ["no_bound", "alias", "reflection", "null", "resonance", "negative"])
def test_auto_declines_unjustified_delay(case):
    from circulax.fitting.delay_selection import fit_auto_delay

    f = np.linspace(0, 10, 101)
    c = static([[0, 0.5], [0.5, 0]], [0.01, 0.01])
    s = c.evaluate(f)
    bound = 0.1
    if case == "no_bound":
        bound = None
    elif case == "alias":
        bound = 6.0
    elif case == "reflection":
        s[:, 0, 0] = 0.1
    elif case == "null":
        s[50, 0, 1] = s[50, 1, 0] = 0
    elif case == "resonance":
        s[:, 0, 1] = s[:, 1, 0] = 0.5 / (1 + 2j * np.pi * f)
    elif case == "negative":
        s = s.conj()
    # Use a loose accuracy target to keep a baseline available for diagnosis.
    try:
        result = fit_auto_delay(
            s, f, 50.0, ModelFitOptions(delay_mode="auto", auto_max_delay=bound, normalized_rmse=1.0, max_absolute_error=1.0)
        )
    except ValueError as exc:
        assert "No constructible candidate" in str(exc)  # noqa: PT017
    else:
        assert result.metadata["auto_selection"]["declined_reason"]
        assert not np.any(result.port_delays)


def test_zero_delay_through_and_invalid_archive(tmp_path):
    c = static([[0, 1], [1, 0]], [0, 0])
    model = component_from_coefficients(c)
    f = np.linspace(0, 10, 10)
    np.testing.assert_allclose(
        model.sp(ports=["p1", "p2"], freqs=jnp.asarray(f), y_dc=jnp.zeros(model.sys_size)), c.evaluate(f), atol=1e-12
    )
    path = tmp_path / "bad.npz"
    c.save(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    arrays["port_delays"] = [-1.0, 0.0]
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="port_delays"):
        ModelCoefficients.load(path)


def test_overdeembedded_fixed_core_fails_accuracy():
    f = np.linspace(0, 10, 101)
    c = static([[0, 0.7], [0.7, 0]], [0.01, 0.01])
    with pytest.raises(ValueError, match="accuracy"):
        fit_model(
            c.evaluate(f),
            f,
            initial_poles=np.array([]),
            options=ModelFitOptions(delay_mode="supplied", port_delays=(0.1, 0.1), normalized_rmse=1e-4, max_absolute_error=1e-3),
        )


def test_reflected_transient_has_round_trip_delay():
    from circulax.components.electronic import Resistor, VoltageSourceAC

    coefficients = static([[0.5]], [0.05])
    net = {
        "instances": {
            "dut": {"component": "dut"},
            "source": {"component": "source", "settings": {"V": 1.0, "freq": 1.0}},
            "rs": {"component": "r", "settings": {"R": 50.0}},
            "g": {"component": "ground"},
        },
        "connections": {"g,p1": "source,p2", "source,p1": "rs,p1", "rs,p2": "dut,p1"},
    }
    circuit = compile_circuit(
        net,
        {"dut": component_from_coefficients(coefficients), "source": VoltageSourceAC, "r": Resistor, "ground": lambda: 0},
        g_leak=0,
    )
    t = jnp.linspace(0, 0.5, 101)
    solution = circuit.transient(t0=0, t1=0.5, dt0=0.002, saveat=t, max_steps=500, throw=True)
    incident = 0.5 * np.sin(2 * np.pi * t)
    reflection = 0.25 * np.where(t >= 0.1, np.sin(2 * np.pi * (t - 0.1)), 0)
    np.testing.assert_allclose(circuit.port(solution.ys, "rs,p2"), incident + reflection, atol=5e-4)
