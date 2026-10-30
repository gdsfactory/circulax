# Copyright (c) 2026 Circulax contributors.
# ruff: noqa: ANN001, ANN201, ANN202 -- concise numerical test fixtures.
"""Independent scattering and loaded-circuit checks for photonic primitives."""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax import compile_circuit
from circulax.components.electronic import Resistor
from circulax.components.optical_passive import (
    NonlinearOpticalWaveguide,
    OpticalAttenuator,
    OpticalInterface,
    OpticalPhaseShifter,
    OpticalTermination,
)
from circulax.components.photonic import DirectionalCoupler, Grating, OpticalSource, OpticalSourcePulse, OpticalWaveguide, Splitter


def scattering(model, z0=1):
    """Recover S by solving independently matched KCL + constitutive equations."""
    n = len(model.ports)

    def residual(y, drive):
        f, _ = model(y=y)
        return jnp.array(
            [f[p] + y[k] / z0 - 2 * drive[k] / np.sqrt(z0) for k, p in enumerate(model.ports)] + [f[s] for s in model.states]
        )

    zero = jnp.zeros(n + len(model.states), dtype=complex)
    matrix = np.asarray(jax.jacfwd(lambda y: residual(y, jnp.zeros(n)), holomorphic=True)(zero))
    solutions = [np.linalg.solve(matrix, -np.asarray(residual(zero, jnp.eye(n)[k]))) for k in range(n)]
    return np.column_stack([y[:n] / np.sqrt(z0) - np.eye(n)[k] for k, y in enumerate(solutions)])


@pytest.mark.parametrize("z0", [1.0, 50.0])
@pytest.mark.parametrize("loss", [0.0, 3.0, 20.0])
def test_matched_attenuator(z0, loss):
    actual = scattering(OpticalAttenuator(loss_dB=loss, z0=z0), z0)
    np.testing.assert_allclose(actual, [[0, 10 ** (-loss / 20)], [10 ** (-loss / 20), 0]], atol=1e-14)


@pytest.mark.parametrize("phase", [0.0, np.pi, 0.73])
def test_phase_and_termination(phase):
    np.testing.assert_allclose(
        scattering(OpticalPhaseShifter(phase=phase)), np.array([[0, 1], [1, 0]]) * np.exp(1j * phase), atol=1e-14
    )
    np.testing.assert_allclose(scattering(OpticalTermination()), [[0]], atol=1e-14)


@pytest.mark.parametrize("r", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("loss", [0.0, 2.0])
def test_interface_full_power_balance_reciprocity_and_endpoints(r, loss):
    delta, phase = 0.4, -0.2
    actual = scattering(OpticalInterface(reflectivity=r, reflection_phase=delta, phase=phase, loss_dB=loss))
    expected = (
        np.exp(1j * phase)
        * 10 ** (-loss / 20)
        * np.array([[np.sqrt(r) * np.exp(1j * delta), np.sqrt(1 - r)], [np.sqrt(1 - r), -np.sqrt(r) * np.exp(-1j * delta)]])
    )
    np.testing.assert_allclose(actual, expected, atol=2e-14)
    np.testing.assert_allclose(actual.T, actual, atol=1e-14)
    np.testing.assert_allclose(actual.conj().T @ actual, np.eye(2) * 10 ** (-loss / 10), atol=1e-14)
    drive = np.array([0.3 + 0.4j, -0.2 + 0.7j])
    np.testing.assert_allclose(np.linalg.norm(actual @ drive) ** 2, np.linalg.norm(drive) ** 2 * 10 ** (-loss / 10), atol=1e-14)


@pytest.mark.parametrize("coupling", [0.0, 0.2, 0.5, 1.0])
def test_directional_coupler_full_unitarity(coupling):
    actual = scattering(DirectionalCoupler(coupling=coupling))
    t, k = np.sqrt(1 - coupling), 1j * np.sqrt(coupling)
    expected = np.array([[0, 0, t, k], [0, 0, k, t], [t, k, 0, 0], [k, t, 0, 0]])
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(actual.conj().T @ actual, np.eye(4), atol=1e-14)
    np.testing.assert_allclose(actual, actual.T, atol=1e-14)


@pytest.mark.parametrize("ratio", [0.0, 0.3, 1.0])
def test_splitter_is_contractive_not_unitary(ratio):
    actual = scattering(Splitter(split_ratio=ratio))
    r, t = np.sqrt(ratio), 1j * np.sqrt(1 - ratio)
    np.testing.assert_allclose(actual, [[0, r, t], [r, 0, 0], [t, 0, 0]], atol=1e-14)
    np.testing.assert_allclose(np.linalg.svd(actual, compute_uv=False), [1, 1, 0], atol=1e-14)
    np.testing.assert_allclose(actual, actual.T, atol=1e-14)


def test_existing_lossless_static_endpoints():
    np.testing.assert_allclose(scattering(Grating()), [[0, 1], [1, 0]], atol=1e-14)
    np.testing.assert_allclose(scattering(OpticalWaveguide(length_um=0, loss_dB_cm=0)), [[0, 1], [1, 0]], atol=1e-14)


def matched_circuit(model, settings, *, pulse=False):
    return compile_circuit(
        {
            "instances": {
                "GND": {"component": "ground"},
                "S": {"component": "src", "settings": {"power": 4e-3, **({"delay": 0.3e-9, "rise": 0.02e-9} if pulse else {})}},
                "R": {"component": "res", "settings": {"R": 1.0}},
                "D": {"component": "dut", "settings": settings},
                "L": {"component": "load"},
            },
            "connections": {"S,p2": "GND,p1", "S,p1": "R,p1", "R,p2": "D,p1", "D,p2": "L,p1"},
            "ports": {"in": "D,p1", "out": "D,p2"},
        },
        {
            "ground": lambda: 0,
            "src": OpticalSourcePulse if pulse else OpticalSource,
            "res": Resistor,
            "dut": model,
            "load": OpticalTermination,
        },
        is_complex=True,
        backend="dense",
        g_leak=0,
    )


@pytest.mark.parametrize("tpa", [0.0, 4000.0])
def test_promoted_waveguide_dc_tpa_and_notebook_equations(tpa):
    settings = {"length_um": 5000.0, "loss_dB_cm": 3.0, "tpa_coeff": tpa, "neff": 0.0}
    circuit = matched_circuit(NonlinearOpticalWaveguide, settings)
    y = circuit.dc()
    length, alpha = 0.5, 3 * np.log(10) / 10
    leff = (1 - np.exp(-alpha * length)) / alpha
    expected = np.sqrt(1e-3) * 10 ** (-3 * length / 20) / np.sqrt(1 + tpa * 1e-3 * leff)
    np.testing.assert_allclose(circuit.port(y, "out"), expected, atol=2e-10)


def test_promoted_zero_length_transmission_and_zero_loss_derivatives():
    circuit = matched_circuit(NonlinearOpticalWaveguide, {"length_um": 0.0, "loss_dB_cm": 0.0, "tpa_coeff": 4000.0})
    np.testing.assert_allclose(circuit.port(circuit.dc(), "out"), np.sqrt(1e-3), atol=1e-11)
    y = jnp.array([0.1 + 0.03j, 0.05j, 0.1 + 0.03j, 0j])

    def loss(params):
        f, q = NonlinearOpticalWaveguide.solver_call(
            0.0, y, NonlinearOpticalWaveguide(length_um=params[0], loss_dB_cm=params[1], tpa_coeff=200.0)
        )
        return jnp.real(jnp.vdot(f, f) + jnp.vdot(q, q))

    point = jnp.array([5000.0, 0.0])
    np.testing.assert_allclose(jax.jacfwd(loss)(point), jax.jacrev(loss)(point), rtol=1e-12)
    assert np.isfinite(np.asarray(jax.jit(jax.grad(loss))(point))).all()
    assert np.isfinite(np.asarray(jax.vmap(loss)(jnp.array([[100.0, 0.0], [5000.0, 2.0]])))).all()


def test_promoted_delay_transient_against_shifted_nonlinear_pulse():
    settings = {"length_um": 5000.0, "loss_dB_cm": 3.0, "tpa_coeff": 4000.0, "neff": 0.0}
    circuit = matched_circuit(NonlinearOpticalWaveguide, settings, pulse=True)
    times = jnp.linspace(0, 0.6e-9, 121)
    tau = settings["length_um"] * 4 / 2.99792458e14
    a = np.sqrt(1e-3) / (1 + np.exp(-(np.asarray(times) - 0.3e-9 - tau) / 0.02e-9))
    leff = -np.expm1(-3 * np.log(10) / 10 * 0.5) / (3 * np.log(10) / 10)
    expected = a * 10 ** (-1.5 / 20) / np.sqrt(1 + 4000 * a * a * leff)
    errors = []
    for dt in [2e-12, 1e-12]:
        sol = circuit.transient(
            t0=0, t1=0.6e-9, dt0=dt, saveat=diffrax.SaveAt(ts=times), stepsize_controller=diffrax.ConstantStepSize(), max_steps=1000
        )
        errors.append(np.max(np.abs(np.asarray(circuit.port(sol.ys, "out")) - expected)))
    assert errors[1] < 4e-6
    assert errors[1] <= errors[0] * 0.7


def test_phase_matched_compiled_ac():
    circuit = matched_circuit(OpticalPhaseShifter, {"phase": 0.4, "loss_dB": 2.0})
    ac = circuit.ac_response(freqs=jnp.array([0.0, 1e9]), sources={"S": 2.0})
    np.testing.assert_allclose(ac["solution"][:, circuit.port_map["out"]], 10 ** (-2 / 20) * np.exp(0.4j), atol=1e-12)


@pytest.mark.parametrize(
    "instance",
    [
        OpticalAttenuator(loss_dB=-1.0),
        OpticalPhaseShifter(loss_dB=-1.0),
        OpticalTermination(z0=0.0),
        OpticalInterface(reflectivity=1.1),
        OpticalInterface(loss_dB=-1.0),
        NonlinearOpticalWaveguide(tpa_coeff=-1.0),
    ],
)
def test_passive_invalid_domains_rejected(instance):
    import equinox as eqx

    y = jnp.zeros(len(instance.ports) + len(instance.states), dtype=complex)
    with pytest.raises(eqx.EquinoxRuntimeError):
        eqx.filter_jit(lambda y: type(instance).solver_call(0.0, y, instance)[0])(y).block_until_ready()


def test_delayed_tpa_quadrature_response_at_finite_bias():
    from circulax.optical_analysis import quadrature_response

    length, loss, tpa = 5000.0, 3.0, 4000.0
    circuit = matched_circuit(NonlinearOpticalWaveguide, {"length_um": length, "loss_dB_cm": loss, "tpa_coeff": tpa, "neff": 0.0})
    frequencies = jnp.array([0.0, 1e9, 5e9])
    alpha = loss * np.log(10) / 10
    leff = -np.expm1(-alpha * length / 1e4) / alpha
    compression = 1 + tpa * 1e-3 * leff
    attenuation = 10 ** (-loss * length / 1e4 / 20)
    delay = length * 4 / 2.99792458e14
    expected_phase = np.exp(-2j * np.pi * np.asarray(frequencies) * delay)
    index = circuit.port_map["out"]
    for drive, observable, gain in [
        ((2.0, 0.0), "in_phase", attenuation / compression**1.5),
        ((0.0, 2.0), "quadrature", attenuation / np.sqrt(compression)),
    ]:
        response = quadrature_response(circuit, freqs=frequencies, sources={"S": drive})
        np.testing.assert_allclose(response[observable][:, index], gain * expected_phase, rtol=3e-8, atol=1e-10)


def test_delayed_tpa_length_gradient_matches_analytic_propagation():
    from circulax.optical_analysis import quadrature_response

    length, loss, tpa, frequency = 5000.0, 3.0, 4000.0, 1e9
    circuit = matched_circuit(NonlinearOpticalWaveguide, {"length_um": length, "loss_dB_cm": loss, "tpa_coeff": tpa, "neff": 0.0})
    bias = circuit.dc()
    index = circuit.port_map["out"]

    # Matched incident states are independent of length. Holding the dc vector
    # here preserves their exact bias and isolates the delay/propagation AD.
    def response(value):
        result = quadrature_response(
            circuit, freqs=jnp.array([frequency]), sources={"S": (2.0, 0.0)}, y_dc=bias, params={"D.length_um": value}
        )
        h = result["in_phase"][0, index]
        return jnp.array([h.real, h.imag])

    alpha = loss * np.log(10) / 10
    leff = -np.expm1(-alpha * length / 1e4) / alpha
    compression = 1 + tpa * 1e-3 * leff
    h = 10 ** (-loss * length / 1e4 / 20) / compression**1.5 * np.exp(-2j * np.pi * frequency * length * 4 / 2.99792458e14)
    logarithmic = (
        -loss * np.log(10) / (20 * 1e4)
        - 1.5 * tpa * 1e-3 * np.exp(-alpha * length / 1e4) / (1e4 * compression)
        - 2j * np.pi * frequency * 4 / 2.99792458e14
    )
    derivative = h * logarithmic
    expected = np.array([derivative.real, derivative.imag])
    forward = jax.jacfwd(response)(jnp.array(length))
    reverse = jax.jacrev(response)(jnp.array(length))
    np.testing.assert_allclose(forward, expected, rtol=3e-8, atol=1e-12)
    np.testing.assert_allclose(reverse, expected, rtol=3e-8, atol=1e-12)
