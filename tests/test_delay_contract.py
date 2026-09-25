"""Analysis-independent contract tests for fixed propagation delay.

These tests use analytic time- and frequency-domain oracles. They deliberately
avoid the S-parameter conversion/fitting code so a shared implementation error
cannot make both the implementation and its expected value agree.
"""

from types import SimpleNamespace
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax.compiler import compile_netlist
from circulax.components.base_component import PhysicsReturn, Signals, component
from circulax.components.electronic import Resistor, TransmissionLine, VoltageSourceAC
from circulax.s_transforms import fdomain_component
from circulax.solvers import analyze_circuit, setup_harmonic_balance, setup_transient
from circulax.solvers.ac_sweep import setup_ac_sweep
from circulax.solvers.assembly import _interp_delayed, assemble_system_real
from circulax.solvers.harmonic_balance import _hb_residual, _periodic_delay_histories

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


@component(ports=("input", "output"), states=("branch",))
def FixedDelayConstraint(  # noqa: N802
    signals: Signals,
    tau: float = 0.25,
) -> PhysicsReturn:
    """Minimal real delay relation: output(t) = input(t - tau)."""
    return {
        "input": 0.0,
        "output": signals.branch,
        "branch": signals.output - signals.at_delay(tau).input,
    }, {}


@fdomain_component(ports=("p1",))
def ComplexAdmittance(f: float, conductance: float = 2.0, susceptance: float = 3.0) -> jax.Array:  # noqa: N802
    """Constant complex admittance used to isolate the complex HB stamp."""
    _ = f
    return jnp.array([[conductance + 1j * susceptance]])


def _compiled_delay(tau: float = 0.25) -> tuple[dict[str, Any], int, dict[str, int]]:
    net = {
        "instances": {
            "DUT": {"component": "delay", "settings": {"tau": tau}},
        },
        "connections": {},
        "ports": {"input": "DUT,input", "output": "DUT,output"},
    }
    return compile_netlist(net, {"delay": FixedDelayConstraint})


def test_inline_delay_is_inferred_from_component_physics() -> None:
    groups, _, _ = _compiled_delay(tau=0.375)
    group = groups["delay"]
    assert group.has_delay
    np.testing.assert_allclose(jax.vmap(group.tau_func)(group.params), [0.375])
    assert not hasattr(FixedDelayConstraint, "delay")


def test_multiple_distinct_inline_delays_are_rejected() -> None:
    @component(ports=("p1", "p2"))
    def MultipleDelays(signals: Signals, tau: float = 0.25) -> PhysicsReturn:  # noqa: N802
        first = signals.at_delay(tau).p1
        second = signals.at_delay(2 * tau).p2
        return {"p1": first, "p2": second}, {}

    net = {
        "instances": {"DUT": {"component": "delay", "settings": {}}},
        "connections": {"DUT,p1": "DUT,p2"},
    }
    with pytest.raises(ValueError, match="multiple distinct delays"):
        compile_netlist(net, {"delay": MultipleDelays})


def test_current_step_interpolation_value_and_jacobian() -> None:
    """A sub-step delay must contribute to the current Newton Jacobian."""
    hist_t = jnp.array([0.0, 1.0, jnp.inf])
    hist_y = jnp.array([[0.0], [2.0], [jnp.inf]])
    idx = jnp.array([[0]])
    tau = jnp.array([0.25])

    def delayed(current: jax.Array) -> jax.Array:
        return _interp_delayed(hist_t, hist_y, tau, idx, t1=2.0, current=current.reshape(1, 1))[0, 0]

    # Query is 75% through [t_previous, t_current].
    np.testing.assert_allclose(delayed(jnp.array(6.0)), 5.0)
    np.testing.assert_allclose(jax.grad(delayed)(jnp.array(6.0)), 0.75)

    def delayed_for_tau(tau_value: jax.Array) -> jax.Array:
        return _interp_delayed(
            hist_t,
            hist_y,
            tau_value.reshape(1),
            idx,
            t1=2.0,
            current=jnp.array([[6.0]]),
        )[0, 0]

    np.testing.assert_allclose(jax.grad(delayed_for_tau)(jnp.array(0.25)), -4.0)


def test_accepted_history_has_zero_current_jacobian() -> None:
    hist_t = jnp.array([0.0, 1.0, 2.0, jnp.inf])
    hist_y = jnp.array([[0.0], [2.0], [8.0], [jnp.inf]])
    idx = jnp.array([[0]])
    tau = jnp.array([1.5])

    def delayed(current: jax.Array) -> jax.Array:
        return _interp_delayed(hist_t, hist_y, tau, idx, t1=3.0, current=current.reshape(1, 1))[0, 0]

    np.testing.assert_allclose(delayed(jnp.array(100.0)), 5.0)
    np.testing.assert_allclose(jax.grad(delayed)(jnp.array(100.0)), 0.0)


def test_delay_is_identity_at_dc_in_residual_and_jacobian() -> None:
    groups, size, _ = _compiled_delay(tau=10.0)
    group = groups["delay"]
    input_idx, output_idx, branch_idx = np.asarray(group.var_indices[0])
    y = jnp.zeros(size).at[input_idx].set(1.25).at[output_idx].set(1.25)

    residual, _, values = assemble_system_real(y, groups, t1=0.0, dt=1.0)
    np.testing.assert_allclose(residual, 0.0, atol=1e-12)

    jac = np.zeros((size, size))
    np.add.at(
        jac,
        (np.asarray(group.jac_rows).reshape(-1), np.asarray(group.jac_cols).reshape(-1)),
        np.asarray(values),
    )
    np.testing.assert_allclose(jac[branch_idx, input_idx], -1.0)
    np.testing.assert_allclose(jac[branch_idx, output_idx], 1.0)


def test_periodic_delay_rotates_every_harmonic() -> None:
    K = 9
    fundamental = 2.0
    tau = 0.075
    t = np.arange(K) / (K * fundamental)
    signal = 0.2 + np.sin(2 * np.pi * fundamental * t) + 0.3 * np.cos(2 * np.pi * 3 * fundamental * t)
    group = SimpleNamespace(
        name="delay",
        has_delay=True,
        tau_func=lambda value: value,
        params=jnp.array([tau]),
        var_indices=jnp.array([[0]]),
    )

    shifted = _periodic_delay_histories(jnp.asarray(signal[:, None]), {"delay": group}, fundamental, is_complex=False)["delay"][
        :, 0, 0
    ]
    expected = 0.2 + np.sin(2 * np.pi * fundamental * (t - tau)) + 0.3 * np.cos(2 * np.pi * 3 * fundamental * (t - tau))
    np.testing.assert_allclose(shifted, expected, atol=1e-12)


def test_hb_delay_constraint_matches_multiharmonic_oracle() -> None:
    tau = 0.075
    fundamental = 2.0
    groups, size, _ = _compiled_delay(tau=tau)
    group = groups["delay"]
    input_idx, output_idx, _ = np.asarray(group.var_indices[0])
    K = 9
    t = np.arange(K) / (K * fundamental)
    input_signal = np.sin(2 * np.pi * fundamental * t) + 0.25 * np.cos(2 * np.pi * 3 * fundamental * t)
    output_signal = np.sin(2 * np.pi * fundamental * (t - tau)) + 0.25 * np.cos(2 * np.pi * 3 * fundamental * (t - tau))
    y_time = jnp.zeros((K, size))
    y_time = y_time.at[:, input_idx].set(input_signal)
    y_time = y_time.at[:, output_idx].set(output_signal)

    residual = _hb_residual(
        y_time,
        groups,
        jnp.asarray(t),
        2 * jnp.pi * fundamental,
        jnp.array([], dtype=jnp.int32),
    )
    np.testing.assert_allclose(residual, 0.0, atol=1e-11)


def test_complex_periodic_delay_does_not_assume_conjugate_symmetry() -> None:
    K = 9
    fundamental = 3.0
    tau = 0.04
    t = np.arange(K) / (K * fundamental)
    envelope = np.exp(2j * np.pi * fundamental * t) + 0.4j * np.exp(-2j * np.pi * 2 * fundamental * t)
    y_unrolled = jnp.column_stack([envelope.real, envelope.imag])
    group = SimpleNamespace(
        name="delay",
        has_delay=True,
        tau_func=lambda value: value,
        params=jnp.array([tau]),
        var_indices=jnp.array([[0]]),
    )

    shifted = _periodic_delay_histories(y_unrolled, {"delay": group}, fundamental, is_complex=True)["delay"][:, 0, 0]
    expected = np.exp(2j * np.pi * fundamental * (t - tau)) + 0.4j * np.exp(-2j * np.pi * 2 * fundamental * (t - tau))
    np.testing.assert_allclose(shifted, expected, atol=1e-12)


def test_complex_fdomain_hb_stamps_both_quadratures() -> None:
    """Complex f-domain HB must preserve Fourier phase and field quadrature."""
    net = {
        "instances": {"DUT": {"component": "admittance", "settings": {}}},
        "connections": {},
        "ports": {"p1": "DUT,p1"},
    }
    groups, size, ports = compile_netlist(net, {"admittance": ComplexAdmittance})
    K = 9
    fundamental = 2.0
    t = np.arange(K) / (K * fundamental)
    voltage = 0.7 * np.exp(2j * np.pi * fundamental * t) + 0.2j * np.exp(-2j * np.pi * 2 * fundamental * t)
    y_time = jnp.zeros((K, 2 * size))
    node = ports["DUT,p1"]
    y_time = y_time.at[:, node].set(voltage.real)
    y_time = y_time.at[:, node + size].set(voltage.imag)

    residual = _hb_residual(
        y_time,
        groups,
        jnp.asarray(t),
        2 * jnp.pi * fundamental,
        jnp.array([], dtype=jnp.int32),
        is_complex=True,
    )
    expected = (2.0 + 3.0j) * voltage
    np.testing.assert_allclose(residual[:, node], expected.real, atol=1e-12)
    np.testing.assert_allclose(residual[:, node + size], expected.imag, atol=1e-12)


def test_transmission_line_ac_uses_same_delay_contract() -> None:
    """The wave-variable stamp reproduces an ideal delayed S21 without S-to-Y."""
    tau = 2.5e-10
    z0 = 50.0
    attenuation = 0.97
    net = {
        "instances": {
            "DUT": {
                "component": "line",
                "settings": {
                    "tau": tau,
                    "z0": z0,
                    "attenuation": attenuation,
                },
            },
        },
        "connections": {},
        "ports": {"p1": "DUT,p1", "p2": "DUT,p2"},
    }
    groups, size, ports = compile_netlist(net, {"line": TransmissionLine})
    linear = analyze_circuit(groups, size, backend="dense")
    y_dc = linear.solve_dc(groups, jnp.zeros(size))
    run_ac = setup_ac_sweep(groups, size, [ports["DUT,p1"], ports["DUT,p2"]], z0=z0)
    freqs = jnp.array([1e6, 1e8, 7e8])
    scattering = run_ac(y_dc, freqs)
    expected = attenuation * jnp.exp(-2j * jnp.pi * freqs * tau)

    np.testing.assert_allclose(scattering[:, 1, 0], expected, atol=1e-10)
    np.testing.assert_allclose(scattering[:, 0, 1], expected, atol=1e-10)
    np.testing.assert_allclose(scattering[:, 0, 0], 0.0, atol=1e-10)
    np.testing.assert_allclose(scattering[:, 1, 1], 0.0, atol=1e-10)


def test_transmission_line_hb_matches_ac_phase() -> None:
    """The same wave-variable line produces the AC delay in HB."""
    fundamental = 1.0
    tau = 0.13
    z0 = 50.0
    attenuation = 0.9
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "V1": {
                "component": "source",
                "settings": {"V": 1.0, "freq": fundamental},
            },
            "RS": {"component": "resistor", "settings": {"R": z0}},
            "TL": {
                "component": "line",
                "settings": {"tau": tau, "z0": z0, "attenuation": attenuation},
            },
            "RL": {"component": "resistor", "settings": {"R": z0}},
        },
        "connections": {
            "GND,p1": ("V1,p2", "RL,p2"),
            "V1,p1": "RS,p1",
            "RS,p2": "TL,p1",
            "TL,p2": "RL,p1",
        },
    }
    models = {
        "ground": lambda: 0,
        "source": VoltageSourceAC,
        "resistor": Resistor,
        "line": TransmissionLine,
    }
    groups, size, ports = compile_netlist(net, models)
    linear = analyze_circuit(groups, size, backend="dense")
    y_dc = linear.solve_dc(groups, jnp.zeros(size))
    run_hb = setup_harmonic_balance(groups, size, fundamental, num_harmonics=3)
    _, spectrum = run_hb(y_dc, max_steps=20, rtol=1e-9, atol=1e-9)

    line_input = spectrum[1, ports["TL,p1"]]
    line_output = spectrum[1, ports["TL,p2"]]
    expected_ratio = attenuation * jnp.exp(-2j * jnp.pi * fundamental * tau)
    np.testing.assert_allclose(line_output / line_input, expected_ratio, atol=2e-7)
    np.testing.assert_allclose(spectrum[2:, ports["TL,p2"]], 0.0, atol=2e-8)


def test_transmission_line_transient_matches_same_phase_delay() -> None:
    """Transient uses the same line equations exercised by AC and HB."""
    fundamental = 1.0
    tau = 0.13
    z0 = 50.0
    attenuation = 0.9
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "V1": {
                "component": "source",
                "settings": {"V": 1.0, "freq": fundamental},
            },
            "RS": {"component": "resistor", "settings": {"R": z0}},
            "TL": {
                "component": "line",
                "settings": {"tau": tau, "z0": z0, "attenuation": attenuation},
            },
            "RL": {"component": "resistor", "settings": {"R": z0}},
        },
        "connections": {
            "GND,p1": ("V1,p2", "RL,p2"),
            "V1,p1": "RS,p1",
            "RS,p2": "TL,p1",
            "TL,p2": "RL,p1",
        },
    }
    models = {
        "ground": lambda: 0,
        "source": VoltageSourceAC,
        "resistor": Resistor,
        "line": TransmissionLine,
    }
    groups, size, ports = compile_netlist(net, models)
    linear = analyze_circuit(groups, size, backend="dense")
    y_dc = linear.solve_dc(groups, jnp.zeros(size))
    run_transient = setup_transient(groups, linear)
    sample_t = jnp.linspace(1.0, 1.5, 101)
    solution = run_transient(
        t0=0.0,
        t1=1.5,
        dt0=0.005,
        y0=y_dc,
        saveat=diffrax.SaveAt(ts=sample_t),
        max_steps=1000,
        throw=True,
    )

    expected = 0.5 * attenuation * jnp.sin(2 * jnp.pi * fundamental * (sample_t - tau))
    np.testing.assert_allclose(solution.ys[:, ports["TL,p2"]], expected, atol=5e-4, rtol=5e-4)
