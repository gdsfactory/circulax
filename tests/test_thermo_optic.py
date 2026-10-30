# Copyright (c) 2026 Circulax contributors
"""Independent thermal energy balance and loaded thermo-optic checks."""

# ruff: noqa: ANN001, ANN201, ANN202, FBT003
import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax import attach_testbench, compile_circuit
from circulax.components.electronic import Resistor, VoltageSource
from circulax.components.photodetectors import OpticalCW
from circulax.components.thermo_optic import JouleHeater, ThermalPhaseShifter, thermo_optic_phase_shifter
from circulax.solvers.transient import BDF2VectorizedTransientSolver

jax.config.update("jax_enable_x64", True)


def fixture():
    package = thermo_optic_phase_shifter(
        resistance=100.0, thermal_resistance=1000.0, thermal_capacitance=1e-6, efficiency=0.02, phase=0.3
    )
    device = {
        "instances": {"TO": {"component": "to"}},
        "ports": {p: f"TO,{p}" for p in ("hp", "hn", "o1", "o2", "dt", "thermal_reference")},
    }
    net = attach_testbench(
        device,
        sources={
            "hp": {"name": "DRIVE", "component": "voltage", "settings": {"V": 1.0}},
            "o1": {"name": "LIGHT", "component": "light", "settings": {"power": 1e-3}},
        },
        loads={"o2": {"name": "MATCH", "component": "resistor", "settings": {"R": 1.0}}},
        gnd=["hn", "thermal_reference"],
    )
    return compile_circuit(
        net,
        {"to": package, "voltage": VoltageSource, "light": OpticalCW, "resistor": Resistor, "ground": lambda: 0},
        is_complex=True,
        backend="dense",
        rtol=1e-9,
        atol=1e-12,
    )


def test_heater_power_current_and_temperature_feedback():
    f, _ = JouleHeater(resistance=100, alpha=0.004)(hp=2.0, hn=0.0, dt=10.0)
    np.testing.assert_allclose(f["hp"], 2 / 104)
    np.testing.assert_allclose(f["dt"], -4 / 104)
    f, _ = JouleHeater()(hp=1 + 0.2j, hn=0.0, dt=3 + 0.1j)
    np.testing.assert_allclose(f["hp"], 0.01 + 0.002j)
    np.testing.assert_allclose(f["dt"], -0.01 + 0.0001j)


def test_loaded_dc_power_phase_and_temperature():
    c = fixture()
    y = c.dc(rtol=1e-12, atol=1e-15)
    dt = c.port(y, "dt")
    np.testing.assert_allclose(dt, 10.0, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(c.port(y, "DRIVE,i_src"), -0.01, atol=1e-10)
    field = c.port(y, "MATCH,p1")
    np.testing.assert_allclose(field, np.sqrt(0.001) * np.exp(0.5j), rtol=1e-7)
    assert abs(np.imag(dt)) < 1e-10
    assert abs(np.imag(c.port(y, "DRIVE,p1"))) < 1e-10


def test_thermal_step_and_cooling_time_constant_convergence():
    c = fixture()
    times = jnp.linspace(0.0, 0.005, 101)
    cold = c.dc(params={"DRIVE.V": 0.0})
    hot = c.dc()
    for y0, voltage, expected in (
        (cold, 1.0, 10 * (1 - np.exp(-np.asarray(times) / 0.001))),
        (hot, 0.0, 10 * np.exp(-np.asarray(times) / 0.001)),
    ):
        outputs = []
        for dt in (1e-5, 5e-6):
            sol = c.transient(
                t0=0.0,
                t1=0.005,
                dt0=dt,
                y0=y0,
                params={"DRIVE.V": voltage},
                saveat=diffrax.SaveAt(ts=times),
                transient_solver=BDF2VectorizedTransientSolver(linear_solver=c.solver, newton_rtol=1e-10, newton_atol=1e-12),
                max_steps=2000,
                throw=True,
            )
            temperature = np.real(c.port(sol.ys, "dt"))
            outputs.append(temperature)
            np.testing.assert_allclose(temperature, expected, atol=0.025, rtol=0.006)
            assert np.max(np.abs(np.imag(c.port(sol.ys, "dt")))) < 1e-9
        np.testing.assert_allclose(outputs[0], outputs[1], atol=0.02)


def test_reciprocal_optical_phase_both_directions_and_ad():
    m = ThermalPhaseShifter(efficiency=0.02, phase=0.3, loss_dB=2.0)
    a1, a2 = 0.03 + 0.01j, 0.02 - 0.04j
    t = 10 ** (-2 / 20) * np.exp(0.5j)
    f, _ = m(o1=a1 + t * a2, o2=a2 + t * a1, dt=10.0, a1=a1, a2=a2)
    assert abs(f["a1"]) < 1e-15
    assert abs(f["a2"]) < 1e-15
    np.testing.assert_allclose((a1 - f["o1"]) / a2, t, rtol=1e-12)
    np.testing.assert_allclose((a2 - f["o2"]) / a1, t, rtol=1e-12)

    def evaluate(x, eff):
        y = x[:5] + 1j * x[5:]
        model = ThermalPhaseShifter(efficiency=eff)
        f, q = model.solver_call(0.0, y, model)
        return jnp.concatenate((jnp.real(f), jnp.imag(f), jnp.real(q), jnp.imag(q)))

    x = jnp.array([0.03, 0.02, 10.0, 0.01, 0.02, 0.01, -0.01, 0.0, 0.01, 0.03])
    np.testing.assert_allclose(jax.jacfwd(evaluate)(x, 0.02), jax.jacrev(evaluate)(x, 0.02), atol=1e-12)
    y = jax.jit(jax.vmap(evaluate, in_axes=(0, None)))(jnp.stack([x, x]), 0.02)
    assert np.isfinite(y).all()
    gradient = jax.jit(jax.grad(lambda efficiency: jnp.sum(evaluate(x, efficiency) ** 2)))(0.02)
    assert np.isfinite(gradient)
    assert abs(gradient) > 0


@pytest.mark.parametrize(
    "settings", [{"resistance": 0.0}, {"thermal_resistance": -1.0}, {"thermal_capacitance": 0.0}, {"loss_dB": -1.0}]
)
def test_unsupported_parameter_domains_rejected(settings):
    with pytest.raises(ValueError):
        thermo_optic_phase_shifter(**settings)


def test_temperature_dependent_heater_equilibrium_and_jacobian():
    c = fixture()
    y = c.dc(params={"TO~HEATER.alpha": 0.004}, rtol=1e-12, atol=1e-15)
    # dt*(1+alpha*dt)=Rth*V²/R0, not a constant-power approximation.
    expected = (-1 + np.sqrt(1 + 4 * 0.004 * 10)) / (2 * 0.004)
    np.testing.assert_allclose(np.real(c.port(y, "dt")), expected, rtol=1e-9)

    def evaluate(x, r):
        m = JouleHeater(resistance=r, alpha=0.004)
        f, _ = m.solver_call(0.0, x[:3] + 1j * x[3:], m)
        return jnp.concatenate((jnp.real(f), jnp.imag(f)))

    x = jnp.array([1.0, 0.0, 10.0, 0.1, -0.1, 0.2])
    np.testing.assert_allclose(jax.jacfwd(evaluate)(x, 100.0), jax.jacrev(evaluate)(x, 100.0), atol=1e-12)
    assert jax.grad(lambda r: evaluate(x, r)[2])(100.0) > 0
