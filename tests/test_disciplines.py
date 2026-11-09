"""Physical convergence bounds, custom domains, and analysis integration."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pytest

from circulax import Discipline, Nature, TolerancePolicy, compile_circuit, electrical, thermal
from circulax.components.base_component import component
from circulax.components.electronic import Capacitor, Resistor, VoltageSource
from circulax.solvers import transient as tran
from circulax.solvers.convergence import newton_solve


def resistor_circuit(*, resistance=1e12, policy=TolerancePolicy(), backend="dense"):
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "V": {"component": "v", "settings": {"V": 1.0}},
            "R": {"component": "r", "settings": {"R": resistance}},
        },
        "connections": {"V,p1": "R,p1", "V,p2": "GND,p1", "R,p2": "GND,p1"},
    }
    return compile_circuit(net, {"ground": lambda: 0, "v": VoltageSource, "r": Resistor}, backend=backend, tolerances=policy)


@pytest.mark.parametrize("backend", ["dense", "klu", "klu_split_linear", "klu_split"])
def test_picoamp_current_and_diagnostics(backend):
    circuit = resistor_circuit(backend=backend)
    y, ok = circuit.dc_checked()
    assert ok
    assert float(y[circuit.port_map["V,i_src"]]) == pytest.approx(-1e-12, abs=1e-15)
    report = circuit.dc_diagnostics(y)
    assert report["residual_error"] < 1
    assert report["update_error"] < 1
    layout = circuit.convergence_layout
    node, state = circuit.port_map["R,p1"], circuit.port_map["V,i_src"]
    assert layout.unknown_natures[node].units == "V"
    assert layout.equation_natures[node].units == "A"
    assert layout.unknown_natures[state].units == "A"
    assert layout.equation_natures[state].units == "V"


def test_custom_discipline_has_no_registry_and_is_immutable():
    velocity = Nature("velocity", "m/s", 1e-8)
    force = Nature("force", "N", 1e-10)
    mechanical = Discipline("mechanical", velocity, force)

    @component(ports=("p", "n"), discipline=mechanical)
    def Damper(signals, drag=2.0):
        f = drag * (signals.p - signals.n)
        return {"p": f, "n": -f}, {}

    @component(ports=("p", "n"), discipline=mechanical)
    def Drive(signals, force=0.1):
        return {"p": -force, "n": force}, {}

    net = {
        "instances": {"GND": {"component": "ground"}, "D": {"component": "damper"}, "F": {"component": "drive"}},
        "connections": {"D,p": "F,p", "D,n": "GND,p1", "F,n": "GND,p1"},
    }
    c = compile_circuit(net, {"damper": Damper, "drive": Drive, "ground": lambda: 0}, backend="dense", tolerances=TolerancePolicy())
    y, ok = c.dc_checked()
    assert ok
    assert y[c.port_map["D,p"]] == pytest.approx(0.05, abs=1e-9)
    with pytest.raises(dataclasses.FrozenInstanceError):
        velocity.atol = 1.0


def test_mixed_domain_shared_ground_and_port_overrides():
    @component(ports=("p", "n", "hot", "cold"), discipline=electrical, port_disciplines={"hot": thermal, "cold": thermal})
    def Heater(signals, resistance=100.0):
        i = (signals.p - signals.n) / resistance
        heat = (signals.p - signals.n) * i
        return {"p": i, "n": -i, "hot": -heat, "cold": heat}, {}

    @component(ports=("hot", "cold"), discipline=thermal)
    def HeatSink(signals, resistance=100.0):
        p = (signals.hot - signals.cold) / resistance
        return {"hot": p, "cold": -p}, {}

    net = {
        "instances": {
            "GND": {"component": "ground"},
            "V": {"component": "v", "settings": {"V": 2.0}},
            "H": {"component": "heater"},
            "T": {"component": "sink"},
        },
        "connections": {"V,p1": "H,p", "V,p2": "GND,p1", "H,n": "GND,p1", "H,cold": "GND,p1", "T,cold": "GND,p1", "H,hot": "T,hot"},
    }
    c = compile_circuit(
        net,
        {"ground": lambda: 0, "v": VoltageSource, "heater": Heater, "sink": HeatSink},
        backend="dense",
        tolerances=TolerancePolicy(),
    )
    y, ok = c.dc_checked()
    assert ok
    assert y[c.port_map["H,hot"]] == pytest.approx(4.0, abs=1e-6)
    assert c.dc_diagnostics(y)["residual_error"] < 1


def test_missing_metadata_and_bad_connections_rejected():
    @component(ports=("p1", "p2"))
    def Unspecified(signals):
        return {"p1": signals.p1, "p2": signals.p2}, {}

    @component(ports=("p1", "p2"), discipline=thermal)
    def Thermal(signals):
        return {"p1": signals.p1, "p2": signals.p2}, {}

    for model, error in [(Unspecified, "Missing.*nature"), (Thermal, "Incompatible disciplines")]:
        net = {
            "instances": {"GND": {"component": "ground"}, "V": {"component": "v"}, "R": {"component": "r"}},
            "connections": {"V,p1": "R,p1", "V,p2": "GND,p1", "R,p2": "GND,p1"},
        }
        with pytest.raises(ValueError, match=error):
            compile_circuit(net, {"ground": lambda: 0, "v": VoltageSource, "r": model}, tolerances=TolerancePolicy())
        assert compile_circuit(net, {"ground": lambda: 0, "v": VoltageSource, "r": model}).convergence_layout is None


def test_overrides_and_conflicts():
    c = resistor_circuit(policy=TolerancePolicy(atol={"voltage": 1e-8}, unknown_atol={"R,p1": 2e-8}, residual_atol={"R,p1": 3e-14}))
    i = c.port_map["R,p1"]
    assert c.convergence_layout.unknown_atol[i] == 2e-8
    assert c.convergence_layout.residual_atol[i] == 3e-14
    with pytest.raises(ValueError, match="Unknown nature"):
        resistor_circuit(policy=TolerancePolicy(atol={"typo": 1e-5}))
    with pytest.raises(ValueError, match="Unknown variable"):
        resistor_circuit(policy=TolerancePolicy(unknown_atol={"typo": 1e-5}))
    with pytest.raises(ValueError, match="Conflicting tolerance"):
        resistor_circuit(policy=TolerancePolicy(unknown_atol={"R,p1": 1e-9, "V,p1": 1e-8}))
    with pytest.raises(ValueError, match="cannot be combined"):
        c.dc(rtol=1e-5)


@pytest.mark.parametrize("kwargs", [{"rtol": -1}, {"rtol": float("nan")}, {"atol": {"v": 0}}, {"local_error_factor": 0}])
def test_policy_validation(kwargs):
    with pytest.raises(ValueError):
        TolerancePolicy(**kwargs)


@pytest.mark.parametrize("large_residual", [True, False])
def test_damped_stagnation_is_not_convergence(large_residual):
    c = resistor_circuit()
    layout = c.convergence_layout
    y = jnp.zeros(c.sys_size)

    def update(y, _):
        raw = jnp.ones_like(y) if not large_residual else jnp.zeros_like(y)
        return y + 1e-30, (raw, jnp.asarray(True))

    def residual(y, _):
        return jnp.ones_like(y) if large_residual else jnp.zeros_like(y)

    sol = newton_solve(update, residual, y, layout=layout, atol=1e-6, rtol=1e-6, max_steps=3)
    assert sol.result != optx.RESULTS.successful


def test_phase_rotation_invariant_norm():
    layout = resistor_circuit().convergence_layout
    z = jnp.array([0, 0.2 + 0.3j, 1e-12j])

    def unroll(z):
        return jnp.concatenate([z.real, z.imag])

    a = layout.errors(unroll(z), unroll(z), unroll(z / 1e6), unroll(z / 1e8))
    b = layout.errors(
        unroll(z * jnp.exp(0.73j)), unroll(z * jnp.exp(0.73j)), unroll(z * jnp.exp(0.73j) / 1e6), unroll(z * jnp.exp(0.73j) / 1e8)
    )
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_jit_vmap_and_implicit_gradient():
    c = resistor_circuit(resistance=100.0)
    idx = c.port_map["V,i_src"]

    def value(r):
        return c.dc(R=r)[idx]

    assert jax.jit(value)(100.0) == pytest.approx(-0.01, abs=1e-12)
    assert jax.grad(value)(100.0) == pytest.approx(1e-4, rel=1e-6)
    np.testing.assert_allclose(jax.jit(jax.vmap(value))(jnp.array([100.0, 200.0])), [-0.01, -0.005], atol=1e-12)


@pytest.mark.parametrize("family", ["Vectorized", "Factorized", "Refactoring"])
@pytest.mark.parametrize("method", ["", "BDF2", "Trap", "SDIRK3"])
def test_transient_physical_convergence_across_families(family, method):
    backend = "dense" if family == "Vectorized" else ("klu_split_linear" if family == "Factorized" else "klu_split")
    c = resistor_circuit(resistance=100.0, backend=backend)
    solver_cls = getattr(tran, f"{method}{family}TransientSolver")
    sol = c.transient(t0=0.0, t1=0.02, dt0=0.01, transient_solver=solver_cls, throw=True)
    assert sol.ys[-1, c.port_map["V,i_src"]] == pytest.approx(-0.01, abs=1e-12)


def test_adaptive_rc_matches_analytic_solution():
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "V": {"component": "v", "settings": {"V": 1.0}},
            "R": {"component": "r", "settings": {"R": 1e3}},
            "C": {"component": "c", "settings": {"C": 1e-6}},
        },
        "connections": {"V,p1": "R,p1", "R,p2": "C,p1", "V,p2": "GND,p1", "C,p2": "GND,p1"},
    }
    c = compile_circuit(
        net,
        {"ground": lambda: 0, "v": VoltageSource, "r": Resistor, "c": Capacitor},
        backend="dense",
        tolerances=TolerancePolicy(local_error_factor=10.0),
    )
    y0 = jnp.zeros(c.sys_size).at[c.port_map["V,p1"]].set(1.0).at[c.port_map["V,i_src"]].set(-1e-3)
    sol = c.transient(t0=0.0, t1=0.003, dt0=1e-5, y0=y0, stepsize_controller=c.adaptive_controller(), throw=True, max_steps=10000)
    assert sol.ys[-1, c.port_map["C,p1"]] == pytest.approx(1 - np.exp(-3), abs=2e-3)
    controller = c.adaptive_controller()
    assert controller.atol[c.port_map["V,i_src"]] == pytest.approx(1e-11)
    assert controller.rtol == pytest.approx(1e-5)


def test_harmonic_balance_uses_physical_layout():
    c = resistor_circuit(resistance=100.0)
    time, freq = c.hb(freq=1.0, harmonics=1)
    np.testing.assert_allclose(time[:, c.port_map["V,i_src"]], -0.01, atol=1e-12)
    assert freq.shape[-1] == c.sys_size


def test_inductor_current_unknown_has_voltage_equation(simple_lrc_netlist):
    net, models = simple_lrc_netlist
    c = compile_circuit(net, models, backend="dense", tolerances=TolerancePolicy())
    idx = c.port_map["L1,i_L"]
    assert c.convergence_layout.unknown_natures[idx].units == "A"
    assert c.convergence_layout.equation_natures[idx].units == "V"


def test_optical_complex_physical_solve():
    from circulax.components.optical_passive import OpticalAttenuator, OpticalTermination
    from circulax.components.photonic import OpticalSource

    net = {
        "instances": {
            "GND": {"component": "ground"},
            "S": {"component": "source", "settings": {"power": 1e-3, "phase": 0.71}},
            "A": {"component": "attenuator", "settings": {"loss_dB": 3.0}},
            "T": {"component": "termination"},
        },
        "connections": {"S,p2": "GND,p1", "S,p1": "A,p1", "A,p2": "T,p1"},
    }
    c = compile_circuit(
        net,
        {"ground": lambda: 0, "source": OpticalSource, "attenuator": OpticalAttenuator, "termination": OpticalTermination},
        backend="dense",
        tolerances=TolerancePolicy(),
    )
    y, ok = c.dc_checked()
    assert ok
    output = c.port(y, "A,p2")
    assert output == pytest.approx(np.sqrt(1e-3) * 10 ** (-3 / 20) * np.exp(0.71j), abs=1e-9)
    assert c.dc_diagnostics(y)["residual_error"] < 1


def test_transient_parameter_gradient_matches_finite_difference():
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "V": {"component": "v", "settings": {"V": 1.0}},
            "R": {"component": "r", "settings": {"R": 1000.0}},
            "C": {"component": "c", "settings": {"C": 1e-6}},
        },
        "connections": {"V,p1": "R,p1", "R,p2": "C,p1", "V,p2": "GND,p1", "C,p2": "GND,p1"},
    }
    c = compile_circuit(
        net, {"ground": lambda: 0, "v": VoltageSource, "r": Resistor, "c": Capacitor}, backend="dense", tolerances=TolerancePolicy()
    )

    def value(r):
        y0 = jnp.zeros(c.sys_size).at[c.port_map["V,p1"]].set(1.0).at[c.port_map["V,i_src"]].set(-1 / r)
        sol = c.transient(t0=0.0, t1=1e-4, dt0=1e-5, y0=y0, params={"R.R": r}, max_steps=20, throw=True)
        return sol.ys[-1, c.port_map["C,p1"]]

    finite_difference = (value(1000.1) - value(999.9)) / 0.2
    assert jax.grad(value)(1000.0) == pytest.approx(float(finite_difference), rel=2e-4)


def test_hb_failure_is_reported_in_physical_mode():
    c = resistor_circuit(resistance=100.0)
    with pytest.raises(Exception, match="failed physical convergence"):
        c.hb(freq=1.0, harmonics=1, y0=jnp.zeros(c.sys_size), max_steps=1)


def test_complex_adaptive_controller_is_phase_invariant():
    from circulax.solvers.step_size import ComplexPIDController, complex_max_norm

    controller = ComplexPIDController(atol=jnp.full(4, 1e-9), rtol=1e-6, norm=complex_max_norm)

    def decision(phase):
        z = jnp.array([0j, jnp.exp(1j * phase)])
        e = jnp.array([0j, 1e-8j * jnp.exp(1j * phase)])
        y = jnp.concatenate([z.real, z.imag])
        error = jnp.concatenate([e.real, e.imag])
        _, state = controller.init(None, 0.0, 1.0, y, 0.1, None, None, 2)
        return controller.adapt_step_size(0.0, 0.1, y, y, None, error, 2, state)

    a, b = decision(0.0), decision(np.pi / 4)
    assert a[0] and b[0]
    assert a[2] == pytest.approx(float(b[2]), rel=1e-12)
    np.testing.assert_allclose(a[4][3:], b[4][3:], rtol=1e-12)


@pytest.mark.parametrize("all_fail", [False, True])
def test_hb_multistart_discards_failed_trials(monkeypatch, all_fail):
    from types import SimpleNamespace
    from circulax.solvers import harmonic_balance as hb

    c = resistor_circuit(resistance=100.0)

    def trial(update, residual, y0, **kwargs):
        success = (jnp.max(jnp.abs(y0)) < 1) & ~jnp.asarray(all_fail)
        result = optx.RESULTS.where(success, optx.RESULTS.successful, optx.RESULTS.max_steps_reached)
        return SimpleNamespace(value=y0, result=result)

    monkeypatch.setattr(hb, "newton_solve", trial)
    kwargs = {"freq": 1.0, "harmonics": 1, "osc_node": "R,p1", "amplitude_tries": jnp.array([0.1, 10.0])}
    if all_fail:
        with pytest.raises(Exception, match="failed physical convergence"):
            c.hb(**kwargs)
    else:
        time, freq = c.hb(**kwargs)
        assert jnp.max(jnp.abs(time[:, c.port_map["R,p1"]])) < 1
