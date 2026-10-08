"""Unified voltage/current sources: DC bias, SIN/PULSE/PWL boundaries, batching and circuit solves."""

import jax
import jax.numpy as jnp
import kfnetlist as kfnl
import numpy as np
import pytest

from circulax import compile_circuit
from circulax.components.electronic import (
    WAVE_DC,
    WAVE_PULSE,
    WAVE_PWL,
    WAVE_SIN,
    Capacitor,
    Diode,
    Resistor,
    VoltageSourceAC,
    WaveformCurrentSource,
    WaveformVoltageSource,
    waveform_value,
)
from circulax.netlist_io import parse_source
from circulax.solvers import setup_harmonic_balance
from circulax.solvers.linear import analyze_circuit
from circulax.solvers.transient import (
    BDF2VectorizedTransientSolver,
    SDIRK3FactorizedTransientSolver,
    SDIRK3VectorizedTransientSolver,
)

MODELS = {
    "vsrc": WaveformVoltageSource,
    "isrc": WaveformCurrentSource,
    "resistor": Resistor,
    "diode": Diode,
    "vac": VoltageSourceAC,
    "ground": lambda: 0,
}
_BASE = {
    "kind": WAVE_DC, "delay": 0.0, "offset": 0.0, "amplitude": 0.0, "freq": 0.0, "damping": 0.0, "phase": 0.0,
    "v1": 0.0, "v2": 0.0, "tr": 0.0, "tf": 0.0, "pw": 0.0, "per": 0.0,
    "pwl_t": (0.0, 0.0), "pwl_v": (0.0, 0.0), "repeat": -1.0, "tstep": 1e-9, "tstop": 1000.0,
}  # fmt: skip


def _w(t: float, **kw: object) -> float:
    return float(waveform_value(t, **{**_BASE, **kw}))


def _divider(settings_by_source: dict[str, dict], component: str = "vsrc") -> dict:
    """One source per instance, each driving its own 1 kΩ load to ground."""
    instances = {"GND": {"component": "ground"}}
    connections = {}
    for name, settings in settings_by_source.items():
        instances[name] = {"component": component, "settings": settings}
        instances[f"R{name}"] = {"component": "resistor", "settings": {"R": 1e3}}
        connections[f"{name},p1"] = f"R{name},p1"
        connections[f"{name},p2"] = "GND,p1"
        connections[f"R{name},p2"] = "GND,p1"
    return {"instances": instances, "connections": connections}


# --- waveform boundaries ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("SIN(0.9 1 1k)", 0.9),
        ("PULSE(2 3)", 2.0),
        ("PWL(0 7 1 8)", 7.0),
        ("DC 0 SIN(0.9 1 1k)", 0.0),
        ("DC 0.5 SIN(0.9 1 1k)", 0.5),
    ],
)
def test_operating_point_uses_override_or_waveform_at_zero(spec: str, expected: float) -> None:
    circuit = compile_circuit(_divider({"V1": parse_source(spec)}), MODELS)
    assert float(circuit.port(circuit.dc(), "V1,p1")) == pytest.approx(expected)


def test_constructor_remembers_an_explicit_zero_dc_override() -> None:
    assert WaveformVoltageSource(kind=WAVE_SIN).dc_given == 0.0
    assert WaveformVoltageSource(kind=WAVE_SIN, dc=0.0).dc_given == 1.0
    assert WaveformCurrentSource(kind=WAVE_PULSE, dc=0.0).dc_given == 1.0


def test_sin_holds_initial_phase_value_before_delay_then_damps() -> None:
    kw = {"kind": WAVE_SIN, "offset": 1.0, "amplitude": 2.0, "freq": 1e3, "delay": 1e-3, "phase": np.pi / 6, "damping": 100.0}
    assert _w(0.5e-3, **kw) == pytest.approx(1.0 + 2.0 * 0.5)  # held at offset + A sin(phase)
    assert _w(1e-3, **kw) == pytest.approx(2.0)
    t = 1.25e-3  # quarter period after delay
    assert _w(t, **kw) == pytest.approx(1.0 + 2.0 * np.exp(-100.0 * 0.25e-3) * np.sin(np.pi / 2 + np.pi / 6))


def test_pulse_boundaries_and_period() -> None:
    kw = {"kind": WAVE_PULSE, "v1": 1.0, "v2": 3.0, "delay": 1.0, "tr": 1.0, "pw": 2.0, "tf": 1.0, "per": 6.0}
    expect = {0.5: 1.0, 1.0: 1.0, 1.5: 2.0, 2.0: 3.0, 3.5: 3.0, 4.5: 2.0, 5.0: 1.0, 6.9: 1.0, 7.5: 2.0, 9.0: 3.0}
    for t, v in expect.items():
        assert _w(t, **kw) == pytest.approx(v), t


def test_pulse_zero_edges_and_period_use_analysis_defaults() -> None:
    pulse = {
        "kind": WAVE_PULSE,
        "v1": -1.0,
        "v2": 2.0,
        "delay": 1.0,
        "tr": 0.0,
        "pw": 2.0,
        "tf": 0.0,
        "per": 0.0,
        "tstep": 1.0,
        "tstop": 6.0,
    }
    expect = {1.0: -1.0, 1.5: 0.5, 2.0: 2.0, 4.0: 2.0, 4.5: 0.5, 5.0: -1.0, 7.5: 0.5}
    for t, value in expect.items():
        assert _w(t, **pulse) == pytest.approx(value)


def test_pwl_holds_ends_and_interpolates_with_delay() -> None:
    kw = {"kind": WAVE_PWL, "pwl_t": (1.0, 2.0, 4.0), "pwl_v": (1.0, 3.0, 0.0), "delay": 1.0}
    expect = {0.5: 1.0, 1.5: 1.0, 2.0: 1.0, 2.5: 2.0, 3.0: 3.0, 4.0: 1.5, 5.0: 0.0, 99.0: 0.0}
    for t, v in expect.items():
        assert _w(t, **kw) == pytest.approx(v), t


def test_pwl_repeat_loops_from_repeat_time() -> None:
    kw = {"kind": WAVE_PWL, "pwl_t": (0.0, 2.0, 4.0), "pwl_v": (0.0, 4.0, 0.0), "repeat": 2.0}
    expect = {1.0: 2.0, 4.0: 0.0, 4.5: 3.0, 5.0: 2.0, 6.0: 4.0, 7.0: 2.0}  # the loop jumps back to v(repeat)
    for t, v in expect.items():
        assert _w(t, **kw) == pytest.approx(v), t


def test_single_point_pwl_is_constant() -> None:
    assert _w(3.0, kind=WAVE_PWL, pwl_t=(1.0,), pwl_v=(2.5,)) == 2.5


def test_gradients_stay_finite_for_unselected_branches() -> None:
    """Every branch runs under select; unselected ones must not poison gradients."""
    for kind in (WAVE_DC, WAVE_SIN, WAVE_PULSE, WAVE_PWL):
        base = {**_BASE, "kind": kind, "v2": 1.0, "amplitude": 1.0, "pwl_t": (0.0, 1.0), "pwl_v": (0.0, 1.0)}

        def value(pw: float, tr: float, per: float, freq: float, tf: float, base: dict = base) -> jax.Array:
            return waveform_value(1.5, **{**base, "pw": pw, "tr": tr, "per": per, "freq": freq, "tf": tf})

        grads = jax.grad(value, argnums=(0, 1, 2, 3, 4))(2.0 if kind == WAVE_PULSE else 0.0, 0.0, 0.0, 1.0, 0.0)
        assert all(np.isfinite(g) for g in grads), kind


# --- circuit solves -----------------------------------------------------------------------------------------------


def test_voltage_source_dc_bias_is_independent_of_transient_waveform() -> None:
    sin = parse_source("DC 0.5 SIN(0.9 1 1k)")
    circuit = compile_circuit(_divider({"V1": sin}), MODELS)
    assert float(circuit.port(circuit.dc(), "V1,p1")) == pytest.approx(0.5)

    ts = jnp.array([0.0, 0.1e-3, 0.25e-3, 0.6e-3])
    sol = circuit.transient(t0=0.0, t1=0.6e-3, dt0=1e-6, saveat=ts)
    v = np.asarray(sol.ys[:, circuit.port_map["V1,p1"]])
    expect = 0.9 + np.sin(2 * np.pi * 1e3 * np.asarray(ts))
    np.testing.assert_allclose(v, expect, atol=1e-6)


def test_current_source_dc_and_pulse_drive_a_resistor() -> None:
    spec = parse_source("DC 1m PULSE(0 2m 0 0 0 1u)")
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "I1": {"component": "isrc", "settings": spec},
            "R1": {"component": "resistor", "settings": {"R": 1e3}},
        },
        "connections": {"I1,p2": "R1,p1", "I1,p1": "GND,p1", "R1,p2": "GND,p1"},
    }
    circuit = compile_circuit(net, MODELS)
    assert float(circuit.port(circuit.dc(), "R1,p1")) == pytest.approx(1.0, rel=1e-6)  # 1 mA * 1 kOhm
    sol = circuit.transient(t0=0.0, t1=2e-6, dt0=1e-8, saveat=jnp.array([0.5e-6, 1.5e-6]))
    v = np.asarray(sol.ys[:, circuit.port_map["R1,p1"]])
    np.testing.assert_allclose(v, [2.0, 0.0], atol=1e-6)  # pulse high, then back to v1


def test_instances_with_different_kinds_and_parameters_share_one_batched_group() -> None:
    settings = {
        "A": parse_source("DC 0.1 SIN(0 1 1k)", pwl_points=3),
        "B": parse_source("DC 0.2 SIN(0 2 2k)", pwl_points=3),
        "C": parse_source("DC 0.3 PULSE(0 5 0 0 0 1m)", pwl_points=3),
        "D": parse_source("DC 0.4 PWL(0 0 0.5m 3)", pwl_points=3),
    }
    circuit = compile_circuit(_divider(settings), MODELS)
    groups = [g for name, g in circuit.groups.items() if name.startswith("vsrc")]
    assert len(groups) == 1
    assert len(groups[0].index_map) == 4

    ts = jnp.array([0.25e-3])
    sol = circuit.transient(t0=0.0, t1=0.3e-3, dt0=1e-6, saveat=ts)
    got = {n: float(sol.ys[0, circuit.port_map[f"{n},p1"]]) for n in settings}
    assert got["A"] == pytest.approx(1.0, abs=1e-6)  # sin(2 pi 1k 0.25m) = 1
    assert got["B"] == pytest.approx(0.0, abs=1e-6)  # sin(pi) = 0
    assert got["C"] == pytest.approx(5.0, abs=1e-6)
    assert got["D"] == pytest.approx(1.5, abs=1e-6)
    dc = circuit.dc()
    assert [round(float(circuit.port(dc, f"{n},p1")), 6) for n in settings] == [0.1, 0.2, 0.3, 0.4]


def test_pwl_instances_of_different_length_split_groups_without_error() -> None:
    settings = {"A": parse_source("PWL(0 0 1m 1)"), "B": parse_source("PWL(0 0 1m 1 2m 0)")}
    circuit = compile_circuit(_divider(settings), MODELS)
    sol = circuit.transient(t0=0.0, t1=1.5e-3, dt0=1e-5, saveat=jnp.array([1.5e-3]))
    assert float(sol.ys[0, circuit.port_map["A,p1"]]) == pytest.approx(1.0, abs=1e-6)
    assert float(sol.ys[0, circuit.port_map["B,p1"]]) == pytest.approx(0.5, abs=1e-6)


def test_source_stepping_ramps_the_dc_bias_and_matches_plain_dc() -> None:
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "V1": {"component": "vsrc", "settings": parse_source("DC 2 SIN(5 1 1k)")},
            "R1": {"component": "resistor", "settings": {"R": 100.0}},
            "D1": {"component": "diode", "settings": {}},
        },
        "connections": {"V1,p1": "R1,p1", "R1,p2": "D1,p1", "D1,p2": "GND,p1", "V1,p2": "GND,p1"},
    }
    circuit = compile_circuit(net, MODELS)
    groups = circuit.groups
    assert groups["vsrc"].amplitude_param == "source_scale"
    solver = analyze_circuit(groups, circuit.sys_size, is_complex=False)
    y0 = jnp.zeros(circuit.sys_size)
    plain = solver.solve_dc(groups, y0)
    stepped = solver.solve_dc_source(groups, y0, n_steps=6)
    np.testing.assert_allclose(stepped, plain, atol=1e-6)
    assert float(plain[circuit.port_map["V1,p1"]]) == pytest.approx(2.0, abs=1e-9)  # waveform offset plays no part


def test_dc_sensitivity_flows_through_batched_source_settings() -> None:
    circuit = compile_circuit(_divider({"V1": parse_source("DC 1 SIN(0 1 1k)")}), MODELS)
    node = circuit.port_map["V1,p1"]
    grad = jax.grad(lambda d: circuit.dc(params={"V1.dc": d})[node])(1.0)
    assert float(grad) == pytest.approx(1.0)


_SCHEMES = [BDF2VectorizedTransientSolver, SDIRK3VectorizedTransientSolver, SDIRK3FactorizedTransientSolver]


@pytest.mark.parametrize("solver", _SCHEMES)
def test_transient_solver_schemes_follow_the_waveform_not_the_bias(solver: type) -> None:
    """Transient stage times see the waveform independently of the DC override."""
    circuit = compile_circuit(_divider({"V1": parse_source("DC 0.5 SIN(0.9 1 1k)")}), MODELS)
    ts = jnp.array([0.1e-3, 0.25e-3, 0.6e-3])
    sol = circuit.transient(t0=0.0, t1=0.6e-3, dt0=1e-6, saveat=ts, transient_solver=solver)
    v = np.asarray(sol.ys[:, circuit.port_map["V1,p1"]])
    np.testing.assert_allclose(v, 0.9 + np.sin(2 * np.pi * 1e3 * np.asarray(ts)), atol=1e-5)


def _hb_node_voltage(component: str, settings: dict) -> np.ndarray:
    circuit = compile_circuit(_divider({"V1": settings}, component=component), MODELS)
    run_hb = setup_harmonic_balance(circuit.groups, circuit.sys_size, freq=1e3, num_harmonics=3)
    y_time, _ = run_hb(circuit.dc())
    return np.asarray(y_time[:, circuit.port_map["V1,p1"]])


def test_harmonic_balance_ignores_the_dc_override_at_every_sample() -> None:
    reference = _hb_node_voltage("vac", {"V": 1.0, "freq": 1e3})
    for spec in ("SIN(0 1 1k)", "DC 0 SIN(0 1 1k)", "DC 0.9 SIN(0 1 1k)"):
        np.testing.assert_allclose(_hb_node_voltage("vsrc", parse_source(spec)), reference, atol=1e-9)


def test_kfnetlist_binding_batches_mixed_kinds_in_one_group() -> None:
    """Parsed settings bind through ``kfnetlist.Netlist`` exactly as through a dict."""
    specs = {"V1": "DC 0.1 SIN(0 1 1k)", "V2": "DC 0.2 PULSE(0 5 0 0 0 1m)", "V3": "DC 0.3 PWL(0 0 0.5m 3)"}
    nl = kfnl.Netlist()
    nl.create_inst(name="GND", kcl="", component="ground")
    gnd = kfnl.PortRef(instance="GND", port="p1")
    for name, spec in specs.items():
        nl.create_inst(name=name, kcl="", component="vsrc", settings=parse_source(spec, pwl_points=3))
        nl.create_inst(name=f"R{name}", kcl="", component="resistor", settings={"R": 1e3})
        nl.create_net(kfnl.PortRef(instance=name, port="p1"), kfnl.PortRef(instance=f"R{name}", port="p1"))
        nl.create_net(kfnl.PortRef(instance=name, port="p2"), kfnl.PortRef(instance=f"R{name}", port="p2"), gnd)
    circuit = compile_circuit(nl, MODELS)
    vgroups = [g for name, g in circuit.groups.items() if name.startswith("vsrc")]
    assert len(vgroups) == 1
    assert len(vgroups[0].index_map) == 3
    sol = circuit.transient(t0=0.0, t1=0.3e-3, dt0=1e-6, saveat=jnp.array([0.25e-3]))
    got = [float(sol.ys[0, circuit.port_map[f"{n},p1"]]) for n in specs]
    np.testing.assert_allclose(got, [1.0, 5.0, 1.5], atol=1e-6)


def test_source_stepping_scales_an_inferred_operating_point() -> None:
    circuit = compile_circuit(_divider({"V1": parse_source("SIN(2 1 1k)")}), MODELS)
    result = circuit.solver.solve_dc_source(circuit.groups, jnp.zeros(circuit.sys_size), n_steps=4)
    assert float(circuit.port(result, "V1,p1")) == pytest.approx(2.0)


def test_dc_parameter_updates_create_an_explicit_override() -> None:
    circuit = compile_circuit(_divider({"V1": parse_source("SIN(0.9 1 1k)")}), MODELS)
    assert float(circuit.port(circuit.dc(params={"V1.dc": 0.0}), "V1,p1")) == pytest.approx(0.0)
    assert float(circuit.port(circuit.dc(dc=0.0), "V1,p1")) == pytest.approx(0.0)
    assert float(circuit.port(circuit.dc(), "V1,p1")) == pytest.approx(0.9)
    assert float(jax.grad(lambda tf: circuit.dc(params={"V1.tf": tf})[circuit.port_map["V1,p1"]])(0.0)) == 0.0


def test_transient_source_defaults_are_bound_under_jit() -> None:
    circuit = compile_circuit(_divider({"V1": parse_source("PULSE(2 3)")}), MODELS)
    times = jnp.array([0.0, 0.1e-6, 0.2e-6, 0.5e-6])
    run = jax.jit(lambda: circuit.transient(t0=0.0, t1=0.5e-6, dt0=0.05e-6, tstep=0.2e-6, saveat=times))
    np.testing.assert_allclose(circuit.port(run().ys, "V1,p1"), [2.0, 2.5, 3.0, 3.0], atol=1e-7)


def test_transient_initializes_capacitor_from_waveform_instead_of_dc_override() -> None:
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "V1": {"component": "vsrc", "settings": parse_source("DC 0.5 SIN(0.9 1 1k)")},
            "R1": {"component": "resistor", "settings": {"R": 1000.0}},
            "C1": {"component": "capacitor", "settings": {"C": 1e-9}},
        },
        "connections": {"V1,p1": "R1,p1", "R1,p2": "C1,p1", "V1,p2": "GND,p1", "C1,p2": "GND,p1"},
    }
    circuit = compile_circuit(net, {**MODELS, "capacitor": Capacitor})
    assert float(circuit.port(circuit.dc(), "C1,p1")) == pytest.approx(0.5)
    solution = circuit.transient(t0=0.0, t1=1e-7, dt0=1e-8, saveat=jnp.array([0.0, 1e-7]))
    assert float(circuit.port(solution.ys, "C1,p1")[0]) == pytest.approx(0.9)
    supplied = circuit.transient(t0=0.0, t1=1e-7, dt0=1e-8, saveat=jnp.array([0.0]), y0=circuit.dc())
    assert float(circuit.port(supplied.ys, "C1,p1")[0]) == pytest.approx(0.5)


@pytest.mark.parametrize("double", [False, True])
def test_all_source_parameter_gradients_are_finite_without_analysis_defaults(double: bool) -> None:
    with jax.experimental.enable_x64(double):
        source = WaveformVoltageSource(kind=WAVE_SIN, offset=0.9, amplitude=1.0, freq=1e3)
        gradients = jax.grad(lambda params: -params(t=0.0)[0]["i_src"])(source)
        assert all(np.all(np.isfinite(leaf)) for leaf in jax.tree.leaves(gradients))
