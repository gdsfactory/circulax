"""Tests for the high-level Circuit / compile_circuit API."""

import jax
import jax.numpy as jnp
import pytest

from circulax import Circuit, compile_circuit
from circulax.compiler import compile_netlist
from circulax.components.electronic import Capacitor, Resistor, VoltageSource
from circulax.solvers.linear import analyze_circuit, backends


def test_compile_circuit_returns_circuit(simple_lrc_netlist):
    net_dict, models_map = simple_lrc_netlist
    circuit = compile_circuit(net_dict, models_map)

    assert isinstance(circuit, Circuit)
    assert circuit.sys_size > 0
    assert isinstance(circuit.port_map, dict)
    assert "V1,p2" in circuit.port_map
    assert circuit.solver.is_complex is False


def test_circuit_scalar_matches_direct_solver(simple_lrc_netlist):
    net_dict, models_map = simple_lrc_netlist

    groups, sys_size, _ = compile_netlist(net_dict, models_map)
    solver = analyze_circuit(groups, sys_size, is_complex=False)
    y_direct = solver.solve_dc(groups, jnp.zeros(sys_size))

    circuit = compile_circuit(net_dict, models_map)
    y_wrapped = circuit()

    assert y_wrapped.shape == (sys_size,)
    assert jnp.allclose(y_wrapped, y_direct, atol=1e-10)


def test_circuit_call_aliases_dc(simple_lrc_netlist):
    net_dict, models_map = simple_lrc_netlist
    circuit = compile_circuit(net_dict, models_map)

    assert jnp.allclose(circuit(), circuit.dc(), atol=1e-10)


def test_circuit_batched_param_vmaps(simple_lrc_netlist):
    net_dict, models_map = simple_lrc_netlist
    circuit = compile_circuit(net_dict, models_map)

    R_sweep = jnp.array([10.0, 20.0, 50.0, 100.0])
    ys = circuit(R=R_sweep)

    assert ys.shape == (R_sweep.shape[0], circuit.sys_size)
    assert jnp.all(jnp.isfinite(ys))


def test_circuit_mismatched_batch_shapes_raise(simple_lrc_netlist):
    net_dict, models_map = simple_lrc_netlist
    circuit = compile_circuit(net_dict, models_map)

    with pytest.raises(ValueError, match="same leading dim"):
        circuit(R=jnp.array([10.0, 20.0]), C=jnp.array([1e-11, 2e-11, 3e-11]))


def test_circuit_mixed_scalar_and_batched_param(simple_lrc_netlist):
    net_dict, models_map = simple_lrc_netlist
    circuit = compile_circuit(net_dict, models_map)

    R_sweep = jnp.array([10.0, 50.0, 100.0])
    ys = circuit(R=R_sweep, C=1e-11)
    assert ys.shape == (3, circuit.sys_size)


def test_circuit_instance_param_update_matches_global_single_instance(simple_lrc_netlist):
    net_dict, models_map = simple_lrc_netlist
    circuit = compile_circuit(net_dict, models_map)

    y_instance = circuit.dc(params={"R1.R": 20.0})
    y_global = circuit.dc(R=20.0)
    assert jnp.allclose(y_instance, y_global, atol=1e-10)


def test_circuit_get_port_field_real(simple_lrc_netlist):
    net_dict, models_map = simple_lrc_netlist
    circuit = compile_circuit(net_dict, models_map)

    y = circuit()
    port = "V1,p2"
    expected = y[circuit.port_map[port]]
    assert jnp.allclose(circuit.get_port_field(y, port), expected)
    assert jnp.allclose(circuit.port(y, port), expected)


def test_circuit_top_level_port_extraction():
    models_map = {
        "resistor": Resistor,
        "source_voltage": VoltageSource,
        "ground": lambda: 0,
    }
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "V1": {"component": "source_voltage", "settings": {"V": 4.0}},
            "R1": {"component": "resistor", "settings": {"R": 100.0}},
            "R2": {"component": "resistor", "settings": {"R": 100.0}},
        },
        "connections": {
            "GND,p1": ("V1,p1", "R2,p2"),
            "V1,p2": "R1,p1",
            "R1,p2": "R2,p1",
        },
        "ports": {"out": "R1,p2"},
    }
    circuit = compile_circuit(net_dict, models_map, backend="dense")
    y = circuit.dc()

    assert "out" in circuit.port_map
    assert jnp.isclose(jnp.abs(circuit.port(y, "out")), 2.0, atol=1e-6)
    assert jnp.allclose(circuit.port(y, "out"), circuit.port(y, "R1,p2"))


def test_circuit_get_port_field_batched(simple_lrc_netlist):
    net_dict, models_map = simple_lrc_netlist
    circuit = compile_circuit(net_dict, models_map)

    ys = circuit(R=jnp.array([10.0, 50.0, 100.0]))
    v = circuit.get_port_field(ys, "V1,p2")
    assert v.shape == (3,)
    assert jnp.allclose(v, ys[:, circuit.port_map["V1,p2"]])


def test_circuit_with_groups_preserves_solver_and_port_map(simple_lrc_netlist):
    net_dict, models_map = simple_lrc_netlist
    circuit = compile_circuit(net_dict, models_map)

    new_circuit = circuit.with_groups(circuit.groups)

    assert isinstance(new_circuit, Circuit)
    assert new_circuit is not circuit
    assert new_circuit.solver is circuit.solver
    assert new_circuit.port_map is circuit.port_map
    assert new_circuit.sys_size == circuit.sys_size


def test_compile_circuit_complex(simple_optical_netlist):
    net_dict, models_map = simple_optical_netlist
    circuit = compile_circuit(net_dict, models_map, is_complex=True)

    assert circuit.solver.is_complex is True

    y = circuit()
    assert y.shape == (2 * circuit.sys_size,)

    # get_port_field should return complex for is_complex circuits
    field = circuit.get_port_field(y, "WG1,p2")
    assert jnp.iscomplexobj(field)


def test_compile_circuit_auto_detects_complex(simple_optical_netlist):
    net_dict, models_map = simple_optical_netlist
    circuit = compile_circuit(net_dict, models_map)

    assert circuit.solver.is_complex is True


def test_circuit_callable_under_jit(simple_lrc_netlist):
    net_dict, models_map = simple_lrc_netlist
    circuit = compile_circuit(net_dict, models_map)

    jitted = jax.jit(lambda R: circuit(R=R))
    y = jitted(jnp.array(25.0))
    assert y.shape == (circuit.sys_size,)
    assert jnp.all(jnp.isfinite(y))


def test_circuit_high_level_transient(simple_lrc_netlist):
    net_dict, models_map = simple_lrc_netlist
    circuit = compile_circuit(net_dict, models_map, backend="dense")
    y0 = circuit.dc()

    sol = circuit.transient(
        t0=0.0,
        t1=1e-10,
        dt0=1e-11,
        y0=y0,
        saveat=jnp.linspace(0.0, 1e-10, 4),
        max_steps=1000,
    )

    assert sol.ys.shape == (4, circuit.sys_size)
    assert jnp.isfinite(sol.ys).all()


def test_circuit_high_level_ac_with_top_level_port():
    models_map = {
        "resistor": Resistor,
        "capacitor": Capacitor,
        "ground": lambda: 0,
    }
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "R1": {"component": "resistor", "settings": {"R": 50.0}},
            "C1": {"component": "capacitor", "settings": {"C": 1e-12}},
        },
        "connections": {
            "R1,p1": "C1,p1",
            "R1,p2": "GND,p1",
            "C1,p2": "GND,p1",
        },
        "ports": {"in": "R1,p1"},
    }
    circuit = compile_circuit(net_dict, models_map, backend="dense")
    S = circuit.ac(ports=["in"], freqs=jnp.array([1e6, 1e9]), z0=50.0)

    assert S.shape == (2, 1, 1)
    assert jnp.iscomplexobj(S)


def test_circuit_high_level_hb():
    models_map = {
        "resistor": Resistor,
        "source_voltage": VoltageSource,
        "ground": lambda: 0,
    }
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "V1": {"component": "source_voltage", "settings": {"V": 1.0}},
            "R1": {"component": "resistor", "settings": {"R": 100.0}},
        },
        "connections": {
            "GND,p1": ("V1,p1", "R1,p2"),
            "V1,p2": "R1,p1",
        },
    }
    circuit = compile_circuit(net_dict, models_map, backend="dense")
    y0 = circuit.dc()
    y_time, y_freq = circuit.hb(freq=1e6, harmonics=1, y0=y0, max_steps=5)

    assert y_time.shape == (3, circuit.sys_size)
    assert y_freq.shape == (2, circuit.sys_size)
    assert jnp.isfinite(y_time).all()


def test_backend_default_is_klu_split_linear():
    assert backends["default"] is backends["klu_split_linear"]
