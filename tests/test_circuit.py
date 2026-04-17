"""Tests for the high-level Circuit / compile_circuit API."""

import jax
import jax.numpy as jnp
import pytest

from circulax import Circuit, compile_circuit
from circulax.compiler import compile_netlist
from circulax.solvers.linear import analyze_circuit


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


def test_circuit_get_port_field_real(simple_lrc_netlist):
    net_dict, models_map = simple_lrc_netlist
    circuit = compile_circuit(net_dict, models_map)

    y = circuit()
    port = "V1,p2"
    expected = y[circuit.port_map[port]]
    assert jnp.allclose(circuit.get_port_field(y, port), expected)


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


def test_circuit_callable_under_jit(simple_lrc_netlist):
    net_dict, models_map = simple_lrc_netlist
    circuit = compile_circuit(net_dict, models_map)

    jitted = jax.jit(lambda R: circuit(R=R))
    y = jitted(jnp.array(25.0))
    assert y.shape == (circuit.sys_size,)
    assert jnp.all(jnp.isfinite(y))
