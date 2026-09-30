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


def test_sax_circuit_inside_model_no_concretization_error():
    """Regression: klujax 0.5.1 raised ConcretizationTypeError when a SAX model
    calls sax.circuit() inside its body (not pre-compiled).

    CSPDK's coupler_ring does this — each call to coupler_ring() builds a fresh
    sax.circuit(), which calls klujax.analyze() via the KLU backend. When circulax
    traces the Newton step (optx.fixed_point → equinox JIT), the analyze() call
    happens inside the trace and int(raw_symbol) fails on traced values.
    """
    import sax
    from sax.models import straight as sax_straight

    def composite_model(wl=1.55, length=100.0, neff=2.34):
        """SAX model that calls sax.circuit() each invocation (like coupler_ring)."""
        netlist = {
            "instances": {"wg": {"component": "straight", "settings": {"length": length, "neff": neff}}},
            "connections": {},
            "ports": {"o1": "wg,in0", "o2": "wg,out0"},
        }
        circuit_fn, _ = sax.circuit(netlist, {"straight": sax_straight}, backend="klu")
        return circuit_fn(wl=wl)

    from circulax.s_transforms import sax_component

    CompositeComp = sax_component(composite_model)
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "R1": {"component": "resistor", "settings": {"R": 1.0}},
            "comp": {"component": "composite"},
        },
        "connections": {
            "GND,p1": ("R1,p2", "comp,o2"),
            "R1,p1": "comp,o1",
        },
    }
    models_map = {"resistor": Resistor, "composite": CompositeComp, "ground": lambda: 0}
    circuit = compile_circuit(net_dict, models_map, is_complex=True)
    y = circuit.dc()
    assert y.shape[0] > 0
    assert jnp.all(jnp.isfinite(y))


def test_backend_default_is_klu_split_linear():
    assert backends["default"] is backends["klu_split_linear"]


@pytest.fixture
def pure_sax_netlist():
    """A coupler + straight waveguide netlist: all-SAX, no sources."""
    from sax.models import coupler, straight

    net_dict = {
        "instances": {
            "c1": {"component": "coupler", "settings": {}},
            "wg": {"component": "straight", "settings": {}},
        },
        "connections": {"c1,out0": "wg,in0"},
        "ports": {"in0": "c1,in0", "in1": "c1,in1", "out0": "wg,out0", "out1": "c1,out1"},
    }
    models = {"coupler": coupler, "straight": straight}
    return net_dict, models


def test_pure_sax_circuit_detected(pure_sax_netlist):
    net_dict, models = pure_sax_netlist
    circuit = compile_circuit(net_dict, models, is_complex=True)
    assert circuit.check_sax_compatibility() is True


def test_pure_sax_dc_still_returns_zero_array(pure_sax_netlist):
    """dc()'s contract is unchanged even for an all-SAX circuit: it always
    returns an Array (all-zero here, since the linear system has no source)."""
    net_dict, models = pure_sax_netlist
    circuit = compile_circuit(net_dict, models, is_complex=True)
    y = circuit.dc()
    assert isinstance(y, jax.Array)
    assert jnp.allclose(y, 0.0)


def test_to_sax_circuit_matches_native_sax_scalar(pure_sax_netlist):
    import sax

    net_dict, models = pure_sax_netlist
    sax_model, _ = sax.circuit(net_dict, models=models)
    S_native = sax_model(wl=1.55)

    circuit = compile_circuit(net_dict, models, is_complex=True)
    S = circuit.to_sax_circuit()(wl=1.55)

    assert isinstance(S, dict)
    for key, native_val in S_native.items():
        assert jnp.allclose(S[key], native_val, atol=1e-9), key


def test_to_sax_circuit_matches_native_sax_batched(pure_sax_netlist):
    import sax

    net_dict, models = pure_sax_netlist
    sax_model, _ = sax.circuit(net_dict, models=models)

    circuit = compile_circuit(net_dict, models, is_complex=True)
    wls = jnp.array([1.5, 1.55, 1.6])
    S_batched = circuit.to_sax_circuit()(wl=wls)

    assert S_batched[("in0", "out0")].shape == (3,)
    for i, wl in enumerate(wls):
        S_native_i = sax_model(wl=float(wl))
        for key, native_val in S_native_i.items():
            assert jnp.allclose(S_batched[key][i], native_val, atol=1e-9), (i, key)


def test_to_sax_circuit_rejects_incompatible_circuit():
    """A circuit with a non-SAX component (resistor/GND) can't become a SAX circuit."""
    import sax
    from sax.models import straight

    from circulax.s_transforms import sax_component

    def composite_model(wl=1.55, length=100.0, neff=2.34):
        netlist = {
            "instances": {"wg": {"component": "straight", "settings": {"length": length, "neff": neff}}},
            "connections": {},
            "ports": {"o1": "wg,in0", "o2": "wg,out0"},
        }
        circuit_fn, _ = sax.circuit(netlist, {"straight": straight}, backend="klu")
        return circuit_fn(wl=wl)

    CompositeComp = sax_component(composite_model)
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "R1": {"component": "resistor", "settings": {"R": 1.0}},
            "comp": {"component": "composite"},
        },
        "connections": {
            "GND,p1": ("R1,p2", "comp,o2"),
            "R1,p1": "comp,o1",
        },
    }
    models_map = {"resistor": Resistor, "composite": CompositeComp, "ground": lambda: 0}
    circuit = compile_circuit(net_dict, models_map, is_complex=True)

    assert circuit.check_sax_compatibility() is False
    y = circuit.dc()
    assert isinstance(y, jax.Array)
    assert jnp.all(jnp.isfinite(y))
    with pytest.raises(ValueError, match="check_sax_compatibility"):
        circuit.to_sax_circuit()


def test_to_sax_circuit_rejects_missing_ports(pure_sax_netlist):
    net_dict, models = pure_sax_netlist
    circuit = compile_circuit(net_dict, models, is_complex=True)
    stripped = circuit.with_groups(circuit.groups)
    with pytest.raises(ValueError, match="no known external ports"):
        stripped.to_sax_circuit()
