"""Tests for setup_ac_sweep: correctness, JIT, vmap, and passivity."""

import jax
import jax.numpy as jnp
import pytest

from circulax.compiler import compile_netlist
from circulax.solvers import analyze_circuit, setup_ac_sweep

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

_R = 50.0  # series resistor (ohms)
_C = 1e-9  # shunt capacitor (farads)
_Z0 = 50.0  # reference impedance


@pytest.fixture
def rc_netlist():
    """Single-port parallel RC circuit: R and C both shunt to GND.

    Port is at the shared node R1,p1 = C1,p1.  The admittance seen from the
    port is Y_circuit = 1/R + jωC, so the analytical S11 is:

        Y_total = 1/Z0 + 1/R + jωC
        S11 = (2/Z0) / Y_total - 1
    """
    from circulax.components.electronic import Capacitor, Resistor

    models_map = {
        "resistor": Resistor,
        "capacitor": Capacitor,
        "ground": lambda: 0,
    }
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "R1": {"component": "resistor", "settings": {"R": _R}},
            "C1": {"component": "capacitor", "settings": {"C": _C}},
        },
        "connections": {
            "R1,p1": "C1,p1",  # port node: R1,p1 == C1,p1
            "R1,p2": "GND,p1",
            "C1,p2": "GND,p1",
        },
    }
    return net_dict, models_map


@pytest.fixture
def rc_setup(rc_netlist):
    """Compiled RC circuit with DC solution and run_ac callable."""
    net_dict, models_map = rc_netlist
    groups, num_vars, pmap = compile_netlist(net_dict, models_map)
    solver = analyze_circuit(groups, num_vars)
    y_dc = solver.solve_dc(groups, jnp.zeros(num_vars))
    port_nodes = [pmap["R1,p1"]]
    run_ac = setup_ac_sweep(groups, num_vars, port_nodes, z0=_Z0)
    return run_ac, y_dc


def _analytical_s11(freqs: jnp.ndarray, R: float = _R, C: float = _C, z0: float = _Z0) -> jnp.ndarray:
    """Analytical S11 for the parallel-RC test circuit.

    Y_circuit = 1/R + jωC, so:
        Y_total = 1/z0 + 1/R + jωC
        S11 = (2/z0) / Y_total - 1
    """
    omega = 2.0 * jnp.pi * freqs
    Y_total = 1.0 / z0 + 1.0 / R + 1j * omega * C
    V_port = (2.0 / z0) / Y_total
    return V_port - 1.0


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

_FREQS = jnp.logspace(6, 10, 50)  # 1 MHz to 10 GHz


def test_ac_sweep_shapes(rc_setup):
    """run_ac returns (N_freqs, N_ports, N_ports) complex array."""
    run_ac, y_dc = rc_setup
    S = run_ac(y_dc, _FREQS)
    assert S.shape == (len(_FREQS), 1, 1)
    assert jnp.iscomplexobj(S)


def test_ac_sweep_finite(rc_setup):
    """All S-parameter values are finite."""
    run_ac, y_dc = rc_setup
    S = run_ac(y_dc, _FREQS)
    assert jnp.isfinite(jnp.abs(S)).all()


def test_ac_sweep_s11_analytical(rc_setup):
    """S11 matches the analytical RC formula to within 1e-6."""
    run_ac, y_dc = rc_setup
    S = run_ac(y_dc, _FREQS)
    S11 = S[:, 0, 0]
    S11_ref = _analytical_s11(_FREQS)
    assert jnp.allclose(S11, S11_ref, atol=1e-6), f"Max error: {jnp.max(jnp.abs(S11 - S11_ref)):.2e}"


def test_ac_sweep_passivity(rc_setup):
    """Passive circuit satisfies |S11| <= 1 at all frequencies."""
    run_ac, y_dc = rc_setup
    S = run_ac(y_dc, _FREQS)
    assert (jnp.abs(S[:, 0, 0]) <= 1.0 + 1e-6).all()


# ---------------------------------------------------------------------------
# Limiting cases
# ---------------------------------------------------------------------------


def test_ac_sweep_dc_limit(rc_netlist):
    """At f→0, S11 → (R_parallel - Z0)/(R_parallel + Z0) (capacitor becomes open).

    For parallel RC, at DC the capacitor is open so Y_circuit = 1/R only.
    """
    net_dict, models_map = rc_netlist
    groups, num_vars, pmap = compile_netlist(net_dict, models_map)
    solver = analyze_circuit(groups, num_vars)
    y_dc = solver.solve_dc(groups, jnp.zeros(num_vars))
    run_ac = setup_ac_sweep(groups, num_vars, [pmap["R1,p1"]], z0=_Z0)

    freqs_low = jnp.array([1e-3])  # essentially DC
    S = run_ac(y_dc, freqs_low)
    S11_dc = S[0, 0, 0]
    # At DC: Y_total = 1/Z0 + 1/R, V_port = (2/Z0)/(1/Z0 + 1/R), S11 = V_port - 1
    Y_total_dc = 1.0 / _Z0 + 1.0 / _R
    expected = (2.0 / _Z0) / Y_total_dc - 1.0  # = 0.0 for R=Z0=50
    assert jnp.abs(S11_dc - expected) < 1e-4


def test_ac_sweep_matched_load():
    """Pure shunt resistive load R = Z0 gives S11 = 0 at all frequencies."""
    from circulax.components.electronic import Resistor

    # Two identical shunt resistors in parallel = R/2.  Set R = 2*Z0 so the
    # parallel combination equals Z0, giving a matched load (S11 = 0).
    R_each = 2.0 * _Z0  # parallel combination = Z0
    models_map = {"resistor": Resistor, "ground": lambda: 0}
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "R1": {"component": "resistor", "settings": {"R": R_each}},
            "R2": {"component": "resistor", "settings": {"R": R_each}},
        },
        "connections": {
            "R1,p1": "R2,p1",  # port node shared between R1 and R2
            "R1,p2": "GND,p1",
            "R2,p2": "GND,p1",
        },
    }
    groups, num_vars, pmap = compile_netlist(net_dict, models_map)
    solver = analyze_circuit(groups, num_vars)
    y_dc = solver.solve_dc(groups, jnp.zeros(num_vars))
    run_ac = setup_ac_sweep(groups, num_vars, [pmap["R1,p1"]], z0=_Z0)

    freqs = jnp.logspace(3, 9, 20)
    S = run_ac(y_dc, freqs)
    assert jnp.allclose(jnp.abs(S[:, 0, 0]), 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# JIT
# ---------------------------------------------------------------------------


def test_ac_sweep_jit(rc_setup):
    """jax.jit(run_ac) matches the eager result."""
    run_ac, y_dc = rc_setup
    S_eager = run_ac(y_dc, _FREQS)
    S_jit = jax.jit(run_ac)(y_dc, _FREQS)
    assert jnp.allclose(S_eager, S_jit, atol=1e-10)


# ---------------------------------------------------------------------------
# vmap
# ---------------------------------------------------------------------------


def test_ac_sweep_vmap_over_ydc(rc_netlist):
    """jax.vmap(run_ac, in_axes=(0, None)) batches correctly over y_dc."""
    net_dict, models_map = rc_netlist
    groups, num_vars, pmap = compile_netlist(net_dict, models_map)
    solver = analyze_circuit(groups, num_vars)
    y_dc = solver.solve_dc(groups, jnp.zeros(num_vars))
    run_ac = setup_ac_sweep(groups, num_vars, [pmap["R1,p1"]], z0=_Z0)

    # Batch of 3 identical DC points
    y_dc_batch = jnp.stack([y_dc, y_dc, y_dc])
    S_batch = jax.vmap(run_ac, in_axes=(0, None))(y_dc_batch, _FREQS)

    assert S_batch.shape == (3, len(_FREQS), 1, 1)
    S_single = run_ac(y_dc, _FREQS)
    assert jnp.allclose(S_batch[0], S_single, atol=1e-10)
    assert jnp.allclose(S_batch[1], S_single, atol=1e-10)


# ---------------------------------------------------------------------------
# F-domain component
# ---------------------------------------------------------------------------


def test_ac_sweep_fdomain():
    """S11 with an f-domain resistor matches the analytical skin-effect formula."""
    from circulax.s_transforms import fdomain_component

    @fdomain_component(ports=("p1", "p2"))
    def SkinResistor(f: float, R0: float = 1.0, a: float = 0.1):
        """Skin-effect resistor: Z(f) = R0 + a*sqrt(|f|)."""
        Z = R0 + a * jnp.sqrt(jnp.abs(f) + 1e-30)
        Y = 1.0 / Z
        return jnp.array([[Y, -Y], [-Y, Y]], dtype=jnp.complex128)

    R0, a = 25.0, 1e-5
    from circulax.components.electronic import Resistor

    models_map = {"skin_r": SkinResistor, "resistor": Resistor, "ground": lambda: 0}
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "SR1": {"component": "skin_r", "settings": {"R0": R0, "a": a}},
            # Huge shunt resistor to create the port node without affecting the result
            "Rbig": {"component": "resistor", "settings": {"R": 1e15}},
        },
        "connections": {
            "SR1,p1": "Rbig,p1",  # port node
            "SR1,p2": "GND,p1",
            "Rbig,p2": "GND,p1",
        },
    }
    groups, num_vars, pmap = compile_netlist(net_dict, models_map)
    solver = analyze_circuit(groups, num_vars)
    y_dc = solver.solve_dc(groups, jnp.zeros(num_vars))
    run_ac = setup_ac_sweep(groups, num_vars, [pmap["SR1,p1"]], z0=_Z0)

    freqs = jnp.logspace(3, 9, 30)
    S = run_ac(y_dc, freqs)
    S11 = S[:, 0, 0]

    # Analytical S11 for skin-effect load
    Z_circuit = R0 + a * jnp.sqrt(jnp.abs(freqs) + 1e-30)
    Y_total = 1.0 / _Z0 + 1.0 / Z_circuit
    S11_ref = (2.0 / _Z0) / Y_total - 1.0

    assert jnp.allclose(S11, S11_ref, atol=1e-6), f"Max fdomain error: {jnp.max(jnp.abs(S11 - S11_ref)):.2e}"


# ---------------------------------------------------------------------------
# Ground node guard
# ---------------------------------------------------------------------------


def test_ac_sweep_ground_node_raises(rc_netlist):
    """Passing node 0 (ground) as a port raises ValueError."""
    net_dict, models_map = rc_netlist
    groups, num_vars, _ = compile_netlist(net_dict, models_map)
    with pytest.raises(ValueError, match="ground"):
        setup_ac_sweep(groups, num_vars, [0], z0=_Z0)
