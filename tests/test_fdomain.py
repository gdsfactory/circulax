"""Tests for frequency-domain (f-domain) component support.

Verifies:
  - fdomain_component decorator creates a valid CircuitComponent subclass with
    _is_fdomain=True and the correct admittance-matrix interface.
  - DC operating-point analysis evaluates the admittance at f=0.
  - Harmonic Balance correctly includes f-domain contributions at each harmonic,
    matching the analytical voltage-divider ratio.
  - Transient simulation raises RuntimeError at setup time.
  - JIT and grad work end-to-end.
"""

import jax
import jax.numpy as jnp
import pytest

from circulax.compiler import compile_netlist
from circulax.s_transforms import fdomain_component
from circulax.solvers import setup_harmonic_balance
from circulax.solvers.linear import DenseSolver, analyze_circuit


# ---------------------------------------------------------------------------
# Shared skin-effect component definition
# ---------------------------------------------------------------------------


@fdomain_component(ports=("p1", "p2"))
def SkinEffectResistor(f: float, R0: float = 1.0, a: float = 0.1):
    """Frequency-dependent resistor: Z(f) = R0 + a * sqrt(|f|).

    Admittance: Y(f) = 1 / (R0 + a * sqrt(|f| + 1e-30))
    """
    Z = R0 + a * jnp.sqrt(jnp.abs(f) + 1e-30)
    Y = 1.0 / Z
    return jnp.array([[Y, -Y], [-Y, Y]], dtype=jnp.complex128)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_FREQ = 1e6  # fundamental frequency (Hz)
_NUM_HARMONICS = 3
_R0 = 1.0
_A = 0.1
_RL = 1.0  # load resistor


def _skin_z(f):
    """Expected impedance of SkinEffectResistor at frequency f."""
    return _R0 + _A * jnp.sqrt(jnp.abs(f) + 1e-30)


def _divider_netlist():
    """Voltage-divider circuit: VoltageSourceAC → SkinEffectResistor → Resistor → GND.

    Nodes (after compilation):
      0 = GND
      1 = node between Vs.p2 and Rs.p1   (driven by source)
      2 = node between Rs.p2 and RL.p1   (measurement point)
      3 = i_src state of Vs
    """
    from circulax.components.electronic import Resistor, VoltageSourceAC

    models_map = {
        "vs": VoltageSourceAC,
        "skin_r": SkinEffectResistor,
        "rl": Resistor,
        "ground": lambda: 0,
    }
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "Vs": {"component": "vs", "settings": {"V": 1.0, "freq": _FREQ}},
            "Rs": {"component": "skin_r", "settings": {"R0": _R0, "a": _A}},
            "RL": {"component": "rl", "settings": {"R": _RL}},
        },
        "connections": {
            "GND,p1": ("Vs,p1", "RL,p2"),
            "Vs,p2": "Rs,p1",
            "Rs,p2": "RL,p1",
        },
    }
    return net_dict, models_map


# ---------------------------------------------------------------------------
# Decorator tests
# ---------------------------------------------------------------------------


def test_fdomain_flag():
    """The fdomain_component decorator sets _is_fdomain=True."""
    assert SkinEffectResistor._is_fdomain is True


def test_fdomain_ports():
    """fdomain_component correctly records the declared port names."""
    assert SkinEffectResistor.ports == ("p1", "p2")


def test_fdomain_states_empty():
    """fdomain_component components have no internal states."""
    assert SkinEffectResistor.states == ()


def test_fdomain_admittance_dc():
    """Y-matrix at f=0 equals [[1/R0, -1/R0], [-1/R0, 1/R0]]."""
    inst = SkinEffectResistor(R0=_R0, a=_A)
    Y = inst.solver_call(0.0, inst)
    Y0 = 1.0 / _R0
    expected = jnp.array([[Y0, -Y0], [-Y0, Y0]], dtype=jnp.complex128)
    assert jnp.allclose(Y, expected, atol=1e-9)


def test_fdomain_admittance_at_freq():
    """Y-matrix at f=_FREQ equals [[Y, -Y], [-Y, Y]] for Y = 1/Z(f)."""
    inst = SkinEffectResistor(R0=_R0, a=_A)
    Y = inst.solver_call(float(_FREQ), inst)
    Z_expected = _skin_z(_FREQ)
    Y_expected = 1.0 / Z_expected
    assert jnp.allclose(Y[0, 0], Y_expected, atol=1e-9)
    assert jnp.allclose(Y[0, 1], -Y_expected, atol=1e-9)
    assert jnp.allclose(Y[1, 0], -Y_expected, atol=1e-9)
    assert jnp.allclose(Y[1, 1], Y_expected, atol=1e-9)


def test_fdomain_bad_signature_raises():
    """Decorating a function whose first arg is not 'f' raises TypeError."""
    with pytest.raises(TypeError, match="'f'"):
        @fdomain_component(ports=("p1", "p2"))
        def BadComponent(x: float, R: float = 1.0):
            return jnp.eye(2, dtype=jnp.complex128)


def test_fdomain_missing_default_raises():
    """Decorating a function with a parameter lacking a default raises TypeError."""
    with pytest.raises(TypeError, match="default"):
        @fdomain_component(ports=("p1",))
        def BadComponent2(f: float, R: float):
            return jnp.eye(1, dtype=jnp.complex128)


# ---------------------------------------------------------------------------
# ComponentGroup flag
# ---------------------------------------------------------------------------


def test_compiled_group_is_fdomain():
    """compile_netlist sets is_fdomain=True on groups containing fdomain components."""
    net_dict, models_map = _divider_netlist()
    groups, _, _ = compile_netlist(net_dict, models_map)
    fdomain_groups = [g for g in groups.values() if g.is_fdomain]
    tdomain_groups = [g for g in groups.values() if not g.is_fdomain]
    assert len(fdomain_groups) == 1
    assert fdomain_groups[0].name == "skin_r"
    assert len(tdomain_groups) >= 2  # VoltageSourceAC + Resistor


# ---------------------------------------------------------------------------
# DC operating-point
# ---------------------------------------------------------------------------


def test_dc_with_fdomain():
    """DC analysis with an fdomain component returns a finite, converged solution."""
    net_dict, models_map = _divider_netlist()
    groups, sys_size, port_map = compile_netlist(net_dict, models_map)
    solver = DenseSolver.from_component_groups(groups, sys_size)
    y_dc = solver.solve_dc(groups, jnp.zeros(sys_size))

    assert jnp.isfinite(y_dc).all()
    # At DC, VoltageSourceAC outputs 0 V → all node voltages should be 0.
    assert jnp.allclose(y_dc, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Harmonic Balance — voltage-divider accuracy
# ---------------------------------------------------------------------------


def test_hb_fdomain_voltage_divider():
    """HB voltage at node2 matches the analytical skin-effect voltage-divider ratio.

    At fundamental frequency f0, the circuit is a series voltage divider:
        V_node2 / V_node1 = RL / (Z_Rs(f0) + RL)
    """
    net_dict, models_map = _divider_netlist()
    groups, sys_size, port_map = compile_netlist(net_dict, models_map)
    solver = DenseSolver.from_component_groups(groups, sys_size)
    y_dc = solver.solve_dc(groups, jnp.zeros(sys_size))

    run_hb = setup_harmonic_balance(groups, sys_size, freq=_FREQ, num_harmonics=_NUM_HARMONICS)
    y_time, y_freq = run_hb(y_dc)

    # y_freq[k, node] is the (normalised) complex amplitude at harmonic k.
    node1 = port_map["Vs,p2"]
    node2 = port_map["Rs,p2"]

    V1_fund = y_freq[1, node1]  # fundamental at driven node
    V2_fund = y_freq[1, node2]  # fundamental at measurement node

    # Analytical voltage-divider ratio at k=1
    Z_skin_fund = _skin_z(_FREQ)
    ratio_expected = _RL / (Z_skin_fund + _RL)

    # Magnitudes should match to within solver tolerance.
    ratio_computed = jnp.abs(V2_fund) / (jnp.abs(V1_fund) + 1e-30)
    assert jnp.allclose(ratio_computed, ratio_expected, atol=1e-4), (
        f"HB ratio {float(ratio_computed):.6f} != expected {float(ratio_expected):.6f}"
    )


def test_hb_fdomain_higher_harmonics_small():
    """Higher harmonics should be near-zero for a linear f-domain circuit driven by a pure sinusoid."""
    net_dict, models_map = _divider_netlist()
    groups, sys_size, port_map = compile_netlist(net_dict, models_map)
    solver = DenseSolver.from_component_groups(groups, sys_size)
    y_dc = solver.solve_dc(groups, jnp.zeros(sys_size))

    run_hb = setup_harmonic_balance(groups, sys_size, freq=_FREQ, num_harmonics=_NUM_HARMONICS)
    y_time, y_freq = run_hb(y_dc)

    node2 = port_map["Rs,p2"]
    # Harmonics k>=2 should be essentially zero for a linear circuit + sinusoidal source.
    for k in range(2, _NUM_HARMONICS + 1):
        amp = jnp.abs(y_freq[k, node2])
        assert float(amp) < 1e-5, f"Harmonic k={k} amplitude {float(amp):.2e} unexpectedly large"


def test_hb_fdomain_shapes():
    """HB with an fdomain component returns arrays with correct shapes."""
    net_dict, models_map = _divider_netlist()
    groups, sys_size, _ = compile_netlist(net_dict, models_map)
    solver = DenseSolver.from_component_groups(groups, sys_size)
    y_dc = solver.solve_dc(groups, jnp.zeros(sys_size))

    run_hb = setup_harmonic_balance(groups, sys_size, freq=_FREQ, num_harmonics=_NUM_HARMONICS)
    y_time, y_freq = run_hb(y_dc)

    K = 2 * _NUM_HARMONICS + 1
    assert y_time.shape == (K, sys_size)
    assert y_freq.shape == (_NUM_HARMONICS + 1, sys_size)
    assert jnp.isfinite(y_time).all()


def test_hb_fdomain_jit():
    """jax.jit(run_hb) gives the same result for a circuit with fdomain components."""
    net_dict, models_map = _divider_netlist()
    groups, sys_size, _ = compile_netlist(net_dict, models_map)
    solver = DenseSolver.from_component_groups(groups, sys_size)
    y_dc = solver.solve_dc(groups, jnp.zeros(sys_size))

    run_hb = setup_harmonic_balance(groups, sys_size, freq=_FREQ, num_harmonics=_NUM_HARMONICS)
    y_time_eager, _ = run_hb(y_dc)
    y_time_jit, _ = jax.jit(run_hb)(y_dc)

    assert jnp.allclose(y_time_eager, y_time_jit, atol=1e-10)


def test_hb_fdomain_grad():
    """jax.grad differentiates a scalar loss through run_hb w.r.t. y_dc."""
    net_dict, models_map = _divider_netlist()
    groups, sys_size, _ = compile_netlist(net_dict, models_map)
    solver = DenseSolver.from_component_groups(groups, sys_size)
    y_dc = solver.solve_dc(groups, jnp.zeros(sys_size))

    run_hb = setup_harmonic_balance(groups, sys_size, freq=_FREQ, num_harmonics=_NUM_HARMONICS)

    def loss(y):
        y_time, _ = run_hb(y)
        return jnp.sum(y_time**2)

    grad = jax.grad(loss)(y_dc)
    assert grad.shape == y_dc.shape
    assert jnp.isfinite(grad).all()


# ---------------------------------------------------------------------------
# Transient guard
# ---------------------------------------------------------------------------


def test_transient_fdomain_raises():
    """setup_transient raises RuntimeError when fdomain components are present."""
    from circulax.solvers import setup_transient

    net_dict, models_map = _divider_netlist()
    groups, sys_size, _ = compile_netlist(net_dict, models_map)
    solver = DenseSolver.from_component_groups(groups, sys_size)

    with pytest.raises(RuntimeError, match="Frequency-domain"):
        setup_transient(groups, solver)
