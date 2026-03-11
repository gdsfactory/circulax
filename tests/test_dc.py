from functools import partial

import jax
import jax.numpy as jnp
import pytest

from circulax.compiler import compile_netlist
from circulax.netlist import Netlist
from circulax.solvers.linear import analyze_circuit, backends

# One canonical backend name per unique solver implementation
_seen_classes: set = set()
_backends: list[str] = []
for _name, _cls in backends.items():
    if _name == "default":
        continue
    if _cls not in _seen_classes:
        _seen_classes.add(_cls)
        _backends.append(_name)


# assemble_total_f is local to this module and uses jax
def assemble_total_f(component_groups: list, y, t:float=0.0)-> None:
    total_f = jnp.zeros(y.shape[0])
    for group in component_groups.values():
        v_locs = y[group.var_indices]
        physics_fn = partial(group.physics_func, t=t)

        def get_f_only(v, p):  # noqa: ANN202
            return physics_fn(y=v, args=p)[0]  # noqa: B023

        f_loc = jax.vmap(get_f_only)(v_locs, group.params)
        total_f = total_f.at[group.eq_indices].add(f_loc)
    return total_f


@pytest.mark.parametrize("backend", _backends)
def test_solve_operation_point_residual_small(simple_lrc_netlist: Netlist, backend)-> None:  # noqa: ANN001
    net_dict, models_map = simple_lrc_netlist
    groups, sys_size, _ = compile_netlist(net_dict, models_map)

    linear_strat = analyze_circuit(groups, sys_size, backend=backend, is_complex=False)
    y_guess = jnp.zeros(sys_size)
    y_op = linear_strat.solve_dc(groups, y_guess)

    assert y_op.shape[0] == sys_size

    total_f = assemble_total_f(groups, y_op, t=0.0)
    G_leak = 1e-9
    residual = total_f + y_op * G_leak

    # Residual should be very small for a converged DC solution
    assert jnp.linalg.norm(residual) < 1e-6


@pytest.mark.parametrize("backend", _backends)
def test_solve_operation_point_residual_small_complex(simple_optical_netlist: Netlist, backend)-> None:  # noqa: ANN001
    net_dict, models_map = simple_optical_netlist
    groups, sys_size, _ = compile_netlist(net_dict, models_map)

    linear_strat = analyze_circuit(groups, sys_size, backend=backend, is_complex=True)
    y_guess = jnp.ones(2 * sys_size)
    y_op = linear_strat.solve_dc(groups, y_guess)

    assert y_op.shape[0] == 2 * sys_size


# ---------------------------------------------------------------------------
# Homotopy convergence tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", _backends)
def test_solve_dc_gmin_matches_direct(simple_lrc_netlist: Netlist, backend) -> None:  # noqa: ANN001
    """GMIN stepping should converge to the same operating point as direct Newton."""
    net_dict, models_map = simple_lrc_netlist
    groups, sys_size, _ = compile_netlist(net_dict, models_map)

    strat = analyze_circuit(groups, sys_size, backend=backend, is_complex=False)
    y_guess = jnp.zeros(sys_size)

    y_direct = strat.solve_dc(groups, y_guess)
    y_gmin = strat.solve_dc_gmin(groups, y_guess, g_start=1e-2, n_steps=10)

    assert y_gmin.shape == y_direct.shape
    assert jnp.allclose(y_gmin, y_direct, atol=1e-4), (
        f"GMIN result differs from direct Newton: max diff {jnp.max(jnp.abs(y_gmin - y_direct))}"
    )


@pytest.mark.parametrize("backend", _backends)
def test_solve_dc_source_matches_direct(simple_lrc_netlist: Netlist, backend) -> None:  # noqa: ANN001
    """Source stepping should converge to the same operating point as direct Newton."""
    net_dict, models_map = simple_lrc_netlist
    groups, sys_size, _ = compile_netlist(net_dict, models_map)

    strat = analyze_circuit(groups, sys_size, backend=backend, is_complex=False)
    y_guess = jnp.zeros(sys_size)

    y_direct = strat.solve_dc(groups, y_guess)
    y_source = strat.solve_dc_source(groups, y_guess, n_steps=10)

    assert y_source.shape == y_direct.shape
    assert jnp.allclose(y_source, y_direct, atol=1e-4), (
        f"Source stepping result differs from direct Newton: max diff {jnp.max(jnp.abs(y_source - y_direct))}"
    )


def test_amplitude_param_tagged() -> None:
    """VoltageSource and CurrentSource should have amplitude_param set after compile."""
    from circulax.components.electronic import CurrentSource, VoltageSource

    assert VoltageSource.amplitude_param == "V"
    assert CurrentSource.amplitude_param == "I"


def test_solve_dc_source_scales_voltage() -> None:
    """At source_scale=0.1, the VoltageSource should impose 10% of V=5.0.

    Uses a minimal R divider (no delay) so source scaling has a measurable effect at DC.
    """
    from circulax.components.electronic import Resistor, VoltageSource

    models_map = {"vs": VoltageSource, "r": Resistor, "ground": lambda: 0}
    # Simple circuit: V+ --[R=1k]-- GND, V source sets V+ to 5V
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "V1": {"component": "vs", "settings": {"V": 5.0}},  # no delay
            "R1": {"component": "r", "settings": {"R": 1000.0}},
        },
        "connections": {
            "GND,p1": ("V1,p1", "R1,p2"),
            "V1,p2": "R1,p1",
        },
    }

    groups, sys_size, port_map = compile_netlist(net_dict, models_map)
    strat = analyze_circuit(groups, sys_size, backend="dense")

    y_10pct, _ = strat._run_newton(groups, jnp.zeros(sys_size), source_scale=0.1)  # noqa: SLF001
    y_full, _ = strat._run_newton(groups, jnp.zeros(sys_size), source_scale=1.0)  # noqa: SLF001

    v1_node = port_map["V1,p2"]
    # V1,p1 = GND = 0, so v_p2 = -(V * scale).
    # 10% scale → -0.5V at V1,p2; full scale → -5.0V
    assert jnp.allclose(y_10pct[v1_node], -0.5, atol=0.01)
    assert jnp.allclose(y_full[v1_node], -5.0, atol=0.01)


@pytest.mark.parametrize("backend", _backends)
def test_solve_dc_checked_returns_converged_flag(simple_lrc_netlist: Netlist, backend) -> None:  # noqa: ANN001
    """solve_dc_checked should return (y, converged) with converged=True for a well-posed circuit."""
    net_dict, models_map = simple_lrc_netlist
    groups, sys_size, _ = compile_netlist(net_dict, models_map)

    strat = analyze_circuit(groups, sys_size, backend=backend, is_complex=False)
    y_guess = jnp.zeros(sys_size)

    y, converged = strat.solve_dc_checked(groups, y_guess)

    assert y.shape == (sys_size,)
    assert converged.shape == ()  # scalar JAX array
    assert bool(converged), "Expected convergence for a linear LRC circuit"

    y_direct = strat.solve_dc(groups, y_guess)
    assert jnp.allclose(y, y_direct, atol=1e-6)


@pytest.mark.parametrize("backend", _backends)
def test_solve_dc_auto_matches_direct(simple_lrc_netlist: Netlist, backend) -> None:  # noqa: ANN001
    """solve_dc_auto should fall through to direct Newton for a well-posed circuit."""
    net_dict, models_map = simple_lrc_netlist
    groups, sys_size, _ = compile_netlist(net_dict, models_map)

    strat = analyze_circuit(groups, sys_size, backend=backend, is_complex=False)
    y_guess = jnp.zeros(sys_size)

    y_auto = strat.solve_dc_auto(groups, y_guess)
    y_direct = strat.solve_dc(groups, y_guess)

    assert y_auto.shape == y_direct.shape
    assert jnp.allclose(y_auto, y_direct, atol=1e-4), (
        f"solve_dc_auto result differs from direct Newton: max diff {jnp.max(jnp.abs(y_auto - y_direct))}"
    )
