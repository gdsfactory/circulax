import diffrax
import jax.numpy as jnp
import pytest

from circulax.compiler import compile_netlist
from circulax.solvers.linear import backends, analyze_circuit
from circulax.solvers.transient import VectorizedTransientSolver

# One canonical backend name per unique solver implementation
_seen_classes: set = set()
_backends: list[str] = []
for _name, _cls in backends.items():
    if _name == "default":
        continue
    if _cls not in _seen_classes:
        _seen_classes.add(_cls)
        _backends.append(_name)


@pytest.mark.parametrize("backend", _backends)
def test_short_transient_runs_float(simple_lrc_netlist, backend):
    net_dict, models_map = simple_lrc_netlist
    groups, sys_size, _ = compile_netlist(net_dict, models_map)

    # Solve DC operating point and run a very short transient with zero forcing
    linear_strat = analyze_circuit(groups, sys_size, backend=backend, is_complex=False)
    y_guess = jnp.zeros(sys_size)
    y_op = linear_strat.solve_dc(groups, y_guess)

    assert y_op.shape[0] == sys_size

    solver = VectorizedTransientSolver(linear_solver=linear_strat)

    term = diffrax.ODETerm(lambda t, y, args: jnp.zeros_like(y))

    t_max = 1e-9
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, t_max, 5))

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t_max,
        dt0=1e-3 * t_max,
        y0=y_op,
        args=(groups, sys_size),
        saveat=saveat,
        max_steps=1000,
    )

    assert sol.ys.shape == (5, sys_size)
    assert jnp.isfinite(sol.ys).all()


@pytest.mark.parametrize("backend", _backends)
def test_short_transient_runs_complex(simple_optical_netlist, backend):
    net_dict, models_map = simple_optical_netlist
    groups, sys_size, port_map = compile_netlist(net_dict, models_map)

    # Solve DC operating point and run a very short transient with zero forcing
    linear_strat = analyze_circuit(groups, sys_size, backend=backend, is_complex=True)

    y_guess_flat = jnp.zeros(sys_size * 2, dtype=jnp.float64)
    y_op_flat = linear_strat.solve_dc(groups, y_guess_flat)

    assert y_op_flat.shape[0] == 2 * sys_size

    solver = VectorizedTransientSolver(linear_solver=linear_strat)
    term = diffrax.ODETerm(lambda t, y, args: jnp.zeros_like(y))

    t_max = 1.0e-9
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, t_max, 500))

    sol = diffrax.diffeqsolve(
        terms=term,
        solver=solver,
        t0=0.0,
        t1=t_max,
        dt0=1e-13,
        y0=y_op_flat,
        args=(groups, sys_size),
        saveat=saveat,
        max_steps=100000,
        throw=False,
        stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
    )

    assert sol.ys.shape == (500, 2 * sys_size)
    assert jnp.isfinite(sol.ys).all()


def test_factorized_transient_matches_vectorized(simple_lrc_netlist) -> None:  # noqa: ANN001
    """FactorizedTransientSolver (frozen-Jacobian) should produce the same trajectory
    as VectorizedTransientSolver (full Newton) on a linear LRC circuit."""
    from circulax.solvers.linear import split_solver_available

    if not split_solver_available:
        pytest.skip("requires klujax KLUHandleManager")

    from circulax.solvers.transient import FactorizedTransientSolver

    net_dict, models_map = simple_lrc_netlist
    groups, sys_size, _ = compile_netlist(net_dict, models_map)

    t_max = 1e-9
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, t_max, 5))
    term = diffrax.ODETerm(lambda t, y, args: jnp.zeros_like(y))

    strat_full = analyze_circuit(groups, sys_size, backend="klu_split")
    y_op = strat_full.solve_dc(groups, jnp.zeros(sys_size))

    sol_full = diffrax.diffeqsolve(
        term,
        VectorizedTransientSolver(linear_solver=strat_full),
        t0=0.0, t1=t_max, dt0=1e-3 * t_max, y0=y_op,
        args=(groups, sys_size), saveat=saveat, max_steps=1000,
    )

    strat_factor = analyze_circuit(groups, sys_size, backend="klu_split_factor")
    sol_factor = diffrax.diffeqsolve(
        term,
        FactorizedTransientSolver(linear_solver=strat_factor),
        t0=0.0, t1=t_max, dt0=1e-3 * t_max, y0=y_op,
        args=(groups, sys_size), saveat=saveat, max_steps=1000,
    )

    assert sol_factor.ys.shape == sol_full.ys.shape
    assert jnp.allclose(sol_factor.ys, sol_full.ys, atol=1e-4)
