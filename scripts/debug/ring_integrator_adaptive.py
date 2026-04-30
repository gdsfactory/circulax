"""Sweep (integrator × step-size controller) combinations on the ring.

The user noted adaptive stepping and integrator choice interact — we've
already shown BDF2 alone gives 299 MHz vs SDIRK3's 166 MHz. This sweeps
tighter tolerances on PID and also fixed very-small dt to see how far
we can push each integrator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "tests"))
sys.path.insert(0, "/home/cdaunt/code/bosdi/src")
sys.path.insert(0, str(_REPO / "scripts"))

from benchmarks.ring.circulax.ring_one_case import build_netlist, dom_freq


def compile_ring(c_load: float = 0.0):
    from circulax.solvers import analyze_circuit
    groups, sys_size, port_map = build_netlist(c_load)
    solver = analyze_circuit(groups, sys_size, backend="klu_split")
    return groups, sys_size, port_map, solver


def dc_init(groups, sys_size, solver):
    G_HOM = 1e-2
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, G_HOM)
    y_src = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    return solver.solve_dc_gmin(groups, y_src, g_start=G_HOM, n_steps=30)


def run(label: str, *, integrator_cls, stepsize_controller, dt0: float = 1e-12,
        t1: float = 30e-9, groups, sys_size, port_map, solver, y0):
    from circulax.solvers import setup_transient

    n1 = port_map["n1,p1"]
    run_fn = setup_transient(groups, solver, transient_solver=integrator_cls)
    try:
        sol = run_fn(
            t0=0.0, t1=t1, dt0=dt0, y0=y0,
            saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t1, 1000)),
            max_steps=5_000_000,
            stepsize_controller=stepsize_controller,
        )
    except Exception as e:
        print(f"  {label:<48s}  FAILED: {type(e).__name__}: {str(e)[:50]}")
        return
    ys = np.asarray(sol.ys)
    t = np.asarray(sol.ts)
    if not np.all(np.isfinite(ys)):
        print(f"  {label:<48s}  NaN")
        return
    v = ys[:, n1]
    freq = dom_freq(t, v)
    n_steps = int(sol.stats.get("num_steps", 0) or 0)
    print(
        f"  {label:<48s}  period={1e9/freq:>6.3f} ns  "
        f"freq={freq/1e6:>7.2f} MHz  swing={v.max()-v.min():.3f}V  "
        f"[{n_steps} steps]"
    )


if __name__ == "__main__":
    from circulax.solvers.transient import (
        BDF2FactorizedTransientSolver,
        SDIRK3FactorizedTransientSolver,
        FactorizedTransientSolver,
    )

    groups, sys_size, port_map, solver = compile_ring(c_load=0.0)
    y0 = dc_init(groups, sys_size, solver)

    # stepsize controllers
    pid_loose  = diffrax.PIDController(rtol=1e-3, atol=1e-5, dtmin=1e-14, dtmax=1e-10)
    pid_med    = diffrax.PIDController(rtol=1e-5, atol=1e-7, dtmin=1e-14, dtmax=1e-10)
    pid_tight  = diffrax.PIDController(rtol=1e-7, atol=1e-9, dtmin=1e-14, dtmax=1e-11)
    pid_xtight = diffrax.PIDController(rtol=1e-9, atol=1e-11, dtmin=1e-14, dtmax=1e-12)
    fix_50ps   = diffrax.ConstantStepSize()  # will use dt0=50e-12
    fix_10ps   = diffrax.ConstantStepSize()  # will use dt0=10e-12
    fix_1ps    = diffrax.ConstantStepSize()  # will use dt0=1e-12

    for iname, icls in (("BE",   FactorizedTransientSolver),
                         ("BDF2", BDF2FactorizedTransientSolver),
                         ("SDIRK3", SDIRK3FactorizedTransientSolver)):
        print(f"[{iname}]")
        run(f"{iname} + fixed dt=50 ps",                groups=groups, sys_size=sys_size,
            port_map=port_map, solver=solver, y0=y0,
            integrator_cls=icls, stepsize_controller=fix_50ps, dt0=50e-12)
        run(f"{iname} + fixed dt=10 ps",                groups=groups, sys_size=sys_size,
            port_map=port_map, solver=solver, y0=y0,
            integrator_cls=icls, stepsize_controller=fix_10ps, dt0=10e-12)
        run(f"{iname} + fixed dt=1 ps",                 groups=groups, sys_size=sys_size,
            port_map=port_map, solver=solver, y0=y0,
            integrator_cls=icls, stepsize_controller=fix_1ps,  dt0=1e-12)
        run(f"{iname} + PID rtol=1e-3",                 groups=groups, sys_size=sys_size,
            port_map=port_map, solver=solver, y0=y0,
            integrator_cls=icls, stepsize_controller=pid_loose)
        run(f"{iname} + PID rtol=1e-5",                 groups=groups, sys_size=sys_size,
            port_map=port_map, solver=solver, y0=y0,
            integrator_cls=icls, stepsize_controller=pid_med)
        run(f"{iname} + PID rtol=1e-7",                 groups=groups, sys_size=sys_size,
            port_map=port_map, solver=solver, y0=y0,
            integrator_cls=icls, stepsize_controller=pid_tight)
        run(f"{iname} + PID rtol=1e-9",                 groups=groups, sys_size=sys_size,
            port_map=port_map, solver=solver, y0=y0,
            integrator_cls=icls, stepsize_controller=pid_xtight)
