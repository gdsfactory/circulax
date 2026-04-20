"""Sweep circulax transient integrators on the ring osc.

VACASK uses trapezoidal (tran_method="trap"). Circulax has BE (order 1),
BDF2 (order 2), SDIRK3 (order 3). If the integrator is the root cause,
we'd expect different periods from each of them.
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

from ring_one_case import build_netlist, dom_freq


def run(label: str, integrator_cls, dt0: float = 1e-12, t1: float = 30e-9):
    from circulax.solvers import analyze_circuit, setup_transient

    groups, sys_size, port_map = build_netlist(c_load=50e-15)
    solver = analyze_circuit(groups, sys_size, backend="klu_split")

    G_HOM = 1e-2
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, G_HOM)
    y_src = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y0 = solver.solve_dc_gmin(groups, y_src, g_start=G_HOM, n_steps=30)

    n1 = port_map["n1,p1"]
    run_fn = setup_transient(groups, solver, transient_solver=integrator_cls)
    try:
        sol = run_fn(
            t0=0.0, t1=t1, dt0=dt0, y0=y0,
            saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t1, 500)),
            max_steps=2_000_000,
            stepsize_controller=diffrax.PIDController(
                rtol=1e-4, atol=1e-6, dtmin=1e-14, dtmax=1e-10,
            ),
        )
    except Exception as e:
        print(f"  {label:<35s}  FAILED: {type(e).__name__}: {str(e)[:50]}")
        return
    ys = np.asarray(sol.ys)
    t = np.asarray(sol.ts)
    if not np.all(np.isfinite(ys)):
        print(f"  {label:<35s}  NaN in transient")
        return
    v = ys[:, n1]
    freq = dom_freq(t, v)
    print(
        f"  {label:<35s}  period={1e9/freq:>6.3f} ns  "
        f"freq={freq/1e6:>7.2f} MHz  swing={v.max() - v.min():.3f} V"
    )


if __name__ == "__main__":
    from circulax.solvers.transient import (
        BDF2FactorizedTransientSolver,
        BDF2RefactoringTransientSolver,
        BDF2VectorizedTransientSolver,
        SDIRK3FactorizedTransientSolver,
        SDIRK3RefactoringTransientSolver,
        SDIRK3VectorizedTransientSolver,
        VectorizedTransientSolver,
        FactorizedTransientSolver,
        RefactoringTransientSolver,
    )

    # Order-1 Backward Euler variants
    run("BE Vectorized",          VectorizedTransientSolver)
    run("BE Factorized",          FactorizedTransientSolver)
    run("BE Refactoring",         RefactoringTransientSolver)
    # Order-2 BDF2 variants
    run("BDF2 Vectorized",        BDF2VectorizedTransientSolver)
    run("BDF2 Factorized",        BDF2FactorizedTransientSolver)
    run("BDF2 Refactoring",       BDF2RefactoringTransientSolver)
    # Order-3 SDIRK3 variants
    run("SDIRK3 Vectorized",      SDIRK3VectorizedTransientSolver)
    run("SDIRK3 Factorized",      SDIRK3FactorizedTransientSolver)
    run("SDIRK3 Refactoring",     SDIRK3RefactoringTransientSolver)
