"""Re-test all integrators on the ring with correct period extraction.

After fixing _dominant_frequency to use rising zero-crossings (FFT was
locking on 3rd harmonic), this script reports each integrator's actual
ring frequency and per-step wall time, with C_load=0 across the board.

Tests both the factor (frozen-Jac) and refactor (per-iter refactor)
variants for each integrator family.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "tests"))
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, "/home/cdaunt/code/bosdi/src")

from benchmarks.ring.circulax.ring_one_case import build_netlist
from circulax.solvers import analyze_circuit, setup_transient
from circulax.solvers.transient import (
    BDF2FactorizedTransientSolver,
    BDF2RefactoringTransientSolver,
    BDF2VectorizedTransientSolver,
    FactorizedTransientSolver,
    RefactoringTransientSolver,
    SDIRK3FactorizedTransientSolver,
    SDIRK3RefactoringTransientSolver,
    SDIRK3VectorizedTransientSolver,
    TrapFactorizedTransientSolver,
    TrapRefactoringTransientSolver,
    TrapVectorizedTransientSolver,
    VectorizedTransientSolver,
)


def period_from_crossings(t: np.ndarray, x: np.ndarray, *, mid: float | None = None) -> float:
    centered = x - x.mean() if mid is None else x - mid
    sign_changes = np.diff(np.sign(centered))
    rising = np.where(sign_changes > 0)[0]
    if len(rising) < 3:
        return float("nan")
    rising = rising[1:]
    times = []
    for i in rising:
        x0, x1 = float(centered[i]), float(centered[i + 1])
        t0, t1 = float(t[i]), float(t[i + 1])
        times.append(t0 - x0 * (t1 - t0) / (x1 - x0))
    return float(np.median(np.diff(np.asarray(times))))


def run_one(label: str, integrator_cls, *, c_load: float, dt0: float, t1: float):
    groups, sys_size, port_map = build_netlist(c_load)
    solver = analyze_circuit(groups, sys_size, backend="klu_split")
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, 1e-2)
    y_src = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y0 = solver.solve_dc_gmin(groups, y_src, g_start=1e-2, n_steps=30)

    n1 = port_map["n1,p1"]
    run = setup_transient(groups, solver, transient_solver=integrator_cls)
    t_start = time.time()
    try:
        sol = run(
            t0=0.0, t1=t1, dt0=dt0, y0=y0,
            saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t1, 1000)),
            max_steps=2_000_000,
            stepsize_controller=diffrax.ConstantStepSize(),
        )
    except Exception as e:
        print(f"  {label:<30s}  c_load={c_load*1e15:>4.0f}fF  FAILED: {type(e).__name__}: {str(e)[:50]}")
        return
    wall = time.time() - t_start

    t = np.asarray(sol.ts)
    v = np.asarray(sol.ys)[:, n1]
    if not np.all(np.isfinite(v)):
        print(f"  {label:<30s}  c_load={c_load*1e15:>4.0f}fF  NaN")
        return

    period = period_from_crossings(t, v)
    swing = float(v.max() - v.min())
    f_mhz = 1e-6 / period if np.isfinite(period) else float("nan")
    ratio = f_mhz / 289.60   # VACASK trap reference
    print(f"  {label:<30s}  c_load={c_load*1e15:>4.0f}fF  "
          f"period={period*1e9:>5.3f}ns  freq={f_mhz:>7.2f}MHz  "
          f"ratio={ratio:.3f}  swing={swing:.3f}V  [{wall:>5.1f}s]")


if __name__ == "__main__":
    print("VACASK trap (fundamental, from zero-crossings): 3.45 ns / 289.60 MHz\n")
    cases: list[tuple[str, type, float]] = [
        # (label, integrator_class, c_load_fF)
        ("Trap-Vectorized",       TrapVectorizedTransientSolver,        0),
        ("Trap-Factorized",       TrapFactorizedTransientSolver,        0),
        ("Trap-Refactoring",      TrapRefactoringTransientSolver,       0),
        ("BDF2-Vectorized",       BDF2VectorizedTransientSolver,        0),
        ("BDF2-Factorized",       BDF2FactorizedTransientSolver,        0),
        ("BDF2-Refactoring",      BDF2RefactoringTransientSolver,       0),
        ("BE-Vectorized",         VectorizedTransientSolver,            0),
        ("BE-Factorized",         FactorizedTransientSolver,            0),
        ("BE-Refactoring",        RefactoringTransientSolver,           0),
        ("SDIRK3-Vectorized",     SDIRK3VectorizedTransientSolver,      0),
        ("SDIRK3-Factorized",     SDIRK3FactorizedTransientSolver,      0),
        ("SDIRK3-Refactoring",    SDIRK3RefactoringTransientSolver,     0),
        # SDIRK3 with the old "stability" cap, as a control
        ("SDIRK3-Refactoring +50fF", SDIRK3RefactoringTransientSolver, 50e-15),
    ]
    for label, cls, c_load in cases:
        run_one(label, cls, c_load=c_load, dt0=5e-12, t1=30e-9)
