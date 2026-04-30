"""Benchmark a parameter-optimisation loop (repeated sim with swapped params).

Simulates the realistic workflow: compile topology once, then loop with
different param values per iteration — the canonical case for device
sizing, biasing optimisation, process-corner sweeps, etc.

Compares two patterns:

  A. NAIVE — rebuild the whole netlist per iter (re-run compile_netlist).
     Handle comes along for free but we pay for union-find, bucket
     assignment, COO sparsity, Jacobian setup, etc. each iteration.

  B. FAST — ``group.with_params(new_params)`` between iters. Topology
     metadata is reused; only the params array and Tier-3 handle change.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import diffrax
import equinox as eqx
import jax.numpy as jnp
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "tests"))
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, "/home/cdaunt/code/bosdi/src")

from benchmarks.ring.circulax.ring_one_case import build_netlist
from circulax.solvers import analyze_circuit, setup_transient
from circulax.solvers.transient import TrapRefactoringTransientSolver


T1 = 50e-9
DT = 5e-12
N_SAVE = 1000


def run_transient(run, groups, sys_size):
    """Invoke the transient driver starting from DC init.  Excludes setup."""
    high_gmin = eqx.tree_at(lambda s: s.g_leak, _SOLVER_CACHE["solver"], 1e-2)
    y_src = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y0 = _SOLVER_CACHE["solver"].solve_dc_gmin(groups, y_src, g_start=1e-2, n_steps=30)
    sol = run(
        t0=0.0, t1=T1, dt0=DT, y0=y0,
        saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, T1, N_SAVE)),
        max_steps=2 * int(T1 / DT),
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    return sol.ys.block_until_ready()


_SOLVER_CACHE: dict = {}


def run_naive(n_iters: int) -> float:
    """Rebuild everything per iter: compile_netlist → solver → setup_transient."""
    t_total = time.time()
    for _ in range(n_iters):
        groups, sys_size, port_map = build_netlist(c_load=0)
        solver = analyze_circuit(groups, sys_size, backend="klu_rs_split")
        _SOLVER_CACHE["solver"] = solver
        run = setup_transient(groups, solver, transient_solver=TrapRefactoringTransientSolver)
        _ = run_transient(run, groups, sys_size)
    return time.time() - t_total


def run_fast(n_iters: int) -> float:
    """Topology + solver + run-fn built ONCE; only with_params per iter."""
    groups, sys_size, port_map = build_netlist(c_load=0)
    solver = analyze_circuit(groups, sys_size, backend="klu_rs_split")
    _SOLVER_CACHE["solver"] = solver
    run = setup_transient(groups, solver, transient_solver=TrapRefactoringTransientSolver)
    baseline = {name: np.asarray(g.params) for name, g in groups.items()
                if hasattr(g, "handle")}

    t_total = time.time()
    for _ in range(n_iters):
        # Swap params on each OSDI group (rebuild handle inside).
        for name, p0 in baseline.items():
            groups[name] = groups[name].with_params(p0)
        _ = run_transient(run, groups, sys_size)
    return time.time() - t_total


def main() -> None:
    # Warmup JIT.  run_fast does it internally; run_naive doesn't cache the
    # run-fn so it re-JITs each iter.  That's exactly the point of the bench.
    groups, sys_size, port_map = build_netlist(c_load=0)
    solver = analyze_circuit(groups, sys_size, backend="klu_rs_split")
    _SOLVER_CACHE["solver"] = solver
    run = setup_transient(groups, solver, transient_solver=TrapRefactoringTransientSolver)
    _ = run_transient(run, groups, sys_size)

    for n_iters in (3, 10):
        t_naive = run_naive(n_iters)
        t_fast = run_fast(n_iters)
        per_naive = t_naive / n_iters
        per_fast = t_fast / n_iters
        print(
            f"n_iters = {n_iters:>2d}  |  "
            f"naive {t_naive:>6.2f} s ({per_naive:>5.2f} s/iter)  |  "
            f"fast {t_fast:>6.2f} s ({per_fast:>5.2f} s/iter)  |  "
            f"speedup {t_naive / t_fast:.2f}×"
        )


if __name__ == "__main__":
    main()
