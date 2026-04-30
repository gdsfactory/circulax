"""Bench TrapFactorized (frozen-Jac Newton inner loop) with and without
Tier-2 residual-only OSDI path.  This is where the new osdi_residual_eval
FFI entrypoint actually gets called — each Newton iter after the first
uses assemble_residual_only_* which now dispatches to osdi_residual_eval.
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
from circulax.solvers.transient import TrapFactorizedTransientSolver


def period_from_crossings(t, x):
    centered = x - x.mean()
    rising = np.where(np.diff(np.sign(centered)) > 0)[0]
    if len(rising) < 3:
        return float("nan")
    rising = rising[1:]
    times = []
    for i in rising:
        x0, x1 = float(centered[i]), float(centered[i + 1])
        t0, t1 = float(t[i]), float(t[i + 1])
        times.append(t0 - x0 * (t1 - t0) / (x1 - x0))
    if len(times) < 2:
        return float("nan")
    return float(np.median(np.diff(np.asarray(times))))


def main():
    t1 = 1e-6
    dt = 5e-11
    n_save = 4000
    n_steps = int(t1 / dt)

    groups, sys_size, port_map = build_netlist(c_load=0)
    solver = analyze_circuit(groups, sys_size, backend="klu_split")

    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, 1e-2)
    y_src = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y0 = solver.solve_dc_gmin(groups, y_src, g_start=1e-2, n_steps=30)
    y0 = jnp.asarray(y0)
    n1 = port_map["n1,p1"]

    run = setup_transient(
        groups, solver, transient_solver=TrapFactorizedTransientSolver,
    )
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, t1, n_save))
    controller = diffrax.ConstantStepSize()

    # Warmup (JIT compile).
    t_c = time.time()
    _ = run(t0=0.0, t1=10 * dt, dt0=dt, y0=y0,
            saveat=diffrax.SaveAt(ts=jnp.linspace(0, 10 * dt, 5)),
            max_steps=20, stepsize_controller=controller).ys.block_until_ready()
    print(f"JIT compile (warmup): {time.time() - t_c:.2f} s")

    # Timed run.
    t0 = time.time()
    sol = run(t0=0.0, t1=t1, dt0=dt, y0=y0, saveat=saveat,
              max_steps=2 * n_steps, stepsize_controller=controller)
    sol.ys.block_until_ready()
    wall = time.time() - t0

    v = np.asarray(sol.ys)[:, n1]
    t = np.asarray(sol.ts)
    period = period_from_crossings(t[t > 100e-9], v[t > 100e-9])
    freq = 1e-6 / period if period > 0 else float("nan")

    print(f"TrapFactorized  scalar  1 µs  dt={dt*1e12:.0f}ps  {n_steps} steps")
    print(f"  wall = {wall:.2f} s   ({wall/n_steps*1e6:.1f} µs/step)")
    print(f"  freq = {freq:.2f} MHz")


if __name__ == "__main__":
    main()
