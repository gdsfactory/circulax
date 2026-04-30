"""Run ONE PID config on the ring osc, print status.  Appends to
reports/bench_pid.csv so a crash in one config doesn't lose others.

Usage:
    pixi run python scripts/pid_one.py <dt0_ps> <dtmax_ps> <rtol> <atol> [t1_ns=100] [dtmin_ps=0] [force_dtmin=0]
"""
from __future__ import annotations

import argparse
import csv
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
from circulax.solvers.transient import TrapRefactoringTransientSolver

CSV = _REPO / "reports" / "bench_pid.csv"


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


def emit(row):
    CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dt0_ps", "dtmin_ps", "dtmax_ps", "rtol", "atol", "t1_ns",
              "status", "last_ok_ns", "accepted", "rejected",
              "freq_MHz", "wall_s"]
    exists = CSV.exists()
    with CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})
    print(" | ".join(f"{k}={row.get(k)!s}" for k in fields if row.get(k, "") != ""))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dt0_ps", type=float)
    p.add_argument("dtmax_ps", type=float)
    p.add_argument("rtol", type=float)
    p.add_argument("atol", type=float)
    p.add_argument("t1_ns", type=float, nargs="?", default=100.0)
    p.add_argument("dtmin_ps", type=float, nargs="?", default=0.0,
                   help="0 = effectively unbounded (1e-3 ps).  >0 sets a floor.")
    p.add_argument("force_dtmin", type=int, nargs="?", default=0,
                   help="1 = accept a step at dtmin rather than aborting.")
    args = p.parse_args()

    dt0 = args.dt0_ps * 1e-12
    dtmax = args.dtmax_ps * 1e-12
    dtmin = args.dtmin_ps * 1e-12 if args.dtmin_ps > 0 else 1e-15
    force_dtmin = bool(args.force_dtmin)
    t1 = args.t1_ns * 1e-9

    groups, sys_size, port_map = build_netlist(c_load=0)
    solver = analyze_circuit(groups, sys_size, backend="klu_split")
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, 1e-2)
    y_src = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y0 = solver.solve_dc_gmin(groups, y_src, g_start=1e-2, n_steps=30)
    y0 = jnp.asarray(y0)
    run = setup_transient(groups, solver, transient_solver=TrapRefactoringTransientSolver)

    ctrl = diffrax.PIDController(
        rtol=args.rtol, atol=args.atol, dtmin=dtmin, dtmax=dtmax,
        force_dtmin=force_dtmin,
    )
    # At least 10 points per fundamental period (assume ~290 MHz ring osc).
    n_save = max(400, int(t1 * 10 * 290e6))
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, t1, n_save))

    row = {"dt0_ps": args.dt0_ps, "dtmin_ps": args.dtmin_ps,
           "dtmax_ps": args.dtmax_ps, "rtol": args.rtol,
           "atol": args.atol, "t1_ns": args.t1_ns}

    t0 = time.time()
    try:
        sol = run(t0=0.0, t1=t1, dt0=dt0, y0=y0, saveat=saveat,
                  max_steps=1_000_000, stepsize_controller=ctrl, throw=False)
        sol.ys.block_until_ready()
    except Exception as e:
        row.update(status=f"EXCEPTION_{type(e).__name__}", wall_s=f"{time.time()-t0:.1f}")
        emit(row); return
    wall = time.time() - t0

    ys = np.asarray(sol.ys); t = np.asarray(sol.ts)
    n_acc = int(sol.stats.get("num_accepted_steps", 0) or 0)
    n_rej = int(sol.stats.get("num_rejected_steps", 0) or 0)

    finite = np.all(np.isfinite(ys), axis=1)
    if not finite.any():
        row.update(status="NaN_from_t0", accepted=n_acc, rejected=n_rej,
                   wall_s=f"{wall:.1f}")
        emit(row); return
    last_ok = float(t[np.where(finite)[0][-1]])

    if not finite.all():
        row.update(status="NaN", last_ok_ns=f"{last_ok*1e9:.2f}",
                   accepted=n_acc, rejected=n_rej, wall_s=f"{wall:.1f}")
        emit(row); return

    if sol.result == diffrax.RESULTS.successful:
        status = "ok"
    elif sol.result == diffrax.RESULTS.max_steps_reached:
        status = "max_steps"
    else:
        status = str(sol.result)

    n1 = port_map["n1,p1"]
    v = ys[:, n1]
    period = period_from_crossings(t[t > 20e-9], v[t > 20e-9])
    row.update(
        status=status, accepted=n_acc, rejected=n_rej, wall_s=f"{wall:.1f}",
        freq_MHz=f"{1e-6/period:.2f}" if period > 0 else "nan",
    )
    emit(row)


if __name__ == "__main__":
    main()
