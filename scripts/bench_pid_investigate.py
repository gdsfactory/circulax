"""Diagnose why the PID adaptive stepper NaN'd on the PSP103 ring.

Sweeps dtmax, dt0, and rtol/atol on the same ring oscillator we benchmarked.
For each case: capture wall time, final time reached (so we see if it NaN'd
mid-run vs completed), number of accepted/rejected steps, oscillation freq.
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

from ring_one_case import build_netlist
from circulax.solvers import analyze_circuit, setup_transient
from circulax.solvers.transient import TrapRefactoringTransientSolver

T1_FULL = 1e-6      # target sim window (VACASK matches)
T1_SHORT = 100e-9   # short window for debugging NaNs


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


_CACHE = {}
def _build_ring():
    if "ring" in _CACHE:
        return _CACHE["ring"]
    groups, sys_size, port_map = build_netlist(c_load=0)
    solver = analyze_circuit(groups, sys_size, backend="klu_split")
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, 1e-2)
    y_src = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y0 = solver.solve_dc_gmin(groups, y_src, g_start=1e-2, n_steps=30)
    y0 = jnp.asarray(y0)
    run = setup_transient(groups, solver, transient_solver=TrapRefactoringTransientSolver)
    _CACHE["ring"] = (groups, sys_size, port_map, run, y0)
    return _CACHE["ring"]


def try_pid(label, *, dt0, dtmax, rtol, atol, t1):
    groups, sys_size, port_map, run, y0 = _build_ring()
    n1 = port_map["n1,p1"]

    controller = diffrax.PIDController(
        rtol=rtol, atol=atol, dtmin=1e-15, dtmax=dtmax, force_dtmin=False,
    )
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, t1, 400))
    t_start = time.time()
    try:
        sol = run(
            t0=0.0, t1=t1, dt0=dt0, y0=y0, saveat=saveat,
            max_steps=500_000, stepsize_controller=controller, throw=False,
        )
        sol.ys.block_until_ready()
    except Exception as e:
        wall = time.time() - t_start
        print(f"  {label:<56s}  EXCEPTION: {type(e).__name__}: {str(e)[:60]}  [{wall:.1f}s]")
        return
    wall = time.time() - t_start

    ys = np.asarray(sol.ys)
    t = np.asarray(sol.ts)

    finite_mask = np.all(np.isfinite(ys), axis=1)
    if not finite_mask.any():
        print(f"  {label:<56s}  NaN from step 0   [{wall:.1f}s]")
        return
    last_ok_idx = int(np.where(finite_mask)[0][-1])
    last_ok_t = t[last_ok_idx]

    # sol.result is a diffrax.EnumerationItem; compare by identity/equality.
    if sol.result == diffrax.RESULTS.successful:
        result_str = "ok"
    elif sol.result == diffrax.RESULTS.max_steps_reached:
        result_str = "max_steps"
    elif sol.result == diffrax.RESULTS.nonlinear_divergence:
        result_str = "divergence"
    else:
        result_str = str(sol.result)

    n_acc = int(sol.stats.get("num_accepted_steps", 0) or 0)
    n_rej = int(sol.stats.get("num_rejected_steps", 0) or 0)

    if not finite_mask.all():
        print(
            f"  {label:<56s}  NaN after t={last_ok_t*1e9:.2f}ns  "
            f"accepted={n_acc} rejected={n_rej}  result={result_str}  [{wall:.1f}s]"
        )
        return

    v = ys[:, n1]
    period = period_from_crossings(t[t > 20e-9], v[t > 20e-9])
    freq_mhz = 1e-6 / period if period > 0 else float("nan")
    print(
        f"  {label:<56s}  ok  t1={t1*1e9:.0f}ns  freq={freq_mhz:7.2f}MHz  "
        f"accepted={n_acc} rejected={n_rej}  [{wall:.1f}s]"
    )


if __name__ == "__main__":
    print("[diagnostic] VACASK reference: 289.6 MHz, 24764 adaptive-trap steps, 1.19s\n")
    print("=== 1. Short window (100 ns) to classify failure mode ===")
    # Find the minimum-friction config that completes the short window.
    for dtmax in (50e-12, 10e-12, 5e-12, 1e-12):
        try_pid(f"dtmax={dtmax*1e12:>4.1f}ps dt0=1e-13 rtol=1e-4 atol=1e-6",
                dt0=1e-13, dtmax=dtmax, rtol=1e-4, atol=1e-6, t1=T1_SHORT)

    print("\n=== 2. With dtmax=5ps, vary rtol/atol ===")
    for rtol, atol in ((1e-3, 1e-5), (1e-4, 1e-6), (1e-5, 1e-7), (1e-6, 1e-8)):
        try_pid(f"dtmax=5ps   dt0=1e-13 rtol={rtol:g} atol={atol:g}",
                dt0=1e-13, dtmax=5e-12, rtol=rtol, atol=atol, t1=T1_SHORT)

    print("\n=== 3. Best case on full 1 µs window ===")
    for rtol, atol in ((1e-4, 1e-6), (1e-5, 1e-7)):
        try_pid(f"dtmax=5ps   dt0=1e-13 rtol={rtol:g} atol={atol:g} t1=1us",
                dt0=1e-13, dtmax=5e-12, rtol=rtol, atol=atol, t1=T1_FULL)

    print("\n=== 4. Initial-step sensitivity (dt0) at same tolerance ===")
    for dt0 in (5e-11, 1e-11, 1e-12, 1e-13, 1e-14):
        try_pid(f"dt0={dt0:.0e} dtmax=50ps rtol=1e-4",
                dt0=dt0, dtmax=50e-12, rtol=1e-4, atol=1e-6, t1=T1_SHORT)
