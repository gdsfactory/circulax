"""Run ONE (integrator × controller) combo, append result to reports/ring_integ.csv.

Invoke in a shell loop. Usage:
    pixi run python scripts/ring_integ_one.py <integrator> <controller> [args...]

<integrator> in {BE, BDF2, SDIRK3}
<controller> in {fixed, pid}
For fixed: pass   dt_ps              e.g. 50
For pid:   pass   rtol atol          e.g. 1e-5 1e-7
Optional: --cload <fF>  --t1-ns <N>   (default c_load=0, t1=30)
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
sys.path.insert(0, "/home/cdaunt/code/bosdi/src")
sys.path.insert(0, str(_REPO / "scripts"))

CSV_PATH = _REPO / "reports" / "ring_integ.csv"


def dom_freq(t: np.ndarray, x: np.ndarray) -> float:
    x = x - x.mean()
    dt = float(t[1] - t[0])
    freqs = np.fft.rfftfreq(len(x), d=dt)
    power = np.abs(np.fft.rfft(x))
    power[0] = 0.0
    return float(freqs[int(np.argmax(power))])


def append_csv(row: dict) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "integrator", "controller", "ctrl_params", "c_load_fF",
        "status", "freq_MHz", "period_ns", "swing_V", "num_steps", "wall_s",
    ]
    exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("integrator", choices=["BE", "BDF2", "SDIRK3"])
    p.add_argument("controller", choices=["fixed", "pid"])
    p.add_argument("args", nargs="+", type=float,
                   help="fixed: dt_ps ; pid: rtol atol")
    p.add_argument("--cload", type=float, default=0.0, help="C_load in fF")
    p.add_argument("--t1-ns", type=float, default=30.0)
    opts = p.parse_args()

    from ring_one_case import build_netlist
    from circulax.solvers import analyze_circuit, setup_transient
    from circulax.solvers.transient import (
        BDF2FactorizedTransientSolver,
        FactorizedTransientSolver,
        SDIRK3FactorizedTransientSolver,
    )

    integrators = {
        "BE":     FactorizedTransientSolver,
        "BDF2":   BDF2FactorizedTransientSolver,
        "SDIRK3": SDIRK3FactorizedTransientSolver,
    }
    icls = integrators[opts.integrator]

    t_start = time.time()
    groups, sys_size, port_map = build_netlist(opts.cload * 1e-15)
    solver = analyze_circuit(groups, sys_size, backend="klu_split")

    G_HOM = 1e-2
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, G_HOM)
    y_src = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y0 = solver.solve_dc_gmin(groups, y_src, g_start=G_HOM, n_steps=30)

    if opts.controller == "fixed":
        dt_ps = opts.args[0]
        ctrl = diffrax.ConstantStepSize()
        dt0 = dt_ps * 1e-12
        ctrl_desc = f"dt={dt_ps:g}ps"
    else:
        rtol, atol = opts.args[0], opts.args[1]
        ctrl = diffrax.PIDController(
            rtol=rtol, atol=atol, dtmin=1e-15, dtmax=1e-10,
        )
        dt0 = 1e-12
        ctrl_desc = f"rtol={rtol:g} atol={atol:g}"

    t1 = opts.t1_ns * 1e-9
    n1 = port_map["n1,p1"]
    run_fn = setup_transient(groups, solver, transient_solver=icls)

    row = {
        "integrator": opts.integrator,
        "controller": opts.controller,
        "ctrl_params": ctrl_desc,
        "c_load_fF": opts.cload,
    }
    try:
        sol = run_fn(
            t0=0.0, t1=t1, dt0=dt0, y0=y0,
            saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t1, 1000)),
            max_steps=5_000_000,
            stepsize_controller=ctrl,
        )
    except Exception as e:
        row["status"] = f"ERROR_{type(e).__name__}"
        row["wall_s"] = time.time() - t_start
        append_csv(row)
        print(f"{opts.integrator:<7} {ctrl_desc:<25} c_load={opts.cload:g}fF  FAILED: {type(e).__name__}")
        return

    ys = np.asarray(sol.ys)
    t = np.asarray(sol.ts)
    n_steps = int(sol.stats.get("num_steps", 0) or 0)
    if not np.all(np.isfinite(ys)):
        row["status"] = "NaN"
        row["num_steps"] = n_steps
        row["wall_s"] = time.time() - t_start
        append_csv(row)
        print(f"{opts.integrator:<7} {ctrl_desc:<25} c_load={opts.cload:g}fF  NaN")
        return

    v = ys[:, n1]
    freq = dom_freq(t, v)
    row.update({
        "status": "ok",
        "freq_MHz": freq / 1e6,
        "period_ns": 1e9 / freq if freq > 0 else float("inf"),
        "swing_V": float(v.max() - v.min()),
        "num_steps": n_steps,
        "wall_s": time.time() - t_start,
    })
    append_csv(row)
    print(
        f"{opts.integrator:<7} {ctrl_desc:<25} c_load={opts.cload:g}fF  "
        f"period={row['period_ns']:>6.3f}ns  freq={row['freq_MHz']:>7.2f}MHz  "
        f"swing={row['swing_V']:.3f}V  [{n_steps} steps, {row['wall_s']:.1f}s]"
    )


if __name__ == "__main__":
    main()
