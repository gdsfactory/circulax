"""Run ONE benchmark configuration for the ring oscillator vs VACASK.

Each invocation runs exactly one row, prints a single CSV-friendly result line,
and writes it (append) to reports/bench_ring.csv.  Designed to be invoked from
a tiny shell loop so a crash in one config doesn't lose the others.

Usage:
    pixi run python scripts/bench_one.py vacask
    pixi run python scripts/bench_one.py circulax <backend> <fixed|adaptive> [t1_ns]
    pixi run python scripts/bench_one.py vmap <backend> <batch>
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
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

from ring_one_case import build_netlist  # noqa: E402

VACASK_BIN = Path("/home/cdaunt/opt/vacask/bin/vacask")
VACASK_DIR = Path("/home/cdaunt/code/vacask/VACASK/benchmark/ring/vacask")
CSV_PATH = _REPO / "reports" / "bench_ring.csv"

T1 = 1e-6     # 1 µs
DT = 5e-11    # 50 ps


def period_from_crossings(t: np.ndarray, x: np.ndarray) -> float:
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


def _emit(row: dict) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case", "backend", "wall_s", "compile_s", "n_steps",
              "us_per_step", "freq_MHz", "batch", "notes"]
    exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})
    bits = " | ".join(f"{k}={row.get(k)!s:>10s}" for k in fields if row.get(k) != "")
    print(bits)


def run_vacask() -> None:
    proc = subprocess.run(
        [str(VACASK_BIN), "--skip-embed", "--skip-postprocess", "runme.sim"],
        cwd=str(VACASK_DIR), capture_output=True, text=True, timeout=600,
        check=False,
    )
    if proc.returncode != 0:
        _emit({"case": "vacask", "backend": "klu", "notes": f"failed_rc={proc.returncode}"})
        return
    elapsed = re.search(r"Elapsed time:\s+([\d.eE+\-]+)", proc.stdout)
    accepted = re.search(r"Accepted timepoints:\s+(\d+)", proc.stdout)
    rejected = re.search(r"Rejected timepoints:\s+(\d+)", proc.stdout)

    raw_in = VACASK_DIR / "tran1.raw"
    raw_local = _REPO / "tests" / "fixtures" / "_tmp_vacask.raw"
    shutil.copy(raw_in, raw_local)
    sys.path.insert(0, "/home/cdaunt/code/vacask/VACASK/python")
    from rawfile import rawread  # type: ignore
    r = rawread(str(raw_local)).get()
    arr = np.asarray(r.data)
    t_arr = arr[:, r.names.index("time")]
    v1_arr = arr[:, r.names.index("1")]
    period = period_from_crossings(t_arr[t_arr > 100e-9], v1_arr[t_arr > 100e-9])
    raw_local.unlink(missing_ok=True)

    n = int(accepted.group(1)) if accepted else 0
    el = float(elapsed.group(1)) if elapsed else float("nan")
    _emit({
        "case": "vacask", "backend": "klu", "wall_s": f"{el:.3f}",
        "n_steps": n, "us_per_step": f"{el / max(n, 1) * 1e6:.1f}",
        "freq_MHz": f"{1e-6 / period:.2f}" if period > 0 else "nan",
        "notes": f"adaptive_trap rejected={rejected.group(1) if rejected else '?'}",
    })


def _build_circulax(backend: str, n_stages: int = 9):
    from circulax.solvers import analyze_circuit, setup_transient
    from circulax.solvers.transient import TrapRefactoringTransientSolver
    groups, sys_size, port_map = build_netlist(c_load=0, n_stages=n_stages)
    solver = analyze_circuit(groups, sys_size, backend=backend)
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, 1e-2)
    y_src = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y0 = solver.solve_dc_gmin(groups, y_src, g_start=1e-2, n_steps=30)
    assert bool(jnp.all(jnp.isfinite(y0)))
    y0 = jnp.asarray(y0)
    run = setup_transient(groups, solver, transient_solver=TrapRefactoringTransientSolver)
    return groups, sys_size, port_map, run, y0


def run_circulax(backend: str, mode: str, t1_ns: float, n_stages: int = 9) -> None:
    t1 = t1_ns * 1e-9
    n_save = min(4000, int(t1 / DT))
    groups, sys_size, port_map, run, y0 = _build_circulax(backend, n_stages=n_stages)
    n1 = port_map["n1,p1"]

    if mode == "fixed":
        controller = diffrax.ConstantStepSize()
        max_steps = int(2 * t1 / DT)
    else:
        controller = diffrax.PIDController(rtol=1e-4, atol=1e-6,
                                           dtmin=1e-14, dtmax=DT)
        max_steps = int(5 * t1 / DT)

    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, t1, n_save))

    # JIT warmup over a short window so we don't pay the long-sim cost twice.
    t_compile = time.time()
    _ = run(t0=0.0, t1=10 * DT, dt0=DT, y0=y0,
            saveat=diffrax.SaveAt(ts=jnp.linspace(0, 10 * DT, 5)),
            max_steps=200, stepsize_controller=controller).ys.block_until_ready()
    compile_wall = time.time() - t_compile

    t_start = time.time()
    sol = run(t0=0.0, t1=t1, dt0=DT, y0=y0, saveat=saveat,
              max_steps=max_steps, stepsize_controller=controller)
    sol.ys.block_until_ready()
    wall = time.time() - t_start

    ys = np.asarray(sol.ys); t = np.asarray(sol.ts)
    if not np.all(np.isfinite(ys)):
        _emit({"case": f"circulax_{mode}", "backend": backend, "notes": "NaN"})
        return
    v = ys[:, n1]
    period = period_from_crossings(t[t > 100e-9], v[t > 100e-9])
    n_actual = int(sol.stats.get("num_accepted_steps") or sol.stats.get("num_steps", 0) or n_save)

    _emit({
        "case": f"circulax_{mode}", "backend": backend,
        "wall_s": f"{wall:.3f}", "compile_s": f"{compile_wall:.2f}",
        "n_steps": n_actual,
        "us_per_step": f"{wall / max(n_actual, 1) * 1e6:.1f}",
        "freq_MHz": f"{1e-6 / period:.2f}" if period > 0 else "nan",
        "notes": f"t1={t1_ns:.0f}ns n_stages={n_stages}",
    })


def run_vmap(backend: str, batch: int, n_stages: int = 9) -> None:
    """Time ``batch`` rings in parallel via jax.vmap on a SHORT 200 ns window.

    For a parameter-sweep workflow the per-ring length matters less than
    relative scaling, and a short window keeps memory low.
    """
    t1 = 200e-9
    n_save = 200
    groups, sys_size, port_map, run, y0 = _build_circulax(backend, n_stages=n_stages)
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, t1, n_save))
    controller = diffrax.ConstantStepSize()

    def sim_one(y_init):
        sol = run(t0=0.0, t1=t1, dt0=DT, y0=y_init, saveat=saveat,
                  max_steps=int(2 * t1 / DT), stepsize_controller=controller)
        n1 = port_map["n1,p1"]
        return sol.ys[:, n1]

    sim_batch = jax.jit(jax.vmap(sim_one))

    rng = np.random.default_rng(0)
    perturb = jnp.asarray(rng.normal(scale=1e-4, size=(batch, sys_size)))
    y0_batch = jnp.broadcast_to(y0, (batch, sys_size)) + perturb

    t_compile = time.time()
    _ = sim_batch(y0_batch).block_until_ready()
    compile_wall = time.time() - t_compile

    t_start = time.time()
    out = sim_batch(y0_batch); out.block_until_ready()
    wall = time.time() - t_start

    n_arr = np.asarray(out)
    n_finite = int(np.sum(np.all(np.isfinite(n_arr), axis=1)))
    _emit({
        "case": "circulax_vmap", "backend": backend, "wall_s": f"{wall:.3f}",
        "compile_s": f"{compile_wall:.2f}", "batch": batch,
        "us_per_step": f"{wall / batch / int(t1 / DT) * 1e6:.1f}",
        "notes": f"t1={t1*1e9:.0f}ns finite={n_finite}/{batch}",
    })


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("case", choices=["vacask", "circulax", "vmap"])
    p.add_argument("backend", nargs="?", default="klu_split")
    p.add_argument("mode", nargs="?", default="fixed")
    p.add_argument("t1_ns", nargs="?", type=float, default=1000.0)
    p.add_argument("batch", nargs="?", type=int, default=4)
    p.add_argument("--n-stages", type=int, default=9,
                   help="Ring osc stage count (odd, ≥ 3).  Default 9 matches the VACASK reference.")
    args = p.parse_args()

    if args.case == "vacask":
        run_vacask()
    elif args.case == "circulax":
        run_circulax(args.backend, args.mode, args.t1_ns, n_stages=args.n_stages)
    elif args.case == "vmap":
        batch = int(args.mode) if args.mode.isdigit() else args.batch
        run_vmap(args.backend, batch, n_stages=args.n_stages)


if __name__ == "__main__":
    main()
