"""Wall-clock benchmark: PSP103 ring oscillator, circulax vs VACASK.

Times the same 1 µs / dt=50 ps transient (matching VACASK's
``benchmark/ring/vacask/runme.sim``) under three configurations:

1. VACASK trap, fixed dt=50 ps, with KLU.
2. circulax Trap-Refactoring + klujax (KLUSplitQuadratic).
3. circulax Trap-Refactoring + klujax-rs (KLURSplitQuadratic).

Trap is circulax's default integrator and matches VACASK's trap to 0.5 %
on this circuit (see docs/bosdi_psp103_ring_oscillator_issue.md).  The
*Refactoring* variant is the right pick for strongly-nonlinear PSP103
stacks because it re-factorises the Jacobian at every Newton iteration
(full quadratic convergence).  klujax-rs (the Rust KLU port) is included
for comparison — both backends produce identical results to roundoff;
this benchmark just measures wall time.

Wall time excludes JAX JIT compilation: each circulax case runs once as a
warmup, then a second call is timed.  VACASK's reported "Elapsed time" is
its end-to-end transient wall (it has no JIT step).

Usage:
    pixi run python scripts/bench_ring_vs_vacask.py
"""

from __future__ import annotations

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

# Sim parameters — keep in sync with VACASK's runme.sim:
#   analysis tran1 tran step=0.05n stop=1u maxstep=0.05n
T1 = 1e-6     # 1 µs
DT = 5e-11    # 50 ps
N_STEPS = int(round(T1 / DT))   # 20 000

# circulax can't store every step (would OOM); save 4000 points evenly.
N_SAVE = 4000


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


def time_vacask() -> dict:
    """Run VACASK on the upstream ring netlist; return wall-time + freq."""
    if not VACASK_BIN.exists():
        return {"status": "vacask binary not found"}
    # VACASK reports its own elapsed time after "Elapsed time:".
    t_wall = time.time()
    proc = subprocess.run(
        [str(VACASK_BIN), "--skip-embed", "--skip-postprocess",
         "runme.sim"],
        cwd=str(VACASK_DIR), capture_output=True, text=True, timeout=600,
        check=False,
    )
    wall = time.time() - t_wall
    if proc.returncode != 0:
        return {"status": f"vacask failed: {proc.returncode}",
                "stderr": proc.stderr[-200:]}

    elapsed_match = re.search(r"Elapsed time:\s+([\d.eE+\-]+)", proc.stdout)
    accepted_match = re.search(r"Accepted timepoints:\s+(\d+)", proc.stdout)
    rejected_match = re.search(r"Rejected timepoints:\s+(\d+)", proc.stdout)

    # Extract frequency from the produced raw file.
    raw_in = VACASK_DIR / "tran1.raw"
    raw_local = _REPO / "tests" / "fixtures" / "vacask_ring_ref.raw"
    shutil.copy(raw_in, raw_local)
    sys.path.insert(0, "/home/cdaunt/code/vacask/VACASK/python")
    from rawfile import rawread  # type: ignore
    r = rawread(str(raw_local)).get()
    arr = np.asarray(r.data)
    t_arr = arr[:, r.names.index("time")]
    v1_arr = arr[:, r.names.index("1")]
    mask = t_arr > 100e-9
    period = period_from_crossings(t_arr[mask], v1_arr[mask])
    raw_local.unlink(missing_ok=True)

    return {
        "status": "ok",
        "subprocess_wall": wall,
        "vacask_elapsed": float(elapsed_match.group(1)) if elapsed_match else None,
        "n_accepted": int(accepted_match.group(1)) if accepted_match else None,
        "n_rejected": int(rejected_match.group(1)) if rejected_match else None,
        "period_ns": period * 1e9,
        "freq_MHz": 1e-6 / period if period > 0 else float("nan"),
    }


def _build_circulax(backend: str):
    """Compile the ring and DC-init it.  Shared setup for the timed runs."""
    from circulax.solvers import analyze_circuit, setup_transient
    from circulax.solvers.transient import TrapRefactoringTransientSolver

    groups, sys_size, port_map = build_netlist(c_load=0)
    solver = analyze_circuit(groups, sys_size, backend=backend)

    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, 1e-2)
    y_src = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y0 = solver.solve_dc_gmin(groups, y_src, g_start=1e-2, n_steps=30)
    assert bool(jnp.all(jnp.isfinite(y0))), "DC diverged"
    y0 = jnp.asarray(y0)
    run = setup_transient(groups, solver,
                          transient_solver=TrapRefactoringTransientSolver)
    return groups, sys_size, port_map, solver, run, y0


def time_circulax_fixed(backend: str) -> dict:
    """Single-circuit, fixed dt=50 ps run (matches VACASK's dt)."""
    try:
        groups, sys_size, port_map, solver, run, y0 = _build_circulax(backend)
    except Exception as e:
        return {"status": f"backend {backend!r} unavailable: {type(e).__name__}"}

    n1 = port_map["n1,p1"]
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, T1, N_SAVE))
    controller = diffrax.ConstantStepSize()

    t_compile = time.time()
    _ = run(t0=0.0, t1=10 * DT, dt0=DT, y0=y0,
            saveat=diffrax.SaveAt(ts=jnp.linspace(0, 10 * DT, 5)),
            max_steps=20, stepsize_controller=controller).ys.block_until_ready()
    compile_wall = time.time() - t_compile

    t_start = time.time()
    sol = run(t0=0.0, t1=T1, dt0=DT, y0=y0, saveat=saveat,
              max_steps=int(2 * N_STEPS), stepsize_controller=controller)
    sol.ys.block_until_ready()
    wall = time.time() - t_start

    ys = np.asarray(sol.ys)
    if not np.all(np.isfinite(ys)):
        return {"status": "NaN in transient"}
    v = np.asarray(ys[:, n1])
    t = np.asarray(sol.ts)
    period = period_from_crossings(t[t > 100e-9], v[t > 100e-9])

    return {"status": "ok", "wall": wall, "compile_wall": compile_wall,
            "n_steps": N_STEPS, "us_per_step": wall / N_STEPS * 1e6,
            "period_ns": period * 1e9,
            "freq_MHz": 1e-6 / period if period > 0 else float("nan")}


def time_circulax_adaptive(backend: str) -> dict:
    """Single-circuit, PID-adaptive run — fairer comparison vs VACASK's
    adaptive stepping.
    """
    try:
        groups, sys_size, port_map, solver, run, y0 = _build_circulax(backend)
    except Exception as e:
        return {"status": f"backend {backend!r} unavailable: {type(e).__name__}"}

    n1 = port_map["n1,p1"]
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, T1, N_SAVE))
    controller = diffrax.PIDController(
        rtol=1e-4, atol=1e-6, dtmin=1e-14, dtmax=DT,
    )

    t_compile = time.time()
    _ = run(t0=0.0, t1=10 * DT, dt0=DT, y0=y0,
            saveat=diffrax.SaveAt(ts=jnp.linspace(0, 10 * DT, 5)),
            max_steps=200, stepsize_controller=controller).ys.block_until_ready()
    compile_wall = time.time() - t_compile

    t_start = time.time()
    sol = run(t0=0.0, t1=T1, dt0=DT, y0=y0, saveat=saveat,
              max_steps=int(5 * N_STEPS), stepsize_controller=controller)
    sol.ys.block_until_ready()
    wall = time.time() - t_start

    ys = np.asarray(sol.ys)
    if not np.all(np.isfinite(ys)):
        return {"status": "NaN in transient"}
    v = np.asarray(ys[:, n1])
    t = np.asarray(sol.ts)
    period = period_from_crossings(t[t > 100e-9], v[t > 100e-9])
    n_steps_actual = int(sol.stats.get("num_accepted_steps") or sol.stats.get("num_steps", 0) or 0)

    return {"status": "ok", "wall": wall, "compile_wall": compile_wall,
            "n_steps": n_steps_actual,
            "us_per_step": wall / max(n_steps_actual, 1) * 1e6,
            "period_ns": period * 1e9,
            "freq_MHz": 1e-6 / period if period > 0 else float("nan")}


def time_circulax_vmap(backend: str, batch: int = 16) -> dict:
    """Time ``batch`` rings in parallel via jax.vmap — circulax's headline
    advantage over scalar simulators like VACASK.

    Each ring has slightly perturbed VDD (parameter sweep); the entire
    batch runs in one JIT-compiled XLA program.
    """
    try:
        groups, sys_size, port_map, solver, run, y0 = _build_circulax(backend)
    except Exception as e:
        return {"status": f"backend {backend!r} unavailable: {type(e).__name__}"}

    # The vmap target is a single transient call; we batch the initial
    # condition (which is the cleanest knob for a parameter sweep without
    # re-running compile_netlist per case).
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, T1, N_SAVE))
    controller = diffrax.ConstantStepSize()

    def sim_one(y_init):
        sol = run(t0=0.0, t1=T1, dt0=DT, y0=y_init, saveat=saveat,
                  max_steps=int(2 * N_STEPS), stepsize_controller=controller)
        n1 = port_map["n1,p1"]
        return sol.ys[:, n1]

    sim_batch = jax.jit(jax.vmap(sim_one))

    # Slightly perturb each replica's initial condition.
    rng = np.random.default_rng(0)
    perturb = jnp.asarray(rng.normal(scale=1e-4, size=(batch, sys_size)))
    y0_batch = jnp.broadcast_to(y0, (batch, sys_size)) + perturb

    t_compile = time.time()
    _ = sim_batch(y0_batch).block_until_ready()
    compile_wall = time.time() - t_compile

    t_start = time.time()
    out = sim_batch(y0_batch)
    out.block_until_ready()
    wall = time.time() - t_start

    return {"status": "ok", "wall": wall, "compile_wall": compile_wall,
            "batch": batch, "us_per_ring": wall / batch * 1e6,
            "speedup_per_ring_vs_serial": batch}


def main() -> None:
    print("=" * 78)
    print("Benchmark: PSP103 9-stage ring oscillator, 1 µs at dt=50 ps")
    print(f"  20 000 timesteps if fixed-dt; matches VACASK runme.sim")
    print(f"  circulax integrator: TrapRefactoring  (default for circulax)")
    print("=" * 78)

    print("\n[VACASK trap, KLU, adaptive]")
    v = time_vacask()
    if v["status"] != "ok":
        print(f"  {v}")
    else:
        print(f"  subprocess wall        : {v['subprocess_wall']:.3f} s")
        if v["vacask_elapsed"] is not None:
            print(f"  vacask 'Elapsed time'  : {v['vacask_elapsed']:.3f} s")
        if v["n_accepted"]:
            us = v["vacask_elapsed"] / v["n_accepted"] * 1e6 if v["vacask_elapsed"] else None
            print(f"  accepted / rejected    : {v['n_accepted']} / {v['n_rejected']}")
            if us is not None:
                print(f"  µs per accepted step   : {us:.1f}")
        print(f"  period / freq          : {v['period_ns']:.3f} ns / {v['freq_MHz']:.2f} MHz")

    vacask_wall = v.get("vacask_elapsed") if v.get("status") == "ok" else None

    for label, backend in (
        ("circulax + klujax (klu_split)",       "klu_split"),
        ("circulax + klujax-rs (klu_rs_split)", "klu_rs_split"),
    ):
        print(f"\n--- {label} ---")
        for run_label, fn in (
            ("fixed dt=50 ps", lambda b=backend: time_circulax_fixed(b)),
            ("adaptive PID rtol=1e-4", lambda b=backend: time_circulax_adaptive(b)),
        ):
            print(f"\n  [{run_label}]")
            c = fn()
            if c["status"] != "ok":
                print(f"    {c}"); continue
            print(f"    JIT compile (warmup)   : {c['compile_wall']:.2f} s  (one-time)")
            print(f"    transient wall         : {c['wall']:.3f} s")
            print(f"    timesteps              : {c['n_steps']}")
            print(f"    µs per step            : {c['us_per_step']:.1f}")
            print(f"    period / freq          : {c['period_ns']:.3f} ns / {c['freq_MHz']:.2f} MHz")
            if vacask_wall:
                ratio = c["wall"] / vacask_wall
                msg = f"{ratio:.1f}× slower" if ratio > 1 else f"{1/ratio:.1f}× FASTER"
                print(f"    vs VACASK wall         : {msg}")

        print(f"\n  [vmap batch=16, fixed dt=50 ps — VACASK has no equivalent]")
        c = time_circulax_vmap(backend, batch=16)
        if c["status"] != "ok":
            print(f"    {c}")
            continue
        print(f"    JIT compile (warmup)   : {c['compile_wall']:.2f} s  (one-time)")
        print(f"    16 rings in parallel   : {c['wall']:.3f} s")
        print(f"    effective µs per ring  : {c['us_per_ring']:.0f}")
        if vacask_wall:
            ratio_per_ring = (c["wall"] / 16) / vacask_wall
            msg = f"{ratio_per_ring:.2f}× slower per ring" if ratio_per_ring > 1 \
                  else f"{1/ratio_per_ring:.1f}× FASTER per ring"
            print(f"    per-ring vs serial VACASK : {msg}")


if __name__ == "__main__":
    main()


def main() -> None:
    print("=" * 78)
    print("Benchmark: PSP103 9-stage ring oscillator, 1 µs at dt=50 ps")
    print(f"  20 000 timesteps, fixed step (matches VACASK runme.sim)")
    print(f"  circulax integrator: TrapRefactoring (the default for v0.X)")
    print("=" * 78)

    print("\n[VACASK]")
    v = time_vacask()
    if v["status"] != "ok":
        print(f"  {v}")
    else:
        print(f"  status                 : ok")
        print(f"  subprocess wall        : {v['subprocess_wall']:.3f} s")
        if v["vacask_elapsed"] is not None:
            print(f"  vacask 'Elapsed time'  : {v['vacask_elapsed']:.3f} s")
        if v["n_accepted"] is not None:
            us = v["vacask_elapsed"] / v["n_accepted"] * 1e6 if v["vacask_elapsed"] else None
            print(f"  accepted / rejected    : {v['n_accepted']} / {v['n_rejected']}")
            if us is not None:
                print(f"  µs per accepted step   : {us:.1f}")
        print(f"  period / freq          : {v['period_ns']:.3f} ns / {v['freq_MHz']:.2f} MHz")

    for label, backend in (("circulax + klujax (klu_split)",      "klu_split"),
                           ("circulax + klujax-rs (klu_rs_split)", "klu_rs_split")):
        print(f"\n[{label}]")
        c = time_circulax(backend)
        if c["status"] != "ok":
            print(f"  {c}")
            continue
        print(f"  status                 : ok")
        print(f"  JAX compile (warmup)   : {c['compile_wall']:.2f} s  (one-time)")
        print(f"  transient wall         : {c['wall']:.3f} s   (excludes compile)")
        print(f"  fixed timesteps        : {c['n_steps']}")
        print(f"  µs per step            : {c['us_per_step']:.1f}")
        print(f"  period / freq          : {c['period_ns']:.3f} ns / {c['freq_MHz']:.2f} MHz")
        if v.get("status") == "ok" and v.get("vacask_elapsed"):
            ratio = c["wall"] / v["vacask_elapsed"]
            print(f"  vs VACASK              : {ratio:.1f}× slower" if ratio > 1
                  else f"  vs VACASK              : {1/ratio:.1f}× FASTER")


if __name__ == "__main__":
    main()
