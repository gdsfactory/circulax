"""Sweep VACASK convergence options to find settings that work for N >= 21.

Diagnostic run showed the default configuration fails with:
  - gdev stepping: all 6 steps fail
  - gshunt stepping: exhausts 100 steps, stuck at gshunt ~3e-4
  - source stepping: exhausts 100 steps, stuck at srcfact ~0.116

Each candidate is tested on N=21; passing candidates are then checked on N=27.

Usage:
    pixi run python benchmarks/ring/vacask_convergence_sweep.py
    pixi run python benchmarks/ring/vacask_convergence_sweep.py --n 27
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _paths import vacask_bin, vacask_repo  # noqa: E402

VACASK = vacask_bin()
VACASK_DIR = vacask_repo() / "benchmark" / "ring" / "vacask"


def _run(sim_path: Path, timeout: int = 300) -> tuple[bool, str, float]:
    """Run VACASK on sim_path; return (converged, stdout, wall_s)."""
    t0 = time.perf_counter()
    proc = subprocess.run(
        [VACASK, "--skip-embed", "--skip-postprocess", sim_path.name],
        cwd=sim_path.parent, capture_output=True, text=True, timeout=timeout, check=False,
    )
    wall = time.perf_counter() - t0
    stdout = proc.stdout
    converged = not bool(re.search(r"[Hh]omotopy failed|Analysis.*aborted", stdout))
    if proc.returncode != 0 and "Homotopy failed" not in stdout:
        converged = False
    # Also check accepted timepoints > 0
    accepted = re.search(r"Accepted timepoints:\s*(\d+)", stdout)
    if accepted and int(accepted.group(1)) == 0:
        converged = False
    return converged, stdout, wall


def _nodeset_all(n_stages: int, v: float) -> dict[str, float]:
    return {str(i): v for i in range(1, n_stages + 1)}


def run_sweep(n_stages: int = 21) -> list[dict]:
    from vacask_gen import emit

    vdd = 1.2
    vhalf = vdd / 2  # 0.6V — the ring's only DC equilibrium

    candidates = [
        # --- Baseline ---
        {
            "label": "baseline",
        },
        # --- Newton damping ---
        {
            "label": "nr_damping=0.5",
            "opts": {"nr_damping": "0.5"},
        },
        # --- More homotopy budget ---
        {
            "label": "op_itlcont=500",
            "opts": {"op_itlcont": "500"},
        },
        # --- Finer source steps ---
        {
            "label": "homotopy_srcstep=0.001",
            "opts": {"homotopy_srcstep": "0.001"},
        },
        # --- Reorder: source stepping first ---
        {
            "label": 'op_homotopy=["src","gdev","gshunt"]',
            "opts": {"op_homotopy": '["src","gdev","gshunt"]'},
        },
        # --- Higher starting gmin ---
        {
            "label": "homotopy_startgmin=0.1",
            "opts": {"homotopy_startgmin": "0.1"},
        },
        # --- Tran-internal nodeset at VDD/2 ---
        {
            "label": f"tran_nodeset={vhalf}V",
            "nodesets": _nodeset_all(n_stages, vhalf),
        },
        # --- Longer OP nodeset enforcement ---
        {
            "label": f"tran_nodeset={vhalf}V + op_nsiter=50",
            "opts": {"op_nsiter": "50"},
            "nodesets": _nodeset_all(n_stages, vhalf),
        },
        # --- Strong nodeset forcing ---
        {
            "label": f"tran_nodeset={vhalf}V + nr_nsforce=1000 + op_nsiter=50",
            "opts": {"nr_nsforce": "1000", "op_nsiter": "50"},
            "nodesets": _nodeset_all(n_stages, vhalf),
        },
        # --- UIC mode: skip OP solve, start transient at VDD/2 ---
        {
            "label": f"icmode=uic ic={vhalf}V",
            "uic": True,
            "ic": _nodeset_all(n_stages, vhalf),
        },
        # --- UIC + also set VDD node ---
        {
            "label": f"icmode=uic ic={vhalf}V+vdd=1.2V",
            "uic": True,
            "ic": {**_nodeset_all(n_stages, vhalf), "vdd": vdd},
        },
        # --- Damping + more budget ---
        {
            "label": "nr_damping=0.5 + op_itlcont=500",
            "opts": {"nr_damping": "0.5", "op_itlcont": "500"},
        },
        # --- Damping + finer steps + more budget ---
        {
            "label": "nr_damping=0.5 + itlcont=500 + srcstep=0.001",
            "opts": {"nr_damping": "0.5", "op_itlcont": "500", "homotopy_srcstep": "0.001"},
        },
        # --- Src-first + damping + more budget ---
        {
            "label": "src_first + nr_damping=0.5 + itlcont=500",
            "opts": {
                "op_homotopy": '["src","gdev","gshunt"]',
                "nr_damping": "0.5",
                "op_itlcont": "500",
            },
        },
    ]

    results = []
    for i, cand in enumerate(candidates):
        label = cand["label"]
        opts = cand.get("opts")
        nodesets = cand.get("nodesets")
        uic = cand.get("uic", False)
        ic = cand.get("ic")
        suffix = f"_sweep{i:02d}"
        sim_path = emit(
            n_stages, extra_options=opts or None, suffix=suffix,
            nodesets=nodesets, uic=uic, ic=ic,
        )

        print(f"[{i+1:2d}/{len(candidates)}] {label!r}", flush=True)
        try:
            converged, stdout, wall = _run(sim_path, timeout=300)
        except subprocess.TimeoutExpired:
            results.append({"label": label, "converged": False, "status": "timeout", "wall_s": 300})
            print("        → TIMEOUT after 300s")
            continue

        status = "ok" if converged else "dc_diverged"
        # Extract which homotopy stage was last attempted
        last_stage = ""
        for m in re.finditer(r"Trying (\w+ stepping)", stdout):
            last_stage = m.group(1)
        results.append({
            "label": label,
            "converged": converged,
            "status": status,
            "wall_s": wall,
            "last_stage_tried": last_stage,
        })
        mark = "PASS" if converged else "FAIL"
        print(f"        → {mark}  wall={wall:.1f}s  last_homotopy={last_stage!r}")

    return results


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("VACASK CONVERGENCE SWEEP SUMMARY")
    print("=" * 70)
    passing = [r for r in results if r["converged"]]
    failing = [r for r in results if not r["converged"]]
    print(f"\nPassing ({len(passing)}):")
    for r in passing:
        print(f"  [PASS]  {r['label']}  ({r['wall_s']:.1f}s)")
    print(f"\nFailing ({len(failing)}):")
    for r in failing:
        print(f"  [FAIL]  {r['label']}  last={r.get('last_stage_tried', '?')!r}")
    if passing:
        fastest = min(passing, key=lambda r: r["wall_s"])
        print(f"\nFastest passing: {fastest['label']!r} ({fastest['wall_s']:.1f}s)")


def main(argv: list[str] | None = None) -> None:
    if not Path(VACASK).exists():
        print(f"ERROR: VACASK not found at {VACASK}", file=sys.stderr)
        sys.exit(1)

    n = 21
    if argv and "--n" in argv:
        idx = argv.index("--n")
        n = int(argv[idx + 1])

    print(f"Sweeping VACASK convergence options for N={n}")
    results = run_sweep(n)
    print_summary(results)


if __name__ == "__main__":
    main(sys.argv[1:])
