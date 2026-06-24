"""Run the mul benchmark on circulax + VACASK + ngspice; update README table.

Invocations:
    pixi run python benchmarks/mul/run.py

Reads the upstream VACASK benchmark templates directly from
/home/cdaunt/code/vacask/VACASK/benchmark/mul/{vacask,ngspice}/ and
times them alongside the local bench_circulax.py runner.
"""

from __future__ import annotations

import csv
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

UPSTREAM = Path("/home/cdaunt/code/vacask/VACASK/benchmark/mul")
CSV_PATH = HERE / "results.csv"
README = HERE / "README.md"


def run_vacask() -> dict:
    vacask = shutil.which("vacask") or "/home/cdaunt/opt/vacask/bin/vacask"
    if not Path(vacask).exists():
        return {"simulator": "vacask", "status": "not_installed"}
    sim_dir = UPSTREAM / "vacask"
    if not (sim_dir / "runme.sim").exists():
        return {"simulator": "vacask", "status": "missing_upstream_template"}

    t0 = time.perf_counter()
    proc = subprocess.run(
        [vacask, "--skip-embed", "--skip-postprocess", "--no-output", "runme.sim"],
        cwd=sim_dir, capture_output=True, text=True, timeout=1800, check=False,
    )
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        return {"simulator": "vacask", "status": f"rc={proc.returncode}",
                "wall_s": wall}
    m = re.search(r"Elapsed time:\s+([\d.eE+\-]+)", proc.stdout)
    accepted = re.search(r"Accepted timepoints:\s+(\d+)", proc.stdout)
    n = int(accepted.group(1)) if accepted else 0
    el = float(m.group(1)) if m else wall
    return {
        "simulator": "vacask", "status": "ok", "wall_s": wall,
        "sim_reported_s": el, "n_steps": n,
        "us_per_step": el / max(n, 1) * 1e6 if n else None,
    }


def run_ngspice() -> dict:
    ngspice = shutil.which("ngspice")
    if ngspice is None:
        return {"simulator": "ngspice", "status": "not_installed"}
    sim_dir = UPSTREAM / "ngspice"
    if not (sim_dir / "runme.sim").exists():
        return {"simulator": "ngspice", "status": "missing_upstream_template"}

    t0 = time.perf_counter()
    proc = subprocess.run(
        [ngspice, "-b", "runme.sim"],
        cwd=sim_dir, capture_output=True, text=True, timeout=1800, check=False,
    )
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        return {"simulator": "ngspice", "status": f"rc={proc.returncode}",
                "wall_s": wall}
    m = re.search(r"Total elapsed time[^\d]*([\d.]+)", proc.stdout)
    # ngspice reports `Accepted timepoints = N` (integer).  length(time)
    # is in scientific notation and must not be parsed with \d+.
    accepted = re.search(r"Accepted timepoints\s*=\s*(\d+)", proc.stdout)
    n = int(accepted.group(1)) if accepted else 0
    el = float(m.group(1)) if m else wall
    return {
        "simulator": "ngspice", "status": "ok", "wall_s": wall,
        "sim_reported_s": el, "n_steps": n,
        "us_per_step": el / max(n, 1) * 1e6 if n else None,
    }


def run_circulax() -> dict:
    import bench_circulax as cxmod
    r = cxmod.run()
    return {
        "simulator": "circulax", "status": "ok",
        "wall_s": r["wall_s"], "compile_s": r["compile_s"],
        "n_steps": r["n_steps"], "us_per_step": r["us_per_step"],
    }


def write_results(rows: list[dict]) -> None:
    fields = ["simulator", "variant", "status", "wall_s", "sim_reported_s",
              "compile_s", "n_steps", "us_per_step", "notes"]
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def render_readme_table(rows: list[dict]) -> str:
    lines = [
        "| Simulator | Wall (s) | Compile (s) | n_steps | µs/step | Status |",
        "|-----------|----------|-------------|---------|---------|--------|",
    ]
    for r in rows:
        sim = r.get("simulator", "?")
        wall = f"{r['wall_s']:.2f}" if isinstance(r.get("wall_s"), (int, float)) else "—"
        comp = f"{r['compile_s']:.2f}" if isinstance(r.get("compile_s"), (int, float)) else "—"
        n = r.get("n_steps") or "—"
        us = f"{r['us_per_step']:.2f}" if isinstance(r.get("us_per_step"), (int, float)) else "—"
        stat = r.get("status", "?")
        lines.append(f"| {sim} | {wall} | {comp} | {n} | {us} | {stat} |")
    return "\n".join(lines)


def update_readme(rows: list[dict]) -> None:
    text = README.read_text()
    table = render_readme_table(rows)
    start = "<!-- RESULTS -->"
    end = "<!-- /RESULTS -->"
    if start in text and end in text:
        new = re.sub(
            rf"{re.escape(start)}.*?{re.escape(end)}",
            f"{start}\n{table}\n_{time.strftime('%Y-%m-%d')}_\n{end}",
            text, flags=re.DOTALL,
        )
        README.write_text(new)


def main() -> None:
    rows = []
    for label, fn in (("vacask", run_vacask), ("ngspice", run_ngspice),
                      ("circulax", run_circulax)):
        print(f"Running {label}…", flush=True)
        try:
            r = fn()
        except Exception as e:
            r = {"simulator": label, "status": f"EXC_{type(e).__name__}",
                 "notes": str(e)[:80]}
        rows.append(r)
        w = r.get("wall_s")
        print(f"  → {r.get('status', '?')}  wall={w if w is None else f'{w:.2f}s'}")

    write_results(rows)
    update_readme(rows)
    print(f"\nWrote {CSV_PATH} and updated {README.name}")


if __name__ == "__main__":
    main()
