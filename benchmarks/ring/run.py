"""Ring-oscillator scaling sweep across circulax + VACASK + ngspice.

Invocations:
    pixi run python benchmarks/ring/run.py              # default N sweep
    pixi run python benchmarks/ring/run.py 9 15 33      # custom N list

For N=9 the upstream VACASK/ngspice templates are used directly; for
other N values we emit ``runme_N.sim`` / ``runme_N.sp`` into the
upstream dirs via vacask_gen.py / ngspice_gen.py before invoking.
A simulator that isn't on PATH, or diverges (e.g. VACASK DC homotopy
at large N), is recorded with a status note — the harness itself
never crashes.
"""

from __future__ import annotations

import csv
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/home/cdaunt/code/vacask/VACASK/python")


def _freq_from_crossings(t: np.ndarray, x: np.ndarray) -> float | None:
    """Rising-zero-crossing frequency (Hz) — same estimator circulax uses."""
    centered = x - x.mean()
    rising = np.where(np.diff(np.sign(centered)) > 0)[0]
    if len(rising) < 3:
        return None
    rising = rising[1:]
    times = []
    for i in rising:
        x0, x1 = float(centered[i]), float(centered[i + 1])
        t0, t1 = float(t[i]), float(t[i + 1])
        times.append(t0 - x0 * (t1 - t0) / (x1 - x0))
    if len(times) < 2:
        return None
    return float(1.0 / np.median(np.diff(np.asarray(times))))


def _vacask_freq(raw_path: Path, ignore_before_s: float = 100e-9) -> float | None:
    """Read VACASK's tran1.raw and extract v(1) oscillation frequency."""
    if not raw_path.exists():
        return None
    try:
        from rawfile import rawread
    except ImportError:
        return None
    # rawread() returns RawData; .get() unwraps to a RawFile with the
    # varname-indexed __getitem__ we want.
    r = rawread(str(raw_path)).get()
    t = np.asarray(r["time"])
    v1 = np.asarray(r["1"])
    mask = t > ignore_before_s
    if mask.sum() < 3:
        return None
    return _freq_from_crossings(t[mask], v1[mask])

VACASK_UPSTREAM = Path("/home/cdaunt/code/vacask/VACASK/benchmark/ring/vacask")
NGSPICE_UPSTREAM = Path("/home/cdaunt/code/vacask/VACASK/benchmark/ring/ngspice")
CSV_PATH = HERE / "results.csv"
README = HERE / "README.md"

DEFAULT_N = [3, 9, 15, 21, 27, 31, 33]


def _vacask_runme(n: int) -> Path:
    if n == 9:
        return VACASK_UPSTREAM / "runme.sim"
    from vacask_gen import emit
    return emit(n)


def _ngspice_runme(n: int) -> Path:
    if n == 9:
        return NGSPICE_UPSTREAM / "runme.sim"
    from ngspice_gen import emit
    return emit(n)


def run_vacask(n: int) -> dict:
    vacask = shutil.which("vacask") or "/home/cdaunt/opt/vacask/bin/vacask"
    if not Path(vacask).exists():
        return {"simulator": "vacask", "n_stages": n, "status": "not_installed"}
    runme = _vacask_runme(n)
    if not runme.exists():
        return {"simulator": "vacask", "n_stages": n, "status": "missing_template"}

    # Drop --no-output so tran1.raw is emitted; we need it for the
    # freq-vs-VACASK comparison.  Raw-file write is small relative to
    # simulation wall, and circulax materialises its saveat array in
    # memory too, so the comparison stays fair.
    t0 = time.perf_counter()
    proc = subprocess.run(
        [vacask, "--skip-embed", "--skip-postprocess", runme.name],
        cwd=runme.parent, capture_output=True, text=True, timeout=3600, check=False,
    )
    wall = time.perf_counter() - t0

    # VACASK logs "Homotopy failed." + "aborted." to stdout on DC failure
    # but exits rc=0.  Catch it by reading stdout, not just the exit code.
    if re.search(r"[Hh]omotopy failed|Analysis.*aborted", proc.stdout):
        return {"simulator": "vacask", "n_stages": n,
                "status": "dc_diverged", "wall_s": wall}
    if proc.returncode != 0:
        return {"simulator": "vacask", "n_stages": n,
                "status": f"rc={proc.returncode}", "wall_s": wall}

    m = re.search(r"Elapsed time:\s+([\d.eE+\-]+)", proc.stdout)
    accepted = re.search(r"Accepted timepoints:\s+(\d+)", proc.stdout)
    nsteps = int(accepted.group(1)) if accepted else 0
    el = float(m.group(1)) if m else wall
    if nsteps == 0:
        return {"simulator": "vacask", "n_stages": n,
                "status": "dc_diverged", "wall_s": wall}

    freq = _vacask_freq(runme.parent / "tran1.raw")
    return {
        "simulator": "vacask", "n_stages": n, "status": "ok",
        "wall_s": wall, "sim_reported_s": el, "n_steps": nsteps,
        "us_per_step": el / max(nsteps, 1) * 1e6,
        "freq_MHz": freq / 1e6 if freq is not None else None,
    }


def _ensure_psp103_osdi(work_dir: Path) -> bool:
    """Compile psp103.va → psp103v4.osdi in work_dir if missing.

    ngspice 45.2 is happy with OSDI 0.4 output from openvaf-r 23.5.0.
    """
    osdi = work_dir / "psp103v4.osdi"
    if osdi.exists():
        return True
    openvaf = shutil.which("openvaf-r") or shutil.which("openvaf")
    if openvaf is None:
        return False
    va = Path("/home/cdaunt/code/vacask/VACASK/devices/psp103v4/psp103.va")
    if not va.exists():
        return False
    proc = subprocess.run(
        [openvaf, str(va), "-o", str(osdi)],
        capture_output=True, text=True, check=False,
    )
    return proc.returncode == 0 and osdi.exists()


def run_ngspice(n: int) -> dict:
    ngspice = shutil.which("ngspice")
    if ngspice is None:
        return {"simulator": "ngspice", "n_stages": n, "status": "not_installed"}
    runme = _ngspice_runme(n)
    if not runme.exists():
        return {"simulator": "ngspice", "n_stages": n, "status": "missing_template"}
    if not _ensure_psp103_osdi(runme.parent):
        return {"simulator": "ngspice", "n_stages": n, "status": "osdi_compile_failed"}

    t0 = time.perf_counter()
    proc = subprocess.run(
        [ngspice, "-b", runme.name],
        cwd=runme.parent, capture_output=True, text=True, timeout=3600, check=False,
    )
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        return {"simulator": "ngspice", "n_stages": n,
                "status": f"rc={proc.returncode}", "wall_s": wall}
    m = re.search(r"Total elapsed time[^\d]*([\d.]+)", proc.stdout)
    # ngspice 45 emits "No. of Data Rows : N" rather than "Accepted timepoints"
    rows = re.search(r"No\.\s*of\s+Data\s+Rows\s*:\s*(\d+)", proc.stdout)
    nsteps = int(rows.group(1)) if rows else 0
    el = float(m.group(1)) if m else wall
    return {
        "simulator": "ngspice", "n_stages": n, "status": "ok",
        "wall_s": wall, "sim_reported_s": el, "n_steps": nsteps,
        "us_per_step": el / max(nsteps, 1) * 1e6 if nsteps else None,
    }


def run_circulax_osdi(n: int) -> dict:
    """circulax with PSP103 via OSDI FFI (bosdi → compiled .osdi)."""
    import bench_circulax as cxmod
    r = cxmod.run(n_stages=n, variant="osdi")
    r["simulator"] = "circulax_osdi"
    return r


def run_circulax_xla(n: int) -> dict:
    """circulax with the simplified pure-JAX MOSFET (no OSDI, no FFI).

    Isolates how much of the circulax wall time is attributable to
    the OSDI FFI boundary.  The device model here is ~20 % off PSP103
    quantitatively, so the oscillation frequency will differ too —
    that's fine, we're measuring per-step cost, not physical fidelity.
    """
    import bench_circulax as cxmod
    r = cxmod.run(n_stages=n, variant="jax-native")
    r["simulator"] = "circulax_xla"
    return r


def write_results(rows: list[dict]) -> None:
    fields = ["simulator", "n_stages", "variant", "status", "wall_s",
              "sim_reported_s", "compile_s", "dc_s", "n_steps",
              "us_per_step", "freq_MHz", "sys_size", "notes"]
    with CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def render_readme_table(rows: list[dict]) -> str:
    """Pivot by N: one row per stage count, columns per simulator.

    All timing columns are µs/step (= wall_s / n_steps × 1e6 for circulax;
    sim_reported_s / n_steps × 1e6 for VACASK).  VA also shows JIT cost.
    """

    def _fmt_us(r: dict) -> str:
        if r.get("status") != "ok":
            return r.get("status", "—")
        u = r.get("us_per_step")
        if isinstance(u, (int, float)):
            return f"{u:.1f}"
        # Fall back to wall_s / n_steps if us_per_step missing.
        w, n = r.get("wall_s"), r.get("n_steps")
        if isinstance(w, (int, float)) and isinstance(n, (int, float)) and n > 0:
            return f"{w / n * 1e6:.1f}"
        return "—"

    def _fmt_va(r: dict) -> str:
        """µs/step; append JIT cost in parens when available."""
        base = _fmt_us(r)
        if base in ("—",) or r.get("status") != "ok":
            return base
        c = r.get("compile_s")
        if isinstance(c, (int, float)):
            return f"{base} (JIT {c:.0f}s)"
        return base

    def _fmt_freq(r: dict) -> str:
        return (f"{r['freq_MHz']:.1f}"
                if isinstance(r.get("freq_MHz"), (int, float)) else "—")

    def _fmt_err(circulax_row: dict, ref_freq_mhz: float | None) -> str:
        f = circulax_row.get("freq_MHz")
        if not isinstance(f, (int, float)) or ref_freq_mhz is None:
            return "—"
        return f"{(f - ref_freq_mhz) / ref_freq_mhz * 100:+.1f} %"

    by_n: dict[int, dict[str, dict]] = {}
    for r in rows:
        n = r.get("n_stages")
        if n is None:
            continue
        by_n.setdefault(int(n), {})[r.get("simulator", "?")] = r

    header = (
        "| N | VACASK (µs/step) | ngspice (µs/step) | circulax (µs/step) | Freq VACASK (MHz) | circulax Δf |\n"
        "|---|------------------|-------------------|--------------------|-------------------|-------------|"
    )
    lines = [header]
    for n in sorted(by_n):
        v = by_n[n].get("vacask", {})
        ng = by_n[n].get("ngspice", {})
        c_osdi = by_n[n].get("circulax_osdi", {})
        ref_f = v.get("freq_MHz") if isinstance(v.get("freq_MHz"), (int, float)) else None
        lines.append(
            f"| {n} | {_fmt_us(v)} | {_fmt_us(ng)} | {_fmt_us(c_osdi)} "
            f"| {_fmt_freq(v)} | {_fmt_err(c_osdi, ref_f)} |"
        )
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


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    n_list = [int(x) for x in argv] if argv else DEFAULT_N

    rows: list[dict] = []
    for n in n_list:
        for label, fn in (("vacask", run_vacask),
                          ("ngspice", run_ngspice),
                          ("circulax_osdi", run_circulax_osdi)):
            print(f"[N={n}] {label}…", flush=True)
            try:
                r = fn(n)
            except Exception as e:
                r = {"simulator": label, "n_stages": n,
                     "status": f"EXC_{type(e).__name__}", "notes": str(e)[:80]}
            rows.append(r)
            w = r.get("wall_s")
            print(f"    → {r.get('status', '?')}  "
                  f"wall={w if w is None else f'{w:.2f}s'}")

    write_results(rows)
    update_readme(rows)
    print(f"\nWrote {CSV_PATH} and updated {README.name}")


if __name__ == "__main__":
    main()
