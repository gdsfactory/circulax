"""Driver for the juncap200 DC sweep — VACASK / circulax-OSDI.

VACASK is the source of truth (industry-standard simulator on the IHP source
file). circulax-OSDI exercises the compiled-binary path through bosdi.

Usage:
    pixi run python benchmarks/juncap200/run.py

Outputs:
    benchmarks/juncap200/results.csv   — wide-format (V_AK rows, simulator cols)
    Stdout                             — comparison table with delta vs VACASK
"""
from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent))
from _paths import vacask_bin as _vacask_bin  # noqa: E402

try:
    from _paths import vacask_repo as _vacask_repo
    sys.path.insert(0, str(_vacask_repo() / "python"))
except OSError:
    pass

import bench_circulax as cx  # noqa: E402

try:
    VACASK_BIN = _vacask_bin()
except OSError:
    VACASK_BIN = None
VACASK_DIR = HERE / "vacask"
RAW_FILE = VACASK_DIR / "dcsweep.raw"
CSV_PATH = HERE / "results.csv"


def _ensure_vacask_osdi() -> None:
    """Compile juncap200.osdi into the vacask dir if missing (deck loads it)."""
    target = VACASK_DIR / "juncap200.osdi"
    if target.exists():
        return
    src_va = str(HERE / "va_source" / "juncap200.va")
    subprocess.run(
        ["openvaf-r", src_va, "-o", str(target)],
        cwd=str(HERE / "va_source"), check=True, capture_output=True, text=True,
    )


def run_vacask() -> dict[float, float] | dict:
    """Run VACASK deck, parse dcsweep.raw, return {V_AK: I_A}."""
    if VACASK_BIN is None or not Path(VACASK_BIN).exists():
        return {"status": "vacask_not_installed"}
    _ensure_vacask_osdi()
    t0 = time.perf_counter()
    proc = subprocess.run(
        [VACASK_BIN, "--skip-embed", "--skip-postprocess", "runme.sim"],
        cwd=VACASK_DIR, capture_output=True, text=True, timeout=300, check=False,
    )
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        return {"status": f"rc={proc.returncode}", "stderr": proc.stderr[:200]}
    if not RAW_FILE.exists():
        return {"status": "no_rawfile", "wall_s": wall}

    from rawfile import rawread

    r = rawread(str(RAW_FILE)).get()
    v = np.asarray(r["v1"])
    # `v1:flow(br)` is the current INTO V+ of v1; current INTO node 'a' is its negation.
    # We report I_A (anode current, +ve = into anode), matching the sign convention
    # used by bench_circulax.py.
    i_into_v_plus = np.asarray(r["v1:flow(br)"])
    return {float(vv): float(-ii) for vv, ii in zip(v, i_into_v_plus)}


def run_circulax(variant: str) -> dict[float, float]:
    """{V_AK: I_A} via circulax bench, sign-aligned with VACASK."""
    res = cx.run(variant)["iv"]
    out: dict[float, float] = {}
    for v, i in res.items():
        if isinstance(i, float):
            # bench_circulax returns y[VS,i_src], current INTO V+ of VS.
            # Negate so I_A is anode current, matching the VACASK column.
            out[float(v)] = -float(i)
    return out


def render_table(by_v: dict[float, dict[str, float]]) -> str:
    """Print a comparison table with deltas vs VACASK."""
    head = (
        f"{'V_AK [V]':>9} | "
        f"{'VACASK [A]':>14} | {'OSDI [A]':>14} | {'OSDI/VACASK':>11}"
    )
    lines = [head, "-" * len(head)]
    for v in sorted(by_v):
        row = by_v[v]
        vac = row.get("vacask")
        osdi = row.get("osdi")

        def _fmt(x):
            return f"{x:>14.4e}" if isinstance(x, float) else f"{x!s:>14}"

        def _ratio(num, den):
            if not isinstance(num, float) or not isinstance(den, float):
                return "—".rjust(11)
            if abs(den) < 1e-20:
                return ("VAC=0" if num == 0 else "den=0").rjust(11)
            return f"{num / den:>11.4f}"

        lines.append(
            f"{v:>9.3f} | {_fmt(vac)} | {_fmt(osdi)} | {_ratio(osdi, vac)}"
        )
    return "\n".join(lines)


def write_csv(by_v: dict[float, dict[str, float]]) -> None:
    """Wide-format CSV: V_AK column + one column per simulator."""
    with CSV_PATH.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["V_AK", "vacask_I_A", "osdi_I_A"])
        for v in sorted(by_v):
            row = by_v[v]
            w.writerow([
                v,
                row.get("vacask", ""),
                row.get("osdi", ""),
            ])


def main() -> None:
    print("[vacask]   running…", flush=True)
    vac = run_vacask()
    if isinstance(vac, dict) and "status" in vac:
        print(f"  → {vac['status']}; aborting comparison")
        return

    print("[circulax-osdi] running…", flush=True)
    osdi = run_circulax("osdi")

    # Merge by V_AK key; VACASK floats may be 0.30000004 etc., so round to
    # mV grid before deduping.
    def _bucket(d):
        return {round(k, 3): v for k, v in d.items()}

    vac_b, osdi_b = _bucket(vac), _bucket(osdi)
    all_v = sorted(set(vac_b) | set(osdi_b))
    by_v: dict[float, dict[str, float]] = {
        v: {
            "vacask": vac_b.get(v),
            "osdi":   osdi_b.get(v),
        }
        for v in all_v
    }

    print()
    print(render_table(by_v))
    write_csv(by_v)
    print(f"\nWrote {CSV_PATH}")


if __name__ == "__main__":
    main()
