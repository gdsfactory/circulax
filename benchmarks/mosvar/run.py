"""Driver for the mosvar DC sweep — VACASK / circulax-OSDI.

Tier 1. mosvar is a 3-port MOS varactor; sweep V_g with bulk tied to GND.
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
sys.path.insert(0, "/home/cdaunt/code/vacask/VACASK/python")

import bench_circulax as cx  # noqa: E402

VACASK_BIN = "/home/cdaunt/opt/vacask/bin/vacask"
VACASK_DIR = HERE / "vacask"
RAW_FILE = VACASK_DIR / "dcsweep.raw"
CSV_PATH = HERE / "results.csv"


def _ensure_vacask_osdi() -> None:
    target = VACASK_DIR / "mosvar.osdi"
    if target.exists():
        return
    src_va = (
        "/home/cdaunt/code/gdsfactory/pdks/IHP-Open-PDK/ihp-sg13g2/"
        "libs.tech/verilog-a/mosvar/mosvar.va"
    )
    subprocess.run(
        ["openvaf-r", src_va, "-o", str(target)],
        cwd=Path(src_va).parent, check=True, capture_output=True, text=True,
    )


def run_vacask() -> dict[float, float] | dict:
    if not Path(VACASK_BIN).exists():
        return {"status": "vacask_not_installed"}
    _ensure_vacask_osdi()
    t0 = time.perf_counter()
    proc = subprocess.run(
        [VACASK_BIN, "--skip-embed", "--skip-postprocess", "runme.sim"],
        cwd=VACASK_DIR, capture_output=True, text=True, timeout=300, check=False,
    )
    if proc.returncode != 0:
        return {"status": f"rc={proc.returncode}", "stderr": proc.stderr[:200]}
    if not RAW_FILE.exists():
        return {"status": "no_rawfile"}

    from rawfile import rawread

    r = rawread(str(RAW_FILE)).get()
    v = np.asarray(r["v1"])
    i_into_v_plus = np.asarray(r["v1:flow(br)"])
    # I_g (current INTO the gate node) = -i_into_v_plus (Vsrc convention).
    return {float(vv): float(-ii) for vv, ii in zip(v, i_into_v_plus)}


def run_circulax(variant: str) -> dict[float, float]:
    res = cx.run(variant)["iv"]
    out: dict[float, float] = {}
    for v, i in res.items():
        if isinstance(i, float):
            # cx.run returns the Vsrc current sinker (current INTO V+ of Vsrc),
            # i.e. -I_g. Negate so the output column is I_g consistently with VACASK.
            out[float(v)] = -float(i)
    return out


def render_table(by_v: dict[float, dict[str, float]]) -> str:
    head = (
        f"{'V_g [V]':>9} | {'VACASK [A]':>14} | {'OSDI [A]':>14} | {'OSDI/VAC':>9}"
    )
    lines = [head, "-" * len(head)]
    for v in sorted(by_v):
        row = by_v[v]
        vac, osdi = row.get("vacask"), row.get("osdi")
        def _fmt(x):
            return f"{x:>14.4e}" if isinstance(x, float) else f"{x!s:>14}"
        def _ratio(num, den):
            if not isinstance(num, float) or not isinstance(den, float):
                return "—".rjust(9)
            if abs(den) < 1e-30:
                return ("VAC=0" if abs(num) < 1e-30 else "den=0").rjust(9)
            return f"{num/den:>9.4f}"
        lines.append(
            f"{v:>9.3f} | {_fmt(vac)} | {_fmt(osdi)} | {_ratio(osdi, vac)}"
        )
    return "\n".join(lines)


def write_csv(by_v: dict[float, dict[str, float]]) -> None:
    with CSV_PATH.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["V_g", "vacask_I_g", "osdi_I_g"])
        for v in sorted(by_v):
            row = by_v[v]
            w.writerow([v, row.get("vacask", ""), row.get("osdi", "")])


def main() -> None:
    print("[vacask]   running…", flush=True)
    vac = run_vacask()
    if isinstance(vac, dict) and "status" in vac:
        print(f"  → {vac['status']}; aborting comparison")
        return

    print("[circulax-osdi] running…", flush=True)
    osdi = run_circulax("osdi")

    def _bucket(d):
        return {round(k, 3): v for k, v in d.items()}

    vac_b, osdi_b = _bucket(vac), _bucket(osdi)
    all_v = sorted(set(vac_b) | set(osdi_b))
    by_v = {
        v: {"vacask": vac_b.get(v), "osdi": osdi_b.get(v)}
        for v in all_v
    }

    print()
    print(render_table(by_v))
    write_csv(by_v)
    print(f"\nWrote {CSV_PATH}")


if __name__ == "__main__":
    main()
