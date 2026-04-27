"""Emit runme_N.sim for the VACASK ring-osc benchmark at arbitrary N.

Writes into this directory so the generated file picks up the local
``models.inc`` and ``psp103v4.osdi`` symlinks (which point at the
upstream VACASK install).

Usage:
    pixi run python benchmarks/ring/vacask/vacask_gen.py 33
"""

from __future__ import annotations

import sys
from pathlib import Path

VACASK_DIR = Path(__file__).resolve().parent


def emit(n_stages: int) -> Path:
    if n_stages < 3 or n_stages % 2 == 0:
        raise ValueError(f"n_stages must be odd and ≥ 3; got {n_stages}")

    lines = [
        f"{n_stages} stage ring oscillator (generated)",
        "",
        'load "psp103v4.osdi"',
        "",
        'include "models.inc"',
        "model vsource vsource",
        "model isource isource",
        "",
        "subckt inverter (in out vdd vss)",
        "parameters w=1u l=0.2u pfact=2",
        "  mp (out in vdd vdd) pmos w=w*pfact l=l",
        "  mn (out in vss vss) nmos w=w l=l",
        "ends",
        "",
        'i0 (0 1) isource type="pulse" val0=0 val1=10u delay=1n rise=1n fall=1n width=1n',
    ]
    for stage in range(1, n_stages + 1):
        lines.append(
            f"u{stage} ({stage} {stage % n_stages + 1} vdd 0)  inverter w=10u l=1u"
        )
    lines += [
        "",
        "vdd (vdd 0) vsource dc=1.2",
        "",
        "control",
        '  options tran_method="trap"',
        "  analysis tran1 tran step=0.05n stop=1u maxstep=0.05n",
        "  print stats",
        "endc",
        "",
    ]
    out = VACASK_DIR / f"runme_{n_stages}.sim"
    out.write_text("\n".join(lines))
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 33
    path = emit(n)
    print(f"wrote {path}")
