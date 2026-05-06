"""Emit runme_N.sim for the ngspice ring-osc benchmark at arbitrary N.

Writes alongside the upstream N=9 template so the generated file can
reuse ``models.inc`` and ngspice's ``pre_osdi psp103v4.osdi`` hook.

Usage:
    pixi run python benchmarks/ring/ngspice_gen.py 33
"""

from __future__ import annotations

import sys
from pathlib import Path

NGSPICE_DIR = Path("/home/cdaunt/code/vacask/VACASK/benchmark/ring/ngspice")


def emit(n_stages: int) -> Path:
    if n_stages < 3 or n_stages % 2 == 0:
        raise ValueError(f"n_stages must be odd and ≥ 3; got {n_stages}")

    lines = [
        f"{n_stages} stage ring oscillator (generated)",
        "",
        '.include "models.inc"',
        "",
        ".subckt inverter in out vdd vss w=1u l=0.2u pfact=2",
        "  xmp out in vdd vdd pmos w={w*pfact} l={l}",
        "  xmn out in vss vss nmos w={w} l={l}",
        ".ends",
        "",
        "i0 0 1 dc 0 pulse 0 10u 1n 1n 1n 1n",
    ]
    for stage in range(1, n_stages + 1):
        lines.append(
            f"xu{stage} {stage} {stage % n_stages + 1} vdd 0  inverter w={{10u}} l={{1u}}"
        )
    lines += [
        "",
        "vdd vdd 0 1.2",
        "",
        ".options method=trap",
        ".options klu",
        "",
        ".control",
        "  pre_osdi psp103v4.osdi",
        "  tran 0.05n 1u 0 0.05n",
        "  rusage all",
        "  set",
        "  set noaskquit",
        "  quit",
        ".endc",
        "",
        ".end",
    ]
    out = NGSPICE_DIR / f"runme_{n_stages}.sp"
    out.write_text("\n".join(lines))
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 33
    path = emit(n)
    print(f"wrote {path}")
