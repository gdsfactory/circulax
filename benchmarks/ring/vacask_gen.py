"""Emit runme_N.sim for the VACASK ring-osc benchmark at arbitrary N.

Writes alongside the upstream N=9 template so the generated file can
reuse the same ``models.inc`` and the ``psp103v4.osdi`` symlink already
present in ``$VACASK_REPO/benchmark/ring/vacask/``.

Usage:
    pixi run python benchmarks/ring/vacask_gen.py 33
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _paths import vacask_repo  # noqa: E402

VACASK_DIR = vacask_repo() / "benchmark" / "ring" / "vacask"


def emit(
    n_stages: int,
    extra_options: dict[str, str] | None = None,
    suffix: str = "",
    nodesets: dict[str, float] | None = None,
    uic: bool = False,
    ic: dict[str, float] | None = None,
) -> Path:
    """Emit a VACASK ring-oscillator netlist.

    Parameters
    ----------
    n_stages:      Number of inverter stages (odd, ≥ 3).
    extra_options: ``key=value`` pairs written as additional ``options`` lines.
    suffix:        Appended to the output filename before ``.sim``.
    nodesets:      Node-name → voltage hint for the internal OP solve.
                   Passed as ``nodeset=[...]`` on the ``tran`` analysis line.
    uic:           If True, use ``icmode="uic"`` to skip the OP solve.
                   Useful when DC convergence fails; transient starts directly
                   from the ``ic`` values.
    ic:            Node-name → voltage initial condition for ``icmode="uic"``.
                   Ignored unless ``uic=True``.

    """
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

    option_lines = ['  options tran_method="trap"']
    if extra_options:
        for k, v in extra_options.items():
            option_lines.append(f"  options {k}={v}")

    # Build the tran analysis line with optional nodeset / ic / icmode
    tran_extras = []
    if uic:
        tran_extras.append('icmode="uic"')
        if ic:
            ic_items = "; ".join(f'"{n}"; {v}' for n, v in ic.items())
            tran_extras.append(f"ic=[ {ic_items} ]")
    elif nodesets:
        ns_items = "; ".join(f'"{n}"; {v}' for n, v in nodesets.items())
        tran_extras.append(f"nodeset=[ {ns_items} ]")

    tran_suffix = (" " + " ".join(tran_extras)) if tran_extras else ""
    tran_line = f"  analysis tran1 tran step=0.05n stop=1u maxstep=0.05n{tran_suffix}"

    lines += [
        "",
        "vdd (vdd 0) vsource dc=1.2",
        "",
        "control",
        *option_lines,
        tran_line,
        "  print stats",
        "endc",
        "",
    ]
    fname = f"runme_{n_stages}{suffix}.sim"
    out = VACASK_DIR / fname
    out.write_text("\n".join(lines))
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 33
    path = emit(n)
    print(f"wrote {path}")
