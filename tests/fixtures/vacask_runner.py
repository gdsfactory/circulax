"""Helpers to run VACASK on a .sim fixture and parse its raw output.

Used by the PSP103 test ladder to cross-check circulax against VACASK
on the same OSDI model card.  Each helper returns a plain dict so the
test assertions read like ``ref["id_A"]`` rather than poking VACASK's
RawFile object directly.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

VACASK_BINARY = Path("/home/cdaunt/opt/vacask/bin/vacask")
VACASK_PYTHON = Path("/home/cdaunt/code/vacask/VACASK/python")
FIXTURE_DIR = Path(__file__).resolve().parent / "vacask"


def vacask_available() -> bool:
    return VACASK_BINARY.exists() and VACASK_PYTHON.is_dir()


def run_vacask(sim_name: str, analysis_name: str, out_name: str | None = None) -> Path:
    """Run VACASK on ``FIXTURE_DIR / sim_name`` and return the path to the
    produced raw file.

    VACASK always writes ``{analysis_name}.raw`` into the working
    directory it was invoked from.  We run from ``FIXTURE_DIR`` so the
    result lands next to the sim file, then rename to ``out_name`` (if
    given) so fixtures can distinguish multiple analyses.
    """
    sim_path = FIXTURE_DIR / sim_name
    if not sim_path.exists():
        msg = f"VACASK fixture not found: {sim_path}"
        raise FileNotFoundError(msg)

    result = subprocess.run(
        [str(VACASK_BINARY), "--skip-embed", "--skip-postprocess", str(sim_path.name)],
        cwd=str(FIXTURE_DIR),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        msg = (
            f"VACASK returned {result.returncode} on {sim_name}:\n"
            f"STDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
        )
        raise RuntimeError(msg)

    raw_in = FIXTURE_DIR / f"{analysis_name}.raw"
    if not raw_in.exists():
        msg = f"VACASK did not produce {raw_in} (check analysis name)."
        raise FileNotFoundError(msg)

    if out_name:
        raw_out = FIXTURE_DIR / out_name
        if raw_out.exists():
            raw_out.unlink()
        shutil.move(str(raw_in), str(raw_out))
        return raw_out
    return raw_in


def read_op_raw(path: Path) -> dict[str, float]:
    """Read a VACASK op-point raw file and return {name: value} as floats."""
    if str(VACASK_PYTHON) not in sys.path:
        sys.path.insert(0, str(VACASK_PYTHON))
    from rawfile import rawread  # type: ignore

    r = rawread(str(path)).get()
    values = np.asarray(r.data[0])
    return {name: float(values[i]) for i, name in enumerate(r.names)}


def read_time_raw(path: Path) -> dict[str, np.ndarray]:
    """Read a VACASK tran raw file and return {name: 1-D array}.

    VACASK's RawFile stores the per-step solution vector as a 2-D
    ``(num_points, num_signals)`` array; this helper splits it into one
    1-D array per signal, indexed by the signal's name.
    """
    if str(VACASK_PYTHON) not in sys.path:
        sys.path.insert(0, str(VACASK_PYTHON))
    from rawfile import rawread  # type: ignore

    r = rawread(str(path)).get()
    arr = np.asarray(r.data)
    return {name: arr[:, i] for i, name in enumerate(r.names)}
