"""Centralised path resolution for benchmark scripts.

Every external dependency (VACASK, IHP-PDK, klujax_rs, circulax-va) is
resolved from an environment variable or ``shutil.which``.  No hardcoded
home-directory paths — contributors set the env vars once in their shell
profile and all benchmarks Just Work.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def _env_path(var: str, hint: str) -> Path:
    val = os.environ.get(var)
    if not val:
        msg = f"${var} is not set. {hint}"
        raise OSError(msg)
    p = Path(val)
    if not p.exists():
        msg = f"${var}={val} does not exist. {hint}"
        raise OSError(msg)
    return p


def vacask_bin() -> str:
    """Resolve the vacask CLI binary: $VACASK_BIN > PATH lookup."""
    env = os.environ.get("VACASK_BIN")
    if env:
        if not Path(env).exists():
            msg = f"$VACASK_BIN={env} does not exist."
            raise OSError(msg)
        return env
    found = shutil.which("vacask")
    if found:
        return found
    msg = "vacask not found. Install it or set $VACASK_BIN to the binary path."
    raise OSError(msg)


def vacask_python() -> str:
    """Resolve the VACASK Python library directory for sys.path insertion."""
    return str(_env_path(
        "VACASK_PYTHON",
        "Set it to the VACASK Python dir (e.g. ~/code/vacask/VACASK/python).",
    ))


def vacask_repo() -> Path:
    """Resolve the VACASK repository root."""
    return _env_path(
        "VACASK_REPO",
        "Set it to the VACASK repo root (e.g. ~/code/vacask/VACASK).",
    )


def ihp_pdk_va() -> Path:
    """Resolve the IHP-Open-PDK verilog-a directory."""
    return _env_path(
        "IHP_PDK_VA",
        "Set it to the IHP-Open-PDK verilog-a dir "
        "(e.g. ~/code/IHP-Open-PDK/ihp-sg13g2/libs.tech/verilog-a).",
    )


def klujax_rs_path() -> Path:
    """Resolve the klujax_rs static build directory."""
    return _env_path(
        "KLUJAX_RS_PATH",
        "Set it to the klujax_rs-static build dir.",
    )


def circulax_va_repo() -> Path:
    """Resolve the circulax-va companion repository."""
    return _env_path(
        "CIRCULAX_VA_REPO",
        "Set it to the circulax-va repo root.",
    )
