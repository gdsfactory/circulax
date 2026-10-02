"""Benchmark-only VACASK reference runner and raw-output reader."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from circulax.netlist_io.library import ResolvedCircuit
from circulax.netlist_io.osdi import compile_va


@dataclass
class ReferenceResult:
    """Named VACASK vectors plus reproducible input and process diagnostics."""

    vectors: dict[str, np.ndarray]
    deck: str
    stdout: str
    stderr: str


def run_vacask(
    deck: str,
    *,
    binary: str | Path,
    module_paths: tuple[Path, ...] = (),
    shared_library_paths: tuple[Path, ...] = (),
    compiler: str | None = None,
    resolved: ResolvedCircuit | None = None,
    raw_filename: str = "op1.raw",
    timeout: float = 120,
) -> ReferenceResult:
    """Run an original native deck and read its output with InSpice.

    `resolved` only provisions the same .va loads into the temporary run folder;
    VACASK still parses and elaborates the original library and wrapper itself.
    """
    from InSpice.Spice.Vacask.RawFile import VacaskRawFile

    executable = shutil.which(str(binary))
    if executable is None:
        msg = f"VACASK binary not found: {binary}"
        raise FileNotFoundError(msg)
    compiler = shutil.which(compiler or "openvaf-r")
    environment = os.environ.copy()
    if shared_library_paths:
        environment["LD_LIBRARY_PATH"] = (
            os.pathsep.join(map(str, shared_library_paths)) + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
        )
    with tempfile.TemporaryDirectory(prefix="circulax-vacask-") as temporary:
        directory = Path(temporary)
        config = "[Paths]\nmodule_path_prefix=" + json.dumps(list(map(str, module_paths))) + "\n"
        if compiler:
            config += "[Binaries]\nopenvaf=" + json.dumps(compiler) + "\n"
        (directory / ".vacaskrc.toml").write_text(config)
        if resolved:
            for reference, base in resolved.loads:
                source = (base / reference).resolve()
                if source.suffix == ".va" and source.is_file():
                    cached = compile_va(source, compiler=compiler)
                    # VACASK checks source mtime before deciding to compile.
                    target = directory / (source.stem + ".osdi")
                    shutil.copyfile(cached, target)
                    os.utime(target, None)
        (directory / "input.sim").write_text(deck)
        run = subprocess.run(  # noqa: S603 -- explicit simulator, no shell
            [executable, "input.sim"], cwd=directory, env=environment, capture_output=True, text=True, timeout=timeout, check=False
        )
        raw_path = directory / raw_filename
        if run.returncode or not raw_path.is_file():
            msg = f"VACASK failed ({run.returncode}); expected {raw_filename}:\n{run.stdout}\n{run.stderr}"
            raise RuntimeError(msg)
        raw = VacaskRawFile(raw_path.read_bytes())
        return ReferenceResult(
            {name: np.asarray(variable.data) for name, variable in raw.variables.items()}, deck, run.stdout, run.stderr
        )
