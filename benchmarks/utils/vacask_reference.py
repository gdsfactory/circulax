"""Benchmark-only VACASK/ngspice reference runners and raw-output readers."""

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
    """Named reference vectors plus input and process diagnostics."""

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
    osdi_modules: tuple[Path, ...] = (),
    va_sources: tuple[Path, ...] = (),
    cache_dir: Path | None = None,
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
        sources = list(va_sources)
        if resolved:
            sources.extend((base / reference).resolve() for reference, base in resolved.loads)
        for source in sources:
            if source.suffix == ".va" and source.is_file():
                cached = compile_va(source, compiler=compiler, cache_dir=cache_dir)
                # VACASK checks source mtime before deciding to compile.
                target = directory / (source.stem + ".osdi")
                shutil.copyfile(cached, target)
                os.utime(target, None)
        if osdi_modules:
            title, body = deck.split("\n", maxsplit=1)
            loads = []
            for module in osdi_modules:
                binary_path = compile_va(module, compiler=compiler) if module.suffix == ".va" else module
                # Keep the original identity so VACASK's automatic primitive
                # loads reuse dlopen handles instead of registering duplicates.
                loads.append("load " + json.dumps(str(binary_path.resolve())))
            deck = title + "\n" + "\n".join(loads) + "\n" + body
        (directory / "input.sim").write_text(deck)
        run = subprocess.run(  # noqa: S603 -- explicit simulator, no shell
            [executable, "input.sim"], cwd=directory, env=environment, capture_output=True, text=True, timeout=timeout, check=False
        )
        raw_path = directory / raw_filename
        if run.returncode or not raw_path.is_file() or "model skipped" in run.stdout + run.stderr:
            msg = f"VACASK failed ({run.returncode}); expected {raw_filename}:\n{run.stdout}\n{run.stderr}"
            raise RuntimeError(msg)
        raw = VacaskRawFile(raw_path.read_bytes())
        return ReferenceResult(
            {name: np.asarray(variable.data) for name, variable in raw.variables.items()}, deck, run.stdout, run.stderr
        )


def run_ngspice(
    deck: str,
    *,
    binary: str | Path = "ngspice",
    osdi_modules: tuple[Path, ...] = (),
    raw_filename: str = "op1.raw",
    timeout: float = 120,
) -> ReferenceResult:
    """Run a native ngspice control deck with explicit ABI-compatible modules."""
    from InSpice.Spice.NgSpice.RawFile import RawFile

    executable = shutil.which(str(binary))
    if executable is None:
        msg = f"ngspice binary not found: {binary}"
        raise FileNotFoundError(msg)
    with tempfile.TemporaryDirectory(prefix="circulax-ngspice-") as temporary:
        directory = Path(temporary)
        commands = []
        for index, module in enumerate(osdi_modules):
            filename = f"module{index}.osdi"
            shutil.copyfile(module, directory / filename)
            commands.append(f"osdi {filename}")
        (directory / ".spiceinit").write_text("\n".join(commands) + "\n")
        (directory / "input.cir").write_text(deck)
        run = subprocess.run(  # noqa: S603 -- explicit simulator, no shell
            [executable, "-b", "input.cir"], cwd=directory, capture_output=True, text=True, timeout=timeout, check=False
        )
        path = directory / raw_filename
        if run.returncode or not path.is_file():
            msg = f"ngspice failed ({run.returncode}); expected {raw_filename}:\n{run.stdout}\n{run.stderr}"
            raise RuntimeError(msg)
        data = path.read_bytes()
        header, payload = data.split(b"Binary:\n", maxsplit=1)
        lines = [line for line in header.splitlines() if not line.startswith((b"Command:", b"Option:"))]
        # Batch AC frequency metadata appends `grid=3` to the unit.
        lines = [line.replace(b"frequency grid=3", b"frequency") for line in lines]
        header = b"\n".join(lines) + b"\n"
        points = int(next(line.split(b":", 1)[1] for line in lines if line.startswith(b"No. Points:")))
        variables = int(next(line.split(b":", 1)[1] for line in lines if line.startswith(b"No. Variables:")))
        # InSpice consumes ngspice's streaming header. Batch `write` omits
        # the process preamble and column-count field; the binary layout agrees.
        header = header.replace(b"Variables:\n", f"Variables:\nNo. of Data Columns : {variables}\n".encode())
        temperature = next((line for line in run.stdout.splitlines() if line.startswith("Doing analysis at TEMP")), None)
        if temperature is None:
            msg = "ngspice output is missing its analysis temperature"
            raise RuntimeError(msg)
        preamble = ("Circuit: IHP parity\n" + temperature + "\n").encode()
        raw = RawFile(preamble + header + b"Binary:\n" + payload, points)
        vectors = {}
        for name, variable in raw.variables.items():
            key = name
            if name.startswith("v("):
                key = name[2:-1]
            elif name.startswith("i("):
                key = name[2:-1] + ":flow(br)"
            if key == "v-sweep":
                key = "vin"
            vectors[key] = np.asarray(variable.data)
        return ReferenceResult(vectors, deck, run.stdout, run.stderr)
