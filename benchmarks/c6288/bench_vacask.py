"""Validate and time native VACASK against the saved Circulax c6288 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from benchmarks.c6288.bench_circulax import check_product  # noqa: E402


def make_deck(root: Path, modules: list[dict], mode: str, stop: float, step: float) -> str:
    """Keep the original native circuit, options and cards with identical binaries."""
    source = root / "benchmark/c6288/vacask"
    deck = (source / "runme.sim").read_text().split('\nembed "runme.py"', 1)[0]
    for name, module in zip(("psp103v4.osdi", "spice/resistor.osdi"), modules, strict=True):
        deck = deck.replace(f'load "{name}"', "load " + json.dumps(module["path"]))
    for name in ("models.inc", "multiplier.inc"):
        deck = deck.replace(f'include "{name}"', "include " + json.dumps(str(source / name)))
    analysis = f'analysis tranmul tran stop={stop:.17g} step={step:.17g} icmode="uic"'
    if mode == "bdf2-capped":
        analysis = 'options tran_method="bdf2"\n  ' + analysis + f" maxstep={step:.17g}"
    deck, count = re.subn(r'analysis tranmul tran stop=2n step=2p icmode="uic"', analysis, deck)
    if count != 1:
        msg = "Upstream transient declaration changed; review benchmark settings"
        raise ValueError(msg)
    return deck


def parse_stats(stdout: str) -> dict:
    """Read named statistics without assuming an optional elapsed-time field."""
    result = {}
    for field, label, convert in (
        ("sim_reported_s", "Elapsed time", float),
        ("nr_solver_s", "NR solver time", float),
        ("nr_iterations", "NR solver iterations", int),
        ("accepted_timepoints", "Accepted timepoints", int),
        ("rejected_timepoints", "Rejected timepoints", int),
        ("unknowns", "Number of unknowns", int),
        ("eval_and_load_s", "Eval and load time", float),
        ("initial_nonzeros", "Initial number of nonzeros", int),
        ("eval_and_load_calls", "Eval and load calls", int),
        ("real_factorizations", "Real factorizations", int),
        ("real_refactorizations", "Real refactorizations", int),
        ("real_solve_calls", "Real solve calls", int),
        ("real_factorization_s", "Real factorization time", float),
        ("real_refactorization_s", "Real refactorization time", float),
        ("real_solve_s", "Real solve time", float),
        ("bypassed_evaluations", "Bypassed evaluations", int),
    ):
        matches = re.findall(re.escape(label) + r":\s+([\d.eE+\-]+)", stdout)
        if matches:
            result[field] = convert(matches[-1])
    return result


def run_once(args: argparse.Namespace, baseline: dict, index: int) -> dict:
    """Run with precompiled shared models, preserving diagnostics and waveforms."""
    executable = str(args.binary.resolve())
    environment = os.environ.copy()
    if args.library_dir:
        environment["LD_LIBRARY_PATH"] = (
            os.pathsep.join(map(str, args.library_dir)) + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
        )
    deck = make_deck(args.vacask_root, baseline["modules"], args.mode, baseline["stop_s"], baseline["step_s"])
    prefix = args.output.with_name(f"{args.output.stem}-{index}")
    prefix.with_suffix(".sim").write_text(deck)
    result = {
        "simulator": "vacask",
        "mode": args.mode,
        "cpus": args.cpus,
        "stepping": "adaptive" if args.mode == "upstream" else "adaptive_bdf2_maxstep_2ps",
        "tolerances": {
            "reltol": 1e-3,
            "abstol_A": 1e-12,
            "vntol_V": 1e-6,
            "chgtol_C": 1e-15,
            "fluxtol_Wb": 1e-14,
            "tran_lteratio": 1.5,
            "source": "lib/options.cpp defaults plus upstream runme.sim",
        },
        "blas_cpus": 1,
        "stop_s": baseline["stop_s"],
        "initial_step_s": baseline["step_s"],
        "initial_conditions": "uic_zero",
        "modules": baseline["modules"],
        "status": "starting",
        "run": index,
        "binary": executable,
        "binary_sha256": hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
        "sources": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in [
                args.vacask_root / "benchmark/c6288/vacask" / name for name in ("runme.sim", "models.inc", "multiplier.inc")
            ]
        ],
    }
    with tempfile.TemporaryDirectory(prefix="circulax-c6288-vacask-") as temporary:
        directory = Path(temporary)
        (directory / "input.sim").write_text(deck)
        (directory / "config.toml").write_text(
            "[Paths]\nmodule_path_prefix=" + json.dumps([str(args.vacask_root / "build/lib/vacask/mod")]) + "\n"
        )
        command = [
            executable,
            "--tomlfile",
            "config.toml",
            "--skip-embed",
            "--skip-postprocess",
            "--quiet-progress",
            "--ncpu",
            str(args.cpus),
            "--blas-ncpu",
            "1",
            "input.sim",
        ]
        started = time.perf_counter()
        try:
            process = subprocess.run(  # noqa: S603 -- explicit simulator, no shell
                command, cwd=directory, env=environment, capture_output=True, text=True, timeout=args.timeout, check=False
            )
        except subprocess.TimeoutExpired:
            result.update(status="timeout", process_wall_s=time.perf_counter() - started)
            return result
        result.update(process_wall_s=time.perf_counter() - started, exit_code=process.returncode)
        prefix.with_suffix(".log").write_text(process.stdout + process.stderr)
        result["version"] = process.stdout.splitlines()[0] if process.stdout else ""
        result.update(parse_stats(process.stdout))
        rawpath = directory / "tranmul.raw"
        if (
            process.returncode
            or not rawpath.exists()
            or re.search(r"Analysis.*aborted|Homotopy failed|model skipped", process.stdout)
        ):
            result["status"] = "simulator_failed"
            return result
        spec = importlib.util.spec_from_file_location("c6288_vacask_rawfile", args.vacask_root / "python/rawfile.py")
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
        raw = reader.rawread(str(rawpath)).get()
        names = [f"{kind}{bit}" for kind, count in (("a", 16), ("b", 16), ("p", 32)) for bit in range(count)]
        times = np.asarray(raw["time"])
        values = np.column_stack([raw[name] for name in names])
        result.update(
            saved_timepoints=len(times),
            final_time_s=float(times[-1]),
            finite=bool(np.isfinite(values).all()),
            final_product=check_product(values[-1, 32:], 0xFFFE0001),
            inputs_high=bool(np.all((values[-1, :32] >= 0.96) & (values[-1, :32] <= 1.32))),
        )
        completed = math.isclose(float(times[-1]), baseline["stop_s"], rel_tol=1e-9, abs_tol=1e-18)
        result["status"] = (
            "ok"
            if completed and result["finite"] and result["final_product"]["valid_logic"] and result["inputs_high"]
            else "validation_failed"
        )
        np.savez_compressed(prefix.with_suffix(".npz"), time=times, voltages=values, names=names)
        if args.circulax.with_suffix(".npz").exists():
            actual = np.load(args.circulax.with_suffix(".npz"))
            interpolated = np.column_stack([np.interp(actual["time"], times, values[:, bit]) for bit in range(64)])
            error = interpolated[:, 32:] - actual["voltages"][:, 32:]
            result["waveform_difference"] = {
                "max_abs_v": float(np.max(np.abs(error))),
                "rms_v": float(np.sqrt(np.mean(error**2))),
                "final_max_abs_v": float(np.max(np.abs(error[-1]))),
                "comparison": "VACASK linearly interpolated at Circulax saved times",
            }
        if result["status"] == "ok":
            result["circulax_transient_over_vacask_process"] = baseline["wall_s"] / result["process_wall_s"]
        return result


def main() -> int:  # noqa: C901 -- reference validation and resumable experiment orchestration
    """Run repeated native reference measurements and write their summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vacask-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, action="append", default=[])
    parser.add_argument("--circulax", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--mode", choices=("upstream", "bdf2-capped"), default="upstream")
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Reuse completed runs from the same output and configuration")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("vacask-results.json"))
    args = parser.parse_args()
    args.vacask_root = args.vacask_root.resolve()
    if args.cpus < 1 or args.repeats < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("cpus, repeats and timeout must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(args.circulax.read_text())
    if baseline["status"] != "ok":
        parser.error("Circulax baseline must have status=ok")
    for module in baseline["modules"]:
        if hashlib.sha256(Path(module["path"]).read_bytes()).hexdigest() != module["sha256"]:
            parser.error("OSDI binary no longer matches Circulax baseline")
    results = []
    if args.resume and args.output.exists():
        results = json.loads(args.output.read_text())["runs"]
        for result in results:
            if result["mode"] != args.mode or result["cpus"] != args.cpus or result["modules"] != baseline["modules"]:
                parser.error("Saved run configuration differs; choose a fresh output")
            prefix = args.output.with_name(f"{args.output.stem}-{result['run']}")
            result.update(parse_stats(prefix.with_suffix(".log").read_text()))
        results = [r for r in results if r["status"] == "ok"]
    for index in range(len(results) + 1, args.repeats + 1):
        result = run_once(args, baseline, index)
        results.append(result)
        args.output.write_text(json.dumps({"runs": results}, indent=2) + "\n")
        print(json.dumps(result), flush=True)  # noqa: T201 -- benchmark progress
        if result["status"] != "ok":
            break
    success = [r for r in results if r["status"] == "ok"]
    summary = {
        "mode": args.mode,
        "runs": results,
        "circulax_transient_s": baseline["wall_s"],
        "circulax_setup_s": baseline["setup_s"],
        "circulax_compile_s": baseline["compile_s"],
    }
    if success:
        summary["median_process_wall_s"] = float(np.median([r["process_wall_s"] for r in success]))
        summary["median_nr_solver_s"] = float(np.median([r["nr_solver_s"] for r in success if "nr_solver_s" in r]))
        summary["circulax_transient_over_vacask_process"] = baseline["wall_s"] / summary["median_process_wall_s"]
        summary["timing_comparison"] = "Circulax timed transient / VACASK whole process; Circulax setup and JIT excluded"
        summary["circulax_measured_total_s"] = baseline["wall_s"] + baseline["setup_s"] + baseline["compile_s"]
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        fields = ["run", "mode", "cpus", "status", "process_wall_s", "nr_solver_s", "accepted_timepoints", "saved_timepoints"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    return 0 if len(success) == args.repeats else 1


if __name__ == "__main__":
    raise SystemExit(main())
