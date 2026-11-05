"""Run the c6288 stress test in a bounded child process; preserve partial results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import traceback
import time

# This benchmark intentionally prints progress and executes its own worker.
# ruff: noqa: T201, S603
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main() -> int:  # noqa: PLR0915 -- command-line orchestration
    """Run a bounded worker and preserve a machine-readable outcome."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vacask-root", type=Path, default=os.environ.get("VACASK_REPO"))
    parser.add_argument("--compiler", default="openvaf-r")
    parser.add_argument("--psp-osdi", type=Path)
    parser.add_argument("--resistor-osdi", type=Path)
    parser.add_argument("--backend", choices=("klu_split", "klu_split_linear"), default="klu_split")
    parser.add_argument("--solver", choices=("bdf2", "trap", "euler"), default="bdf2")
    parser.add_argument("--newton", choices=("full", "frozen"), default="full")
    parser.add_argument("--max-newton", type=int, default=100)
    parser.add_argument("--stop", type=float, default=2e-9)
    parser.add_argument("--step", type=float, default=2e-12)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--representation", choices=("full", "reduced"), default="reduced")
    parser.add_argument("--audit-only", action="store_true", help="Warm up and run two short calls to audit JIT reuse")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.vacask_root is None:
        parser.error("supply --vacask-root or set VACASK_REPO")
    args.vacask_root = args.vacask_root.resolve()
    args.output = args.output.resolve()
    if not all(math.isfinite(x) and x > 0 for x in (args.stop, args.step, args.timeout)) or args.step > args.stop:
        parser.error("stop, step and timeout must be finite and positive; step must not exceed stop")
    if args.max_newton < 1:
        parser.error("max-newton must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def record(result: dict) -> None:
        temporary = args.output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output)
        print(json.dumps(result), flush=True)

    if args.worker:
        from benchmarks.c6288.bench_circulax import run

        try:
            result = run(args, record)
        except Exception as exc:  # noqa: BLE001 -- preserve benchmark failures
            result = json.loads(args.output.read_text()) if args.output.exists() else {}
            result.update(status="error", error=str(exc))
            record(result)
            traceback.print_exc()
        return 0 if result["status"] in {"ok", "prepared", "jit_audited"} else 1
    record({"status": "starting"})
    log = args.output.with_suffix(".log")
    process_start = time.perf_counter()
    with log.open("w") as stream:
        try:
            child = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"],
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=False,
                env={**os.environ, "JAX_PLATFORMS": "cpu"},
            )
            code = child.returncode
        except subprocess.TimeoutExpired:
            code = 124
    result = json.loads(args.output.read_text())
    result["process_wall_s"] = time.perf_counter() - process_start
    record(result)
    if code and result["status"] not in {"error", "transient_failed", "warmup_failed", "logic_mismatch"}:
        result.update(status="timeout" if code == 124 else "process_failed", exit_code=code)
        record(result)
    with args.output.with_suffix(".csv").open("w", newline="") as stream:
        fields = [
            "status",
            "sys_size",
            "setup_s",
            "compile_s",
            "wall_s",
            "n_steps",
            "us_per_step",
            "solver",
            "backend",
            "newton",
            "max_newton",
            "stop_s",
            "step_s",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(result)
    print(json.dumps(result, indent=2))
    print(f"Log: {log}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
