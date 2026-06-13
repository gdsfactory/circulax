"""Unified benchmark entrypoint.

Examples:
    pixi run python benchmarks/run.py list
    pixi run python benchmarks/run.py run ring -- 3 9
    pixi run python benchmarks/run.py run release
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


@dataclass(frozen=True)
class BenchmarkCase:
    path: Path
    description: str
    release: bool = False
    default_args: tuple[str, ...] = ()


CASES: dict[str, BenchmarkCase] = {
    "rc": BenchmarkCase(ROOT / "rc" / "run.py", "VACASK/ngspice/Circulax RC benchmark", release=True),
    "mul": BenchmarkCase(ROOT / "mul" / "run.py", "VACASK/ngspice/Circulax diode multiplier benchmark", release=True),
    "ring": BenchmarkCase(ROOT / "ring" / "run.py", "PSP103 ring oscillator via OSDI", release=True),
    "juncap200": BenchmarkCase(ROOT / "juncap200" / "run.py", "IHP juncap200 DC sweep via OSDI"),
    "mosvar": BenchmarkCase(ROOT / "mosvar" / "run.py", "IHP mosvar DC sweep via OSDI"),
    "ring-bsim4": BenchmarkCase(ROOT / "ring_bsim4" / "bench_bsim4_osdi.py", "BSIM4 ring oscillator via OSDI"),
    "legacy-rc-pulse": BenchmarkCase(ROOT / "legacy" / "rc_pulse_testbench.py", "Legacy RC pulse testbench"),
    "legacy-diode-cascade": BenchmarkCase(ROOT / "legacy" / "diode_cascade_testbench.py", "Legacy diode cascade testbench"),
    "legacy-rectifier": BenchmarkCase(ROOT / "legacy" / "fullwave_rect_testbench.py", "Legacy full-wave rectifier testbench"),
    "legacy-lc-ladder": BenchmarkCase(ROOT / "legacy" / "lc_ladder_testbench.py", "Legacy LC ladder scalability testbench"),
    "legacy-diode-clipper-hb": BenchmarkCase(ROOT / "legacy" / "diode_clipper_hb_testbench.py", "Legacy diode clipper HB benchmark"),
    "legacy-stiff-newton": BenchmarkCase(ROOT / "legacy" / "stiff_newton_benchmark.py", "Legacy stiff Newton solver benchmark"),
    "legacy-diode-solver": BenchmarkCase(ROOT / "legacy" / "diode_cascade_solver_benchmark.py", "Legacy diode solver comparison"),
}


def _run_case(name: str, extra_args: list[str]) -> int:
    case = CASES[name]
    cmd = [sys.executable, str(case.path), *case.default_args, *extra_args]
    print(f"[{name}] {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=REPO, check=False).returncode


def list_cases() -> int:
    width = max(len(name) for name in CASES)
    for name, case in CASES.items():
        marker = "release" if case.release else ""
        print(f"{name:<{width}}  {marker:<7}  {case.description}")
    return 0


def run_cases(target: str, extra_args: list[str]) -> int:
    names = [name for name, case in CASES.items() if case.release] if target == "release" else [target]
    unknown = [name for name in names if name not in CASES]
    if unknown:
        print(f"Unknown benchmark: {unknown[0]}", file=sys.stderr)
        return 2

    failed: list[tuple[str, int]] = []
    for name in names:
        rc = _run_case(name, extra_args if target != "release" else [])
        if rc:
            failed.append((name, rc))

    if failed:
        for name, rc in failed:
            print(f"[{name}] failed with exit code {rc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List available benchmark cases")

    run_p = sub.add_parser("run", help="Run one benchmark case or the release suite")
    run_p.add_argument("target", choices=tuple(CASES) + ("release",))
    run_p.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the benchmark after '--'")

    args = parser.parse_args(argv)
    if args.command == "list":
        return list_cases()
    if args.command == "run":
        forwarded = args.args[1:] if args.args[:1] == ["--"] else args.args
        return run_cases(args.target, forwarded)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
