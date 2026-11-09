"""Sequential repeated comparison; every subprocess command is saved before execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vacask-root", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--phases", nargs="+", choices=("circulax", "vacask", "ring"), default=["circulax", "vacask", "ring"])
    parser.add_argument("--ring-worker", action="store_true")
    parser.add_argument("--stages", type=int)
    parser.add_argument("--representation", choices=("full", "reduced"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.ring_worker:
        binary = REPO / "circulax/components/osdi/compiled/psp103v4_psp103.osdi"
        if not binary.exists():
            subprocess.run([sys.executable, str(REPO / "scripts/compile_osdi.py")], check=True)
        from benchmarks.ring.bench_circulax import run

        result = run(
            n_stages=args.stages,
            variant="osdi",
            osdi_reduction=args.representation == "reduced",
            waveform_path=args.output.with_suffix(".npz"),
        )
        result["model"] = {"path": str(binary), "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
        result["fixture_sha256"] = hashlib.sha256((REPO / "tests/fixtures/psp103_models.py").read_bytes()).hexdigest()
        cards = args.vacask_root / "benchmark/ring/vacask/models.inc"
        result["model_cards"] = {"path": str(cards), "sha256": hashlib.sha256(cards.read_bytes()).hexdigest()}
        result["representation"] = args.representation
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return int(result["status"] != "ok")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    common = ["--vacask-root", str(args.vacask_root.resolve())]
    environment = {
        **os.environ,
        "JAX_LOG_COMPILES": "1",
        "JAX_PLATFORMS": "cpu",
        "CIRCULAX_RING_CACHE": str(out / "ring-jax-cache"),
    }

    # Avoid persistent cache hits hiding the compiler audit. Ring's configured
    # persistent cache is retained for reproducibility; its logs distinguish hits.
    def execute(command, output):
        if args.resume and output.exists() and json.loads(output.read_text()).get("status") == "ok":
            print(f"Resume: {output.name}", flush=True)
            return
        entry = {
            "command": command,
            "cwd": str(REPO),
            "started_unix": time.time(),
            "environment": {
                k: environment.get(k)
                for k in (
                    "JAX_LOG_COMPILES",
                    "JAX_PLATFORMS",
                    "CIRCULAX_RING_CACHE",
                    "RAYON_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                )
            },
        }
        with (out / "commands.jsonl").open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(f"Start: {output.name}", flush=True)
        with output.with_name(output.stem + "-driver.log").open("w") as stream:
            process = subprocess.run(command, cwd=REPO, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=7200)
        print(f"Exit {process.returncode}: {output.name}", flush=True)
        if process.returncode:
            raise RuntimeError(f"Benchmark failed: {output}; inspect its logs")

    if "circulax" in args.phases:
        for repeat in range(1, args.repeats + 1):
            for representation in ("full", "reduced"):
                output = out / f"{representation}-{repeat}.json"
                execute(
                    [
                        sys.executable,
                        str(REPO / "benchmarks/c6288/run.py"),
                        *common,
                        "--representation",
                        representation,
                        "--timeout",
                        "1200",
                        "--output",
                        str(output),
                    ],
                    output,
                )
    if "vacask" in args.phases:
        for mode in ("upstream", "bdf2-capped"):
            output = out / f"native-{mode}.json"
            execute(
                [
                    sys.executable,
                    str(REPO / "benchmarks/c6288/bench_vacask.py"),
                    *common,
                    "--binary",
                    str(args.binary.resolve()),
                    "--library-dir",
                    str(args.library_dir.resolve()),
                    "--circulax",
                    str(out / "reduced-1.json"),
                    "--repeats",
                    str(args.repeats),
                    "--mode",
                    mode,
                    "--output",
                    str(output),
                ],
                output,
            )
    if "ring" in args.phases:
        for stages in (3, 9, 15):
            for repeat in range(1, args.repeats + 1):
                for representation in ("full", "reduced"):
                    output = out / f"ring-{stages}-{representation}-{repeat}.json"
                    execute(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            *common,
                            "--binary",
                            str(args.binary),
                            "--library-dir",
                            str(args.library_dir),
                            "--ring-worker",
                            "--stages",
                            str(stages),
                            "--representation",
                            representation,
                            "--output",
                            str(output),
                        ],
                        output,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
