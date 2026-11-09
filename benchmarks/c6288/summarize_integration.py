"""Summarize repeated measurements and compare waveforms on the saved fixed grid."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent


def distribution(values):
    array = np.asarray(values, dtype=float)
    return {
        "values": array.tolist(),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "sample_std": float(np.std(array, ddof=1)) if len(array) > 1 else None,
    }


def difference(reference, actual):
    a = np.load(reference)
    b = np.load(actual)
    interpolated = np.column_stack([np.interp(a["time"], b["time"], b["voltages"][:, i]) for i in range(64)])
    error = interpolated[:, 32:] - a["voltages"][:, 32:]
    return {
        "max_abs_V": float(abs(error).max()),
        "rms_V": float(np.sqrt(np.mean(error**2))),
        "final_max_abs_V": float(abs(error[-1]).max()),
        "worst_output_bit": int(np.argmax(np.max(abs(error), axis=0))),
    }


def main():
    result = {"circulax": {}, "vacask": {}, "rings": {}, "jit_audit": {}, "waveform_comparison": {}}
    for representation in ("full", "reduced"):
        files = sorted(ROOT.glob(f"{representation}-[0-9].json"))
        runs = [json.loads(path.read_text()) for path in files]
        if runs:
            result["circulax"][representation] = {
                "statuses": [run["status"] for run in runs],
                "unknowns": [run["sys_size"] for run in runs],
                **{
                    field: distribution([run[field] for run in runs if field in run])
                    for field in ("setup_s", "compile_s", "wall_s")
                },
                "runs": [p.name for p in files],
            }
        for path in files:
            log = path.with_suffix(".log").read_text()
            count = len(re.findall(r"Compiling jit\(circuit_diffeqsolve\)", log))
            result["jit_audit"][path.name] = {"transient_compilations": count, "warmup_and_timed_share_executable": count == 1}
    for mode in ("upstream", "bdf2-capped"):
        path = ROOT / f"native-{mode}.json"
        if path.exists():
            runs = json.loads(path.read_text())["runs"]
            result["vacask"][mode] = {
                "statuses": [r["status"] for r in runs],
                "process_wall_s": distribution([r["process_wall_s"] for r in runs]),
                "steps": [r.get("accepted_timepoints") for r in runs],
                "unknowns": [r.get("unknowns") for r in runs],
            }
    for stages in (3, 9, 15):
        for representation in ("full", "reduced"):
            paths = sorted(ROOT.glob(f"ring-{stages}-{representation}-[0-9].json"))
            runs = [json.loads(p.read_text()) for p in paths]
            if runs:
                result["rings"][f"{stages}-{representation}"] = {
                    "statuses": [r["status"] for r in runs],
                    "unknowns": [r.get("sys_size") for r in runs],
                    "frequencies_MHz": [r.get("freq_MHz") for r in runs],
                    "wall_s": distribution([r["wall_s"] for r in runs if "wall_s" in r])
                    if any("wall_s" in r for r in runs)
                    else None,
                }
                for path in paths:
                    log = path.with_name(path.stem + "-driver.log").read_text()
                    result["jit_audit"][path.name] = {
                        "transient_compilations": len(re.findall(r"Compiling jit\(circuit_diffeqsolve\)", log)),
                        "persistent_cache_hits": len(
                            re.findall(r"Persistent compilation cache hit for 'jit_circuit_diffeqsolve", log)
                        ),
                    }
    reference = ROOT / "reduced-1.npz"
    for pattern in ("full-[0-9].npz", "reduced-[2-9].npz", "native-*-capped-[0-9].npz", "native-upstream-[0-9].npz"):
        for path in sorted(ROOT.glob(pattern)):
            result["waveform_comparison"][path.name] = difference(reference, path)
    result["kernel_profiles"] = {
        representation: json.loads((ROOT / f"profile-{representation}.json").read_text())
        for representation in ("full", "reduced")
        if (ROOT / f"profile-{representation}.json").exists()
    }
    result["validation"] = {}
    for representation in ("full", "reduced"):
        for index in range(1, 4):
            name = f"{representation}-{index}"
            record = json.loads((ROOT / f"{name}.json").read_text())
            waveform = np.load(ROOT / f"{name}.npz")
            result["validation"][name] = {
                "status": record["status"],
                "steps": record["n_steps"],
                "samples": len(waveform["time"]),
                "final_time_s": float(waveform["time"][-1]),
                "finite": bool(np.isfinite(waveform["voltages"]).all()),
                "product": record["final_product"],
                "inputs_high": record["inputs_high"],
            }
    for stages in (3, 9, 15):
        a = np.load(ROOT / f"ring-{stages}-full-1.npz")
        b = np.load(ROOT / f"ring-{stages}-reduced-1.npz")
        error = a["voltages"] - b["voltages"]
        full = result["rings"][f"{stages}-full"]["frequencies_MHz"][0]
        reduced = result["rings"][f"{stages}-reduced"]["frequencies_MHz"][0]
        result["waveform_comparison"][f"ring-{stages}"] = {
            "max_abs_V": float(abs(error).max()),
            "rms_V": float(np.sqrt(np.mean(error**2))),
            "frequency_relative_difference": float(abs(full - reduced) / full),
        }
    for key, filename in (
        ("ring_startup_audit", "ring-startup-audit.json"),
        ("source_topology_comparison", "source-topology-comparison.json"),
    ):
        if (ROOT / filename).exists():
            result[key] = json.loads((ROOT / filename).read_text())
    result["test_runs"] = {}
    for key, filename in (
        ("circulax", "integration-tests-complete.log"),
        ("released_bosdi", "released-bosdi-tests.log"),
        ("bosdi", "bosdi-tests.log"),
    ):
        log = (ROOT / filename).read_text()
        passed = re.findall(r"(\d+) passed", log)
        skipped = re.findall(r"(\d+) skipped", log)
        result["test_runs"][key] = {"log": filename, "passed": int(passed[-1]), "skipped": int(skipped[-1]) if skipped else 0}
    (ROOT / "integration-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
