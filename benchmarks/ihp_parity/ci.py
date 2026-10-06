"""Provision the locked IHP/VACASK packages and require both parity matrices."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

VA_SOURCES = ("psp103/psp103.va", "psp103/psp103_nqs.va", "r3_cmc/r3_cmc.va", "mosvar/mosvar.va")


def stage_pdk(source: Path, destination: Path) -> None:
    """Copy package resources and correct the known converted resistor drift."""
    shutil.copytree(source / "models", destination / "ihp/models", dirs_exist_ok=True)
    native = destination / "ihp/models/ngspice/models/resistors_mod.lib"
    converted = destination / "ihp/models/vacask/models/resistors_mod.lib"
    original, text = native.read_text(), converted.read_text()
    if original.count("sw_mman=1") != 3:
        msg = "IHP native resistor settings changed; review the temporary parity correction"
        raise ValueError(msg)
    if text.count("sw_mman=0 nsmm_rsh=1") != 3:
        msg = "IHP converted resistor settings changed; review/remove the temporary parity correction"
        raise ValueError(msg)
    converted.write_text(text.replace("sw_mman=0 nsmm_rsh=1", "sw_mman=1 nsmm_rsh=1"))


def fingerprint(path: Path) -> dict[str, str]:
    """Record an external artifact's exact bytes."""
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    """Run parity with no workstation checkout or simulator paths."""
    import vacask_bin

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path(".pixi/ihp-parity"))
    parser.add_argument("--output-dir", type=Path, default=Path("parity-results"))
    parser.add_argument("--ngspice", type=Path, help="OSDI-enabled ngspice; defaults to the pinned CI build")
    parser.add_argument("--only", default="", help="Optional local smoke selection; CI runs the full matrix")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build, output = args.build_dir.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("MPLCONFIGDIR", str(build / "matplotlib"))
    package = importlib.util.find_spec("ihp")
    if package is None or package.origin is None:
        parser.error("install the ihp-parity Pixi environment")
    pdk = build / "pdk"
    stage_pdk(Path(package.origin).parent, pdk)
    compiler, simulator = Path(vacask_bin.OPENVAF_CMD), Path(vacask_bin.VACASK_CMD)
    module_root = Path(vacask_bin.MOD_DIR)
    runtime = Path(vacask_bin.BIN_DIR) / "lib"
    # Prefer the locked C++ runtime: the wheel's older libstdc++ cannot run
    # ngspice built by current system compilers. Its LLVM libraries remain available.
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
        (str(Path(sys.prefix) / "lib"), str(runtime), os.environ.get("LD_LIBRARY_PATH", ""))
    )

    # Import the native bridge after configuring the wheel's runtime libraries.
    from circulax.netlist_io.osdi import compile_va, module_metadata

    cache = build / "osdi-cache"
    compiled = []
    for name in VA_SOURCES:
        logger.info("Compiling/caching %s", name)
        compiled.append(compile_va(pdk / "ihp/models/ngspice/va" / name, compiler=str(compiler), cache_dir=cache))
    primitives = [module_root / "spice" / f"{name}.osdi" for name in ("resistor", "capacitor", "inductor")]
    modules = compiled[:3] + primitives
    for module in modules:
        module_metadata(module)  # Fail early if the simulator/compiler ABI drifts.
    ngspice = (args.ngspice or build / "toolchain/ngspice/install/bin/ngspice").resolve()
    if not ngspice.is_file():
        parser.error("run the ihp-build-ngspice Pixi task or supply --ngspice with OSDI support")
    common = [
        sys.executable,
        "-m",
        "benchmarks.ihp_parity.run",
        "--pdk",
        str(pdk),
        "--vacask",
        str(simulator),
        "--ngspice",
        str(ngspice),
        "--compiler",
        str(compiler),
        "--vacask-compiler",
        str(compiler),
        "--cache-dir",
        str(cache),
        "--module-path",
        str(module_root),
        "--shared-library-path",
        str(runtime),
    ]
    for module in modules:
        common.extend(("--osdi-module", str(module)))
    for module in compiled[:3]:
        common.extend(("--ngspice-osdi-module", str(module)))
    if args.only:
        common.extend(("--only", args.only))
    provenance = {
        "ihp": json.loads(importlib.metadata.distribution("ihp-gdsfactory").read_text("direct_url.json")),
        "vacask_bin": importlib.metadata.version("vacask-bin"),
        "compiler": fingerprint(compiler),
        "simulator": fingerprint(simulator),
        "ngspice": fingerprint(ngspice),
        "ngspice_revision": "724dc77b9153dc75eaec474b1f7a44f1fcf5f362",
        "modules": [fingerprint(module) for module in modules],
        "temporary_correction": "Copied IHP converted resistor cards: three sw_mman=0 -> sw_mman=1 settings",
        "generic_cpu": True,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    failed = False
    for backend in ("ngspice", "vacask"):
        result = output / f"{backend}.json"
        logger.info("Running %s parity", backend)
        result.unlink(missing_ok=True)
        status = subprocess.run([*common, "--simulator", backend, "--output", str(result)], check=False)  # noqa: S603
        data = json.loads(result.read_text()) if result.is_file() else {}
        passed = sum(case.get("passed", False) for case in data.values())
        logger.info("%s: %s/%s passed", backend, passed, len(data))
        failed |= status.returncode != 0 or not data or passed != len(data) or (not args.only and len(data) != 60)
    raise SystemExit(int(failed))


if __name__ == "__main__":
    main()
