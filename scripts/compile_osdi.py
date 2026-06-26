"""Compile Verilog-A test fixtures to OSDI using openvaf-r.

Compiles every .va file under tests/data/va/ (non-recursive) to a
platform-native .osdi binary alongside the source.  Uses --target-cpu generic
so the output runs on any x86-64 machine without AVX2.

Requires openvaf-r in PATH.  Install from:
    https://github.com/arpadbuermen/OpenVAF
"""

import subprocess
import sys
from pathlib import Path

VA_DIR = Path(__file__).parent.parent / "tests" / "data" / "va"
VA_FILES = sorted(VA_DIR.glob("*.va"))

if not VA_FILES:
    print("No .va files found in", VA_DIR)
    sys.exit(0)

errors = []
for va in VA_FILES:
    osdi = va.with_suffix(".osdi")
    print(f"  {va.name} -> {osdi.name}")
    result = subprocess.run(
        ["openvaf-r", "--target_cpu", "generic", str(va), "-o", str(osdi)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr.strip()}", file=sys.stderr)
        errors.append(va.name)

if errors:
    print(f"\n{len(errors)} file(s) failed to compile: {errors}", file=sys.stderr)
    sys.exit(1)

print(f"Compiled {len(VA_FILES)} file(s) successfully.")
