"""Compile Verilog-A sources to OSDI using openvaf-r --target_cpu generic.

Compiles two sets of VA files:
  1. tests/data/va/*.va  → alongside the source (test fixtures)
  2. circulax/components/osdi/psp103v4/psp103.va
     → circulax/components/osdi/compiled/psp103v4_psp103.osdi

Using --target_cpu generic avoids AVX2 issues on heterogeneous CI runners.

Requires openvaf-r in PATH.  Install from:
    https://github.com/arpadbuermen/OpenVAF
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

# (va_source, osdi_output)
TARGETS: list[tuple[Path, Path]] = []

# Test fixtures — every .va in tests/data/va/
VA_DIR = ROOT / "tests" / "data" / "va"
for va in sorted(VA_DIR.glob("*.va")):
    TARGETS.append((va, va.with_suffix(".osdi")))

# PSP103 component used by notebooks / ring-oscillator examples
PSP103_VA = ROOT / "circulax" / "components" / "osdi" / "psp103v4" / "psp103.va"
PSP103_OSDI = ROOT / "circulax" / "components" / "osdi" / "compiled" / "psp103v4_psp103.osdi"
if PSP103_VA.exists():
    TARGETS.append((PSP103_VA, PSP103_OSDI))

if not TARGETS:
    print("No .va files found to compile.")
    sys.exit(0)

errors = []
for va, osdi in TARGETS:
    print(f"  {va.name} -> {osdi}")
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

print(f"Compiled {len(TARGETS)} file(s) successfully.")
