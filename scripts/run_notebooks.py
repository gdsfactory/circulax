"""Execute all example notebooks in-place with papermill.

Replaces the old Unix-only:
  find examples -name '*.ipynb' ... | xargs -P4 -I NB papermill NB NB -k circulax
"""

import subprocess
import sys
from pathlib import Path

try:
    from bosdi.circulax import osdi_component as _  # noqa: F401

    _HAS_OSDI = True
except ImportError:
    _HAS_OSDI = False

_NEEDS_OSDI = {"ring_oscillator_osdi.ipynb", "05_psp103_ring_param_fitting.ipynb"}

ROOT = Path(__file__).parent.parent
notebooks = sorted(
    p for p in (ROOT / "examples").rglob("*.ipynb")
    if ".ipynb_checkpoints" not in p.parts
)

if not notebooks:
    print("No notebooks found in examples/")
    sys.exit(0)

skipped = [nb for nb in notebooks if nb.name in _NEEDS_OSDI and not _HAS_OSDI]
to_run = [nb for nb in notebooks if nb not in skipped]

if skipped:
    print(f"Skipping {len(skipped)} OSDI notebook(s) — bosdi.circulax not installed:")
    for nb in skipped:
        print(f"  {nb.relative_to(ROOT)}")

print(f"Running {len(to_run)} notebooks…")
failed = []
for nb in to_run:
    print(f"  {nb.relative_to(ROOT)}")
    result = subprocess.run(
        [sys.executable, "-m", "papermill", str(nb), str(nb), "-k", "circulax"],
        check=False,
    )
    if result.returncode != 0:
        failed.append(nb)

if failed:
    print(f"\n{len(failed)} notebook(s) failed:")
    for nb in failed:
        print(f"  {nb.relative_to(ROOT)}")
    sys.exit(1)

print("All notebooks executed successfully.")
