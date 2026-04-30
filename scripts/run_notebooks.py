"""Execute all example notebooks in-place with papermill.

Replaces the old Unix-only:
  find examples -name '*.ipynb' ... | xargs -P4 -I NB papermill NB NB -k circulax
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
notebooks = sorted(
    p for p in (ROOT / "examples").rglob("*.ipynb")
    if ".ipynb_checkpoints" not in p.parts
)

if not notebooks:
    print("No notebooks found in examples/")
    sys.exit(0)

print(f"Running {len(notebooks)} notebooks…")
failed = []
for nb in notebooks:
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
