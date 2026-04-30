"""Remove the dist/ build artefact directory.

Replaces the old Unix-only: rm -rf dist/
"""

import shutil
from pathlib import Path

dist = Path(__file__).parent.parent / "dist"
shutil.rmtree(dist, ignore_errors=True)
print(f"Removed {dist}")
