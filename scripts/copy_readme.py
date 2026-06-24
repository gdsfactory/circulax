"""Copy ReadMe.md into docs/index.md and strip 'docs/' URL prefixes.

Replaces the old Unix-only: cp ReadMe.md docs/index.md && sed -i 's|docs/||g' docs/index.md
"""

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent
src = ROOT / "ReadMe.md"
dst = ROOT / "docs" / "index.md"

shutil.copy(src, dst)
dst.write_text(re.sub(r"docs/", "", dst.read_text()))
print(f"Copied {src} → {dst} (stripped docs/ prefixes)")
