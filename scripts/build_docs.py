"""Convert executed example notebooks to Markdown in docs/examples/.

Replaces the old Unix-only:
  rm -rf docs/examples && mkdir -p docs/examples
  && jupyter nbconvert --config docs/nbconvert_config.py --to markdown
  && find examples -name '*.gif' | xargs -I GIF cp GIF docs/examples/
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
docs_examples = ROOT / "docs" / "examples"

shutil.rmtree(docs_examples, ignore_errors=True)
docs_examples.mkdir(parents=True)

subprocess.run(
    [sys.executable, "-m", "jupyter", "nbconvert",
     "--config", str(ROOT / "docs" / "nbconvert_config.py"),
     "--to", "markdown"],
    check=True,
    cwd=ROOT,
)

for gif in (ROOT / "examples").rglob("*.gif"):
    dest = docs_examples / gif.name
    shutil.copy(gif, dest)
    print(f"Copied {gif.name}")

print(f"Docs built in {docs_examples}")
