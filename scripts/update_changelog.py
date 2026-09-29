# ruff: noqa: INP001
"""Refresh only the Unreleased section of CHANGELOG.md with git-cliff."""

from __future__ import annotations

import subprocess
from pathlib import Path
from shutil import which

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"
UNRELEASED = "## [Unreleased]"


def main() -> None:
    """Generate current release notes without rewriting tagged history."""
    git = which("git")
    if git is None:
        msg = "git is required to update the changelog"
        raise RuntimeError(msg)
    generated = subprocess.run(  # noqa: S603
        [git, "cliff", "--unreleased", "--strip", "header"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    current = CHANGELOG.read_text()
    start = current.index(UNRELEASED)
    next_release = current.index("\n## [", start + len(UNRELEASED))
    updated = f"{current[:start]}{generated}\n\n{current[next_release + 1 :]}"
    CHANGELOG.write_text(updated)


if __name__ == "__main__":
    main()
