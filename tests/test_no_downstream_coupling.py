"""Circulax is a library; it must not name or depend on the applications that consume it."""

import re
from pathlib import Path

import circulax

_PACKAGE = Path(circulax.__file__).parent
_FORBIDDEN = re.compile(r"gdsfactoryplus|gfp\b|@tags\b")


def test_package_does_not_reference_downstream_apps() -> None:
    """No imports, issue references or doc-tool markers for a consuming application."""
    offenders = [
        f"{path.relative_to(_PACKAGE)}:{number}: {line.strip()}"
        for path in sorted(_PACKAGE.rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if _FORBIDDEN.search(line)
    ]
    assert not offenders, "downstream-app coupling in circulax:\n" + "\n".join(offenders)
