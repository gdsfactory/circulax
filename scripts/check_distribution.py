"""Reject direct URL dependencies in built distributions before a PyPI upload.

Twine checks metadata rendering but does not enforce PyPI's dependency URL ban.
"""

# ruff: noqa: INP001 -- standalone release check

import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

from packaging.requirements import Requirement


def check_distribution(path: Path) -> None:
    """Validate every Requires-Dist field, including optional extras."""
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
            metadata = archive.read(name)
    else:
        with tarfile.open(path) as archive:
            name = next(name for name in archive.getnames() if name.count("/") == 1 and name.endswith("/PKG-INFO"))
            member = archive.extractfile(name)
            if member is None:
                msg = f"Missing PKG-INFO in {path}"
                raise ValueError(msg)
            metadata = member.read()
    requirements = BytesParser().parsebytes(metadata).get_all("Requires-Dist", [])
    for text in requirements:
        if Requirement(text).url is not None:
            msg = f"{path}: PyPI does not accept direct URL requirement {text!r}"
            raise ValueError(msg)


if __name__ == "__main__":
    for filename in sys.argv[1:]:
        check_distribution(Path(filename))
