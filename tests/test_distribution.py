import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.check_distribution import check_distribution


@pytest.mark.parametrize("suffix", [".whl", ".tar.gz"])
@pytest.mark.parametrize("direct_url", [False, True])
def test_distribution_dependencies(tmp_path: Path, suffix: str, *, direct_url: bool) -> None:
    requirement = (
        'parser @ git+https://example.com/parser.git ; extra == "netlists"' if direct_url else 'parser>=1 ; extra == "netlists"'
    )
    metadata = f"Metadata-Version: 2.4\nName: example\nVersion: 1\nRequires-Dist: {requirement}\n\n".encode()
    path = tmp_path / ("example-1" + suffix)
    if suffix == ".whl":
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("example-1.dist-info/METADATA", metadata)
    else:
        with tarfile.open(path, "w:gz") as archive:
            member = tarfile.TarInfo("example-1/PKG-INFO")
            member.size = len(metadata)
            archive.addfile(member, io.BytesIO(metadata))
    if direct_url:
        with pytest.raises(ValueError, match="PyPI does not accept direct URL"):
            check_distribution(path)
    else:
        check_distribution(path)
