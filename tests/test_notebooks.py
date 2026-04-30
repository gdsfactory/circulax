"""Test execution of notebooks in the examples directory.

These tests are marked ``@pytest.mark.long`` and are only run in the full
``pytest_all`` task (not ``pytest_run``).  They also require ``papermill``
to be installed — if it is not, the whole module is skipped with a clear
message rather than failing with an ImportError.
"""

import shutil
from collections.abc import Generator
from pathlib import Path

import pytest

papermill = pytest.importorskip(
    "papermill",
    reason="papermill not installed — skipping notebook tests (install circulax[docs])",
)

TEST_DIR = Path(__file__).resolve().parent.parent
NBS_DIR = TEST_DIR / "examples"
NBS_FAIL_DIR = TEST_DIR / "failed"

shutil.rmtree(NBS_FAIL_DIR, ignore_errors=True)
NBS_FAIL_DIR.mkdir(exist_ok=True)


def _find_notebooks(*dir_parts: str) -> Generator[Path, None, None]:
    base_dir = TEST_DIR.joinpath(*dir_parts).resolve()
    for path in base_dir.rglob("*.ipynb"):
        if "checkpoint" in path.name:
            continue
        yield path


@pytest.mark.long
@pytest.mark.parametrize("path", sorted(_find_notebooks("examples")))
def test_nbs(path: Path | str) -> None:
    fn = Path(path).name
    nb = papermill.iorw.load_notebook_node(str(path))
    nb = papermill.engines.papermill_engines.execute_notebook_with_engine(
        engine_name=None,
        nb=nb,
        kernel_name="circulax",
        input_path=str(path),
        output_path=None,
    )
    papermill.execute.raise_for_execution_errors(nb, str(NBS_FAIL_DIR / fn))
