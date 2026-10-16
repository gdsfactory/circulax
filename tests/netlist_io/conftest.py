"""Skip parser integration tests when the optional extension is absent."""

import pytest


@pytest.fixture(autouse=True)
def require_netlist_parser(request: pytest.FixtureRequest) -> None:
    if request.path.name not in {"test_sources.py", "test_source_waveforms.py", "test_expressions.py"}:
        pytest.importorskip("netlist_parser")
