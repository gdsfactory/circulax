"""Skip parser integration tests when the optional extension is absent."""

import pytest


@pytest.fixture(autouse=True)
def require_netlist_parser() -> None:
    pytest.importorskip("netlist_parser")
