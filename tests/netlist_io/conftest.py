"""Skip parser integration tests when the optional extension is absent."""

import pytest


@pytest.fixture(autouse=True)
def require_netlist_parser() -> None:
    parser = pytest.importorskip("netlist_parser")
    if not hasattr(parser, "parse_spectre"):
        pytest.skip("NetlistParse Python Spectre binding is required")
