"""Scalar expressions remain usable without the optional file parser."""

import pytest

from circulax.netlist_io.expressions import Scope, evaluate, evaluate_source
from circulax.netlist_io.syntax import NetlistError, children, parameters


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("width * 1u", 2e-6),
        ("WIDTH + 3", 5),
        ("{2 * (width + 3)}", 10),
        ("1 + 2 * 3", 7),
        ("2^3^2", 64),
        ("-2**2", 4),
        ("2**-2", 0.25),
        ("max(1m, min(2m, 3m))", 2e-3),
        ("sqrt(4) + abs(-2) + pow(2, 3)", 12),
        ("1e-3 + .5m + 2mil", 0.0015508),
        ("!0 && 2 > 1", 1),
        ("0 && missing", 0),
        ("1 || missing", 1),
        ("width == 2 ? 3 : missing", 3),
        ("0 ? missing : 1 ? 4 : 5", 4),
        ("5 % 2", 1),
    ],
)
def test_scalar_expression(text: str, expected: float) -> None:
    assert evaluate_source(text, {"width": 2}) == pytest.approx(expected)


@pytest.mark.parametrize("dialect", ["spice", "ngspice", "spectre"])
def test_suffix_dialect(dialect: str) -> None:
    assert evaluate_source("5M", {}, dialect) == (5e6 if dialect == "spectre" else 5e-3)
    assert evaluate_source("5meg", {}, dialect) == 5e6


@pytest.mark.parametrize(
    "text",
    [
        "",
        "1 2",
        "1kHz",
        "1e",
        "1 +",
        "(1",
        "1)",
        "{1)",
        "max(1,)",
        "missing",
        "unknown(1)",
        "1\nparameters other=2",
        "x.y",
        "x[0]",
        "__import__('os')",
        "[x for x in y]",
        "1; 2",
    ],
)
def test_invalid_expressions(text: str) -> None:
    with pytest.raises(NetlistError):
        evaluate_source(text, {})


@pytest.mark.parametrize(
    "text",
    [
        "width * 1u",
        "1 + 2 * 3",
        "2^3^2",
        "-2**2",
        "2**-2",
        "max(1m, min(2m, 3m))",
        "1 ? 5 : missing",
        "5 % 2",
        "5M",
        "1 == 2 || 3 > 2",
        "2 < 1 && missing",
    ],
)
def test_matches_optional_file_parser(text: str) -> None:
    parser = pytest.importorskip("netlist_parser")
    root = parser.parse_netlist("parameters value=" + text + "\n", "spectre")
    assert not parser.errors(root)
    scope = Scope()
    scope.bindings["width"] = 2
    expected = evaluate(parameters(children(root, "Parameters")[0])["value"], scope)
    assert evaluate_source(text, {"width": 2}) == expected
