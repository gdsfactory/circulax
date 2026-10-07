"""Direct NetlistParse syntax access, preserving symbolic expressions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import netlist_parser


class NetlistError(ValueError):
    """A netlist could not be parsed or resolved without changing its meaning."""


@dataclass(frozen=True)
class Statement:
    """One parsed statement and the library that owns its relative paths."""

    kind: str
    node: Any
    path: Path


def children(node: Any, kind: str | None = None) -> list[Any]:
    """Return significant children, optionally selecting a syntax kind."""
    return [c for c in node.children if not c.is_trivia and (kind is None or c.kind == kind)]


def parameters(node: Any) -> dict[str, Any]:
    """Return parameter names and their NetlistParse expression nodes."""
    result = {}
    for p in children(node, "Parameter"):
        parts = children(p)
        name = parts[0].text
        if name in result:
            msg = f"duplicate parameter {name!r}"
            raise NetlistError(msg)
        result[name] = parts[-1]
    return result


def parse_file(path: Path, dialect: str = "ngspice") -> list[Statement]:
    """Parse original UTF-8 or legacy Latin-1 model cards and includes.

    @tags circulax-simulation
    """
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    if dialect == "ngspice":
        root = netlist_parser.parse_spice("* Circulax library\n" + text)
    elif dialect == "spectre":
        if not hasattr(netlist_parser, "parse_netlist"):
            msg = (
                "This library requires NetlistParse with the Python parse_netlist binding. "
                "Install circulax[netlists] to obtain the supported parser binding."
            )
            raise ImportError(msg)
        root = netlist_parser.parse_netlist(text, "spectre")
    else:
        msg = f"unsupported dialect {dialect!r}"
        raise NetlistError(msg)
    errors = netlist_parser.errors(root)

    def incomplete(n: Any) -> bool:
        return n.kind == "Incomplete" or any(incomplete(c) for c in n.children)

    if errors or incomplete(root):
        msg = f"{path}: NetlistParse rejected syntax; spans={errors}; source={text[:100]!r}"
        raise NetlistError(msg)
    aliases = {"Parameters": "ParamStatement", "Instance": "SubcktCall", "Include": "IncludeStatement"}
    return [Statement(aliases.get(n.kind, n.kind), n, path) for n in children(root) if n.kind not in {"Title", "Notation"}]
