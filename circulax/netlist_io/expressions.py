"""Evaluate NetlistParse expression nodes with explicit, scoped name lookup."""

from __future__ import annotations

import math
import operator
import re
from typing import Any

import netlist_parser

from circulax.netlist_io.syntax import NetlistError, children, parameters

_SCALE = {"t": 1e12, "g": 1e9, "meg": 1e6, "k": 1e3, "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15, "mil": 25.4e-6}
_BINARY = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "**": operator.pow,
    "^": operator.pow,
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    "%": operator.mod,
}
_FUNCTIONS = {
    "max": max,
    "min": min,
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "ln": math.log,
    "log": math.log,
    "log10": math.log10,
    "pow": pow,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}


class Scope:
    """Lazy lexical parameter environment, with overrides and cycle detection."""

    def __init__(self, parent: Scope | None = None) -> None:
        """Create a lazy parameter scope with an optional lexical parent."""
        self.parent = parent
        self.dialect = parent.dialect if parent else "ngspice"
        self.statistical_mode = parent.statistical_mode if parent else "reject"
        self.bindings: dict[str, Any] = {}
        self._active: set[str] = set()

    def lookup(self, name: str) -> float | str:
        """Resolve a parameter, evaluating its definition in the declaring scope.

        SPICE identifiers are case-insensitive, so a name is matched exactly
        first and falls back to a case-insensitive scan of this scope only.
        """
        if name not in self.bindings:
            folded = next((k for k in self.bindings if k.lower() == name.lower()), None)
            if folded is not None:
                name = folded
            elif self.parent is not None:
                return self.parent.lookup(name)
            else:
                msg = f"unresolved parameter {name!r}"
                raise NetlistError(msg)
        value = self.bindings[name]
        if isinstance(value, (float, int)):
            return value
        if name in self._active:
            msg = f"parameter cycle involving {name!r}"
            raise NetlistError(msg)
        self._active.add(name)
        try:
            return evaluate(value, self)
        finally:
            self._active.remove(name)


def evaluate(node: Any, scope: Scope) -> float | str:  # noqa: C901, PLR0911, PLR0912 -- CST operation dispatch
    """Interpret only supported CST operations; never execute source as Python."""
    c = children(node)
    kind = node.kind
    if kind in {"Brace", "Parens", "ParenthesizedExpression", "LiteralExpr", "FunctionArgs", "Condition", "Quote", "Prime"}:
        expressions = [p for p in c if p.kind != "Notation"]
        if len(expressions) != 1:
            msg = f"invalid expression {node.text!r}"
            raise NetlistError(msg)
        return evaluate(expressions[0], scope)
    if kind in {"Literal", "StringLiteral"} and node.text.startswith('"'):
        return node.text[1:-1]
    if kind == "NumberLiteral":
        match = re.fullmatch(r"([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)([A-Za-z]*)", node.text)
        if not match:
            msg = f"invalid number {node.text!r}"
            raise NetlistError(msg)
        suffix = match[2].lower()
        if match[2] == "M" and scope.dialect not in {"spice", "ngspice"}:
            return float(match[1]) * 1e6
        if suffix and suffix not in _SCALE:
            msg = f"unsupported unit suffix {suffix!r}"
            raise NetlistError(msg)
        return float(match[1]) * _SCALE.get(suffix, 1)
    if kind in {"NameRef", "Identifier"}:
        return scope.lookup(node.text)
    if kind == "BinaryExpression":
        left, op, right = c
        a = evaluate(left, scope)
        if op.text == "&&":
            return float(bool(a) and bool(evaluate(right, scope)))
        if op.text == "||":
            return float(bool(a) or bool(evaluate(right, scope)))
        if op.text not in _BINARY:
            msg = f"unsupported operator {op.text!r}"
            raise NetlistError(msg)
        return float(_BINARY[op.text](a, evaluate(right, scope)))
    if kind in {"UnaryExpression", "UnaryOp"}:
        op, expression = c
        value = evaluate(expression, scope)
        if op.text == "-":
            return -value
        if op.text == "+":
            return value
        if op.text == "!":
            return float(not value)
        msg = f"unsupported unary operator {op.text!r}"
        raise NetlistError(msg)
    if kind == "TernaryExpr":
        return evaluate(c[2] if evaluate(c[0], scope) else c[4], scope)
    if kind == "FunctionCall":
        function = c[0].text
        if function.lower() == "agauss" and scope.statistical_mode == "nominal":
            arguments = children(node, "FunctionArgs")
            if len(arguments) != 3:
                msg = "agauss requires nominal, variation and sigma arguments"
                raise NetlistError(msg)
            values = [evaluate(argument, scope) for argument in arguments]
            return float(values[0])
        if function not in _FUNCTIONS:
            msg = f"unsupported function {function!r}"
            raise NetlistError(msg)
        return float(_FUNCTIONS[function](*(evaluate(a, scope) for a in children(node, "FunctionArgs"))))
    msg = f"unsupported expression {kind}: {node.text!r}"
    raise NetlistError(msg)


def evaluate_source(expression: str, settings: dict[str, float]) -> float:
    """Parse a parameter expression using the same safe interpreter.

    @tags circulax-simulation
    """
    root = netlist_parser.parse_netlist("parameters value=" + expression + "\n", "spectre")
    if netlist_parser.errors(root) or not children(root, "Parameters"):
        msg = f"invalid property expression {expression!r}"
        raise NetlistError(msg)
    scope = Scope()
    scope.bindings.update(settings)
    return evaluate(parameters(children(root, "Parameters")[0])["value"], scope)


def parse_sine_waveform(waveform: str) -> dict[str, float]:
    """Resolve SPICE-style SIN(offset amplitude frequency [delay damping phase]).

    Phase is supplied in degrees and returned in radians. Numeric arguments use
    the same safe expression/unit interpreter as model cards. Other waveform
    kinds are rejected so callers cannot silently substitute a DC source.

    @tags circulax-simulation
    """
    match = re.fullmatch(r"(?i:sin)\s*\(([^()]*)\)", waveform.strip())
    if match is None:
        msg = f"unsupported transient source waveform {waveform!r}"
        raise NetlistError(msg)
    arguments = [part for part in re.split(r"[\s,]+", match[1].strip()) if part]
    if not 3 <= len(arguments) <= 6:
        msg = "SIN requires offset, amplitude, frequency and at most three optional arguments"
        raise NetlistError(msg)
    names = ("offset", "amplitude", "freq", "delay", "damping", "phase")
    result = dict.fromkeys(names, 0.0)
    result.update({name: float(evaluate_source(value, {})) for name, value in zip(names, arguments, strict=False)})
    if any(not math.isfinite(value) for value in result.values()) or any(result[name] < 0 for name in ("freq", "delay", "damping")):
        msg = "SIN requires finite values and nonnegative frequency, delay and damping"
        raise NetlistError(msg)
    result["phase"] = math.radians(result["phase"])
    return result
