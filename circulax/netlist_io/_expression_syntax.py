"""Small scalar grammar producing nodes for the shared safe CST interpreter.

Full SPICE/Spectre documents still use NetlistParse. Schematic settings and
waveform arguments need only numbers, names, arithmetic and scalar functions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from circulax.netlist_io.syntax import NetlistError

_TOKEN = re.compile(
    r"\s*(?:(\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?[A-Za-z]*|"
    r"[A-Za-z_]\w*|\*\*|&&|\|\||>=|<=|==|!=|[+*/^%<>!?:(),{}-])"
)
_PRECEDENCE = {
    "||": 1,
    "&&": 2,
    "==": 3,
    "!=": 3,
    "<": 4,
    "<=": 4,
    ">": 4,
    ">=": 4,
    "+": 5,
    "-": 5,
    "*": 6,
    "/": 6,
    "%": 6,
    "**": 8,
    "^": 8,
}


@dataclass
class _Node:
    """The significant-node interface consumed by expressions.evaluate."""

    kind: str
    text: str
    children: tuple[_Node, ...] = ()
    is_trivia: bool = False


class _Parser:
    """Precedence parser; evaluation remains in the shared CST interpreter."""

    def __init__(self, text: str) -> None:
        self.tokens: list[str] = []
        position = 0
        text = text.strip()
        while position < len(text):
            match = _TOKEN.match(text, position)
            if match is None:
                msg = f"invalid property expression {text!r}"
                raise NetlistError(msg)
            self.tokens.append(match[0].strip())
            position = match.end()
        self.tokens.append("")
        self.position = 0

    def peek(self) -> str:
        return self.tokens[self.position]

    def take(self) -> str:
        token = self.peek()
        if not token:
            msg = "unexpected end of property expression"
            raise NetlistError(msg)
        self.position += 1
        return token

    def expect(self, token: str) -> None:
        if self.take() != token:
            msg = f"expected {token!r} in property expression"
            raise NetlistError(msg)

    def atom(self) -> _Node:
        token = self.take()
        if token in {"+", "-", "!"}:
            # Match NetlistParse: unary operators bind tighter than power.
            return _Node("UnaryExpression", token, (_Node("Notation", token), self.expression(9)))
        if token in {"(", "{"}:
            node = self.expression()
            self.expect({"(": ")", "{": "}"}[token])
            return node
        if token[0].isdigit() or token[0] == ".":
            return _Node("NumberLiteral", token)
        if token[0].isalpha() or token[0] == "_":
            if self.peek() != "(":
                return _Node("NameRef", token)
            self.take()
            args = [_Node("Identifier", token)]
            if self.peek() != ")":
                while True:
                    args.append(_Node("FunctionArgs", "", (self.expression(),)))
                    if self.peek() != ",":
                        break
                    self.take()
            self.expect(")")
            return _Node("FunctionCall", token, tuple(args))
        msg = f"unexpected token {token!r} in property expression"
        raise NetlistError(msg)

    def expression(self, minimum: int = 0) -> _Node:
        left = self.atom()
        while (precedence := _PRECEDENCE.get(self.peek(), -1)) >= minimum:
            op = self.take()
            # NetlistParse treats every binary operator, including power,
            # as left-associative. Keep schematic and model-card values equal.
            right = self.expression(precedence + 1)
            left = _Node("BinaryExpression", op, (left, _Node("Notation", op), right))
        if minimum == 0 and self.peek() == "?":
            self.take()
            yes = self.expression()
            self.expect(":")
            no = self.expression()
            left = _Node("TernaryExpr", "", (left, _Node("Notation", "?"), yes, _Node("Notation", ":"), no))
        return left


def parse_expression(text: str) -> _Node:
    """Parse exactly one scalar expression, rejecting all trailing input."""
    parser = _Parser(text)
    node = parser.expression()
    if parser.peek():
        msg = f"invalid property expression {text!r}"
        raise NetlistError(msg)
    return node
