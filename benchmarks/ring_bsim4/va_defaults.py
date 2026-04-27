"""Pull parameter type + default values out of Verilog-A source text.

OpenVAF's MIR dump doesn't cleanly expose the ``.va``-declared defaults
(they survive as one branch of a parameter-clamping phi in the setup
function), and the MIR alone can't tell us that a parameter is ``integer``
vs ``real`` — the type only matters at the Python signature level (so JAX
doesn't try to differentiate through a non-float). So we parse the ``.va``
source directly for ``parameter real|integer|string NAME = default``
declarations.

For source files whose parameter defaults are given via preprocessor
macros (BSIM4's ``parameter integer verbose = `INT_NOT_GIVEN`` or
PSP103's macro expansions like ``MPIcc(NAME, 103, ...)``), we optionally
shell out to ``openvaf-r --print-expansion`` first and parse the
fully-expanded source — all macros resolved, defaults turned into
literals.

The results feed into the emitter's function-signature defaults and
tell the emitter whether a param should become a plain ``float``
kwarg (JAX-differentiable) or an ``equinox.field(static=True)``
kwarg (treated as constant at trace time).
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParamSpec:
    """One ``.va`` parameter: its Python-world type and its default literal.

    ``type_`` is ``"float"``, ``"int"``, or ``"str"`` — the Python type
    annotation to emit. ``default`` is a Python-literal string that can
    drop straight into a function signature (``"300.0"``, ``"103"``,
    ``'"nmos"'`` — note the inner quotes for strings).
    """

    type_: str
    default: str


_PARAM_RE = re.compile(
    r"parameter\s+(real|integer|string)\s+(\w+)\s*=\s*([^;]+?)\s*(?:from\s|exclude\s|;)",
    re.MULTILINE,
)


def parse_va_defaults(va_text: str) -> dict[str, ParamSpec]:
    """Extract ``{param_name: ParamSpec}`` from Verilog-A source text.

    Parameters whose default is a numeric / string literal are captured;
    anything else (macro references without preprocessor expansion,
    constant expressions) is silently skipped. For those, the caller
    supplies a ``0.0`` fallback when planning the component surface.
    """
    out: dict[str, ParamSpec] = {}
    for match in _PARAM_RE.finditer(va_text):
        va_type, name, raw_value = match.groups()
        raw = raw_value.strip()

        if va_type == "real":
            default = _maybe_float_literal(raw)
            if default is None:
                continue
            out[name] = ParamSpec(type_="float", default=default)
        elif va_type == "integer":
            default = _maybe_int_literal(raw)
            if default is None:
                continue
            out[name] = ParamSpec(type_="int", default=default)
        else:  # "string"
            default = _maybe_string_literal(raw)
            if default is None:
                continue
            out[name] = ParamSpec(type_="str", default=default)
    return out


def parse_va_defaults_expanded(va_path: Path) -> dict[str, ParamSpec]:
    """Preprocess via ``openvaf-r --print-expansion`` before parsing.

    Falls back to the raw source when ``openvaf-r`` isn't on ``$PATH`` or
    the expansion fails — macro-free sources still parse correctly.
    """
    try:
        completed = subprocess.run(  # noqa: S603
            ["openvaf-r", "--print-expansion", str(va_path)],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
        return parse_va_defaults(completed.stdout)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return parse_va_defaults(va_path.read_text())


def _maybe_float_literal(raw: str) -> str | None:
    """Return a Python-normalised float literal, or ``None`` if unparseable.

    Accepts ``1e-12``, ``300``, ``-0.5``, etc. Integer-valued reals are
    rendered as ``N.0`` to keep the Python type annotation honest.

    BSIM4's preprocessor emits signed literals with whitespace
    (``- 12345789``); we collapse internal whitespace before parsing so
    those resolve to a clean ``-12345789.0``.
    """
    compact = re.sub(r"\s+", "", raw)
    try:
        value = float(compact)
    except ValueError:
        return None
    if "." not in compact and "e" not in compact.lower() and value == int(value):
        return f"{int(value)}.0"
    return compact


def _maybe_int_literal(raw: str) -> str | None:
    """Return a Python-style int literal, or ``None`` if unparseable.

    BSIM4's ``INT_NOT_GIVEN`` macro expands to ``- 9999999`` (note the
    space between the sign and the digits), so we strip internal
    whitespace before converting. Un-parseable defaults fall back to the
    emitter's ``0`` fallback.
    """
    compact = re.sub(r"\s+", "", raw)
    try:
        return str(int(compact))
    except ValueError:
        return None


def _maybe_string_literal(raw: str) -> str | None:
    """Return a Python-quoted string literal, or ``None`` if unparseable.

    Verilog-A strings are double-quoted in source; we keep the quotes
    so the emitter can drop the token directly into a Python signature.
    """
    if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
        # Strip + re-add quotes via ``repr`` so Python-hostile characters
        # (backslashes, newlines) are escaped correctly.
        return repr(raw[1:-1])
    return None


__all__ = ["ParamSpec", "parse_va_defaults", "parse_va_defaults_expanded"]
