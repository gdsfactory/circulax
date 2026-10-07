"""Explicit scalar unit conversion at the schematic/library boundary."""

from __future__ import annotations

import math

_PREFIXES = {"": 1.0, "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3, "k": 1e3, "M": 1e6, "G": 1e9}
_BASES = {"m": "length", "s": "time", "V": "voltage", "A": "current", "F": "capacitance", "H": "inductance", "Ohm": "resistance"}


def _unit(unit: str) -> tuple[str, int, float]:
    """Resolve a supported engineering unit without inferring parameter names.

    @tags circulax-simulation
    """
    if unit in {"", "1"}:
        return "dimensionless", 1, 1.0
    base, _, power = unit.partition("^")
    exponent = int(power) if power else 1
    if exponent < 1:
        msg = f"unsupported unit {unit!r}"
        raise ValueError(msg)
    for symbol, dimension in _BASES.items():
        if base.endswith(symbol) and base[: -len(symbol)] in _PREFIXES:
            return dimension, exponent, _PREFIXES[base[: -len(symbol)]] ** exponent
    msg = f"unsupported unit {unit!r}"
    raise ValueError(msg)


def convert_unit(value: float, source: str, target: str) -> float:
    """Convert explicitly declared units; reject incompatible dimensions.

    @tags circulax-simulation
    """
    source_dimension, source_power, source_scale = _unit(source)
    target_dimension, target_power, target_scale = _unit(target)
    if (source_dimension, source_power) != (target_dimension, target_power):
        msg = f"incompatible units {source!r} and {target!r}"
        raise ValueError(msg)
    result = float(value) * source_scale / target_scale
    if not math.isfinite(result):
        msg = "unit conversion requires finite values"
        raise ValueError(msg)
    return result
