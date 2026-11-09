# Copyright (c) 2026 Circulax contributors.
"""Physical quantity metadata and simulation tolerance policies.

Definitions are immutable Python values: custom disciplines need no registration.
Units document the model contract; Circulax does not perform unit conversion.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Nature:
    """A quantity's units, negligible magnitude, and optional residual reference.

    ``nominal`` is an explicit magnitude in the same units. A zero nominal
    makes residual checks absolute-only; relative update checks still use the
    actual solution magnitude. Equation rescaling must rescale this metadata.
    """

    name: str
    units: str
    atol: float
    nominal: float = 0.0

    def __post_init__(self) -> None:
        if not self.name or not self.units:
            msg = "Nature requires nonempty name and units"
            raise ValueError(msg)
        if not math.isfinite(self.atol) or self.atol <= 0:
            msg = "Nature.atol must be finite and positive"
            raise ValueError(msg)
        if not math.isfinite(self.nominal) or self.nominal < 0:
            msg = "Nature.nominal must be finite and nonnegative"
            raise ValueError(msg)


@dataclass(frozen=True)
class Discipline:
    """The potential unknown and conserved flow equation at a port."""

    name: str
    potential: Nature
    flow: Nature

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.potential, Nature) or not isinstance(self.flow, Nature):
            msg = "Discipline requires a name and potential/flow Nature values"
            raise ValueError(msg)


voltage = Nature("voltage", "V", 1e-6)
current = Nature("current", "A", 1e-12)
temperature = Nature("temperature", "K", 1e-4)
power = Nature("power", "W", 1e-9)
wave_amplitude = Nature("wave_amplitude", "sqrt(W)", 1e-9)
electrical = Discipline("electrical", voltage, current)
thermal = Discipline("thermal", temperature, power)


@dataclass(frozen=True, init=False)
class TolerancePolicy:
    """Opt into physical Newton checks and independently scaled local errors.

    ``atol`` overrides nature defaults by nature name. ``unknown_atol`` and
    ``residual_atol`` override exact ``instance,port_or_state`` names (not
    aliases). Unknown/equation overrides take precedence over nature overrides.
    Residuals use ``atol + rtol * nature.nominal``; unlike solution updates,
    their relative scale cannot be inferred from the net residual.

    ``local_error_factor`` multiplies unknown absolute and relative tolerances
    for adaptive local error control, independently of Newton termination.
    """

    rtol: float
    atol: tuple[tuple[str, float], ...]
    unknown_atol: tuple[tuple[str, float], ...]
    residual_atol: tuple[tuple[str, float], ...]
    local_error_factor: float

    def __init__(
        self,
        rtol: float = 1e-6,
        *,
        atol: Mapping[str, float] | None = None,
        unknown_atol: Mapping[str, float] | None = None,
        residual_atol: Mapping[str, float] | None = None,
        local_error_factor: float = 1.0,
    ) -> None:
        """Validate policy settings and freeze mappings as immutable entries."""
        if not math.isfinite(rtol) or rtol < 0:
            msg = "TolerancePolicy.rtol must be finite and nonnegative"
            raise ValueError(msg)
        if not math.isfinite(local_error_factor) or local_error_factor <= 0:
            msg = "local_error_factor must be finite and positive"
            raise ValueError(msg)
        object.__setattr__(self, "rtol", float(rtol))
        object.__setattr__(self, "local_error_factor", float(local_error_factor))
        for name, values in (("atol", atol), ("unknown_atol", unknown_atol), ("residual_atol", residual_atol)):
            entries = tuple(sorted((key, float(value)) for key, value in (values or {}).items()))
            if any(not key or not math.isfinite(value) or value <= 0 for key, value in entries):
                msg = f"{name} overrides must have nonempty names and finite positive values"
                raise ValueError(msg)
            object.__setattr__(self, name, entries)


# Pseudo-potential and flow used by the power-wave optical stamps. Their
# product is power; wave amplitudes themselves have units sqrt(W).
optical_potential = Nature("optical_potential", "sqrt(W*ohm)", 1e-9)
optical_flow = Nature("optical_flow", "sqrt(W/ohm)", 1e-9)
optical = Discipline("optical", optical_potential, optical_flow)
normalized_carrier = Nature("normalized_carrier", "1", 1e-9)
