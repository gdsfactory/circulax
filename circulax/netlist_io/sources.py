"""Parse SPICE independent-source specifications into waveform source settings.

The returned dictionaries use the keyword names of
:class:`circulax.components.electronic.WaveformVoltageSource` and
:class:`~circulax.components.electronic.WaveformCurrentSource`, so an SDK client
binds parsed text with ``WaveformVoltageSource(**parse_source("DC 0.5 SIN(...)"))``.
Numeric arguments use the shared safe expression evaluator with SPICE suffix
semantics (``M`` is milli, ``meg`` is mega). Unit letters after a suffix, AC
magnitudes and unsupported waveform kinds raise :class:`NetlistError` instead of
silently degrading to a DC source.
"""

from __future__ import annotations

import itertools
import math
import re

from circulax.components.electronic import WAVE_DC, WAVE_PULSE, WAVE_PWL, WAVE_SIN
from circulax.netlist_io.expressions import evaluate_source, parse_sine_waveform
from circulax.netlist_io.syntax import NetlistError

_CALL = re.compile(r"(?P<name>[A-Za-z_]\w*)\s*\((?P<args>[^()]*)\)")
_PULSE_NAMES = ("v1", "v2", "delay", "tr", "tf", "pw", "per")


def _number(token: str, dialect: str, what: str) -> float:
    try:
        value = float(evaluate_source(token, {}, dialect))
    except (NetlistError, ArithmeticError, ValueError) as exc:
        msg = f"invalid {what} value {token!r}: {exc}"
        raise NetlistError(msg) from exc
    if not math.isfinite(value):
        msg = f"{what} must be finite; got {token!r}"
        raise NetlistError(msg)
    return value


def _tokens(arguments: str) -> list[str]:
    return [part for part in re.split(r"[\s,]+", arguments.strip()) if part]


def _pulse(arguments: str, dialect: str, tstep: float | None, tstop: float | None) -> dict[str, float]:
    tokens = _tokens(arguments)
    if not 2 <= len(tokens) <= len(_PULSE_NAMES):
        msg = "PULSE requires v1 and v2 and at most five optional arguments (td tr tf pw per)"
        raise NetlistError(msg)
    # SPICE defaults tr/tf to TSTEP and pw to TSTOP; a component cannot know them,
    # so they come from the caller and otherwise mean ideal edges and a held pulse.
    result = {"v1": 0.0, "v2": 0.0, "delay": 0.0, "tr": tstep or 0.0, "tf": tstep or 0.0, "pw": tstop or math.inf, "per": 0.0}
    result.update({name: _number(token, dialect, f"PULSE {name}") for name, token in zip(_PULSE_NAMES, tokens, strict=False)})
    if any(result[name] < 0 for name in ("delay", "tr", "tf", "pw", "per")):
        msg = "PULSE requires nonnegative delay, rise, fall, width and period"
        raise NetlistError(msg)
    if 0 < result["per"] < result["tr"] + result["pw"] + result["tf"]:
        msg = "PULSE period must be at least rise + width + fall"
        raise NetlistError(msg)
    return result


def _pwl(arguments: str, dialect: str) -> dict[str, float | tuple[float, ...]]:
    options = {"r": -1.0, "td": 0.0}
    values: list[float] = []
    for token in _tokens(arguments):
        if "=" in token:
            key, _, text = token.partition("=")
            option = key.lower()
            if option not in options:
                msg = f"unsupported PWL option {key!r}"
                raise NetlistError(msg)
            options[option] = _number(text, dialect, f"PWL {key}")
        else:
            values.append(_number(token, dialect, "PWL point"))
    if not values or len(values) % 2:
        msg = "PWL requires one or more (time, value) pairs"
        raise NetlistError(msg)
    times, levels = values[0::2], values[1::2]
    if times[0] < 0 or any(b <= a for a, b in itertools.pairwise(times)):
        msg = "PWL times must be nonnegative and strictly increasing"
        raise NetlistError(msg)
    if options["td"] < 0:
        msg = "PWL td must be nonnegative"
        raise NetlistError(msg)
    if options["r"] >= 0 and options["r"] >= times[-1]:
        msg = "PWL repeat time r must be nonnegative and earlier than the last point"
        raise NetlistError(msg)
    return {"pwl_t": tuple(times), "pwl_v": tuple(levels), "delay": options["td"], "repeat": options["r"]}


def _pad(values: tuple[float, ...], length: int) -> tuple[float, ...]:
    """Pad to ``length`` by repeating the last value; an empty input becomes zeros."""
    if length < 2:  # interpolation needs two samples
        msg = "pwl_points must be at least 2"
        raise NetlistError(msg)
    if not values:
        return (0.0,) * length
    return (*values, *([values[-1]] * (length - len(values))))


def parse_waveform(
    waveform: str,
    *,
    dialect: str = "spice",
    tstep: float | None = None,
    tstop: float | None = None,
    pwl_points: int | None = None,
) -> dict[str, float | tuple[float, ...]]:
    """Parse ``SIN(...)``, ``PULSE(...)`` or ``PWL(...)`` into waveform source settings.

    Args:
        waveform: SPICE waveform text, e.g. ``"PULSE(0 1 1n 1n 1n 10n 20n)"``.
        dialect: Number-suffix dialect; ``"spice"`` reads ``M`` as milli.
        tstep: Default PULSE rise/fall time when omitted (SPICE ``TSTEP``);
            otherwise omitted edges are ideal.
        tstop: Default PULSE width when omitted (SPICE ``TSTOP``), otherwise the
            pulse stays high; also the default SIN frequency ``1 / tstop``, without
            which an omitted SIN frequency raises.
        pwl_points: When set, emit ``pwl_t``/``pwl_v`` of exactly this length for
            *every* kind (PWL padded by repeating its last point, others zeros) so
            instances of different kinds stack into one batched group. PWL
            instances with more points raise.

    Returns:
        Keyword settings for the waveform source components, including ``kind``.
        Fields not needed by the waveform keep their component defaults.

    @tags circulax-simulation

    """
    match = _CALL.fullmatch(waveform.strip())
    if match is None:
        msg = f"unsupported transient source waveform {waveform!r}"
        raise NetlistError(msg)
    name = match["name"].lower()
    result: dict[str, float | tuple[float, ...]]
    if name == "sin":
        arguments = _tokens(match["args"])
        if len(arguments) == 2 and tstop:  # SPICE defaults an omitted FREQ to 1/TSTOP
            waveform = f"SIN({' '.join(arguments)} {1.0 / tstop!r})"
        result = {"kind": WAVE_SIN, **parse_sine_waveform(waveform, dialect)}
    elif name == "pulse":
        result = {"kind": WAVE_PULSE, **_pulse(match["args"], dialect, tstep, tstop)}
    elif name == "pwl":
        result = {"kind": WAVE_PWL, **_pwl(match["args"], dialect)}
    else:
        msg = f"unsupported transient source waveform {match['name']!r}; expected SIN, PULSE or PWL"
        raise NetlistError(msg)
    if pwl_points is not None:
        times = result.get("pwl_t", ())
        if len(times) > pwl_points:
            msg = f"PWL has {len(times)} points but pwl_points={pwl_points}"
            raise NetlistError(msg)
        result["pwl_t"] = _pad(times, pwl_points)
        result["pwl_v"] = _pad(result.get("pwl_v", ()), pwl_points)
    return result


def parse_source(
    spec: str,
    *,
    dialect: str = "spice",
    tstep: float | None = None,
    tstop: float | None = None,
    pwl_points: int | None = None,
) -> dict[str, float | tuple[float, ...]]:
    """Parse the value part of a SPICE V/I source card: ``[DC] value`` and/or a waveform.

    Accepts ``"1.2"``, ``"DC 1.2"``, ``"SIN(...)"`` and ``"DC 0.5 SIN(...)"`` (in
    either order). The DC bias is independent of the waveform: with no ``DC``
    clause ``dc`` is ``0.0`` (the SPICE default), it is never inferred from the
    waveform. ``AC`` clauses are rejected (source-driven AC is out of scope).

    Returns:
        Keyword settings with ``kind`` and ``dc`` plus the waveform fields, ready
        for ``WaveformVoltageSource(**settings)`` / ``WaveformCurrentSource(**settings)``.

    @tags circulax-simulation

    """
    calls = list(_CALL.finditer(spec))
    if len(calls) > 1:
        msg = f"only one transient waveform is allowed per source; got {spec!r}"
        raise NetlistError(msg)
    call = calls[0] if calls else None
    rest = spec if call is None else f"{spec[: call.start()]} {spec[call.end() :]}"
    tokens = _tokens(rest)
    if tokens and tokens[0].lower() == "dc":
        tokens = tokens[1:]
    if tokens and tokens[0].lower() == "ac":
        msg = "AC source magnitudes are not supported"
        raise NetlistError(msg)
    if len(tokens) > 1:
        msg = f"unsupported source clauses {' '.join(tokens[1:])!r}"
        raise NetlistError(msg)
    if not tokens and call is None:
        msg = f"source specification {spec!r} has neither a DC value nor a waveform"
        raise NetlistError(msg)
    dc = _number(tokens[0], dialect, "DC") if tokens else 0.0
    if call is not None:
        settings = parse_waveform(call.group(), dialect=dialect, tstep=tstep, tstop=tstop, pwl_points=pwl_points)
    else:
        settings = {"kind": WAVE_DC}
        if pwl_points is not None:
            settings.update(pwl_t=_pad((), pwl_points), pwl_v=_pad((), pwl_points))
    return {**settings, "dc": dc}
