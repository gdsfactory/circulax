"""Schematic source expressions share the netlist scalar/unit interpreter."""

import math

import pytest

from circulax.netlist_io import expressions
from circulax.netlist_io.syntax import NetlistError


def test_sine_waveform_preserves_offset_and_engineering_units() -> None:
    """SIN offset, amplitude and frequency are independent instance values.

    @tags circulax-simulation
    """
    assert expressions.parse_sine_waveform("sin(0.9 1 1k)") == {
        "offset": 0.9,
        "amplitude": 1.0,
        "freq": 1000.0,
        "delay": 0.0,
        "damping": 0.0,
        "phase": 0.0,
    }


def test_sine_waveform_optional_arguments_use_seconds_and_radians() -> None:
    """SPICE-style delayed damped SIN declarations translate phase degrees.

    @tags circulax-simulation
    """
    settings = expressions.parse_sine_waveform("SIN(1, 2, 10k, 3u, 4, 90)")
    assert settings["delay"] == pytest.approx(3e-6)
    assert settings["damping"] == 4
    assert settings["phase"] == pytest.approx(math.pi / 2)


@pytest.mark.parametrize("waveform", ["pulse(0 1 1n)", "sin(1 2)", "sin(0 1 -1)"])
def test_unsupported_or_invalid_sine_waveforms_fail(waveform: str) -> None:
    """Unsupported source syntax must never become a silent DC stimulus.

    @tags circulax-simulation
    """
    with pytest.raises(NetlistError):
        expressions.parse_sine_waveform(waveform)
