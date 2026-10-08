"""SPICE source specifications parse to waveform-source settings or fail explicitly."""

import math

import pytest

from circulax.components.electronic import WAVE_DC, WAVE_PULSE, WAVE_PWL, WAVE_SIN
from circulax.netlist_io import parse_source, parse_waveform
from circulax.netlist_io.syntax import NetlistError


def test_sin_settings_carry_kind_and_engineering_units() -> None:
    """@tags circulax-simulation"""
    s = parse_waveform("SIN(0.9 1m 10k 2u 3 90)")
    assert s["kind"] == WAVE_SIN
    assert s["amplitude"] == pytest.approx(1e-3)
    assert s["freq"] == 1e4
    assert s["delay"] == pytest.approx(2e-6)
    assert s["damping"] == 3
    assert s["phase"] == pytest.approx(math.pi / 2)


def test_uppercase_m_is_milli_in_source_cards() -> None:
    """SPICE sources read ``M`` as milli; ``meg`` is mega.

    @tags circulax-simulation
    """
    assert parse_waveform("PULSE(0 1 5M)")["delay"] == pytest.approx(5e-3)
    assert parse_waveform("PULSE(0 1 5meg)")["delay"] == 5e6


def test_pulse_all_arguments_and_defaults() -> None:
    """@tags circulax-simulation"""
    s = parse_waveform("pulse(0, 3.3, 1n, 2n, 3n, 10n, 40n)")
    assert (s["kind"], s["v1"], s["v2"]) == (WAVE_PULSE, 0.0, 3.3)
    assert (s["delay"], s["tr"], s["tf"], s["pw"], s["per"]) == pytest.approx((1e-9, 2e-9, 3e-9, 10e-9, 40e-9))
    minimal = parse_waveform("PULSE(0 1)")
    assert (minimal["tr"], minimal["tf"], minimal["per"], minimal["delay"]) == (0.0, 0.0, 0.0, 0.0)
    assert math.isinf(minimal["pw"])


def test_pulse_spice_defaults_come_from_caller() -> None:
    """@tags circulax-simulation"""
    s = parse_waveform("PULSE(0 1)", tstep=1e-9, tstop=1e-6)
    assert (s["tr"], s["tf"], s["pw"]) == (1e-9, 1e-9, 1e-6)


def test_pwl_points_and_options() -> None:
    """@tags circulax-simulation"""
    s = parse_waveform("PWL(0 0, 1m 2, 3m -1 r=1m td=10u)")
    assert s["kind"] == WAVE_PWL
    assert s["pwl_t"] == (0.0, pytest.approx(1e-3), pytest.approx(3e-3))
    assert s["pwl_v"] == (0.0, 2.0, -1.0)
    assert s["repeat"] == pytest.approx(1e-3)
    assert s["delay"] == pytest.approx(1e-5)


def test_pwl_points_pad_every_kind_for_batching() -> None:
    """@tags circulax-simulation"""
    pwl = parse_waveform("PWL(0 0 1 1)", pwl_points=4)
    assert pwl["pwl_t"] == (0.0, 1.0, 1.0, 1.0)
    assert pwl["pwl_v"] == (0.0, 1.0, 1.0, 1.0)
    assert parse_waveform("SIN(0 1 1k)", pwl_points=4)["pwl_t"] == (0.0,) * 4
    with pytest.raises(NetlistError, match="pwl_points"):
        parse_waveform("PWL(0 0 1 1 2 2)", pwl_points=2)


def test_source_dc_bias_is_independent_of_waveform() -> None:
    """@tags circulax-simulation"""
    both = parse_source("DC 0.5 SIN(0.9 1 1k)")
    assert both["dc"] == 0.5
    assert both["offset"] == 0.9
    assert parse_source("SIN(0.9 1 1k) DC 0.5")["dc"] == 0.5
    assert parse_source("SIN(0.9 1 1k)")["dc"] == 0.0
    assert parse_source("2.5") == {"kind": WAVE_DC, "dc": 2.5}
    assert parse_source("dc 1m") == {"kind": WAVE_DC, "dc": pytest.approx(1e-3)}


@pytest.mark.parametrize(
    "text",
    [
        "",
        "SIN(0 1)",
        "SIN(0 1 -1k)",
        "PULSE(0)",
        "PULSE(0 1 0 1 1 1 1 1)",
        "PULSE(0 1 0 -1n)",
        "PULSE(0 1 0 1n 1n 10n 5n)",
        "PWL()",
        "PWL(0 0 1)",
        "PWL(1 0 0 1)",
        "PWL(0 0 1 1 1 2)",
        "PWL(0 0 1 1 r=2)",
        "PWL(0 0 1 1 q=1)",
        "PWL(-1 0 1 1)",
        "EXP(0 1 1n 1n 2n 1n)",
        "SIN(0 1 1k) PULSE(0 1)",
        "AC 1",
        "DC 1 AC 1",
        "SIN(0 1 1k",
        "SIN(0 1 1kHz)",
        "SIN(0 1 {1/0})",
        "1 2",
    ],
)
def test_invalid_source_specs_fail(text: str) -> None:
    """Malformed or unsupported specs never become a silent DC source.

    @tags circulax-simulation
    """
    with pytest.raises(NetlistError):
        parse_source(text)


@pytest.mark.parametrize("text", ["1", "SIN(0 1 1k)", "PWL(0 0 1 1)"])
def test_pwl_points_below_two_fails_for_every_kind(text: str) -> None:
    """@tags circulax-simulation"""
    with pytest.raises(NetlistError, match="pwl_points"):
        parse_source(text, pwl_points=1)


def test_sin_frequency_defaults_to_inverse_tstop_like_spice() -> None:
    """@tags circulax-simulation"""
    assert parse_waveform("SIN(0 1)", tstop=2e-3)["freq"] == pytest.approx(500.0)
    with pytest.raises(NetlistError):
        parse_waveform("SIN(0 1)")
