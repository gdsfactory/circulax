# Copyright (c) 2026 GDSFactory
"""Netlist assemblies for reconfigurable gain and passive-feedback laser cavities.

Assemblies use only public components and compile_circuit. No separate time
solver is used. Threshold helpers describe the matched-interface SG assembly;
they do not apply unchanged after replacing a reflector or adding reflections.
"""

from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from itertools import pairwise
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

from circulax.components.laser_sections import (
    BidirectionalWaveguide,
    MatchedOpticalLoad,
    OpticalDirectionalCoupler,
    OpticalInterface,
    ThermoOpticPhaseShifter,
    TravelingWaveGainCell,
    sampled_bragg_grating,
    sampled_grating_response,
)
from circulax.components.soa import ELECTRON_CHARGE, LIGHT_SPEED
from circulax.components.traveling_wave_laser import LaserParameters

if TYPE_CHECKING:
    from circulax import Circuit


class HybridSettings(NamedTuple):
    """Passive-feedback settings: comb spacings Hz, lengths m, phase rad/K, loss dB.

    samples is a physical number of identical thin grating patches. Each
    grating's length is (samples-1)*c/(2*n_g*FSR). Loss is per optical interface.
    """

    left_fsr_hz: float = 200e9
    right_fsr_hz: float = 220e9
    samples: int = 5
    grating_strength: float = 0.9
    passive_group_index: float = 4.0
    phase_length_m: float = 100e-6
    phase_per_k: float = 0.15
    interface_loss_db: float = 0.5
    shift_m_per_k: float = 0.08e-9


DEFAULT_HYBRID_SETTINGS = HybridSettings()


def grating_settings(p: LaserParameters, s: HybridSettings, side: str) -> dict:
    """Settings shared by the native grating and its exact scattering helper."""
    fsr = s.left_fsr_hz if side == "left" else s.right_fsr_hz
    return {
        "spacing_m": LIGHT_SPEED / (2 * s.passive_group_index * fsr),
        "group_index": s.passive_group_index,
        "strength": s.grating_strength,
        "center_m": p.wavelength_m,
        "reference_m": p.wavelength_m,
        "shift_m_per_k": s.shift_m_per_k,
    }


def hybrid_laser_netlist(
    p: LaserParameters,
    s: HybridSettings = DEFAULT_HYBRID_SETTINGS,
    *,
    segments: int = 4,
    left_reflector: object = None,
    right_reflector: object = None,
) -> tuple[dict, dict]:
    """Editable cavity netlist and model map, without drives or terminations.

    Left output--SG_L--interface--gain cells--interface--phase--SG_R--right output.
    All gain diodes share anode/cathode. Reflector and phase heaters are separate
    electrical ports. Either reflector can be a compatible component or compiled
    Circuit with ports p1,p2,heater,heater_return, e.g. ring_reflector(). Its
    settings are then caller-owned; SG threshold helpers no longer apply.
    A replacement uses p1 facing the cavity and p2 facing the exterior.
    p.r_left/r_right belong to the older integrated model and are ignored;
    the separate reflector components own all reflection/transmission physics.
    """
    if not isinstance(segments, int) or segments < 1:
        msg = "segments must be a positive integer"
        raise ValueError(msg)
    gain_settings = {key: value for key, value in p._asdict().items() if key not in {"r_left", "r_right"}}
    gain_settings["length_m"] = p.length_m / segments
    instances = {
        "SG_L": {"component": "grating", "settings": grating_settings(p, s, "left")},
        "SG_R": {"component": "grating", "settings": grating_settings(p, s, "right")},
        "CHIP_L": {"component": "interface", "settings": {"power_loss_db": s.interface_loss_db}},
        "CHIP_R": {"component": "interface", "settings": {"power_loss_db": s.interface_loss_db}},
        "PHASE": {
            "component": "phase",
            "settings": {"length_m": s.phase_length_m, "group_index": s.passive_group_index, "phase_per_k": s.phase_per_k},
        },
        **{f"GAIN{j}": {"component": "gain", "settings": gain_settings.copy()} for j in range(segments)},
    }
    chain = ["SG_L", "CHIP_L", *(f"GAIN{j}" for j in range(segments)), "CHIP_R", "PHASE", "SG_R"]
    connections = {f"{a},p2": f"{b},p1" for a, b in pairwise(chain)}
    if segments > 1:
        for terminal in ("anode", "cathode"):
            connections[f"GAIN0,{terminal}"] = tuple(f"GAIN{j},{terminal}" for j in range(1, segments))
    connections["PHASE,heater_return"] = ("SG_L,heater_return", "SG_R,heater_return")
    ports = {
        "p1": "SG_L,p1",
        "p2": "SG_R,p2",
        "anode": "GAIN0,anode",
        "cathode": "GAIN0,cathode",
        "heater_left": "SG_L,heater",
        "heater_right": "SG_R,heater",
        "heater_phase": "PHASE,heater",
        "heater_return": "PHASE,heater_return",
    }
    models = {
        "grating": sampled_bragg_grating(s.samples),
        "interface": OpticalInterface,
        "phase": ThermoOpticPhaseShifter,
        "gain": TravelingWaveGainCell,
    }
    if left_reflector is not None:
        models["left_feedback"] = left_reflector
        instances["SG_L"] = {"component": "left_feedback"}
        del connections["SG_L,p2"]
        connections["SG_L,p1"] = "CHIP_L,p1"
        ports["p1"] = "SG_L,p2"
    if right_reflector is not None:
        models["right_feedback"] = right_reflector
        instances["SG_R"] = {"component": "right_feedback"}
    return {"instances": instances, "connections": connections, "ports": ports}, models


def hybrid_laser(
    p: LaserParameters,
    s: HybridSettings = DEFAULT_HYBRID_SETTINGS,
    *,
    segments: int = 4,
    left_reflector: object = None,
    right_reflector: object = None,
) -> Circuit:
    """Compile a reusable laser subcircuit; parent supplies pump/heaters/loads."""
    from circulax import compile_circuit

    net, models = hybrid_laser_netlist(p, s, segments=segments, left_reflector=left_reflector, right_reflector=right_reflector)
    return compile_circuit(net, models, is_complex=True, backend="klu", g_leak=0)


@lru_cache(maxsize=16)
def add_drop_ring(
    *, length_m: float = 400e-6, group_index: float = 4.0, power_coupling: float = 0.2, loss_m1: float = 100.0
) -> Circuit:
    """Bidirectional add/drop ring assembled from two couplers and two arms.

    Ports input/through/add/drop plus heater/heater_return. One arm supplies
    thermal phase tuning. All unexcited optical ports must be terminated.
    """
    from circulax import compile_circuit

    arm = {"length_m": length_m / 2, "group_index": group_index, "loss_m1": loss_m1}
    net = {
        "instances": {
            "C1": {"component": "coupler", "settings": {"power_coupling": power_coupling}},
            "C2": {"component": "coupler", "settings": {"power_coupling": power_coupling}},
            "A": {"component": "waveguide", "settings": arm.copy()},
            "B": {"component": "phase", "settings": arm.copy()},
        },
        "connections": {"C1,p4": "A,p1", "A,p2": "C2,p2", "C2,p4": "B,p1", "B,p2": "C1,p2"},
        "ports": {
            "input": "C1,p1",
            "through": "C1,p3",
            "add": "C2,p1",
            "drop": "C2,p3",
            "heater": "B,heater",
            "heater_return": "B,heater_return",
        },
    }
    return compile_circuit(
        net,
        {"coupler": OpticalDirectionalCoupler, "waveguide": BidirectionalWaveguide, "phase": ThermoOpticPhaseShifter},
        is_complex=True,
        backend="klu",
        g_leak=0,
    )


@lru_cache(maxsize=16)
def ring_reflector(*, length_m: float = 400e-6, power_coupling: float = 0.2) -> Circuit:
    """Ring drop port followed by a mirror: a replaceable spectral reflector.

    Through/add ports are matched; drop sees a frequency-flat lossless mirror.
    p1 faces the cavity, p2 is the transmitted mirror output. This deliberately
    has a different spectrum and loss budget from a sampled Bragg reflector.
    """
    from circulax import compile_circuit

    ring = add_drop_ring(length_m=length_m, power_coupling=power_coupling)
    ring_model = "ring_" + sha256(repr((length_m, power_coupling)).encode()).hexdigest()[:16]
    net = {
        "instances": {
            "RING": {"component": ring_model},
            "M": {"component": "mirror", "settings": {"amplitude_reflection": 0.9}},
            "THROUGH": {"component": "ring_load"},
            "ADD": {"component": "ring_load"},
        },
        "connections": {
            "RING,drop": "M,p1",
            "RING,through": "THROUGH,p1",
            "RING,add": "ADD,p1",
        },
        "ports": {"p1": "RING,input", "p2": "M,p2", "heater": "RING,heater", "heater_return": "RING,heater_return"},
    }
    return compile_circuit(
        net,
        {ring_model: ring, "mirror": OpticalInterface, "ring_load": MatchedOpticalLoad},
        is_complex=True,
        backend="klu",
        g_leak=0,
    )


class HybridThreshold(NamedTuple):
    """Required modal gain (1/m), injection (A), unit-magnitude round-trip field."""

    gain_m1: object
    current_a: object
    roundtrip: object


def hybrid_threshold(p: LaserParameters, s: HybridSettings, omega: object, temperatures: object) -> HybridThreshold:
    """Threshold of the SG assembly with matched, lossy optical interfaces.

    temperatures = left grating, right grating, phase rises (K). This helper
    follows the assembled cell equations; no separate optical dynamics solver.
    Its roots must also satisfy phase, not just high reflection overlap.
    """
    tl, tr, tp = temperatures
    rl = sampled_grating_response(omega, tl, samples=s.samples, **grating_settings(p, s, "left"))[0]
    rr = sampled_grating_response(omega, tr, samples=s.samples, **grating_settings(p, s, "right"))[0]
    interface_roundtrip_amplitude = 10 ** (-2 * s.interface_loss_db / 10)
    feedback = rl * rr * interface_roundtrip_amplitude
    gain = p.loss_m1 - jnp.log(jnp.maximum(jnp.abs(feedback), 1e-300)) / p.length_m
    density = p.transparency_m3 + gain / (p.confinement * p.differential_gain_m2)
    current = ELECTRON_CHARGE * p.length_m * p.width_m * p.thickness_m * density / (p.injection_efficiency * p.lifetime_s)
    flight = p.length_m * p.group_index / LIGHT_SPEED
    phase_flight = s.phase_length_m * s.passive_group_index / LIGHT_SPEED
    phase = (
        -p.alpha_h * gain * p.length_m
        + 2 * (p.detuning_rad_s - jnp.asarray(omega)) * flight
        + 2 * s.phase_per_k * tp
        - 2 * jnp.asarray(omega) * phase_flight
    )
    roundtrip = feedback / jnp.maximum(abs(feedback), 1e-300) * jnp.exp(1j * phase)
    return HybridThreshold(gain, current, roundtrip)


def hybrid_modes(
    p: LaserParameters,
    s: HybridSettings,
    temperatures: object,
    *,
    bounds_hz: tuple[float, float] = (-1.1e12, 1.1e12),
    points: int = 22001,
    max_current_a: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate phase crossings in a stated validity window; refine to qualify.

    max_current_a optionally restricts the returned threshold-current range.
    Near exact grating zeros the flat unbounded gain law produces arbitrarily
    large thresholds and rapid Henry-phase windings; qualify a finite current
    range as well as the frequency grid. Sign crossing can miss tangent roots.
    Comb mirrors repeat indefinitely in
    this thin-patch approximation. No global mode-selection or SMSR claim.
    """
    import jax

    evaluate = jax.jit(lambda f: hybrid_threshold(p, s, 2 * jnp.pi * f, temperatures))
    grid = np.linspace(*bounds_hz, points)
    loops = np.asarray(evaluate(grid).roundtrip)
    roots = []
    for i in np.flatnonzero(np.signbit(loops[:-1].imag) != np.signbit(loops[1:].imag)):
        root = brentq(lambda f: float(jnp.imag(evaluate(f).roundtrip)), grid[i], grid[i + 1], xtol=1e-2)
        value = evaluate(root)
        if float(jnp.real(value.roundtrip)) > 0.99 and (not roots or abs(root - roots[-1]) > 1):
            roots.append(root)
    frequencies = np.asarray(roots)
    currents = np.asarray(evaluate(frequencies).current_a)
    if max_current_a is not None:
        keep = currents <= max_current_a
        frequencies, currents = frequencies[keep], currents[keep]
    return frequencies, currents


def hybrid_frame_parameters(omega: float, segments: int, *, prefix: str = "") -> dict:
    """Shift every delayed leaf of the default SG assembly into one envelope frame.

    Prefix is the flattened hierarchy path, e.g. "LASER~". Replacement feedback
    networks must shift all their own delayed leaves consistently as well.
    """
    return {
        f"{prefix}{name}.frame_offset_rad_s": omega for name in ("SG_L", "SG_R", "PHASE", *(f"GAIN{j}" for j in range(segments)))
    }
