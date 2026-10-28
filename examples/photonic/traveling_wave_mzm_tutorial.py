# ruff: noqa: INP001, ANN001, ANN201
"""Repeated wiring and numerical utilities for the push-pull MZM tutorial.

The notebook defines component physics, optical connectivity and analyses.
These helpers repeat the section topology, generate binary/PAM4 waveforms,
fold eyes and calculate the loaded-cell group index.
"""

from typing import Any

import numpy as np


def electrode_netlist(
    *, n_sections: int, dz: float, rj_product: float, cj_per_m: float, phase_settings: dict | None = None
) -> dict:
    """Repeat modal half-lines, an adapter and two Rj/junction branches.

    Supplying phase_settings replaces each capacitor with a PN phase shifter
    and connects the optical sections along each arm.
    """
    optical = phase_settings is not None
    net = {"instances": {"GND": {"component": "ground"}}, "connections": {}, "ports": {}}
    ins, con, ports = net["instances"], net["connections"], net["ports"]
    for k in range(n_sections):
        ins[f"X{k}"] = {"component": "adapter"}
        for mode, model in (("d", "d_line"), ("c", "c_line")):
            left, right = f"{mode}L{k}", f"{mode}R{k}"
            ins[left], ins[right] = {"component": model}, {"component": model}
            con[f"{left},p2"] = f"X{k},{mode}"
            con[f"{right},p1"] = f"X{k},{mode}"
            if k:
                con[f"{left},p1"] = f"{mode}R{k - 1},p2"
            else:
                ports[f"{mode}_in"] = f"{left},p1"
            ports[f"{mode}_out"] = f"{right},p2"
        for arm in ("a", "b"):
            resistor, junction = f"R{arm}{k}", f"PS{arm}{k}"
            ins[resistor] = {"component": "R", "settings": {"R": rj_product / dz}}
            ins[junction] = (
                {"component": "pn", "settings": phase_settings} if optical else {"component": "C", "settings": {"C": cj_per_m * dz}}
            )
            positive, negative = ("anode", "cathode") if optical else ("p1", "p2")
            con[f"{resistor},p1"] = f"X{k},{arm}"
            con[f"{resistor},p2"] = f"{junction},{positive}"
            con[f"{junction},{negative}"] = "GND,p1"
            ports[f"junction_{arm}_{k}"] = f"{junction},{positive}"
    if optical:
        for arm in ("a", "b"):
            for k in range(n_sections - 1):
                con[f"PS{arm}{k},o2"] = f"PS{arm}{k + 1},o1"
    return net


def loaded_group_index(line: Any, frequencies: Any, *, scale: float, dz: float, rj_product: float, cj_per_m: float) -> np.ndarray:
    """Differentiate the Bloch phase of a symmetric, junction-loaded RLC cell.

    scale converts the physical junction to differential (2) or common (0.5)
    mode: R_modal = scale * Rj and C_modal = Cj / scale.
    """
    f = np.asarray(frequencies)
    s_ac = 2j * np.pi * f
    d = dz / 2
    series = np.tile(np.eye(2, dtype=complex), (len(f), 1, 1))
    shunt = series.copy()
    series[:, 0, 1] = line.series.evaluate(f) * d / 2
    shunt[:, 1, 0] = line.shunt.evaluate(f) * d
    half = series @ shunt @ series
    junction = np.tile(np.eye(2, dtype=complex), (len(f), 1, 1))
    junction[:, 1, 0] = s_ac * (cj_per_m * dz / scale) / (1 + s_ac * rj_product * cj_per_m)
    cell = half @ junction @ half
    gamma_loaded = np.arccosh((cell[:, 0, 0] + cell[:, 1, 1]) / 2) / dz
    return 299_792_458.0 * np.gradient(gamma_loaded.imag, 2 * np.pi * f)


def prbs_drive(baud, *, preamble_bits=16, vpp=4.0):
    """Return a PRBS7 pattern and 0.2-UI-edge differential PWL drive."""
    ui = 1 / baud
    state, pattern = 0x7F, []
    for _ in range(127):
        pattern.append(state & 1)
        state = (state >> 1) | (((state ^ (state >> 1)) & 1) << 6)
    if state != 0x7F or sum(pattern) != 64:
        msg = "Invalid PRBS7 period or balance"
        raise ValueError(msg)
    bits = np.r_[np.full(preamble_bits, pattern[0]), pattern]
    levels = vpp * (bits - 0.5)
    times, values = [0.0], [float(levels[0])]
    for k in range(1, len(bits)):
        if levels[k] != levels[k - 1]:
            times.extend([k * ui, (k + 0.2) * ui])
            values.extend([float(levels[k - 1]), float(levels[k])])
    times.append(len(bits) * ui)
    values.append(float(levels[-1]))
    return bits, np.asarray(times), np.asarray(values)


def fold_eye(times, power, *, baud, samples_per_ui, preamble_bits, flight_time):
    """Fold two UI with one central eye and half an eye at each edge."""
    ui = 1 / baud
    n_bits = (len(times) - 1) // samples_per_ui
    eye_ui = np.arange(2 * samples_per_ui + 1) / samples_per_ui
    fold_bits = np.arange(preamble_bits + 8, n_bits - 5)
    # Start at mid-symbol: center one full eye in the two-UI display.
    query = flight_time + (fold_bits[:, None] + 0.5 + eye_ui[None, :]) * ui
    if query.max() >= float(times[-1]):
        msg = "Eye windows extend beyond the simulated record"
        raise ValueError(msg)
    traces = np.interp(query.ravel(), times, power).reshape(query.shape)
    return eye_ui, traces


def pam4_drive(
    baud,
    *,
    n_symbols=128,
    preamble_symbols=16,
    vpp=4.0,
    random_jitter_ui=0.025,
    periodic_jitter_ui=0.06,
    periodic_jitter_hz=2e9,
    seed=7,
):
    """Return seeded PAM4 symbols and a PWL drive with jittered transition times.

    Random jitter is Gaussian with the specified RMS in UI, clipped at ±3 RMS.
    Periodic jitter is a sinusoidal time displacement with peak amplitude in UI.
    Both shift complete 0.2-UI ramps, preserving their duration and voltage levels.
    """
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 4, n_symbols)
    symbols = np.r_[np.full(preamble_symbols, data[0]), data]
    levels = vpp * (symbols / 3 - 0.5)
    ui = 1 / baud
    edges = np.arange(1, len(symbols)) * ui
    random = np.clip(rng.normal(size=len(edges)), -3, 3) * random_jitter_ui
    periodic = periodic_jitter_ui * np.sin(2 * np.pi * periodic_jitter_hz * edges)
    shifts = ui * (random + periodic)
    shifts[edges < preamble_symbols * ui] = 0.0
    times, values = [0.0], [float(levels[0])]
    for k, start in enumerate(edges + shifts, start=1):
        if levels[k] != levels[k - 1]:
            times.extend([start, start + 0.2 * ui])
            values.extend([float(levels[k - 1]), float(levels[k])])
    times.append(len(symbols) * ui)
    values.append(float(levels[-1]))
    if np.any(np.diff(times) <= 0):
        msg = "Jitter is too large: PWL edges overlap or leave the simulation interval"
        raise ValueError(msg)
    return symbols, np.asarray(times), np.asarray(values)
