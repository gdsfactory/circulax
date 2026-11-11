# Copyright (c) 2026 GDSFactory
"""Longitudinal threshold conditions for the native thermally tuned laser.

These equations are the frequency-domain limits of its delayed cells and
single-pole mirror, not a modal power/noise model. Threshold discrimination is
not SMSR. Mode enumeration is an explicitly bounded frequency search.
"""

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

from circulax.components.laser_mirrors import MirrorParameters, mirror_detuning, reflection_response
from circulax.components.soa import ELECTRON_CHARGE, LIGHT_SPEED

if TYPE_CHECKING:
    from circulax.components.traveling_wave_laser import LaserParameters


class SpectralThreshold(NamedTuple):
    """Gain (1/m), carrier density (1/m³), current (A), unwrapped phase (rad)."""

    gain_m1: jax.Array
    density_m3: jax.Array
    current_a: jax.Array
    phase_rad: jax.Array


def spectral_threshold(
    p: "LaserParameters",
    mirrors: MirrorParameters,
    omega_rad_s: jax.Array,
    phase_temperature_k: jax.Array,
    filter_temperature_k: jax.Array,
) -> SpectralThreshold:
    """Threshold magnitude/phase at a physical frequency offset, differentiable.

    The nominal propagation calibration p.detuning_rad_s stays fixed when
    tuning. A longitudinal mode satisfies phase=2*pi*integer, not merely a
    reflector peak. Both the Henry phase and spectral reflection phase enter.
    """
    center = mirror_detuning(mirrors.mirror_center_m, mirrors.mirror_shift_m_per_k, filter_temperature_k, p.wavelength_m)
    rr = reflection_response(p.r_right, mirrors.mirror_bandwidth_hz, omega_rad_s, center)
    gain = p.loss_m1 - jnp.log(jnp.abs(p.r_left * rr)) / p.length_m
    density = p.transparency_m3 + gain / (p.confinement * p.differential_gain_m2)
    current = ELECTRON_CHARGE * p.length_m * p.width_m * p.thickness_m * density / (p.injection_efficiency * p.lifetime_s)
    phase = (
        jnp.angle(p.r_left * p.r_right)
        + mirrors.phase_shift_rad_per_k * phase_temperature_k
        - jnp.arctan((omega_rad_s - center) / (2 * jnp.pi * mirrors.mirror_bandwidth_hz))
        - p.alpha_h * gain * p.length_m
        + (p.detuning_rad_s - omega_rad_s) * 2 * p.length_m * p.group_index / LIGHT_SPEED
    )
    return SpectralThreshold(gain, density, current, phase)


class LaserModes(NamedTuple):
    """Phase-root frequencies (Hz offsets), threshold currents (A), branch integers."""

    frequency_hz: np.ndarray
    threshold_a: np.ndarray
    branch: np.ndarray


def find_laser_modes(
    p: "LaserParameters",
    mirrors: MirrorParameters,
    phase_temperature_k: float,
    filter_temperature_k: float,
    *,
    frequency_bounds_hz: tuple[float, float] = (-250e9, 250e9),
    samples: int = 4001,
) -> LaserModes:
    """Enumerate phase crossings in a stated interval using bracketed roots.

    Refine samples/bounds to qualify a design. Tangent roots exactly at a phase
    extremum can be missed by sign-crossing enumeration; no global mode-count
    or stability claim is made. Evaluate well away from such branch folds.
    """
    frequency = np.linspace(*frequency_bounds_hz, samples)
    evaluate = jax.jit(lambda w: spectral_threshold(p, mirrors, w, phase_temperature_k, filter_temperature_k))
    phase = np.asarray(evaluate(2 * np.pi * frequency).phase_rad) / (2 * np.pi)
    roots = []
    for i in range(samples - 1):
        low, high = sorted((phase[i], phase[i + 1]))
        for branch in range(int(np.ceil(low)), int(np.floor(high)) + 1):
            if low == high:
                continue
            root = brentq(
                lambda f, b=branch: float(evaluate(2 * np.pi * f).phase_rad) - 2 * np.pi * b,
                frequency[i],
                frequency[i + 1],
                xtol=1e-2,
            )
            if not roots or abs(root - roots[-1][0]) > 1:
                roots.append((root, branch))
    roots.sort()
    frequencies = np.array([r[0] for r in roots])
    currents = np.asarray(evaluate(2 * np.pi * frequencies).current_a)
    return LaserModes(frequencies, currents, np.array([r[1] for r in roots], dtype=int))
