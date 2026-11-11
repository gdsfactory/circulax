# Copyright (c) 2026 GDSFactory
"""Passive single-pole spectral reflection and electrically loaded thermal tuning.

Envelope convention is exp(+i*omega*t). Frequencies are offsets from the fixed
optical reference; heating redshifts mirror wavelength. Rejected light is a
dump channel, not an additional lossless transmitted output.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from circulax.components.base_component import PhysicsReturn, Signals, component
from circulax.components.soa import LIGHT_SPEED


class MirrorParameters(NamedTuple):
    """Illustrative reflector/heater settings; SI units throughout."""

    mirror_bandwidth_hz: float = 12e9  # Power-reflection HWHM; FWHM is twice this value.
    mirror_center_m: float = 1540e-9
    mirror_shift_m_per_k: float = 0.08e-9
    phase_shift_rad_per_k: float = -0.6
    heater_resistance_ohm: float = 100.0
    thermal_resistance_k_per_w: float = 200.0
    thermal_tau_s: float = 1e-6


def mirror_detuning(center_m: ArrayLike, shift_m_per_k: ArrayLike, temperature_k: ArrayLike, reference_m: ArrayLike) -> jax.Array:
    """Mirror center angular frequency offset from the fixed reference (rad/s)."""
    return 2 * jnp.pi * LIGHT_SPEED * (1 / (center_m + shift_m_per_k * temperature_k) - 1 / reference_m)


def reflection_response(r_peak: ArrayLike, bandwidth_hz: ArrayLike, omega_rad_s: ArrayLike, center_rad_s: ArrayLike) -> jax.Array:
    """Complex reflection for exp(+i*omega*t), |r| <= |r_peak|."""
    gamma = 2 * jnp.pi * bandwidth_hz
    return r_peak * gamma / (gamma + 1j * (omega_rad_s - center_rad_s))


def thermal_terms(voltage: ArrayLike, temperature: ArrayLike, resistance: ArrayLike, rth: ArrayLike, tau: ArrayLike) -> tuple:
    """Return heater current, thermal residual/storage for real electrical drive."""
    current = voltage / resistance
    power = jnp.real(voltage) ** 2 / resistance
    return current, temperature - rth * power, tau * temperature


def mirror_terms(field: ArrayLike, reflected: ArrayLike, r_peak: ArrayLike, gamma: ArrayLike, detuning: ArrayLike) -> tuple:
    """Residual/storage realizing dm/dt=-(gamma-i*detuning)m+gamma*r_peak*u."""
    return (1 - 1j * detuning / gamma) * reflected - r_peak * field, reflected / gamma


def mirror_power_balance(
    field: ArrayLike, reflected: ArrayLike, derivative: ArrayLike, r_peak: ArrayLike, gamma: ArrayLike
) -> tuple:
    """Return tap, rejected power, stored energy, storage rate (W,W,J,W).

    Pin-Pref-Ptap-Pdump=dU/dt exactly for the single-pole mirror ODE.
    """
    tap = (1 - jnp.abs(r_peak) ** 2) * jnp.abs(field) ** 2
    dump = jnp.abs(r_peak * field - reflected) ** 2
    return tap, dump, jnp.abs(reflected) ** 2 / gamma, 2 * jnp.real(jnp.conj(reflected) * derivative) / gamma


@component(
    ports=("incident", "reflected", "tap", "heater", "heater_return"),
    states=("m", "temperature", "i_reflected", "i_tap"),
    holomorphic=False,
)
def ThermalReflector(
    signals: Signals,
    r_peak: complex = 0.7 + 0j,
    mirror_bandwidth_hz: float = 12e9,
    mirror_center_m: float = 1540e-9,
    mirror_shift_m_per_k: float = 0.08e-9,
    reference_wavelength_m: float = 1540e-9,
    frame_offset_rad_s: float = 0.0,
    heater_resistance_ohm: float = 100.0,
    thermal_resistance_k_per_w: float = 200.0,
    thermal_tau_s: float = 1e-6,
) -> PhysicsReturn:
    """Native reflector for independent AC, thermal and passive-energy checks.

    Reflected/tap nodes impose fields for observation or matched downstream
    networks. The tap coefficient is fixed sqrt(1-|r_peak|²); detuned rejection
    goes to an unmonitored dump. Heater power is not optical absorption heat.
    """
    voltage = signals.heater - signals.heater_return
    current, thermal, storage = thermal_terms(
        voltage, signals.temperature, heater_resistance_ohm, thermal_resistance_k_per_w, thermal_tau_s
    )
    delta = mirror_detuning(mirror_center_m, mirror_shift_m_per_k, jnp.real(signals.temperature), reference_wavelength_m)
    fm, qm = mirror_terms(signals.incident, signals.m, r_peak, 2 * jnp.pi * mirror_bandwidth_hz, delta - frame_offset_rad_s)
    return {
        "incident": 0.0,
        "reflected": signals.i_reflected,
        "tap": signals.i_tap,
        "heater": current,
        "heater_return": -current,
        "temperature": thermal,
        "m": fm,
        "i_reflected": signals.reflected - signals.m,
        "i_tap": signals.tap - jnp.sqrt(1 - jnp.abs(r_peak) ** 2) * signals.incident,
    }, {"m": qm, "temperature": storage}
