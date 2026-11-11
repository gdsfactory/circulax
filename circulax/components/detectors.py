# Copyright 2026 Circulax contributors
"""Absorbed-power photodetectors with retained optical waves.

Compile with is_complex=True. Optical envelopes are sqrt(W); only optical_in
supports total traveling-wave observables. Electrical and carrier coordinates
retain the notebook laws, including complex electrical I/Q and populations.
Use small_signal for full real-block derivatives; legacy AC is unsupported.
"""

# ruff: noqa: PLR0917 -- preserve notebook positional settings and component signatures

from __future__ import annotations

import jax
import jax.numpy as jnp

from circulax.components.base_component import CircuitComponent, PhysicsReturn, Signals, component
from circulax.power_waves import power_wave_constraints, validate_reference

QE = 1.602176634e-19  # C
PORTS = ("optical_in", "anode", "cathode")


def spectral_response(
    responsivity: jax.Array | float = 0.8,
    wavelength_nm: jax.Array | float = 1550.0,
    center_wavelength_nm: jax.Array | float = 1550.0,
    responsivity_bandwidth_nm: jax.Array | float = 250.0,
    reflectivity: jax.Array | float = 0.04,
    reflection_bandwidth_nm: jax.Array | float = 180.0,
    reflection_phase_rad: jax.Array | float = 0.0,
    reflection_phase_slope_rad_nm: jax.Array | float = 0.01,
) -> tuple[jax.Array, jax.Array]:
    """Absorbed-power responsivity (A/W) and complex amplitude reflection vs nm."""
    detuning_nm = wavelength_nm - center_wavelength_nm
    r_abs = responsivity * (wavelength_nm / center_wavelength_nm)
    r_abs *= jnp.exp(-0.5 * (detuning_nm / responsivity_bandwidth_nm) ** 2)
    reflection = jnp.sqrt(reflectivity) * jnp.exp(-0.25 * (detuning_nm / reflection_bandwidth_nm) ** 2)
    reflection = reflection * jnp.exp(1j * (reflection_phase_rad + reflection_phase_slope_rad_nm * detuning_nm))
    return r_abs, reflection


def optical_conversion(
    incident: jax.Array | complex,
    responsivity: jax.Array | float = 0.8,
    wavelength_nm: jax.Array | float = 1550.0,
    center_wavelength_nm: jax.Array | float = 1550.0,
    responsivity_bandwidth_nm: jax.Array | float = 250.0,
    reflectivity: jax.Array | float = 0.04,
    reflection_bandwidth_nm: jax.Array | float = 180.0,
    reflection_phase_rad: jax.Array | float = 0.0,
    reflection_phase_slope_rad_nm: jax.Array | float = 0.01,
) -> tuple[jax.Array, jax.Array]:
    """Return reflected envelope and generated current from incident sqrt(W).

    This takes a traveling envelope, replacing the old potential-based helper.
    Generated current equals absorbed-power responsivity times net optical power.
    """
    r_abs, reflection = spectral_response(
        responsivity,
        wavelength_nm,
        center_wavelength_nm,
        responsivity_bandwidth_nm,
        reflectivity,
        reflection_bandwidth_nm,
        reflection_phase_rad,
        reflection_phase_slope_rad_nm,
    )
    absorbed_power = (1 - jnp.abs(reflection) ** 2) * jnp.abs(incident) ** 2
    return reflection * incident, r_abs * absorbed_power


def dark_iv(
    v: jax.Array | complex,
    i_s: jax.Array | float = 1e-12,
    ideality: jax.Array | float = 1.5,
    v_t: jax.Array | float = 0.02585,
    i_leak: jax.Array | float = 1e-9,
    v_leak: jax.Array | float = 0.2,
    r_sh: jax.Array | float = 1e12,
) -> jax.Array:
    """Anode-to-cathode dark current in A; retain complex voltage continuation."""
    return i_s * jnp.expm1(v / (ideality * v_t)) + i_leak * jnp.tanh(v / v_leak) + v / r_sh


def junction_charge(
    v: jax.Array | complex, c0: jax.Array | float = 20e-15, v_bi: jax.Array | float = 0.8, fc: jax.Array | float = 0.5
) -> jax.Array:
    """Junction charge in C, with the original real-part forward-branch selection."""
    v_join = fc * v_bi
    # A complex V is retained for the electrical imaginary-block equations.
    v_depletion = jnp.where(jnp.real(v) <= v_join, v, v_join)
    q_depletion = 2 * c0 * v_bi * (1 - jnp.sqrt(1 - v_depletion / v_bi))
    dv = jnp.where(jnp.real(v) > v_join, v - v_join, 0.0)
    c_join = c0 / jnp.sqrt(1 - fc)
    dc_join = c0 / (2 * v_bi * (1 - fc) ** 1.5)
    return q_depletion + c_join * dv + 0.5 * dc_join * dv**2


def internal_field(
    x_e: jax.Array | complex,
    x_h: jax.Array | complex,
    v: jax.Array | complex,
    v_bi: jax.Array | float = 0.8,
    screening: jax.Array | float = 1.0,
    x_screen: jax.Array | float = 1.0,
    depletion_width_m: jax.Array | float = 1e-6,
) -> tuple[jax.Array, jax.Array]:
    """Unscreened and screened fields in V/m from real nonnegative populations."""
    population = (jnp.maximum(jnp.real(x_e), 0) + jnp.maximum(jnp.real(x_h), 0)) / 2
    reverse_bias = jnp.maximum(-jnp.real(v), 0)
    unscreened = (v_bi + reverse_bias) / depletion_width_m
    effective = unscreened / (1 + screening * population / x_screen)
    return unscreened, effective


def carrier_rates(
    x_e: jax.Array | complex,
    x_h: jax.Array | complex,
    v: jax.Array | complex,
    tau_e0: jax.Array | float = 8e-12,
    tau_h0: jax.Array | float = 40e-12,
    v_bi: jax.Array | float = 0.8,
    v_ref: jax.Array | float = 3.0,
    screening: jax.Array | float = 1.0,
    x_screen: jax.Array | float = 1.0,
    tau_rec: jax.Array | float = 500e-12,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Electron/hole sweep times (s) and normalized pair recombination rate (1/s)."""
    _, effective_field = internal_field(x_e, x_h, v, v_bi, screening, x_screen)
    reference_field = (v_bi + v_ref) / 1e-6
    tau_e, tau_h = tau_e0 * reference_field / effective_field, tau_h0 * reference_field / effective_field
    recombination = x_e * x_h / tau_rec
    return tau_e, tau_h, recombination


def ideal_photodetector(*, r0: float = 1.0) -> type[CircuitComponent]:
    """Build IdealPhotodetector with a static finite positive real optical reference.

    Runtime settings and units match the original notebook. Reflection is
    specified by power reflectivity (0 through 1) and phase in radians.
    The reference changes only optical potential/flow coordinates.
    """
    r0 = validate_reference(r0)

    @component(ports=PORTS, states=("wave_optical_in",), holomorphic=False)
    def IdealPhotodetector(
        signals: Signals,
        responsivity: float = 0.8,
        wavelength_nm: float = 1550.0,
        center_wavelength_nm: float = 1550.0,
        responsivity_bandwidth_nm: float = 250.0,
        reflectivity: float = 0.04,
        reflection_bandwidth_nm: float = 180.0,
        reflection_phase_rad: float = 0.0,
        reflection_phase_slope_rad_nm: float = 0.01,
    ) -> PhysicsReturn:
        """IdealPhotodetector with absorbed-power generation and retained incident wave."""
        outgoing, i_photo = optical_conversion(
            signals.wave_optical_in,
            responsivity,
            wavelength_nm,
            center_wavelength_nm,
            responsivity_bandwidth_nm,
            reflectivity,
            reflection_bandwidth_nm,
            reflection_phase_rad,
            reflection_phase_slope_rad_nm,
        )
        optical_f, _ = power_wave_constraints(signals, ("optical_in",), jnp.array([outgoing]), r0=r0)
        return {**optical_f, "anode": -i_photo, "cathode": i_photo}, {}

    IdealPhotodetector._wave_r0 = r0  # noqa: SLF001 -- compiler optical metadata
    IdealPhotodetector._wave_ports = ("optical_in",)  # noqa: SLF001 -- optical terminals only
    return IdealPhotodetector


def photodiode(*, r0: float = 1.0) -> type[CircuitComponent]:
    """Build Photodiode with a static finite positive real optical reference.

    Runtime settings and units match the original notebook. Reflection is
    specified by power reflectivity (0 through 1) and phase in radians.
    The reference changes only optical potential/flow coordinates.
    """
    r0 = validate_reference(r0)

    @component(ports=PORTS, states=("wave_optical_in",), holomorphic=False)
    def Photodiode(
        signals: Signals,
        responsivity: float = 0.8,
        wavelength_nm: float = 1550.0,
        center_wavelength_nm: float = 1550.0,
        responsivity_bandwidth_nm: float = 250.0,
        reflectivity: float = 0.04,
        reflection_bandwidth_nm: float = 180.0,
        reflection_phase_rad: float = 0.0,
        reflection_phase_slope_rad_nm: float = 0.01,
        i_s: float = 1e-12,
        ideality: float = 1.5,
        v_t: float = 0.02585,
        i_leak: float = 1e-9,
        v_leak: float = 0.2,
        r_sh: float = 1e12,
        c0: float = 20e-15,
        v_bi: float = 0.8,
        fc: float = 0.5,
    ) -> PhysicsReturn:
        """Photodiode with absorbed-power generation and retained incident wave."""
        v = signals.anode - signals.cathode
        current = dark_iv(v, i_s, ideality, v_t, i_leak, v_leak, r_sh)
        outgoing, i_photo = optical_conversion(
            signals.wave_optical_in,
            responsivity,
            wavelength_nm,
            center_wavelength_nm,
            responsivity_bandwidth_nm,
            reflectivity,
            reflection_bandwidth_nm,
            reflection_phase_rad,
            reflection_phase_slope_rad_nm,
        )
        optical_f, _ = power_wave_constraints(signals, ("optical_in",), jnp.array([outgoing]), r0=r0)
        current -= i_photo
        charge = junction_charge(v, c0, v_bi, fc)
        return {**optical_f, "anode": current, "cathode": -current}, {"anode": charge, "cathode": -charge}

    Photodiode._wave_r0 = r0  # noqa: SLF001 -- compiler optical metadata
    Photodiode._wave_ports = ("optical_in",)  # noqa: SLF001 -- optical terminals only
    return Photodiode


def carrier_photodiode(*, r0: float = 1.0) -> type[CircuitComponent]:
    """Build CarrierPhotodiode with a static finite positive real optical reference.

    Runtime settings and units match the original notebook. Reflection is
    specified by power reflectivity (0 through 1) and phase in radians.
    The reference changes only optical potential/flow coordinates.
    """
    r0 = validate_reference(r0)

    @component(ports=PORTS, states=("x_e", "x_h", "wave_optical_in"), holomorphic=False)
    def CarrierPhotodiode(
        signals: Signals,
        responsivity: float = 0.8,
        wavelength_nm: float = 1550.0,
        center_wavelength_nm: float = 1550.0,
        responsivity_bandwidth_nm: float = 250.0,
        reflectivity: float = 0.04,
        reflection_bandwidth_nm: float = 180.0,
        reflection_phase_rad: float = 0.0,
        reflection_phase_slope_rad_nm: float = 0.01,
        i_s: float = 1e-12,
        ideality: float = 1.5,
        v_t: float = 0.02585,
        i_leak: float = 1e-9,
        v_leak: float = 0.2,
        r_sh: float = 1e12,
        c0: float = 20e-15,
        v_bi: float = 0.8,
        fc: float = 0.5,
        volume_m3: float = 100e-18,
        density_scale_m3: float = 1e21,
        tau_e0: float = 8e-12,
        tau_h0: float = 40e-12,
        v_ref: float = 3.0,
        screening: float = 1.0,
        x_screen: float = 1.0,
        tau_rec: float = 500e-12,
        electron_weight: float = 0.5,
        tau_ref: float = 20e-12,
    ) -> PhysicsReturn:
        """CarrierPhotodiode with absorbed-power generation and retained incident wave."""
        v = signals.anode - signals.cathode
        x_e, x_h = signals.x_e, signals.x_h
        charge_scale = QE * volume_m3 * density_scale_m3
        outgoing, generation_current = optical_conversion(
            signals.wave_optical_in,
            responsivity,
            wavelength_nm,
            center_wavelength_nm,
            responsivity_bandwidth_nm,
            reflectivity,
            reflection_bandwidth_nm,
            reflection_phase_rad,
            reflection_phase_slope_rad_nm,
        )
        optical_f, _ = power_wave_constraints(signals, ("optical_in",), jnp.array([outgoing]), r0=r0)
        generation = generation_current / charge_scale
        tau_e, tau_h, recombination = carrier_rates(x_e, x_h, v, tau_e0, tau_h0, v_bi, v_ref, screening, x_screen, tau_rec)
        i_photo = charge_scale * (electron_weight * x_e / tau_e + (1 - electron_weight) * x_h / tau_h)
        current = dark_iv(v, i_s, ideality, v_t, i_leak, v_leak, r_sh) - i_photo
        charge = junction_charge(v, c0, v_bi, fc)
        return {
            **optical_f,
            "anode": current,
            "cathode": -current,
            "x_e": tau_ref * (x_e / tau_e + recombination - generation),
            "x_h": tau_ref * (x_h / tau_h + recombination - generation),
        }, {"anode": charge, "cathode": -charge, "x_e": tau_ref * x_e, "x_h": tau_ref * x_h}

    CarrierPhotodiode._wave_r0 = r0  # noqa: SLF001 -- compiler optical metadata
    CarrierPhotodiode._wave_ports = ("optical_in",)  # noqa: SLF001 -- optical terminals only
    CarrierPhotodiode._non_wave_states = ("x_e", "x_h")  # noqa: SLF001 -- populations are not normalized fields
    return CarrierPhotodiode


IdealPhotodetector = ideal_photodetector()
Photodiode = photodiode()
CarrierPhotodiode = carrier_photodiode()
