# Copyright (c) 2026 GDSFactory

"""Bidirectional diode-driven SOA with a shared, integrated optical gain state.

Optical ports use the same unit-reference-impedance wave convention as SAX
components: V=a+b and I=a-b, with I entering the device. Incident and outgoing
waves a/b have units sqrt(W). All material and diode parameters are illustrative.
"""

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from circulax.components.base_component import PhysicsReturn, Signals, component

ELECTRON_CHARGE = 1.602176634e-19
PLANCK = 6.62607015e-34
LIGHT_SPEED = 299792458.0


def soa_scales(
    *,
    length_um: float = 600.0,
    width_um: float = 2.0,
    thickness_um: float = 0.1,
    confinement: float = 0.3,
    differential_gain_m2: float = 6e-20,
    wavelength_nm: float = 1540.0,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Return active volume, modal differential gain and saturation energy.

    The confinement factor belongs in the modal differential gain, including
    the saturation energy. At fixed cross-section, saturation energy does not
    scale with length.
    """
    area = width_um * thickness_um * 1e-12
    modal_slope = confinement * differential_gain_m2
    photon_energy = PLANCK * LIGHT_SPEED / (wavelength_nm * 1e-9)
    return area * length_um * 1e-6, modal_slope, photon_energy * area / modal_slope


def unsaturated_gain(
    current_a: ArrayLike,
    *,
    lifetime_s: float = 280e-12,
    length_um: float = 600.0,
    width_um: float = 2.0,
    thickness_um: float = 0.1,
    confinement: float = 0.3,
    differential_gain_m2: float = 6e-20,
    transparency_m3: float = 5.5e23,
    injection_efficiency: float = 0.8,
) -> ArrayLike:
    """Integrated logarithmic power gain at the supplied conduction current."""
    volume, slope, _ = soa_scales(
        length_um=length_um,
        width_um=width_um,
        thickness_um=thickness_um,
        confinement=confinement,
        differential_gain_m2=differential_gain_m2,
    )
    density = injection_efficiency * current_a * lifetime_s / (ELECTRON_CHARGE * volume)
    return slope * (density - transparency_m3) * length_um * 1e-6


def stimulated_power(gain: ArrayLike, loss: ArrayLike, input_power: ArrayLike) -> ArrayLike:
    """Stimulated optical generation (W) in a uniform logarithmic-gain cell.

    Integrate g*P along the cell. Passive absorption is excluded from carrier
    extraction. The series preserves finite derivatives at zero net gain.
    """
    net = gain - loss
    small = jnp.abs(net) < 1e-5
    safe_net = jnp.where(small, 1.0, net)
    exprel = jnp.where(small, 1 + net / 2 + net**2 / 6 + net**3 / 24, jnp.expm1(net) / safe_net)
    return input_power * gain * exprel


@component(
    ports=("anode", "cathode", "optical_in", "optical_out"),
    states=("v_j", "h", "wave_optical_in", "wave_optical_out"),
    holomorphic=False,
)
def SemiconductorOpticalAmplifier(
    signals: Signals,
    length_um: float = 600.0,
    width_um: float = 2.0,
    thickness_um: float = 0.1,
    lifetime_s: float = 280e-12,
    confinement: float = 0.3,
    differential_gain_m2: float = 6e-20,
    transparency_m3: float = 5.5e23,
    injection_efficiency: float = 0.8,
    wavelength_nm: float = 1540.0,
    alpha_h: float = 3.5,
    loss_db_per_cm: float = 0.0,
    saturation: float = 1.0,
    saturation_current_a: float = 1e-12,
    ideality: float = 1.8,
    thermal_voltage_v: float = 0.02585,
    resistance_ohm: float = 5.0,
    capacitance_f: float = 150e-15,
) -> PhysicsReturn:
    """Reciprocal optical transmission with diode injection and shared saturation.

    ``optical_in`` and ``optical_out`` retain their names for connectivity but
    either port accepts light. States ``wave_optical_in/out`` are the incident
    fields a_L/a_R; outgoing fields are b_L=T*a_R and b_R=T*a_L. Optical port
    values are total fields V=a+b, not incident power amplitudes when reflected
    light is present. Port flows are I=a-b at unit reference impedance. Use a
    matched source and matched termination for a reflection-free experiment.

    ``h`` is the integral of modal gain over length. Both incident powers
    deplete the same reservoir: tau*dh/dt=h0-h-saturation*tau*P_stim/E_sat.
    P_stim=(|a_L|^2+|a_R|^2)*h*exprel(h-loss), using a uniform gain per section.
    With zero loss and a_R=0 this recovers the original forward-only equation.
    Counterpropagating interference is spatially averaged: standing-wave
    carrier gratings and spatial hole burning are not resolved. Bidirectional
    loading generally needs a segment-convergence check even at zero loss.

    Set saturation=0 for a linear amplifier, and alpha_h=0 for no chirp.
    The junction's displacement current is excluded from carrier injection.
    Diode Is and C scale with contact area; vertical series R scales inversely.
    ASE is sampled separately with :func:`ase_field`, never inside Newton.
    The optical interfaces are reflection-free internally; external components
    may reflect light back through them. Transmission has no propagation delay
    or cold-waveguide phase. Cavity round-trip timing and laser dynamics need
    a traveling-wave model. Thermal effects and ASE depletion are omitted.
    """
    contact_scale = length_um * width_um / (600.0 * 2.0)
    vj = signals.v_j
    diode_current = saturation_current_a * contact_scale * jnp.expm1(vj / (ideality * thermal_voltage_v))
    terminal_current = (signals.anode - signals.cathode - vj) / (resistance_ohm / contact_scale)
    h0 = unsaturated_gain(
        jnp.real(diode_current),
        length_um=length_um,
        width_um=width_um,
        thickness_um=thickness_um,
        lifetime_s=lifetime_s,
        confinement=confinement,
        differential_gain_m2=differential_gain_m2,
        transparency_m3=transparency_m3,
        injection_efficiency=injection_efficiency,
    )
    _, _, energy = soa_scales(
        length_um=length_um,
        width_um=width_um,
        thickness_um=thickness_um,
        confinement=confinement,
        differential_gain_m2=differential_gain_m2,
        wavelength_nm=wavelength_nm,
    )
    h_real = jnp.real(signals.h)
    left_in, right_in = signals.wave_optical_in, signals.wave_optical_out
    pin = jnp.abs(left_in) ** 2 + jnp.abs(right_in) ** 2
    loss = loss_db_per_cm * length_um / 10000.0 * jnp.log(10.0) / 10.0
    depletion = saturation * lifetime_s * stimulated_power(h_real, loss, pin) / energy
    transmission = jnp.exp(0.5 * ((1 - 1j * alpha_h) * h_real - loss))
    left_out, right_out = transmission * right_in, transmission * left_in
    return {
        "anode": terminal_current,
        "cathode": -terminal_current,
        "v_j": (diode_current - terminal_current) / 0.1,
        "h": signals.h - h0 + depletion,
        "optical_in": left_in - left_out,
        "optical_out": right_in - right_out,
        "wave_optical_in": left_in + left_out - signals.optical_in,
        "wave_optical_out": right_in + right_out - signals.optical_out,
    }, {"v_j": capacitance_f * contact_scale * vj / 0.1, "h": lifetime_s * signals.h}


def ase_field(
    gain: ArrayLike, bandwidth_hz: float, *, noise_figure_db: float = 6.5, wavelength_nm: float = 1540.0, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Sample output-only ASE, one polarization, in a stated complex bandwidth.

    Return (complex field, expected power). The complex white samples assume
    bandwidth_hz=1/dt for a uniformly sampled envelope; an actual receiver
    filter must be applied before interpreting in-band noise. The high-gain
    NF approximation n_sp=10**(NF/10)/2 is used. Absorbing sections add no ASE.
    """
    gain = np.asarray(gain)
    nsp = 10 ** (noise_figure_db / 10) / 2
    power = nsp * np.maximum(gain - 1, 0) * PLANCK * LIGHT_SPEED / (wavelength_nm * 1e-9) * bandwidth_hz
    noise = (rng.normal(size=gain.shape) + 1j * rng.normal(size=gain.shape)) * np.sqrt(power / 2)
    return noise, power
