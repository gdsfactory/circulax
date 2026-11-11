# Copyright (c) 2026 GDSFactory
"""Composable bidirectional sections for reconfigurable semiconductor laser circuits.

Unit reference impedance: V=a+b, I=a-b, with I entering the component and
waves in sqrt(W). Reference-envelope convention: exp(+i*omega*t). Positive
phase_rad multiplies the envelope by exp(+i*phase_rad). All lengths use SI.
"""

from functools import lru_cache

import jax.numpy as jnp

from circulax.components.base_component import CircuitComponent, PhysicsReturn, Signals, component, source
from circulax.components.laser_mirrors import mirror_detuning, thermal_terms
from circulax.components.soa import LIGHT_SPEED, soa_scales, stimulated_power, unsaturated_gain


def _waves(signals: Signals, left: object, right: object) -> dict:
    return {
        "p1": signals.a1 - left,
        "p2": signals.a2 - right,
        "a1": signals.a1 + left - signals.p1,
        "a2": signals.a2 + right - signals.p2,
    }


@component(ports=("p1", "p2"), states=("a1", "a2"), holomorphic=True)
def OpticalInterface(signals: Signals, amplitude_reflection: float = 0.0, power_loss_db: float = 0.0) -> PhysicsReturn:
    """Reciprocal optical interface; r=i*rho, t=sqrt(1-rho²), common excess loss.

    rho is a real amplitude magnitude in [-1,1], not power reflectance.
    This also supplies a frequency-flat mirror when excess loss is zero.
    """
    amplitude = 10 ** (-power_loss_db / 20)
    r = 1j * amplitude_reflection * amplitude
    t = jnp.sqrt(1 - amplitude_reflection**2) * amplitude
    return _waves(signals, r * signals.a1 + t * signals.a2, t * signals.a1 + r * signals.a2), {}


@component(ports=("p1", "p2"), states=("a1", "a2"), holomorphic=True)
def BidirectionalWaveguide(
    signals: Signals,
    length_m: float = 50e-6,
    group_index: float = 4.0,
    loss_m1: float = 0.0,
    phase_rad: float = 0.0,
    frame_offset_rad_s: float = 0.0,
) -> PhysicsReturn:
    """Reflection-free reciprocal propagation with real flight time and loss.

    phase_rad is the phase at the fixed optical reference, not a recomputed
    carrier phase when changing envelope coordinates. loss_m1 is power loss.
    """
    tau = length_m * group_index / LIGHT_SPEED
    past = signals.at_delay(tau)
    transmission = jnp.exp(-loss_m1 * length_m / 2 + 1j * (phase_rad - frame_offset_rad_s * tau))
    return _waves(signals, transmission * past.a2, transmission * past.a1), {}


@component(ports=("p1", "p2", "heater", "heater_return"), states=("a1", "a2", "temperature"), holomorphic=False)
def ThermoOpticPhaseShifter(
    signals: Signals,
    length_m: float = 100e-6,
    group_index: float = 4.0,
    loss_m1: float = 0.0,
    phase_rad: float = 0.0,
    phase_per_k: float = 0.15,
    frame_offset_rad_s: float = 0.0,
    heater_resistance_ohm: float = 100.0,
    thermal_resistance_k_per_w: float = 200.0,
    thermal_tau_s: float = 1e-6,
) -> PhysicsReturn:
    """Reciprocal delayed phase section, uniform arrival-time temperature.

    temperature is the rise above ambient (K); phase_per_k is one-pass rad/K.
    Thermal variation during a flight is neglected (flight << thermal time).
    """
    tau = length_m * group_index / LIGHT_SPEED
    past = signals.at_delay(tau)
    phase = phase_rad + phase_per_k * jnp.real(signals.temperature) - frame_offset_rad_s * tau
    transmission = jnp.exp(-loss_m1 * length_m / 2 + 1j * phase)
    f = _waves(signals, transmission * past.a2, transmission * past.a1)
    current, ft, qt = thermal_terms(
        signals.heater - signals.heater_return,
        signals.temperature,
        heater_resistance_ohm,
        thermal_resistance_k_per_w,
        thermal_tau_s,
    )
    f.update(heater=current, heater_return=-current, temperature=ft)
    return f, {"temperature": qt}


@component(ports=("p1", "p2", "anode", "cathode"), states=("a1", "a2", "v_j", "h"), holomorphic=False)
def TravelingWaveGainCell(
    signals: Signals,
    length_m: float = 75e-6,
    width_m: float = 2e-6,
    thickness_m: float = 0.1e-6,
    group_index: float = 3.6,
    lifetime_s: float = 280e-12,
    confinement: float = 0.3,
    differential_gain_m2: float = 6e-20,
    transparency_m3: float = 5.5e23,
    injection_efficiency: float = 0.8,
    wavelength_m: float = 1540e-9,
    alpha_h: float = 3.5,
    loss_m1: float = 1000.0,
    detuning_rad_s: float = 0.0,
    frame_offset_rad_s: float = 0.0,
    saturation: float = 1.0,
    saturation_current_a: float = 1e-12,
    ideality: float = 1.8,
    thermal_voltage_v: float = 0.02585,
    resistance_ohm: float = 5.0,
    capacitance_f: float = 150e-15,
) -> PhysicsReturn:
    """One retarded-gain cell; both delayed incident powers share one reservoir.

    Connect cells in optical series and their diode terminals in electrical
    parallel for a uniformly contacted gain region. Diode area scales with cell length.
    Frozen arrival-time gain is first order in cell length. Interference is
    averaged; carrier gratings, gain dispersion and spontaneous noise omitted.
    """
    scale = length_m * width_m / (600e-6 * 2e-6)
    diode = saturation_current_a * scale * jnp.expm1(signals.v_j / (ideality * thermal_voltage_v))
    current = (signals.anode - signals.cathode - signals.v_j) * scale / resistance_ohm
    settings = {
        "length_um": length_m * 1e6,
        "width_um": width_m * 1e6,
        "thickness_um": thickness_m * 1e6,
        "confinement": confinement,
        "differential_gain_m2": differential_gain_m2,
    }
    h0 = unsaturated_gain(
        jnp.real(diode),
        lifetime_s=lifetime_s,
        transparency_m3=transparency_m3,
        injection_efficiency=injection_efficiency,
        **settings,
    )
    energy = soa_scales(wavelength_nm=wavelength_m * 1e9, **settings)[2]
    tau = length_m * group_index / LIGHT_SPEED
    past = signals.at_delay(tau)
    gain = jnp.real(signals.h)
    loss = loss_m1 * length_m
    transmission = jnp.exp(((1 - 1j * alpha_h) * gain - loss) / 2 + 1j * (detuning_rad_s - frame_offset_rad_s) * tau)
    f = _waves(signals, transmission * past.a2, transmission * past.a1)
    pin = abs(past.a1) ** 2 + abs(past.a2) ** 2
    f.update(
        anode=current,
        cathode=-current,
        v_j=(diode - current) / 0.1,
        h=signals.h - h0 + saturation * lifetime_s * stimulated_power(gain, loss, pin) / energy,
    )
    return f, {"v_j": capacitance_f * scale * signals.v_j / 0.1, "h": lifetime_s * signals.h}


@lru_cache(maxsize=16)
def sampled_bragg_grating(samples: int = 5) -> type[CircuitComponent]:
    """Lossless thin-patch sampled-grating approximation with causal gap delays.

    Identical zero-length Bragg patches have r=i*tanh(strength/samples),
    t=sech(strength/samples). Delayed gaps retain all multiple reflections.
    Peak power reflectance=tanh(strength)^2; comb FSR=c/(2*n_g*spacing).
    Finite patch bandwidth/dispersion is omitted: qualify a finite wavelength
    window near the reference. Increasing samples changes the physical device;
    it is not numerical mesh refinement of the same grating.
    """
    if not isinstance(samples, int) or samples < 2:
        msg = "samples must be an integer >= 2"
        raise ValueError(msg)
    states = ("a1", "a2", "temperature", *(f"{d}{j}" for j in range(samples) for d in ("f", "b")))

    @component(ports=("p1", "p2", "heater", "heater_return"), states=states, holomorphic=False)
    def SampledBraggGrating(
        signals: Signals,
        spacing_m: float = 187.37028625e-6,
        group_index: float = 4.0,
        strength: float = 0.9,
        center_m: float = 1540e-9,
        shift_m_per_k: float = 0.08e-9,
        reference_m: float = 1540e-9,
        frame_offset_rad_s: float = 0.0,
        heater_resistance_ohm: float = 100.0,
        thermal_resistance_k_per_w: float = 200.0,
        thermal_tau_s: float = 1e-6,
    ) -> PhysicsReturn:
        """Two optical scattering ports and a real resistive thermal control."""
        tau = spacing_m * group_index / LIGHT_SPEED
        past = signals.at_delay(tau)
        center = mirror_detuning(center_m, shift_m_per_k, jnp.real(signals.temperature), reference_m)
        gap = jnp.exp(1j * (center - frame_offset_rad_s) * tau)
        r = 1j * jnp.tanh(strength / samples)
        t = 1 / jnp.cosh(strength / samples)
        f = _waves(signals, signals.b0, getattr(signals, f"f{samples - 1}"))
        for j in range(samples):
            u = signals.a1 if j == 0 else gap * getattr(past, f"f{j - 1}")
            v = signals.a2 if j == samples - 1 else gap * getattr(past, f"b{j + 1}")
            f[f"f{j}"] = getattr(signals, f"f{j}") - (t * u + r * v)
            f[f"b{j}"] = getattr(signals, f"b{j}") - (r * u + t * v)
        current, ft, qt = thermal_terms(
            signals.heater - signals.heater_return,
            signals.temperature,
            heater_resistance_ohm,
            thermal_resistance_k_per_w,
            thermal_tau_s,
        )
        f.update(heater=current, heater_return=-current, temperature=ft)
        return f, {"temperature": qt}

    return SampledBraggGrating


SampledBraggGrating = sampled_bragg_grating()


def sampled_grating_response(
    omega: object,
    temperature: object = 0.0,
    *,
    samples: int = 5,
    spacing_m: float = 187.37028625e-6,
    group_index: float = 4.0,
    strength: float = 0.9,
    center_m: float = 1540e-9,
    shift_m_per_k: float = 0.08e-9,
    reference_m: float = 1540e-9,
) -> tuple:
    """Exact frequency-domain r,t of the same native delayed-patch network."""
    center = mirror_detuning(center_m, shift_m_per_k, temperature, reference_m)
    z = jnp.exp(-1j * (jnp.asarray(omega) - center) * spacing_m * group_index / LIGHT_SPEED)
    r = 1j * jnp.tanh(strength / samples)
    t = 1 / jnp.cosh(strength / samples)
    reflection = r + jnp.zeros_like(z)
    transmission = t + jnp.zeros_like(z)
    for _ in range(samples - 1):
        denominator = 1 - r * z * z * reflection
        transmission = t * z * transmission / denominator
        reflection = r + t * t * z * z * reflection / denominator
    return reflection, transmission


@component(ports=("p1", "p2", "p3", "p4"), states=("a1", "a2", "a3", "a4"), holomorphic=True)
def OpticalDirectionalCoupler(signals: Signals, power_coupling: float = 0.2) -> PhysicsReturn:
    """Reciprocal unitary four-port coupler, suitable for connecting ring loops.

    Forward inputs p1,p2 exit at p3,p4; reverse paths follow reciprocity.
    Cross coupling has +i phase. The physical ring delay belongs in waveguides.
    """
    t = jnp.sqrt(1 - power_coupling)
    k = 1j * jnp.sqrt(power_coupling)
    a = [getattr(signals, f"a{j}") for j in range(1, 5)]
    b = [t * a[2] + k * a[3], k * a[2] + t * a[3], t * a[0] + k * a[1], k * a[0] + t * a[1]]
    return {
        **{f"p{j}": a[j - 1] - b[j - 1] for j in range(1, 5)},
        **{f"a{j}": a[j - 1] + b[j - 1] - getattr(signals, f"p{j}") for j in range(1, 5)},
    }, {}


@source(ports=("p1", "p2"), states=("branch",), holomorphic=True, ac_source_stamp={"branch": 2.0})
def MatchedOpticalPulse(
    signals: Signals, t: float, power_w: float = 1e-12, start_s: float = 10e-12, duration_s: float = 30e-12
) -> PhysicsReturn:
    """Matched coherent pulse source: prescribed incoming wave, absorbs return.

    Compact sin² FIELD, sin⁴ power; exactly zero outside its support. This is
    deterministic startup, not spontaneous emission. p2 is optical reference.
    """
    x = jnp.clip((t - start_s) / duration_s, 0.0, 1.0)
    field = jnp.where((t > start_s) & (t < start_s + duration_s), jnp.sqrt(power_w) * jnp.sin(jnp.pi * x) ** 2, 0.0)
    return {"p1": signals.branch, "p2": -signals.branch, "branch": signals.p1 - signals.p2 - 2 * field - signals.branch}, {}


@component(ports=("p1",), holomorphic=True)
def MatchedOpticalLoad(signals: Signals) -> PhysicsReturn:
    """Absorb a wave at unit reference impedance; no incident optical field."""
    return {"p1": signals.p1}, {}
