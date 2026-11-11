# Copyright (c) 2026 GDSFactory
"""Electrically pumped, bidirectional delayed-gain laser circuit component.

Fields are power-normalized (sqrt(W)); reservoirs store integrated modal gain.
Cell flight times are native Circulax inline delays. The optical gain/carrier
law and diode conventions match the SOA. Facets are internal to the component;
left/right are outgoing field outputs. Seed is a transmitted coherent
envelope added at the left inner boundary.

For a reconfigurable cavity, use hybrid_laser_netlist and the separate
bidirectional gain, interface, phase and reflector sections in laser_sections.
This integrated-facet model remains useful as a restricted reference case.
"""

from functools import lru_cache
from typing import TYPE_CHECKING, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from circulax.components.base_component import CircuitComponent, PhysicsReturn, Signals, component, source
from circulax.components.laser_mirrors import MirrorParameters, mirror_detuning, mirror_terms, reflection_response, thermal_terms
from circulax.components.soa import ELECTRON_CHARGE, LIGHT_SPEED, soa_scales, stimulated_power, unsaturated_gain

if TYPE_CHECKING:
    from circulax import Circuit


class LaserParameters(NamedTuple):
    """Illustrative material/cavity settings in SI units, for netlist settings."""

    length_m: float = 600e-6
    width_m: float = 2e-6
    thickness_m: float = 0.1e-6
    group_index: float = 3.6
    lifetime_s: float = 280e-12
    confinement: float = 0.3
    differential_gain_m2: float = 6e-20
    transparency_m3: float = 5.5e23
    injection_efficiency: float = 0.8
    wavelength_m: float = 1540e-9
    alpha_h: float = 3.5
    loss_m1: float = 1000.0
    r_left: complex = 0.9 + 0j
    r_right: complex = 0.7 + 0j
    detuning_rad_s: float = 0.0
    saturation: float = 1.0


def material_scales(p: LaserParameters) -> tuple:
    """Return active volume (m³), modal gain slope (m²), saturation energy (J)."""
    return soa_scales(
        length_um=p.length_m * 1e6,
        width_um=p.width_m * 1e6,
        thickness_um=p.thickness_m * 1e6,
        confinement=p.confinement,
        differential_gain_m2=p.differential_gain_m2,
        wavelength_nm=p.wavelength_m * 1e9,
    )


def threshold(p: LaserParameters) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Uniform dark-state threshold: gain (1/m), carrier density (1/m³), current (A)."""
    gain = p.loss_m1 - jnp.log(jnp.abs(p.r_left * p.r_right)) / p.length_m
    density = p.transparency_m3 + gain / (p.confinement * p.differential_gain_m2)
    current = ELECTRON_CHARGE * material_scales(p)[0] * density / (p.injection_efficiency * p.lifetime_s)
    return gain, density, current


def resonant_detuning(p: LaserParameters) -> jax.Array:
    """Propagation calibration (rad/s) making the nominal threshold mode stationary."""
    gain, _, _ = threshold(p)
    return LIGHT_SPEED / p.group_index * (p.alpha_h * gain / 2 - jnp.angle(p.r_left * p.r_right) / (2 * p.length_m))


@lru_cache(maxsize=16)
def traveling_wave_laser(segments: int = 8, *, tunable_mirrors: bool = False) -> type[CircuitComponent]:
    """Return a native Circulax component class with a static segment count.

    Ports: anode/cathode (V/A), seed/left/right (sqrt(W)). Seed is added
    inside the left facet; left reports intracavity outcoupling only, not
    the externally reflected part of an incident seed beam. Optical outputs
    impose outgoing fields independent of their numerical load; they are not
    bidirectional scattering ports. Facets and all cavity feedback are inside
    the component. With tunable_mirrors=True, add heater_phase/filter/return
    electrical ports, two temperature-rise states, and a causal filtered right
    reflector. A physical external cavity would require separate incoming
    and outgoing ports and is outside this model's contract.

    States f_j and b_j are fields arriving at each cell's right/left edge;
    h_j is the local integrated modal gain. Both incoming directions see the
    same h_j and deplete that reservoir. Transmission uses the gain at arrival
    time, uniform along a cell: a first-order retarded-gain approximation that
    converges as flight time per cell decreases. Constant-gain transport and
    round-trip phase/loss are exact, apart from solver/history interpolation.
    """
    if not isinstance(segments, int) or segments < 1:
        msg = "segments must be a positive integer"
        raise ValueError(msg)
    states = ("v_j", "i_left", "i_right", *(f"{kind}{j}" for j in range(segments) for kind in ("f", "b", "h")))

    ports = ("anode", "cathode", "seed", "left", "right")
    if tunable_mirrors:
        ports += ("heater_phase", "heater_filter", "heater_return")
        states += ("temperature_phase", "temperature_filter", "mirror_right")

    @component(ports=ports, states=states, holomorphic=False)
    def TravelingWaveLaser(
        signals: Signals,
        length_m: float = 600e-6,
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
        r_left: complex = 0.9 + 0j,
        r_right: complex = 0.7 + 0j,
        detuning_rad_s: float = 0.0,
        saturation: float = 1.0,
        frame_offset_rad_s: float = 0.0,
        mirror_bandwidth_hz: float = 12e9,
        mirror_center_m: float = 1540e-9,
        mirror_shift_m_per_k: float = 0.08e-9,
        phase_shift_rad_per_k: float = -0.6,
        heater_resistance_ohm: float = 100.0,
        thermal_resistance_k_per_w: float = 200.0,
        thermal_tau_s: float = 1e-6,
        saturation_current_a: float = 1e-12,
        ideality: float = 1.8,
        thermal_voltage_v: float = 0.02585,
        resistance_ohm: float = 5.0,
        capacitance_f: float = 150e-15,
    ) -> PhysicsReturn:
        """Diode injection and segmented delayed bidirectional optical propagation."""
        contact_scale = length_m * width_m / (600e-6 * 2e-6)
        diode = saturation_current_a * contact_scale * jnp.expm1(signals.v_j / (ideality * thermal_voltage_v))
        terminal = (signals.anode - signals.cathode - signals.v_j) * contact_scale / resistance_ohm
        h0 = (
            unsaturated_gain(
                jnp.real(diode),
                lifetime_s=lifetime_s,
                length_um=length_m * 1e6,
                width_um=width_m * 1e6,
                thickness_um=thickness_m * 1e6,
                confinement=confinement,
                differential_gain_m2=differential_gain_m2,
                transparency_m3=transparency_m3,
                injection_efficiency=injection_efficiency,
            )
            / segments
        )
        energy = soa_scales(
            length_um=length_m * 1e6,
            width_um=width_m * 1e6,
            thickness_um=thickness_m * 1e6,
            confinement=confinement,
            differential_gain_m2=differential_gain_m2,
            wavelength_nm=wavelength_m * 1e9,
        )[2]
        tau = length_m * group_index / (LIGHT_SPEED * segments)
        past = signals.at_delay(tau)
        loss = loss_m1 * length_m / segments
        f = {
            "anode": terminal,
            "cathode": -terminal,
            "seed": 0.0,
            "v_j": (diode - terminal) / 0.1,
            "left": signals.i_left,
            "right": signals.i_right,
            "i_left": signals.left - jnp.sqrt(1 - jnp.abs(r_left) ** 2) * signals.b0,
            "i_right": signals.right - jnp.sqrt(1 - jnp.abs(r_right) ** 2) * getattr(signals, f"f{segments - 1}"),
        }
        q = {"v_j": capacitance_f * contact_scale * signals.v_j / 0.1}
        left_reflection = r_left
        if tunable_mirrors:
            filter_t = jnp.real(signals.temperature_filter)
            left_reflection = r_left * jnp.exp(1j * phase_shift_rad_per_k * jnp.real(past.temperature_phase))
            f["heater_return"] = 0.0
            for kind in ("phase", "filter"):
                voltage = getattr(signals, f"heater_{kind}") - signals.heater_return
                current, ft, qt = thermal_terms(
                    voltage,
                    getattr(signals, f"temperature_{kind}"),
                    heater_resistance_ohm,
                    thermal_resistance_k_per_w,
                    thermal_tau_s,
                )
                f[f"heater_{kind}"] = current
                f["heater_return"] -= current
                f[f"temperature_{kind}"] = ft
                q[f"temperature_{kind}"] = qt
            center = mirror_detuning(mirror_center_m, mirror_shift_m_per_k, filter_t, wavelength_m)
            f["mirror_right"], q["mirror_right"] = mirror_terms(
                getattr(signals, f"f{segments - 1}"),
                signals.mirror_right,
                r_right,
                2 * jnp.pi * mirror_bandwidth_hz,
                center - frame_offset_rad_s,
            )
        for j in range(segments):
            forward_in = left_reflection * past.b0 + past.seed if j == 0 else getattr(past, f"f{j - 1}")
            backward_in = (
                (past.mirror_right if tunable_mirrors else r_right * getattr(past, f"f{segments - 1}"))
                if j == segments - 1
                else getattr(past, f"b{j + 1}")
            )
            h = getattr(signals, f"h{j}")
            gain = jnp.real(h)
            transmission = jnp.exp(((1 - 1j * alpha_h) * gain - loss) / 2 + 1j * (detuning_rad_s - frame_offset_rad_s) * tau)
            pin = jnp.abs(forward_in) ** 2 + jnp.abs(backward_in) ** 2
            f[f"f{j}"] = getattr(signals, f"f{j}") - transmission * forward_in
            f[f"b{j}"] = getattr(signals, f"b{j}") - transmission * backward_in
            f[f"h{j}"] = h - h0 + saturation * lifetime_s * stimulated_power(gain, loss, pin) / energy
            q[f"h{j}"] = lifetime_s * h
        return f, q

    return TravelingWaveLaser


TravelingWaveLaser = traveling_wave_laser()
TunableTravelingWaveLaser = traveling_wave_laser(tunable_mirrors=True)


@source(ports=("out",), states=("branch",), holomorphic=True)
def CoherentSeed(
    signals: Signals,
    t: float,
    power_w: float = 1e-12,
    start_s: float = 20e-12,
    duration_s: float = 40e-12,
) -> PhysicsReturn:
    """Single compact sin² field pulse, deterministic; power_w is peak power (W)."""
    phase = jnp.clip((t - start_s) / duration_s, 0.0, 1.0)
    field = jnp.where((t > start_s) & (t < start_s + duration_s), jnp.sqrt(power_w) * jnp.sin(jnp.pi * phase) ** 2, 0.0)
    return {"out": signals.branch, "branch": signals.out - field}, {}


def carrier_density(p: LaserParameters, integrated_gain: ArrayLike, segments: int) -> jax.Array:
    """Convert cell integrated gain to local carrier density (m⁻³)."""
    return p.transparency_m3 + jnp.real(integrated_gain) * segments / (p.length_m * p.confinement * p.differential_gain_m2)


def seeded_initial_state(
    circuit: "Circuit",
    dc_state: jax.Array,
    p: LaserParameters,
    segments: int = 8,
    *,
    instance: str = "LASER",
    seed_power_w: ArrayLike = 1e-12,
    mirrors: MirrorParameters | None = None,
    frame_offset_rad_s: ArrayLike = 0.0,
) -> jax.Array:
    """Prepare a coherent threshold-mode initial history on a circuit DC state.

    Retain electrical bias, set every reservoir to threshold, and initialize
    both optical directions consistently with the facets. The native solver
    extends this state constantly before t=0. This is a specified coherent
    initial condition, not spontaneous emission. The seed input must be zero
    during the subsequent free-running simulation. The principal mode assumes
    detuning_rad_s=resonant_detuning(p). For filtered mirrors, mirrors must
    match the component settings; segments and frame_offset_rad_s must match
    its topology/frame. The supplied frame must be an actual longitudinal
    phase root, not just a desired wavelength. This condition is checked for
    nonzero seeds. dc_state uses the circuit's standard
    real/imaginary block representation.
    """
    gain, _, _ = threshold(p)
    rl, rr = p.r_left, p.r_right
    if mirrors is not None:
        from circulax.components.laser_modes import spectral_threshold

        tp = jnp.real(circuit.port(dc_state, f"{instance},temperature_phase"))
        tf = jnp.real(circuit.port(dc_state, f"{instance},temperature_filter"))
        condition = spectral_threshold(p, mirrors, frame_offset_rad_s, tp, tf)
        gain = condition.gain_m1
        phase_error = jnp.angle(jnp.exp(1j * condition.phase_rad))
        dc_state = eqx.error_if(
            dc_state,
            (jnp.abs(phase_error) > 1e-6) & (jnp.asarray(seed_power_w) > 0),
            "Filtered laser initial frame must satisfy a longitudinal phase root",
        )
        rl = p.r_left * jnp.exp(1j * mirrors.phase_shift_rad_per_k * tp)
        center = mirror_detuning(mirrors.mirror_center_m, mirrors.mirror_shift_m_per_k, tf, p.wavelength_m)
        rr = reflection_response(p.r_right, mirrors.mirror_bandwidth_hz, frame_offset_rad_s, center)
    dz = p.length_m / segments
    kappa = (gain - p.loss_m1) / 2 + 1j * (
        (p.detuning_rad_s - frame_offset_rad_s) * p.group_index / LIGHT_SPEED - p.alpha_h * gain / 2
    )
    amplitude = jnp.sqrt(seed_power_w)
    values = {
        "left": jnp.sqrt(1 - abs(p.r_left) ** 2) * amplitude / rl,
        "right": jnp.sqrt(1 - abs(p.r_right) ** 2) * amplitude * jnp.exp(kappa * p.length_m),
        "i_left": 0.0,
        "i_right": 0.0,
    }
    for j in range(segments):
        values[f"f{j}"] = amplitude * jnp.exp(kappa * (j + 1) * dz)
        values[f"b{j}"] = amplitude / rl * jnp.exp(-kappa * j * dz)
        values[f"h{j}"] = gain * dz
    if mirrors is not None:
        values["mirror_right"] = rr * amplitude * jnp.exp(kappa * p.length_m)
    result = dc_state
    for name, value in values.items():
        index = circuit.port_map[f"{instance},{name}"]
        result = result.at[index].set(jnp.real(value)).at[index + circuit.sys_size].set(jnp.imag(value))
    return result
