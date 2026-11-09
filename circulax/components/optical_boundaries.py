# Copyright 2026 Circulax contributors

"""Explicit single-mode launched-wave boundaries, independent of returning light."""

from collections.abc import Callable

import jax

from circulax.components.base_component import CircuitComponent, PhysicsReturn, Signals, component, source
from circulax.power_waves import validate_reference


def optical_wave_source(*, r0: float = 1.0, waveform: Callable[[jax.Array], jax.Array] | None = None) -> type[CircuitComponent]:
    """Build a matched source class with static reference and optional envelope.

    The one terminal ``p`` faces the circuit. Relative to the source itself,
    ``b = amplitude * waveform(t)`` is emitted and ``a`` is returning light.
    Thus F_p = potential/r0 - 2*b/sqrt(r0), Q_p = 0. This reduction follows
    solely from the source's own equation b=prescribed, for arbitrary a.

    ``amplitude`` is complex, in sqrt(W), and defaults to 1. The optional
    JAX-compatible waveform returns a dimensionless complex multiplier;
    omitted means CW. Like existing @source models, DC evaluates it at t=0,
    transient and HB at their sample times. AC excitation metadata instead
    specifies an independent emitted-wave perturbation in sqrt(W), without
    multiplying by the DC amplitude or waveform. No optical ground is needed.

    Use the same finite positive real r0, carrier frame, and single mode as
    the attached wave component. Connect one boundary to one device port;
    use an explicit splitter for branching.
    """
    r0 = validate_reference(r0)
    root = r0**0.5

    @source(ports=("p",), amplitude_param="amplitude", holomorphic=True, ac_source_stamp={"p": 2 / root})
    def OpticalWaveSource(signals: Signals, t: float, amplitude: complex = 1.0 + 0j) -> PhysicsReturn:
        """Matched emitter; amplitude is the launched envelope in sqrt(W)."""
        emitted = amplitude if waveform is None else amplitude * waveform(t)
        return {"p": signals.p / r0 - 2 * emitted / root}, {}

    OpticalWaveSource._wave_r0 = r0  # noqa: SLF001 -- compiler wave-reference metadata
    return OpticalWaveSource


def optical_wave_termination(*, r0: float = 1.0) -> type[CircuitComponent]:
    """Build a matched absorber class, with a single circuit-facing terminal p.

    Its own outgoing wave is identically zero for arbitrary incoming light:
    F_p = potential/r0, Q_p = 0. No auxiliary state or ground is required.
    Reference, mode, carrier, and connection assumptions match the source.
    """
    r0 = validate_reference(r0)

    @component(ports=("p",), holomorphic=True)
    def OpticalWaveTermination(signals: Signals) -> PhysicsReturn:
        """Matched absorber of arbitrary incident light, emitting no wave."""
        return {"p": signals.p / r0 + 0j}, {}

    OpticalWaveTermination._wave_r0 = r0  # noqa: SLF001 -- compiler wave-reference metadata
    return OpticalWaveTermination


OpticalWaveSource = optical_wave_source()
OpticalWaveTermination = optical_wave_termination()
