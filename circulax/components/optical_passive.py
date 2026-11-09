# Copyright (c) 2026 Circulax contributors.
# ruff: noqa: PLR0917 -- component parameter signatures follow the public decorator API.
"""Power-normalized reciprocal optical primitives and delayed waveguide.

See docs/components/photonic_conventions.md. Optical ports are single ended;
``z0`` is a real positive reference impedance. No carrier oscillation is resolved.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from circulax.components.base_component import PhysicsReturn, Signals, component
from circulax.disciplines import optical, optical_potential, wave_amplitude


def _wave_stamp(
    signals: Signals, ports: tuple[str, ...], states: tuple[str, ...], scattering: jax.Array, z0: float = 1.0
) -> PhysicsReturn:
    """Exact b=S a constraints, including singular ideal S-to-Y endpoints."""
    z0 = eqx.error_if(jnp.asarray(z0), (z0 <= 0) | ~jnp.isfinite(z0), "Optical reference impedance z0 must be finite and positive")
    a = jnp.array([getattr(signals, name) for name in states])
    b = scattering @ a
    root_z = jnp.sqrt(z0)
    result = {p: (a[k] - b[k]) / root_z for k, p in enumerate(ports)}
    result.update({s: getattr(signals, ports[k]) - root_z * (a[k] + b[k]) for k, s in enumerate(states)})
    return result, {}


@component(
    discipline=optical,
    state_natures={"a1": wave_amplitude, "a2": wave_amplitude},
    equation_natures={"a1": optical_potential, "a2": optical_potential},
    ports=("p1", "p2"),
    states=("a1", "a2"),
    holomorphic=True,
)
def OpticalAttenuator(signals: Signals, loss_dB: float = 0.0, z0: float = 1.0) -> PhysicsReturn:
    """Matched reciprocal attenuator.

    Connect p1/p2; loss_dB is nonnegative power loss in dB, and z0 is a positive real reference impedance.

    Device group:
        Photonic / Passive routing and phase

    Device reference:
        optical_passive.md

    Examples:
        ```python
        import numpy as np

        from circulax.components.optical_passive import OpticalAttenuator

        transmission = 10 ** (-3 / 20)
        # Matched through excitation: incident a1=1 sqrt(W), a2=0.
        f, q = OpticalAttenuator(loss_dB=3.0)(p1=1.0, p2=transmission, a1=1.0, a2=0.0)
        assert abs(f["a1"]) < 1e-12
        assert abs(f["a2"]) < 1e-12
        np.testing.assert_allclose(transmission**2, 10 ** (-3 / 10))
        print("Output power for 1 W incident:", transmission**2, "W")
        ```

    """
    loss_dB = eqx.error_if(
        jnp.asarray(loss_dB), (loss_dB < 0) | ~jnp.isfinite(loss_dB), "Passive loss_dB must be finite and nonnegative"
    )
    t = 10.0 ** (-loss_dB / 20.0)
    return _wave_stamp(signals, ("p1", "p2"), ("a1", "a2"), jnp.array([[0, t], [t, 0]]), z0)


@component(
    discipline=optical,
    state_natures={"a1": wave_amplitude, "a2": wave_amplitude},
    equation_natures={"a1": optical_potential, "a2": optical_potential},
    ports=("p1", "p2"),
    states=("a1", "a2"),
    holomorphic=True,
)
def OpticalPhaseShifter(signals: Signals, phase: float = 0.0, loss_dB: float = 0.0, z0: float = 1.0) -> PhysicsReturn:
    """Matched reciprocal phase bias, transmission exp(+i phase-loss_dB ln(10)/20).

    Phase is radians, matching the example-local PhaseBias sign convention.
    This static bias has no group delay, electrical loading or thermal dynamics.

    Static reciprocal phase bias. Connect p1/p2; phase is in radians and transmission is exp(+i*phase), with
    optional loss_dB. It adds no propagation delay.

    Device group:
        Photonic / Passive routing and phase

    Device reference:
        optical_passive.md

    Examples:
        ```python
        import numpy as np

        from circulax.components.optical_passive import OpticalPhaseShifter

        transmission = np.exp(0.7j)
        f, q = OpticalPhaseShifter(phase=0.7)(p1=1.0, p2=transmission, a1=1.0, a2=0.0)
        assert abs(f["a1"]) < 1e-12
        assert abs(f["a2"]) < 1e-12
        np.testing.assert_allclose(abs(transmission) ** 2, 1.0)
        print("Phase shift:", np.angle(transmission), "rad")
        ```

    """
    loss_dB = eqx.error_if(
        jnp.asarray(loss_dB), (loss_dB < 0) | ~jnp.isfinite(loss_dB), "Passive loss_dB must be finite and nonnegative"
    )
    t = 10.0 ** (-loss_dB / 20.0) * jnp.exp(1j * phase)
    return _wave_stamp(signals, ("p1", "p2"), ("a1", "a2"), jnp.array([[0, t], [t, 0]]), z0)


@component(discipline=optical, ports=("p1",), holomorphic=True)
def OpticalTermination(signals: Signals, z0: float = 1.0) -> PhysicsReturn:
    """Matched absorbing one-port termination.

    Connect p1 to an unused optical node; z0 matches its positive real optical reference impedance.

    Device group:
        Photonic / Passive routing and phase

    Device reference:
        optical_passive.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.optical_passive import OpticalTermination

        jax.config.update("jax_enable_x64", True)
        model = OpticalTermination()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.complex128)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(OpticalTermination.__name__, "ports:", model.ports)
        ```

    """
    z0 = eqx.error_if(jnp.asarray(z0), (z0 <= 0) | ~jnp.isfinite(z0), "Optical reference impedance z0 must be finite and positive")
    return {"p1": signals.p1 / z0}, {}


@component(
    discipline=optical,
    state_natures={"a1": wave_amplitude, "a2": wave_amplitude},
    equation_natures={"a1": optical_potential, "a2": optical_potential},
    ports=("p1", "p2"),
    states=("a1", "a2"),
    holomorphic=True,
)
def OpticalInterface(
    signals: Signals,
    reflectivity: float = 0.0,
    loss_dB: float = 0.0,
    phase: float = 0.0,
    reflection_phase: float = 0.0,
    z0: float = 1.0,
) -> PhysicsReturn:
    """Reciprocal partially reflecting interface with coherent reflection phases.

    S=exp(i phase)*10**(-loss_dB/20)*[[sqrt(R)*exp(i delta), sqrt(1-R)],
    [sqrt(1-R), -sqrt(R)*exp(-i delta)]]. This is unitary before uniform
    loss for 0<=R<=1; the opposite reflection phases are essential for power
    conservation under simultaneous excitation. delta=reflection_phase radians.
    Full transmission/reflection endpoints are exact; no hidden leakage is added.

    Reciprocal partial reflector with coherent phases. Connect p1/p2; exact full transmission and reflection
    endpoints are supported.

    Device group:
        Photonic / Passive routing and phase

    Device reference:
        optical_passive.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.optical_passive import OpticalInterface

        jax.config.update("jax_enable_x64", True)
        model = OpticalInterface()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.complex128)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(OpticalInterface.__name__, "ports:", model.ports)
        ```

    """
    invalid = (
        (reflectivity < 0) | (reflectivity > 1) | (loss_dB < 0) | ~jnp.isfinite(reflectivity + loss_dB + phase + reflection_phase)
    )
    reflectivity = eqx.error_if(
        jnp.asarray(reflectivity), invalid, "Interface requires reflectivity in [0,1], nonnegative loss and finite phases"
    )
    r, t = jnp.sqrt(reflectivity), jnp.sqrt(1 - reflectivity)
    scale = 10.0 ** (-loss_dB / 20.0) * jnp.exp(1j * phase)
    scattering = scale * jnp.array([[r * jnp.exp(1j * reflection_phase), t], [t, -r * jnp.exp(-1j * reflection_phase)]])
    return _wave_stamp(signals, ("p1", "p2"), ("a1", "a2"), scattering, z0)


@component(
    discipline=optical,
    state_natures={"a1": wave_amplitude, "a2": wave_amplitude},
    equation_natures={"a1": optical_potential, "a2": optical_potential},
    ports=("p1", "p2"),
    states=("a1", "a2"),
)
def NonlinearOpticalWaveguide(
    signals: Signals,
    length_um: float = 100.0,
    loss_dB_cm: float = 1.0,
    tpa_coeff: float = 0.0,
    neff: float = 2.4,
    n_group: float = 4.0,
    wavelength_nm: float = 1310.0,
    z0: float = 1.0,
) -> PhysicsReturn:
    """Promoted bidirectional delayed waveguide with linear loss and TPA.

    From examples/photonic/photonics_transient.ipynb. ``a1/a2`` retain the
    notebook's incident pseudo-voltage convention: incident power = |a|²/z0.
    ``tpa_coeff`` is in 1/(W cm). Counterpropagating beams are attenuated
    independently (no cross-TPA, Kerr, carrier or thermal coupling).
    Nonnegative length/loss/TPA and group index, positive wavelength/z0 required.
    Carrier propagation exp(-i beta L) and group delay are each applied once.

    Bidirectional delayed waveguide with independent directional two-photon absorption. Length is in um and linear
    loss in dB/cm. No cross-TPA, Kerr or carrier-heating physics.

    Device group:
        Photonic / Optical sources and propagation

    Device reference:
        optical_passive.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.optical_passive import NonlinearOpticalWaveguide

        jax.config.update("jax_enable_x64", True)
        model = NonlinearOpticalWaveguide()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.complex128)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(NonlinearOpticalWaveguide.__name__, "ports:", model.ports)
        ```

    """
    invalid = (length_um < 0) | (loss_dB_cm < 0) | (tpa_coeff < 0) | (n_group < 0) | (wavelength_nm <= 0) | (z0 <= 0)
    invalid = invalid | ~jnp.isfinite(length_um + loss_dB_cm + tpa_coeff + neff + n_group + wavelength_nm + z0)
    length_um = eqx.error_if(
        jnp.asarray(length_um), invalid, "Waveguide requires finite nonnegative length/loss/TPA/group index, positive wavelength/z0"
    )
    length_cm = length_um / 10_000.0
    alpha_cm = loss_dB_cm * jnp.log(10.0) / 10.0
    # Analytic zero-loss limit with a safe denominator, retaining finite AD.
    x = alpha_cm * length_cm
    safe_x = jnp.where(jnp.abs(x) > 1e-6, x, 1.0)
    ratio = jnp.where(jnp.abs(x) > 1e-6, -jnp.expm1(-x) / safe_x, 1 - x / 2 + x**2 / 6 - x**3 / 24)
    effective_length = length_cm * ratio
    linear_amplitude = 10.0 ** (-loss_dB_cm * length_cm / 20.0)
    phase = 2.0 * jnp.pi * neff * length_um * 1000.0 / wavelength_nm
    delay = length_um * n_group / 2.99792458e14
    past = signals.at_delay(delay)

    def propagate(incident: jax.Array) -> jax.Array:
        tpa_amplitude = 1.0 / jnp.sqrt(1.0 + tpa_coeff * jnp.abs(incident) ** 2 / z0 * effective_length)
        return linear_amplitude * tpa_amplitude * jnp.exp(-1j * phase) * incident

    b1, b2 = propagate(past.a2), propagate(past.a1)
    return {
        "p1": (signals.a1 - b1) / z0,
        "p2": (signals.a2 - b2) / z0,
        "a1": signals.p1 - signals.a1 - b1,
        "a2": signals.p2 - signals.a2 - b2,
    }, {}
