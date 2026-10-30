# Copyright (c) 2026 Circulax contributors.
"""Electrical phase shifters promoted from the traveling-wave MZM examples.

Optical states are incident pseudo-voltages; power is |a|²/z0 in watts.
Positive control phase multiplies the envelope by exp(+i phase). Electrical
charge uses the full complex voltage to constrain both electrical quadratures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from circulax.components.base_component import PhysicsReturn, Signals, component

if TYPE_CHECKING:
    from circulax.circuit import Circuit


@component(
    ports=("anode", "cathode", "o1", "o2"),
    port_aliases={"anode": "anode_0", "cathode": "cathode_0"},
    states=("a_forward", "a_reverse", "m_forward", "m_reverse"),
)
def PNJunctionPhaseShifter(  # noqa: PLR0917 -- retain promoted example API
    signals: Signals,
    length: float = 200.0,
    loss_dB_cm: float = 1.0,
    neff0: float = 2.4,
    n_group: float = 4.05,
    wl: float = 1.55,
    a1: float = 1.2,
    a2: float = 0.15,
    bias: float = -1.0,
    Cj0_per_um_fF: float = 0.32,
    z0: float = 1.0,
) -> PhysicsReturn:
    """Delayed quadratic-phase PN junction section.

    Promoted from the original traveling-wave notebook, preserving its parameters.
    Optical ports are o1/o2; anode/cathode provide real electrical control.
    length and wl are in um; Cj0_per_um_fF is junction capacitance in fF/um.
    The capacitance is constant and the optical flight delay is bidirectional.

    Device group:
        Photonic / Electrical and thermal phase control

    Device reference:
        phase_shifters.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.phase_shifters import PNJunctionPhaseShifter

        jax.config.update("jax_enable_x64", True)
        model = PNJunctionPhaseShifter()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.complex128)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(PNJunctionPhaseShifter.__name__, "ports:", model.ports)
        ```

    """
    half_delay = length * n_group / (2 * 2.99792458e14)
    past = signals.at_delay(half_delay)
    carrier_phase = 2 * jnp.pi * neff0 * length / wl
    half_transmission = 10 ** (-loss_dB_cm * length / 10_000 / 40) * jnp.exp(-0.5j * carrier_phase)
    v = (past.anode - past.cathode).real
    delta_phase = length * 1e-3 * (a1 * (v - bias) + a2 * (v**2 - bias**2))
    modulation = jnp.exp(1j * delta_phase)
    b1 = half_transmission * past.m_reverse * modulation
    b2 = half_transmission * past.m_forward * modulation
    q_j = Cj0_per_um_fF * length * 1e-15 * (signals.anode - signals.cathode)
    return {
        "o1": (signals.a_forward - b1) / z0,
        "o2": (signals.a_reverse - b2) / z0,
        "anode": 0.0,
        "cathode": 0.0,
        "a_forward": signals.o1 - signals.a_forward - b1,
        "a_reverse": signals.o2 - signals.a_reverse - b2,
        "m_forward": signals.m_forward - half_transmission * past.a_forward,
        "m_reverse": signals.m_reverse - half_transmission * past.a_reverse,
    }, {"anode": q_j, "cathode": -q_j}


@component(ports=("anode", "cathode", "o1", "o2"), states=("a_forward", "a_reverse", "m_forward", "m_reverse"))
def PNPhaseShifter(  # noqa: PLR0917 -- retain promoted example API
    signals: Signals,
    length_um: float = 200.0,
    capacitance: float = 64e-15,
    a1: float = 1.2,
    a2: float = 0.15,
    bias: float = -1.0,
    loss_db_cm: float = 1.0,
    n_group: float = 4.05,
    neff: float = 2.4,
    wavelength_um: float = 1.55,
) -> PhysicsReturn:
    """The reference notebook's corresponding delayed PN-section API.

    Device group:
        Photonic / Electrical and thermal phase control

    Device reference:
        phase_shifters.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.phase_shifters import PNPhaseShifter

        jax.config.update("jax_enable_x64", True)
        model = PNPhaseShifter()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.complex128)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(PNPhaseShifter.__name__, "ports:", model.ports)
        ```

    """
    past = signals.at_delay(length_um * n_group / (2 * 2.99792458e14))
    half_t = 10 ** (-loss_db_cm * length_um / 10_000 / 40) * jnp.exp(-1j * jnp.pi * neff * length_um / wavelength_um)
    voltage = (past.anode - past.cathode).real
    phase = length_um * 1e-3 * (a1 * (voltage - bias) + a2 * (voltage**2 - bias**2))
    b1 = half_t * past.m_reverse * jnp.exp(1j * phase)
    b2 = half_t * past.m_forward * jnp.exp(1j * phase)
    charge = capacitance * (signals.anode - signals.cathode)
    return {
        "o1": signals.a_forward - b1,
        "o2": signals.a_reverse - b2,
        "a_forward": signals.o1 - signals.a_forward - b1,
        "a_reverse": signals.o2 - signals.a_reverse - b2,
        "m_forward": signals.m_forward - half_t * past.a_forward,
        "m_reverse": signals.m_reverse - half_t * past.a_reverse,
    }, {"anode": charge, "cathode": -charge}


@component(ports=("anode", "cathode", "o1", "o2"), states=("a1", "a2"))
def ElectricalPhaseShifter(  # noqa: PLR0917 -- physical component parameter API
    signals: Signals,
    phase: float = 0.0,
    efficiency: float = 1.0,
    bias: float = 0.0,
    loss_dB: float = 0.0,
    loss_per_v2: float = 0.0,
    z0: float = 1.0,
) -> PhysicsReturn:
    """Ideal control-only reciprocal phase shifter (efficiency rad/V).

    No electrical loading or propagation delay. loss_per_v2 is nonnegative
    excess power attenuation in dB/V² about bias, with baseline loss_dB.
    The real control voltage affects both optical quadratures.

    Voltage-controlled phase (`efficiency` in rad/V) and optional voltage-dependent loss.

    Device group:
        Photonic / Electrical and thermal phase control

    Device reference:
        phase_shifters.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.phase_shifters import ElectricalPhaseShifter

        jax.config.update("jax_enable_x64", True)
        model = ElectricalPhaseShifter()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.complex128)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(ElectricalPhaseShifter.__name__, "ports:", model.ports)
        ```

    """
    voltage = (signals.anode - signals.cathode).real - bias
    transmission = 10 ** (-(loss_dB + loss_per_v2 * voltage**2) / 20) * jnp.exp(1j * (phase + efficiency * voltage))
    b1, b2 = transmission * signals.a2, transmission * signals.a1
    return {
        "anode": 0.0,
        "cathode": 0.0,
        "o1": (signals.a1 - b1) / z0,
        "o2": (signals.a2 - b2) / z0,
        "a1": signals.o1 - signals.a1 - b1,
        "a2": signals.o2 - signals.a2 - b2,
    }, {}


def loaded_phase_shifter(*, resistance: float = 10.0, capacitance: float = 64e-15, **settings: float) -> Circuit:
    """Compile ideal phase control behind a genuine series-R/shunt-C load.

    Ports anode/cathode are external drive terminals; junction exposes the
    internal capacitor voltage. No additional electrical pole is introduced.
    Parameters are SI ohms/farads, plus ElectricalPhaseShifter settings.

    The ideal control core behind actual series resistance and shunt capacitance.

    Device group:
        Photonic / Electrical and thermal phase control

    Device reference:
        phase_shifters.md

    Examples:
        ```python
        from circulax.components.phase_shifters import loaded_phase_shifter

        device = loaded_phase_shifter()
        assert device.sys_size > 0
        print("External ports:", tuple(device.ports))
        ```

    """
    import numpy as np

    from circulax import compile_circuit
    from circulax.components.electronic import Capacitor, Resistor

    if not np.isfinite(resistance) or resistance <= 0 or not np.isfinite(capacitance) or capacitance < 0:
        msg = "resistance must be positive and capacitance nonnegative"
        raise ValueError(msg)
    net = {
        "instances": {
            "R": {"component": "R", "settings": {"R": resistance}},
            "C": {"component": "C", "settings": {"C": capacitance}},
            "PS": {"component": "PS", "settings": settings},
        },
        "connections": {"R,p2": ("PS,anode", "C,p1"), "C,p2": "PS,cathode"},
        "ports": {"anode": "R,p1", "cathode": "PS,cathode", "junction": "C,p1", "o1": "PS,o1", "o2": "PS,o2"},
    }
    return compile_circuit(net, {"R": Resistor, "C": Capacitor, "PS": ElectricalPhaseShifter}, is_complex=True)
