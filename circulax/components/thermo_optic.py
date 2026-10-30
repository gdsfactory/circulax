# Copyright (c) 2026 Circulax contributors
"""Lumped Joule-heated thermo-optic phase shifter and explicit thermal network.

These SI constitutive equations implement Ohm's law, thermal energy balance,
and linear thermo-optic phase tuning; no external Verilog-A implementation is
used. Thermal node voltage is temperature rise in K and inward flow is W.
"""

# ruff: noqa: ANN001, ANN201
import jax.numpy as jnp

from circulax.components.base_component import component


@component(ports=("hp", "hn", "dt"), holomorphic=False)
def JouleHeater(signals, resistance=100.0, alpha=0.0, ambient_temperature_K=300.0, imaginary_thermal_conductance=1e-3):
    """Ohmic heater; resistance is specified at ambient temperature in kelvin.

    ``alpha`` is fractional resistance change per K. Physical heat uses the
    real drive voltage, while full complex conduction restores imaginary
    electrical voltages. The imaginary thermal term constrains the auxiliary
    complex representation and injects no physical heat.

    Ohmic electrical conduction and local Joule heat, optionally with resistance versus temperature.

    Device group:
        Photonic / Electrical and thermal phase control

    Device reference:
        thermo_optic.md

    Examples:
        ```python
        import numpy as np

        from circulax.components.thermo_optic import JouleHeater

        f, q = JouleHeater(resistance=100.0)(hp=1.0, hn=0.0, dt=0.0)
        np.testing.assert_allclose(f["hp"], 0.01)
        np.testing.assert_allclose(-f["dt"], 0.01)
        print("Heater current: 10 mA; generated heat: 10 mW")
        ```

    """
    temperature = ambient_temperature_K + jnp.real(signals.dt)
    r = resistance * (1 + alpha * (temperature - ambient_temperature_K))
    voltage = signals.hp - signals.hn
    current = voltage / r
    power = jnp.real(voltage) ** 2 / r
    return {"hp": current, "hn": -current, "dt": -power + 1j * imaginary_thermal_conductance * jnp.imag(signals.dt)}, {}


@component(ports=("o1", "o2", "dt"), states=("a1", "a2"), holomorphic=False)
def ThermalPhaseShifter(signals, efficiency=0.01, phase=0.0, loss_dB=0.0, z0=1.0):
    """Matched reciprocal optical core with phase efficiency in rad/K.

    Positive phase multiplies the field by exp(+i*phase). Incident internal
    states are pseudo-voltages; their power is |a|²/z0. There is no propagation
    delay or optical absorption heating. The dt port must connect to a thermal
    network or prescribed temperature-rise boundary.

    Optical phase controlled by local temperature, with efficiency in rad/K.

    Device group:
        Photonic / Electrical and thermal phase control

    Device reference:
        thermo_optic.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.thermo_optic import ThermalPhaseShifter

        jax.config.update("jax_enable_x64", True)
        model = ThermalPhaseShifter()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.complex128)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(ThermalPhaseShifter.__name__, "ports:", model.ports)
        ```

    """
    t = 10 ** (-loss_dB / 20) * jnp.exp(1j * (phase + efficiency * jnp.real(signals.dt)))
    b1, b2 = t * signals.a2, t * signals.a1
    return {
        "o1": (signals.a1 - b1) / z0,
        "o2": (signals.a2 - b2) / z0,
        "a1": signals.o1 - signals.a1 - b1,
        "a2": signals.o2 - signals.a2 - b2,
        "dt": 0.0,
    }, {}


def thermo_optic_phase_shifter(
    *,
    resistance=100.0,
    alpha=0.0,
    ambient_temperature_K=300.0,
    thermal_resistance=1e3,
    thermal_capacitance=1e-6,
    efficiency=0.01,
    phase=0.0,
    loss_dB=0.0,
    z0=1.0,
):
    """Return a reusable Circuit with heater, optical core and thermal R/C.

    Ports are hp, hn, o1, o2, dt and thermal_reference. The exposed dt measurement/control node is
    allocated explicitly. Ground thermal_reference to define zero temperature
    rise at ambient. Positive Rth/Cth are required for the claimed dynamics.
    Parameters are setup constants; the flattened heater/optical/Rth/Cth instance
    parameters can subsequently be updated through the ordinary Circuit API.

    Heater, optical phase core and a real thermal R/C network.

    Device group:
        Photonic / Electrical and thermal phase control

    Device reference:
        thermo_optic.md

    Examples:
        ```python
        from circulax.components.thermo_optic import thermo_optic_phase_shifter

        device = thermo_optic_phase_shifter()
        assert device.sys_size > 0
        print("External ports:", tuple(device.ports))
        ```

    """
    from circulax import compile_circuit
    from circulax.components.electronic import Capacitor, Resistor

    if resistance <= 0 or thermal_resistance <= 0 or thermal_capacitance <= 0 or z0 <= 0 or ambient_temperature_K <= 0:
        msg = "resistance, Rth, Cth, z0 and ambient_temperature_K must be positive"
        raise ValueError(msg)
    if loss_dB < 0:
        msg = "loss_dB must be nonnegative for a passive phase shifter"
        raise ValueError(msg)
    net = {
        "instances": {
            "HEATER": {
                "component": "thermo_heater",
                "settings": {"resistance": resistance, "alpha": alpha, "ambient_temperature_K": ambient_temperature_K},
            },
            "OPTICAL": {
                "component": "thermo_optical",
                "settings": {"efficiency": efficiency, "phase": phase, "loss_dB": loss_dB, "z0": z0},
            },
            "Rth": {"component": "thermo_resistor", "settings": {"R": thermal_resistance}},
            "Cth": {"component": "thermo_capacitor", "settings": {"C": thermal_capacitance}},
        },
        "connections": {},
        "nets": [
            {"p1": "HEATER,dt", "p2": "OPTICAL,dt"},
            {"p1": "HEATER,dt", "p2": "Rth,p1"},
            {"p1": "HEATER,dt", "p2": "Cth,p1"},
            {"p1": "Rth,p2", "p2": "Cth,p2"},
        ],
        "ports": {
            "hp": "HEATER,hp",
            "hn": "HEATER,hn",
            "o1": "OPTICAL,o1",
            "o2": "OPTICAL,o2",
            "dt": "HEATER,dt",
            "thermal_reference": "Rth,p2",
        },
    }
    return compile_circuit(
        net,
        {
            "thermo_heater": JouleHeater,
            "thermo_optical": ThermalPhaseShifter,
            "thermo_resistor": Resistor,
            "thermo_capacitor": Capacitor,
        },
        is_complex=True,
        backend="dense",
    )
