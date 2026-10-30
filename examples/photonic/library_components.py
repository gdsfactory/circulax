# Copyright (c) 2026 Circulax contributors
# ruff: noqa: INP001
"""Small runnable passive, phase, filter, ring and thermal library examples.

Run: python examples/photonic/library_components.py
Detector/MZM example: library_link.py; distributed example: library_traveling_wave_mzm.py.
Every optical source below is matched with incident power in watts at z0=1.
"""

import jax
import jax.numpy as jnp
import numpy as np

from circulax import Circuit, attach_testbench, compile_circuit, quadrature_response
from circulax.components import (
    AllPassRing,
    OpticalAttenuator,
    OpticalCW,
    OpticalFilter,
    OpticalPhaseShifter,
    thermo_optic_phase_shifter,
)
from circulax.components.electronic import Resistor, VoltageSource


def matched_device(model: object, settings: dict) -> Circuit:
    """Terminate a two-port optical model with a matched source and load."""
    net = {"instances": {"D": {"component": "D", "settings": settings}}, "ports": {"in": "D,p1", "out": "D,p2"}}
    net = attach_testbench(
        net,
        sources={"in": {"name": "LIGHT", "component": "LIGHT", "settings": {"power": 1e-3}}},
        loads={"out": {"name": "LOAD", "component": "R", "settings": {"R": 1.0}}},
    )
    return compile_circuit(net, {"D": model, "LIGHT": OpticalCW, "R": Resistor}, is_complex=True)


def main() -> None:
    """Compare observable DC fields and dynamic filter transfer with formulas."""
    jax.config.update("jax_enable_x64", True)  # noqa: FBT003
    examples = (
        (OpticalAttenuator, {"loss_dB": 3.0}, 10 ** (-3 / 20)),
        (OpticalPhaseShifter, {"phase": 0.4}, np.exp(0.4j)),
        (AllPassRing, {"gamma_i": 1e9, "gamma_e": 2e9, "detuning": 1e9}, 1 - 4e9 / (3e9 - 1e9j)),
        (OpticalFilter, {"bandwidth_hz": 1e9}, 1.0),
    )
    for model, settings, transfer in examples:
        circuit = matched_device(model, settings)
        dc = circuit.dc()
        actual = circuit.port(dc, "out")
        np.testing.assert_allclose(actual, np.sqrt(1e-3) * transfer, atol=1e-10)
        print(f"{model.__name__}: {circuit.sys_size} complex unknowns, field {complex(actual):.6g}")  # noqa: T201
        if model is OpticalFilter:
            f = jnp.array([1e8, 1e9, 1e10])
            ac = quadrature_response(circuit, freqs=f, parameter_drives={"LIGHT.power": 1e-4}, y_dc=dc)
            observed = ac["in_phase"][:, circuit.port_map["out"]]
            expected = 1e-4 / (2 * np.sqrt(1e-3)) / (1 + 1j * np.asarray(f) / 1e9)
            np.testing.assert_allclose(observed, expected, atol=1e-10)
    device = thermo_optic_phase_shifter(efficiency=0.02)
    net = {
        "instances": {"TO": {"component": "TO"}},
        "ports": {p: f"TO,{p}" for p in ("hp", "hn", "o1", "o2", "dt", "thermal_reference")},
    }
    net = attach_testbench(
        net,
        sources={
            "hp": {"name": "HEAT", "component": "V", "settings": {"V": 1.0}},
            "o1": {"name": "LIGHT", "component": "LIGHT", "settings": {"power": 1e-3}},
        },
        loads={"o2": {"name": "LOAD", "component": "R", "settings": {"R": 1.0}}},
        gnd=["hn", "thermal_reference"],
    )
    circuit = compile_circuit(net, {"TO": device, "V": VoltageSource, "LIGHT": OpticalCW, "R": Resistor}, is_complex=True)
    dc = circuit.dc()
    np.testing.assert_allclose(circuit.port(dc, "dt"), 10.0, atol=1e-8)
    np.testing.assert_allclose(circuit.port(dc, "LOAD,p1"), np.sqrt(0.001) * np.exp(0.2j), atol=1e-9)
    print("Thermo-optic: 10 K from 10 mW, optical phase 0.2 rad")  # noqa: T201


if __name__ == "__main__":
    main()
