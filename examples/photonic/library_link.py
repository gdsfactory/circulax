# Copyright (c) 2026 Circulax contributors
# ruff: noqa: INP001
"""Run a loaded CW -> push-pull MZM -> absorbing photodiode link.

Run from the repository: python examples/photonic/library_link.py
Power is incident watts from a matched source; electrical outputs are volts.
This deterministic envelope demonstration is not a calibrated transceiver.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np

from circulax import Circuit, compile_circuit, quadrature_response
from circulax.components.electronic import Resistor, WaveformVoltageSource
from circulax.components.modulators import lumped_mzm
from circulax.components.photodetectors import IdealPhotodetector, OpticalCW


def build_link(*, voltage: float = 0.2, power: float = 0.002) -> Circuit:
    """Build a finite arm-RC MZM with 50-ohm detector load and matched dumps."""
    mzm = lumped_mzm(efficiency=0.7, resistance=100.0, capacitance=1e-12)
    net = {
        "instances": {
            "G": {"component": "ground"},
            "LIGHT": {"component": "LIGHT", "settings": {"power": power}},
            "DRIVE": {"component": "DRIVE", "settings": {"dc": voltage}},
            "MZM": {"component": "MZM"},
            "PD": {"component": "PD", "settings": {"reflectivity": 0.0}},
            "LOAD": {"component": "R", "settings": {"R": 50.0}},
            "DUMP_IN": {"component": "R", "settings": {"R": 1.0}},
            "DUMP_OUT": {"component": "R", "settings": {"R": 1.0}},
        },
        "connections": {
            "LIGHT,p1": "MZM,in",
            "MZM,in_aux": "DUMP_IN,p1",
            "MZM,out_aux": "DUMP_OUT,p1",
            "MZM,out": "PD,optical_in",
            "DRIVE,p1": "MZM,p",
            "PD,anode": "LOAD,p1",
            "G,p1": ("LIGHT,p2", "DRIVE,p2", "MZM,n", "PD,cathode", "LOAD,p2", "DUMP_IN,p2", "DUMP_OUT,p2"),
        },
        "ports": {"output": "LOAD,p1", "optical": "PD,optical_in", "junction": "MZM,junction_a"},
    }
    return compile_circuit(
        net,
        {"LIGHT": OpticalCW, "DRIVE": WaveformVoltageSource, "MZM": mzm, "PD": IdealPhotodetector, "R": Resistor},
        is_complex=True,
    )


def main() -> None:
    """Check independent static transfer and RC-limited EO/OE conversion."""
    jax.config.update("jax_enable_x64", True)  # noqa: FBT003
    start = time.perf_counter()
    circuit = build_link()
    dc = circuit.dc()
    elapsed = time.perf_counter() - start
    voltage = circuit.port(dc, "output").real
    expected = 50 * 0.8 * 0.002 * (1 + np.sin(0.7 * 0.2)) / 2
    np.testing.assert_allclose(voltage, expected, atol=1e-9)
    frequencies = jnp.array([1e6, 1e9, 1e10])
    ac = quadrature_response(circuit, freqs=frequencies, sources={"DRIVE": 1.0}, y_dc=dc)
    response = ac["in_phase"][:, circuit.port_map["output"]]
    slope = 50 * 0.8 * 0.002 * 0.7 * np.cos(0.7 * 0.2) / 2
    expected_ac = slope / (1 + 2j * np.pi * np.asarray(frequencies) * 100 * 1e-12)
    np.testing.assert_allclose(response, expected_ac, atol=1e-8)
    print(f"{circuit.sys_size} complex unknowns; build + first DC {elapsed:.3f}s")  # noqa: T201
    print(f"Detector voltage {float(voltage):.8f} V; EO/OE response {np.asarray(response)} V/V")  # noqa: T201


if __name__ == "__main__":
    main()
