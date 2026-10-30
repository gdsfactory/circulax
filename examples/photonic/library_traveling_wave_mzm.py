# Copyright (c) 2026 Circulax contributors.
"""Small reproducible loaded-electrode MZM with an independent ABCD reference.

Run from the repository: .venv/bin/python examples/photonic/library_traveling_wave_mzm.py
Synthetic constant attenuation/index data are illustrative, not calibrated data.
"""

# ruff: noqa: ANN001, ANN201, INP001
import jax
import jax.numpy as jnp
import numpy as np

from circulax import compile_circuit
from circulax.components.electronic import Resistor, WaveformVoltageSource
from circulax.components.modulators import traveling_wave_mzm
from circulax.components.photonic import OpticalSource
from circulax.components.transmission_line import mode_transmission_line
from circulax.optical_analysis import quadrature_response

jax.config.update("jax_enable_x64", True)  # noqa: FBT003
C0 = 299792458.0


def build_bench(n_sections=2, *, rf_index=2.2, attenuation=200.0, source_r=70.0, load_r=100.0, optical_index=4.05):
    """Create a terminated distributed modulator, with exposed junction probes."""
    samples = np.r_[0, np.geomspace(1e6, 8e10, 40)]
    line = mode_transmission_line(samples, attenuation_db_per_m=attenuation, n_eff=rf_index, impedance=55.0)
    device = traveling_wave_mzm(
        differential_line=line,
        common_line=line,
        n_sections=n_sections,
        phase_settings={"a1": 1.0, "a2": 0.0, "n_group": optical_index},
    )
    instances = {
        "GND": {"component": "ground"},
        "M": {"component": "M"},
        "VD": {"component": "V"},
        "VC": {"component": "V", "settings": {"dc": -2.0}},
        "LIGHT": {"component": "S", "settings": {"power": 1.0}},
    }
    connections = {
        "VD,p1": "RD,p1",
        "RD,p2": "M,d_in",
        "M,d_out": "LD,p1",
        "LD,p2": "GND,p1",
        "VC,p1": "RC,p1",
        "RC,p2": "M,c_in",
        "M,c_out": "LC,p1",
        "LC,p2": "GND,p1",
        "VD,p2": "GND,p1",
        "VC,p2": "GND,p1",
        "LIGHT,p1": "M,optical_in",
        "LIGHT,p2": "GND,p1",
        "M,optical_out": "OPTICAL_LOAD,p1",
        "OPTICAL_LOAD,p2": "GND,p1",
    }
    for name, value in [("RD", source_r), ("LD", load_r), ("RC", 20.0), ("LC", 30.0), ("OPTICAL_LOAD", 1.0)]:
        instances[name] = {"component": "R", "settings": {"R": value}}
    ports = {"out": "M,optical_out"}
    ports.update({f"junction_{arm}_{k}": f"M,junction_{arm}_{k}" for arm in ("a", "b") for k in range(n_sections)})
    return compile_circuit(
        {"instances": instances, "connections": connections, "ports": ports},
        {"M": device, "V": WaveformVoltageSource, "S": OpticalSource, "R": Resistor},
        is_complex=True,
    )


def numpy_eo_reference(
    frequencies, n_sections, *, rf_index=2.2, attenuation=200.0, source_r=70.0, load_r=100.0, optical_index=4.05
):
    """Independent cascaded symmetric-T ABCD and midpoint optical propagation."""
    length = 0.002
    dz = length / n_sections
    rj = 0.0047
    cj = 320e-12
    impedance = 55.0
    response = []
    for f in frequencies:
        s = 2j * np.pi * f
        gamma = attenuation * np.log(10) / 20 + s * rf_index / C0
        series = np.array([[1, gamma * impedance * dz / 4], [0, 1]])
        shunt = np.array([[1, 0], [gamma / impedance * dz / 2, 1]])
        half = series @ shunt @ series
        junction = np.array([[1, 0], [s * cj * dz / 2 / (1 + s * rj * cj), 1]])
        cell = half @ junction @ half
        total = np.linalg.matrix_power(cell, n_sections)
        vout = 1 / (total[0, 0] + total[0, 1] / load_r + source_r * (total[1, 0] + total[1, 1] / load_r))
        state = np.array([vout, vout / load_r])
        centers = []
        for _ in range(n_sections):
            center = half @ state
            centers.append(center[0] / (1 + s * rj * cj))
            state = half @ junction @ center
        voltages = np.asarray(centers[::-1])
        delay = (length - (np.arange(n_sections) + 0.5) * dz) * optical_index / C0
        # Both arm phase efficiencies are 1 rad/(V mm), and d=a-b.
        response.append(10 ** (-length * 100 / 10) / 2 * dz * 1e3 * np.sum(voltages * np.exp(-s * delay)))
    return np.asarray(response)


def run(n_sections=2):
    """Solve DC and full optical/electrical quadrature response and compare."""
    frequencies = jnp.asarray([1e8, 4e9, 15e9])
    circuit = build_bench(n_sections)
    dc = circuit.dc()
    result = quadrature_response(circuit, freqs=frequencies, sources={"VD": 1.0}, y_dc=dc)
    index = circuit.port_map["out"]
    field = complex(circuit.port(dc, "out"))
    power_response = 2 * (
        field.real * np.asarray(result["in_phase"][:, index]) + field.imag * np.asarray(result["quadrature"][:, index])
    )
    reference = numpy_eo_reference(np.asarray(frequencies), n_sections)
    np.testing.assert_allclose(power_response, reference, rtol=2e-6, atol=2e-9)
    np.testing.assert_allclose(abs(field) ** 2, 10 ** (-0.2 / 10) / 2, atol=1e-8)
    return circuit, power_response


if __name__ == "__main__":
    circuit, response = run()
    print(f"{circuit.sys_size} complex unknowns; EO power response (W/V): {response}")  # noqa: T201
