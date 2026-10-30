# Copyright (c) 2026 Circulax contributors.
# ruff: noqa: ANN001, ANN201, ANN202
"""Independent coherent matrix references for composed photonic modulators."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax import compile_circuit
from circulax.components.electronic import Resistor, WaveformVoltageSource
from circulax.components.modulators import lumped_mzm, mach_zehnder, traveling_wave_mzm
from circulax.components.photonic import OpticalSource
from circulax.components.transmission_line import mode_transmission_line

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


def real_block_ac(circuit, dc, frequencies, source="V"):
    """Independent dense solve of compiled F/Q quadrature Jacobians."""
    from circulax.solvers.assembly import assemble_residual_only_complex

    def residual(y):
        return assemble_residual_only_complex(y, circuit.groups, 0.0, 1.0)

    jf, jq = jax.jacfwd(residual)(dc)
    jf = np.array(jf)
    jq = np.array(jq)
    for index in (0, circuit.sys_size):
        jf[index, :] = 0
        jf[index, index] = 1
        jq[index, :] = 0
    rhs = np.zeros(len(dc))
    rhs[circuit.port_map[f"{source},i_src"]] = 1
    solutions = np.stack([np.linalg.solve(np.asarray(jf) + 2j * np.pi * f * np.asarray(jq), rhs) for f in frequencies])
    return solutions


def optical_bench(device, voltage=None):
    ins = {"GND": {"component": "ground"}, "D": {"component": "D"}, "S": {"component": "S", "settings": {"power": 1.0}}}
    con = {"S,p1": "D,in", "S,p2": "GND,p1"}
    for i, port in enumerate(("in_aux", "out", "out_aux")):
        ins[f"R{i}"] = {"component": "R", "settings": {"R": 1.0}}
        con[f"R{i},p1"] = f"D,{port}"
        con[f"R{i},p2"] = "GND,p1"
    if voltage is not None:
        ins["V"] = {"component": "V", "settings": {"dc": voltage}}
        con["V,p1"] = "D,p"
        con["V,p2"] = ("D,n", "GND,p1")
    net = {"instances": ins, "connections": con, "ports": {"out": "D,out", "aux": "D,out_aux"}}
    return compile_circuit(net, {"D": device, "S": OpticalSource, "R": Resistor, "V": WaveformVoltageSource}, is_complex=True)


@pytest.mark.parametrize(("phases", "loss"), [((0, 0), (0, 0)), ((0.3, -1.2), (0, 0)), ((1.1, -0.4), (1.2, 3.1))])
def test_mzi_coherent_complementary_outputs(phases, loss):
    device = mach_zehnder(phase_a=phases[0], phase_b=phases[1], loss_a_dB=loss[0], loss_b_dB=loss[1])
    circuit = optical_bench(device)
    y = circuit.dc()
    matrix = np.array([[1, 1j], [1j, 1]]) / np.sqrt(2)
    arms = 10 ** (-np.asarray(loss) / 20) * np.exp(1j * np.asarray(phases))
    expected = matrix @ np.diag(arms) @ matrix @ np.array([1, 0])
    actual = np.array([circuit.port(y, "out"), circuit.port(y, "aux")])
    np.testing.assert_allclose(actual, expected, atol=2e-9)
    assert np.sum(abs(actual) ** 2) == pytest.approx(np.mean(10 ** (-np.asarray(loss) / 10)))


@pytest.mark.parametrize("push_pull", [True, False])
def test_loaded_mzm_phase_and_rc_small_signal(push_pull):
    resistance, capacitance, efficiency, voltage = 20.0, 1e-6, 0.7, 0.2
    c = optical_bench(
        lumped_mzm(push_pull=push_pull, efficiency=efficiency, resistance=resistance, capacitance=capacitance), voltage
    )
    y = c.dc()
    ga, gb = (0.5, -0.5) if push_pull else (1, 0)
    phase = np.array([efficiency * ga * voltage, efficiency * gb * voltage - np.pi / 2])
    expected = (np.exp(1j * phase[0]) - np.exp(1j * phase[1])) / 2
    np.testing.assert_allclose(c.port(y, "out"), expected, atol=1e-8)
    f = jnp.array([10.0, 1e3, 1e4])
    ac = {"solution": real_block_ac(c, y, f)}
    # Sum optical I/Q phasors: this is twice the upper envelope sideband.
    derivative = 0.5j * efficiency * (ga * np.exp(1j * phase[0]) - gb * np.exp(1j * phase[1]))
    np.testing.assert_allclose(
        ac["solution"][:, c.port_map["out"]] + 1j * ac["solution"][:, c.port_map["out"] + c.sys_size],
        derivative / (1 + 2j * np.pi * np.asarray(f) * resistance * capacitance),
        atol=2e-7,
    )
    idx = c.port_map["out"]
    lower = ac["solution"][:, idx] - 1j * ac["solution"][:, idx + c.sys_size]
    np.testing.assert_allclose(
        lower, derivative.conjugate() / (1 + 2j * np.pi * np.asarray(f) * resistance * capacitance), atol=2e-7
    )
    input_admittance = (
        (ga**2 + gb**2) * 2j * np.pi * np.asarray(f) * capacitance / (1 + 2j * np.pi * np.asarray(f) * resistance * capacitance)
    )
    from circulax.optical_analysis import quadrature_response

    public = quadrature_response(c, freqs=f, sources={"V": 1.0}, y_dc=y)
    np.testing.assert_allclose(public["in_phase"], ac["solution"][:, : c.sys_size], atol=2e-8)
    np.testing.assert_allclose(public["quadrature"], ac["solution"][:, c.sys_size :], atol=2e-8)
    source_current = ac["solution"][:, c.port_map["V,i_src"]]
    np.testing.assert_allclose(source_current, -input_admittance, atol=2e-9)


def test_distributed_wrapper_small_circuit_and_section_validation():
    frequencies = np.r_[0, np.geomspace(1e5, 1e10, 30)]
    line = mode_transmission_line(frequencies, attenuation_db_per_m=20.0, n_eff=2.2, impedance=50.0)
    c = traveling_wave_mzm(differential_line=line, common_line=line, n_sections=1)
    assert {"d_in", "d_out", "c_in", "c_out", "optical_in", "optical_out", "junction_a_0", "junction_b_0"} <= set(c.ports)
    with pytest.raises(ValueError, match="n_sections"):
        traveling_wave_mzm(differential_line=line, common_line=line, n_sections=0)


@pytest.mark.parametrize("n_sections", [2, 4])
def test_loaded_electrode_against_independent_abcd(n_sections):
    """RF loss, source/load mismatch and junction loading from NumPy ABCD."""
    from circulax.components.electronic import Capacitor, DifferentialAdapter
    from circulax.components.modulators import electrode_netlist

    f = np.array([1e8, 4e9, 15e9])
    length = 0.002
    dz = length / n_sections
    attenuation = 200.0
    index = 2.2
    impedance = 55.0
    rj = 0.0047
    cj = 320e-12
    samples = np.r_[0, np.geomspace(1e5, 3e10, 40)]
    line = mode_transmission_line(samples, attenuation_db_per_m=attenuation, n_eff=index, impedance=impedance)
    electrode = compile_circuit(
        electrode_netlist(n_sections=n_sections, dz=dz, rj_product=rj, cj_per_m=cj),
        {
            "adapter": DifferentialAdapter,
            "R": Resistor,
            "C": Capacitor,
            "d_line": line(length=dz / 2),
            "c_line": line(length=dz / 2),
        },
    )
    ins = {"GND": {"component": "ground"}, "E": {"component": "E"}, "V": {"component": "V"}}
    con = {"V,p2": "GND,p1"}
    # Drive differential mode, terminate common mode independently.
    for name, value in [("RS", 70.0), ("RL", 100.0), ("RC", 20.0), ("LC", 30.0)]:
        ins[name] = {"component": "R", "settings": {"R": value}}
    con.update(
        {
            "V,p1": "RS,p1",
            "RS,p2": "E,d_in",
            "E,d_out": "RL,p1",
            "RL,p2": "GND,p1",
            "E,c_in": "RC,p1",
            "RC,p2": "GND,p1",
            "E,c_out": "LC,p1",
            "LC,p2": "GND,p1",
        }
    )
    bench = compile_circuit(
        {"instances": ins, "connections": con, "ports": {f"j{k}": f"E,junction_a_{k}" for k in range(n_sections)}},
        {"E": electrode, "V": WaveformVoltageSource, "R": Resistor},
    )
    ac = bench.ac_response(freqs=jnp.asarray(f), sources={"V": 1.0})
    expected = []
    for freq in f:
        s = 2j * np.pi * freq
        gamma = attenuation * np.log(10) / 20 + s * index / 299792458
        z = gamma * impedance
        y = gamma / impedance
        series = np.array([[1, z * dz / 4], [0, 1]])
        shunt = np.array([[1, 0], [y * dz / 2, 1]])
        half = series @ shunt @ series
        junction = np.array([[1, 0], [s * (cj * dz / 2) / (1 + s * rj * cj), 1]])
        cell = half @ junction @ half
        total = np.linalg.matrix_power(cell, n_sections)
        vout = 1 / (total[0, 0] + total[0, 1] / 100 + 70 * (total[1, 0] + total[1, 1] / 100))
        state = np.array([vout, vout / 100])
        values = []
        for _ in range(n_sections):
            center = half @ state
            values.append(center[0] / 2 / (1 + s * rj * cj))
            state = half @ junction @ center
        expected.append(values[::-1])
    actual = np.stack([bench.port(ac["solution"], f"j{k}") for k in range(n_sections)], axis=-1)
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-9)


def test_full_distributed_eo_reference_and_section_convergence():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "examples/photonic/library_traveling_wave_mzm.py"
    spec = importlib.util.spec_from_file_location("library_tw_mzm", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    f = np.array([1e8, 4e9, 15e9])
    target = reference.numpy_eo_reference(f, 128)
    errors = []
    for n in (1, 2, 4):
        circuit, response = reference.run(n)
        errors.append(np.linalg.norm(response - target))
        assert circuit.sys_size > 0
    assert errors[2] < errors[1] < errors[0]
    # Deliberate severe velocity mismatch, RF loss and unequal terminations.
    from circulax.optical_analysis import quadrature_response

    settings = {"rf_index": 6.0, "optical_index": 1.5, "attenuation": 400.0, "source_r": 50.0, "load_r": 120.0}
    circuit = reference.build_bench(2, **settings)
    dc = circuit.dc()
    result = quadrature_response(circuit, freqs=jnp.asarray(f), sources={"VD": 1.0}, y_dc=dc)
    idx = circuit.port_map["out"]
    field = complex(circuit.port(dc, "out"))
    actual = 2 * (field.real * np.asarray(result["in_phase"][:, idx]) + field.imag * np.asarray(result["quadrature"][:, idx]))
    np.testing.assert_allclose(actual, reference.numpy_eo_reference(f, 2, **settings), rtol=2e-6, atol=2e-9)
