# ruff: noqa: INP001
"""Circuit builders for the traveling-wave MZM tutorial.

The voltage-integration reference follows the supplied scikit-rf example.
The RLC builder realizes the tutorial's nonnegative Foster expansion.
"""

import copy

import numpy as np
import skrf as rf
from scipy.constants import speed_of_light as c
from skrf.circuit import Circuit as SkCircuit
from skrf.mathFunctions import ALMOST_ZERO
from skrf.media import DefinedGammaZ0
from skrf.media.definedAEpTandZ0 import DefinedAEpTandZ0
from skrf.network import connect


def _vcvs(name: str, frequency: rf.Frequency, gain: float = 1) -> rf.Network:
    """Create a Voltage-Controlled Voltage Source (VCVS) 4-port network."""
    s = ALMOST_ZERO * np.ones([len(frequency), 4, 4], dtype=complex)
    s[:, 0, 0] = 1
    s[:, 1, 1] = 1
    s[:, 2, 0] = gain
    s[:, 2, 1] = -gain
    s[:, 2, 3] = 1
    s[:, 3, 0] = -gain
    s[:, 3, 1] = gain
    s[:, 3, 2] = 1
    return rf.Network(name=name, frequency=frequency, s=s)


def _delay_line(
    frequency: rf.Frequency, time_delay: float, zc: float = 50, z0: float = 50, name: str = "_delay_line"
) -> rf.Network:
    """Create a transmission line delay 2-port network."""
    gamma = 1j * frequency.w * time_delay
    p = (zc - z0) / (zc + z0)
    prop = np.exp(-gamma)
    denom = 1.0 - (p * prop) ** 2
    s11 = p * (1.0 - prop**2) / denom
    s21 = (1.0 - p**2) * prop / denom
    s = np.ones([len(frequency), 2, 2], dtype=complex)
    s[:, 0, 0] = s[:, 1, 1] = s11
    s[:, 0, 1] = s[:, 1, 0] = s21
    return rf.Network(name=name, frequency=frequency, s=s)


def _voltage_amplifier_2port(
    frequency: rf.Frequency, G: float, Zi: float, Zo: float, Z0: float = 50.0, name: str = "amp"
) -> rf.Network:
    """Create a two-port voltage amplifier network."""
    s = np.ones([len(frequency), 2, 2], dtype=complex)
    s[:, 0, 0] = (Zi - Z0) / (Zi + Z0)
    s[:, 0, 1] = 0
    s[:, 1, 0] = 2.0 * G * Zi * Z0 / ((Zi + Z0) * (Zo + Z0))
    s[:, 1, 1] = (Zo - Z0) / (Zo + Z0)
    return rf.Network(name=name, frequency=frequency, s=s)


def _generate_loaded_circuit(
    half_unloaded_section: rf.Network,
    r_length: float = 0.0047,
    c_per_length: float = 3.2e-10,
    d: float = 0.0001,
    unit: str = "m",
    z0: float = 50,
    gain: float = 1 / 20.0,
    optical_group_index: float = 4.05,
) -> SkCircuit:
    """Generate a loaded MZM unit cell circuit.

    Args:
        half_unloaded_section: 2-port skrf.Network for half the unloaded line.
        r_length: Sheet resistance of the diode [Ohm*m].
        c_per_length: Capacitance per length of the diode [F/m].
        d: Subsection length.
        unit: Length unit for d.
        z0: Port reference impedance.
        gain: VCVS gain (typically 1/n_sections).
        optical_group_index: Optical group index for the delay line.

    Returns:
        skrf.Circuit: The loaded unit cell circuit.

    """
    half_unloaded_section_in = copy.deepcopy(half_unloaded_section)
    half_unloaded_section_in.name = "unloaded_in"
    half_unloaded_section_out = copy.deepcopy(half_unloaded_section_in)
    half_unloaded_section_out.name = "unloaded_out"
    freq = half_unloaded_section.frequency
    media = DefinedGammaZ0(frequency=freq, z0=half_unloaded_section.z0[0, 0])
    d_m = rf.constants.to_meters(d=d, unit=unit)
    cap_per_section = d_m * c_per_length
    r_per_section = r_length / d_m
    time_delay = d_m * optical_group_index / c
    vcvs1 = _vcvs(frequency=freq, gain=gain, name="vcvs1")
    delay = _delay_line(frequency=freq, time_delay=time_delay, zc=50, name="delay1")
    amp_in = _voltage_amplifier_2port(frequency=freq, G=1, Zi=50, Zo=0, name="amp_in")
    amp_out = _voltage_amplifier_2port(frequency=freq, G=2, Zi=50, Zo=50, name="amp_out")
    port1 = SkCircuit.Port(frequency=freq, name="port1", z0=z0)
    port2 = SkCircuit.Port(frequency=freq, name="port2", z0=z0)
    port3 = SkCircuit.Port(frequency=freq, name="port3", z0=z0)
    port4 = SkCircuit.Port(frequency=freq, name="port4", z0=z0)
    cap = media.capacitor(cap_per_section, name="cap1")
    res1 = media.resistor(r_per_section, name="res1")
    ground = SkCircuit.Ground(frequency=freq, z0=z0, name="ground")
    loaded_connections = [
        [(port1, 0), (half_unloaded_section_in, 0)],
        [(port2, 0), (amp_in, 0)],
        [(port3, 0), (half_unloaded_section_out, 1)],
        [(port4, 0), (amp_out, 1)],
        [(half_unloaded_section_in, 1), (res1, 0), (half_unloaded_section_out, 0)],
        [(res1, 1), (cap, 0), (vcvs1, 1)],
        [(cap, 1), (vcvs1, 0), (ground, 0)],
        [(vcvs1, 2), (amp_in, 1)],
        [(vcvs1, 3), (delay, 0)],
        [(amp_out, 0), (delay, 1)],
    ]
    return SkCircuit(loaded_connections, name="loaded_section")


def voltage_integration_reference(
    frequencies: np.ndarray,
    *,
    n_sections: int,
    dz: float,
    z_line: float,
    z_ref: float,
    n_opt: float,
    rj_product: float,
    cj_per_m: float,
    ep_r: float = 4.04,
    tanD: float = 0.11,
    A: float = 0.0009,
) -> tuple[DefinedAEpTandZ0, rf.Network]:
    """Return the dispersive medium and matched-monitor four-port reference."""
    frequency = rf.Frequency.from_f(frequencies, unit="Hz")
    medium = DefinedAEpTandZ0(
        frequency=frequency,
        ep_r=ep_r,
        tanD=tanD,
        A=A,
        f_A=1.0,
        z0=np.full_like(frequencies, z_line),
        z0_port=50.0,
        f_low=1000000000.0,
        f_high=100000000000.0,
        f_ep=20000000000.0,
        model="djordjevicsvensson",
    )
    cell = _generate_loaded_circuit(
        medium.line(d=dz / 2, unit="m"),
        gain=1 / n_sections,
        d=dz,
        unit="m",
        r_length=rj_product,
        c_per_length=cj_per_m,
        optical_group_index=n_opt,
    ).network
    cascade = rf.network.cascade_list([cell] * n_sections)
    reference = connect(cascade, 3, _voltage_amplifier_2port(frequency, G=2, Zi=50, Zo=z_ref), 0)
    reference.renormalize(z_ref)
    reference.port_names = ["rf_in", "integration_in", "rf_out", "integrated_voltage"]
    return (medium, reference)


def rlc_netlist(
    foster_coeffs: np.ndarray,
    foster_poles: np.ndarray,
    foster_scale: float,
    *,
    n_sections: int,
    dz: float,
    z_line: float,
    z_ref: float,
    rj_product: float,
    cj_per_m: float,
    v_bias: float,
    m: int = 1,
) -> dict:
    """Build an RLC T ladder with a series-R/shunt-C junction per section."""
    coef, scale = (foster_coeffs, foster_scale)
    active = coef[2:] > 1e-08
    weights, poles = (coef[2:][active], foster_poles[active])
    ins = {
        "GND": {"component": "ground"},
        "VS": {"component": "drive", "settings": {"dc": 2 * v_bias}},
        "RS": {"component": "R", "settings": {"R": z_ref}},
        "RL": {"component": "R", "settings": {"R": z_ref}},
    }
    con = {"VS,p1": "RS,p1", "VS,p2": "GND,p1", "RL,p2": "GND,p1"}
    ports = {}
    serial = 0

    def put(kind: str, value: float, start: str, end: str | None = None) -> str:
        nonlocal serial
        serial += 1
        name = f"{kind}{serial}"
        ins[name] = {"component": kind, "settings": {kind: value}}
        p1 = name + ",p1"
        p2 = name + ",p2"
        con[p1] = start
        if end is not None:
            con[p2] = end
        return p2

    def series(start: str, d: float) -> str:
        z = z_line * d
        tip = put("R", coef[0] * z, start)
        tip = put("L", coef[1] / scale * z, tip)
        for w, p in zip(weights, poles, strict=True):
            node = tip
            end = put("R", w * z, node)
            put("L", w / p * z, node, end)
            tip = end
        return tip

    def shunt(node: str, d: float) -> None:
        y = d / z_line
        put("R", 1 / (coef[0] * y), node, "GND,p1")
        put("C", coef[1] / scale * y, node, "GND,p1")
        for w, p in zip(weights, poles, strict=True):
            end = put("R", 1 / (w * y), node)
            put("C", w * y / p, end, "GND,p1")

    def half(start: str) -> str:
        d = dz / (2 * m)
        for _ in range(m):
            start = series(start, d / 2)
            shunt(start, d)
            start = series(start, d / 2)
        return start

    last = "RS,p2"
    for k in range(n_sections):
        last = half(last)
        tip = put("R", rj_product / dz, last)
        put("C", cj_per_m * dz, tip, "GND,p1")
        ports[f"junction_{k}"] = tip
        last = half(last)
    con["RL,p1"] = last
    ports.update(rf_input="RS,p2", rf_output="RL,p1")
    return {"instances": ins, "connections": con, "ports": ports}
