# Copyright (c) 2026 Circulax contributors.
"""Composed MZI and distributed MZM builders promoted from photonic examples."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from circulax.components.base_component import PhysicsReturn, Signals, component

if TYPE_CHECKING:
    from circulax.circuit import Circuit
from circulax.components.optical_passive import OpticalPhaseShifter as PhaseBias


@component(ports=("active", "reference", "out"), states=("i_out",))
def QuadratureCombiner(signals: Signals) -> PhysicsReturn:
    """Retained directed mixing/combining blocks from older demonstrations.

    Device group:
        Photonic / Passive routing and phase

    Device reference:
        modulators.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.modulators import QuadratureCombiner

        jax.config.update("jax_enable_x64", True)
        model = QuadratureCombiner()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.complex128)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(QuadratureCombiner.__name__, "ports:", model.ports)
        ```

    """
    return {
        "active": 0.0,
        "reference": 0.0,
        "out": signals.i_out,
        "i_out": signals.out - (signals.active + 1j * signals.reference) / 2,
    }, {}


def optical_netlist(arm_a: Any, arm_b: Any, *, quadrature_phase: float = -jnp.pi / 2) -> dict:
    """Connect the legacy two-arm interferometer with explicit quadrature bias."""
    return {
        "instances": {
            "GND": {"component": "ground"},
            "SPLIT": {"component": "coupler", "settings": {"coupling": 0.5}},
            "COMBINE": {"component": "coupler", "settings": {"coupling": 0.5}},
            "BIAS": {"component": "bias", "settings": {"phase": quadrature_phase}},
            "DUMP_IN": {"component": "R", "settings": {"R": 1.0}},
            "DUMP_OUT": {"component": "R", "settings": {"R": 1.0}},
        },
        "connections": {
            "SPLIT,p2": "DUMP_IN,p1",
            "DUMP_IN,p2": "GND,p1",
            "DUMP_OUT,p2": "GND,p1",
            "SPLIT,p3": arm_a[0],
            "SPLIT,p4": arm_b[0],
            arm_a[1]: "COMBINE,p1",
            arm_b[1]: "BIAS,p1",
            "BIAS,p2": "COMBINE,p2",
            "COMBINE,p4": "DUMP_OUT,p1",
        },
        "ports": {"optical_in": "SPLIT,p1", "optical_out": "COMBINE,p3"},
    }


def electrode_netlist(
    *, n_sections: int, dz: float, rj_product: float, cj_per_m: float, phase_settings: dict | None = None
) -> dict:
    """Repeat modal half-lines, an adapter and two Rj/junction branches.

    Supplying phase_settings replaces each capacitor with a PN phase shifter
    and connects the optical sections along each arm.
    """
    optical = phase_settings is not None
    net = {"instances": {"GND": {"component": "ground"}}, "connections": {}, "ports": {}}
    ins, con, ports = net["instances"], net["connections"], net["ports"]
    for k in range(n_sections):
        ins[f"X{k}"] = {"component": "adapter"}
        for mode, model in (("d", "d_line"), ("c", "c_line")):
            left, right = f"{mode}L{k}", f"{mode}R{k}"
            ins[left], ins[right] = {"component": model}, {"component": model}
            con[f"{left},p2"] = f"X{k},{mode}"
            con[f"{right},p1"] = f"X{k},{mode}"
            if k:
                con[f"{left},p1"] = f"{mode}R{k - 1},p2"
            else:
                ports[f"{mode}_in"] = f"{left},p1"
            ports[f"{mode}_out"] = f"{right},p2"
        for arm in ("a", "b"):
            resistor, junction = f"R{arm}{k}", f"PS{arm}{k}"
            ins[resistor] = {"component": "R", "settings": {"R": rj_product / dz}}
            ins[junction] = (
                {"component": "pn", "settings": phase_settings} if optical else {"component": "C", "settings": {"C": cj_per_m * dz}}
            )
            positive, negative = ("anode", "cathode") if optical else ("p1", "p2")
            con[f"{resistor},p1"] = f"X{k},{arm}"
            con[f"{resistor},p2"] = f"{junction},{positive}"
            con[f"{junction},{negative}"] = "GND,p1"
            ports[f"junction_{arm}_{k}"] = f"{junction},{positive}"
    if optical:
        for arm in ("a", "b"):
            for k in range(n_sections - 1):
                con[f"PS{arm}{k},o2"] = f"PS{arm}{k + 1},o1"
    return net


def loaded_group_index(line: Any, frequencies: Any, *, scale: float, dz: float, rj_product: float, cj_per_m: float) -> np.ndarray:
    """Differentiate the Bloch phase of a symmetric, junction-loaded RLC cell.

    scale converts the physical junction to differential (2) or common (0.5)
    mode: R_modal = scale * Rj and C_modal = Cj / scale.
    """
    f = np.asarray(frequencies)
    s_ac = 2j * np.pi * f
    d = dz / 2
    series = np.tile(np.eye(2, dtype=complex), (len(f), 1, 1))
    shunt = series.copy()
    series[:, 0, 1] = line.series.evaluate(f) * d / 2
    shunt[:, 1, 0] = line.shunt.evaluate(f) * d
    half = series @ shunt @ series
    junction = np.tile(np.eye(2, dtype=complex), (len(f), 1, 1))
    junction[:, 1, 0] = s_ac * (cj_per_m * dz / scale) / (1 + s_ac * rj_product * cj_per_m)
    cell = half @ junction @ half
    gamma_loaded = np.arccosh((cell[:, 0, 0] + cell[:, 1, 1]) / 2) / dz
    return 299_792_458.0 * np.gradient(gamma_loaded.imag, 2 * np.pi * f)


def mach_zehnder(
    *,
    phase_a: float = 0.0,
    phase_b: float = 0.0,
    loss_a_dB: float = 0.0,
    loss_b_dB: float = 0.0,
    coupling_in: float = 0.5,
    coupling_out: float = 0.5,
) -> Circuit:
    """Compile reciprocal four-optical-port MZI with two genuine couplers.

    Both complementary outputs are exposed. Arms have independent +phase
    and power insertion loss; interference preserves the coherent output phase.

    Two couplers and two independently phased/lossy arms.

    Device group:
        Photonic / Interferometers and modulators

    Device reference:
        modulators.md

    Examples:
        ```python
        from circulax.components.modulators import mach_zehnder

        device = mach_zehnder()
        assert device.sys_size > 0
        print("External ports:", tuple(device.ports))
        ```

    """
    from circulax import compile_circuit
    from circulax.components.photonic import DirectionalCoupler

    net = {
        "instances": {
            "SPLIT": {"component": "coupler", "settings": {"coupling": coupling_in}},
            "COMBINE": {"component": "coupler", "settings": {"coupling": coupling_out}},
            "ARM_A": {"component": "phase", "settings": {"phase": phase_a, "loss_dB": loss_a_dB}},
            "ARM_B": {"component": "phase", "settings": {"phase": phase_b, "loss_dB": loss_b_dB}},
        },
        "connections": {"SPLIT,p3": "ARM_A,p1", "SPLIT,p4": "ARM_B,p1", "ARM_A,p2": "COMBINE,p1", "ARM_B,p2": "COMBINE,p2"},
        "ports": {"in": "SPLIT,p1", "in_aux": "SPLIT,p2", "out": "COMBINE,p3", "out_aux": "COMBINE,p4"},
    }
    return compile_circuit(net, {"coupler": DirectionalCoupler, "phase": PhaseBias}, is_complex=True)


@component(ports=("p", "n", "a", "b"), states=("ia", "ib"), holomorphic=True)
def _PushPullDrive(signals: Signals, gain_a: float = 1.0, gain_b: float = -1.0) -> PhysicsReturn:
    v = signals.p - signals.n
    return {
        "p": -gain_a * signals.ia - gain_b * signals.ib,
        "n": gain_a * signals.ia + gain_b * signals.ib,
        "a": signals.ia,
        "b": signals.ib,
        "ia": signals.a - gain_a * v,
        "ib": signals.b - gain_b * v,
    }, {}


def lumped_mzm(
    *,
    push_pull: bool = True,
    efficiency: float = 1.0,
    bias_phase: float = -jnp.pi / 2,
    resistance: float = 10.0,
    capacitance: float = 64e-15,
    loss_a_dB: float = 0.0,
    loss_b_dB: float = 0.0,
) -> Circuit:
    """Compile couplers and loaded phase arms with an ideal transformer drive.

    Drive voltage p-n creates arm voltages +V/2,-V/2 (push-pull) or V,0.
    Each arm retains its series R and shunt C, so its optical control has an RC
    response. The lossless drive transformer reflects arm currents into p-n, preserving
    finite input loading and the work delivered to each RC arm.

    A Mach–Zehnder with finite RC-loaded phase arms and single-arm or push-pull drive.

    Device group:
        Photonic / Interferometers and modulators

    Device reference:
        modulators.md

    Examples:
        ```python
        import jax
        import numpy as np

        from circulax import attach_testbench, compile_circuit
        from circulax.components import OpticalCW, lumped_mzm
        from circulax.components.electronic import Resistor, VoltageSource

        jax.config.update("jax_enable_x64", True)
        device = lumped_mzm(efficiency=0.7, resistance=100.0, capacitance=1e-12)
        net = {
            "instances": {"MZM": {"component": "mzm"}},
            "ports": {p: f"MZM,{p}" for p in ("in", "in_aux", "out", "out_aux", "p", "n")},
        }
        net = attach_testbench(
            net,
            sources={
                "in": {"name": "LIGHT", "component": "light", "settings": {"power": 0.002}},
                "p": {"name": "DRIVE", "component": "voltage", "settings": {"V": 0.2}},
            },
            loads={
                p: {"name": name, "component": "resistor", "settings": {"R": 1.0}}
                for p, name in (("in_aux", "DUMP_IN"), ("out", "LOAD"), ("out_aux", "DUMP_OUT"))
            },
            gnd=["n"],
        )
        circuit = compile_circuit(
            net,
            {"mzm": device, "light": OpticalCW, "voltage": VoltageSource, "resistor": Resistor},
            is_complex=True,
        )
        field = complex(circuit.port(circuit.dc(), "LOAD,p1"))
        np.testing.assert_allclose(abs(field) ** 2, 0.002 * (1 + np.sin(0.7 * 0.2)) / 2, atol=1e-10)
        print("Loaded MZM output power:", abs(field) ** 2, "W")
        ```

    """
    from circulax import compile_circuit
    from circulax.components.phase_shifters import loaded_phase_shifter
    from circulax.components.photonic import DirectionalCoupler

    arm_a = loaded_phase_shifter(resistance=resistance, capacitance=capacitance, efficiency=efficiency, loss_dB=loss_a_dB)
    arm_b = loaded_phase_shifter(
        resistance=resistance, capacitance=capacitance, efficiency=efficiency, phase=bias_phase, loss_dB=loss_b_dB
    )
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "SPLIT": {"component": "coupler"},
            "COMBINE": {"component": "coupler"},
            "A": {"component": "a"},
            "B": {"component": "b"},
            "DRIVE": {
                "component": "drive",
                "settings": {"gain_a": 0.5 if push_pull else 1.0, "gain_b": -0.5 if push_pull else 0.0},
            },
        },
        "connections": {
            "SPLIT,p3": "A,o1",
            "SPLIT,p4": "B,o1",
            "A,o2": "COMBINE,p1",
            "B,o2": "COMBINE,p2",
            "DRIVE,a": "A,anode",
            "DRIVE,b": "B,anode",
            "GND,p1": ("A,cathode", "B,cathode"),
        },
        "ports": {
            "in": "SPLIT,p1",
            "in_aux": "SPLIT,p2",
            "out": "COMBINE,p3",
            "out_aux": "COMBINE,p4",
            "p": "DRIVE,p",
            "n": "DRIVE,n",
            "junction_a": "A,junction",
            "junction_b": "B,junction",
        },
    }
    return compile_circuit(net, {"a": arm_a, "b": arm_b, "drive": _PushPullDrive, "coupler": DirectionalCoupler}, is_complex=True)


def traveling_wave_mzm(
    *,
    differential_line: Any,
    common_line: Any,
    length_m: float = 2e-3,
    n_sections: int = 6,
    rj_product: float = 4.7e-3,
    cj_per_m: float = 320e-12,
    phase_settings: dict | None = None,
    bias_phase: float = -jnp.pi / 2,
) -> Circuit:
    """Compile the tutorial modal-electrode MZM; no source/termination added.

    Supply fitted RLGCTransmissionLine objects (mode_transmission_line).
    Each cell uses two half-line ladders, a differential adapter and two real
    RC-loaded delayed PN arms. n_sections controls spatial resolution.

    A distributed electrode coupled to delayed loaded PN sections.

    Device group:
        Photonic / Interferometers and modulators

    Device reference:
        modulators.md

    Examples:
        ```python
        import numpy as np

        from circulax.components.modulators import traveling_wave_mzm
        from circulax.components.transmission_line import mode_transmission_line

        frequencies = np.r_[0.0, np.geomspace(1e6, 20e9, 20)]
        line = mode_transmission_line(frequencies, attenuation_db_per_m=200.0, n_eff=2.2, impedance=55.0)
        device = traveling_wave_mzm(differential_line=line, common_line=line, n_sections=1)
        assert {"optical_in", "optical_out", "d_in", "d_out", "c_in", "c_out"} <= set(device.ports)
        print("Distributed sections: 1; complex unknowns:", device.sys_size)
        ```

    """
    from circulax import compile_circuit
    from circulax.components.electronic import Capacitor, DifferentialAdapter, Resistor
    from circulax.components.phase_shifters import PNPhaseShifter
    from circulax.components.photonic import DirectionalCoupler

    if isinstance(n_sections, bool) or not isinstance(n_sections, int) or n_sections < 1:
        msg = "n_sections must be a positive integer"
        raise ValueError(msg)
    if not np.isfinite(length_m) or length_m <= 0 or rj_product <= 0 or cj_per_m <= 0:
        msg = "length_m, rj_product and cj_per_m must be positive"
        raise ValueError(msg)
    dz = length_m / n_sections
    settings = {"length_um": dz * 1e6, "capacitance": cj_per_m * dz, **(phase_settings or {})}
    net = electrode_netlist(n_sections=n_sections, dz=dz, rj_product=rj_product, cj_per_m=cj_per_m, phase_settings=settings)
    optical = optical_netlist(
        ("PSa0,o1", f"PSa{n_sections - 1},o2"), ("PSb0,o1", f"PSb{n_sections - 1},o2"), quadrature_phase=bias_phase
    )
    for key in ("instances", "connections", "ports"):
        net[key].update(optical[key])
    models = {
        "adapter": DifferentialAdapter,
        "R": Resistor,
        "C": Capacitor,
        "pn": PNPhaseShifter,
        "d_line": differential_line(length=dz / 2),
        "c_line": common_line(length=dz / 2),
        "coupler": DirectionalCoupler,
        "bias": PhaseBias,
    }
    return compile_circuit(net, models, is_complex=True)
