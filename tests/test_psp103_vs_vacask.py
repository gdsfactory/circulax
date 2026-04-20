"""Cross-check PSP103 ladder stages against VACASK references.

For each stage in the existing PSP103 verification ladder we run VACASK
on a matched netlist and assert circulax's converged value agrees.  The
``models.inc`` under ``tests/fixtures/vacask/`` is a byte-for-byte copy
of VACASK's ring-oscillator card, so both simulators read the exact
same 269 model-card parameter overrides on top of the 783 canonical
PSP103 defaults from the compiled OSDI binary.

Tests are skipped (not failed) when:
  * the bosdi package cannot be imported, or
  * the VACASK binary / Python rawfile reader are not installed locally.
"""

from __future__ import annotations

import sys
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import pytest

from circulax.components.osdi import _BOSDI_AVAILABLE

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from fixtures.vacask_runner import (  # noqa: E402
    FIXTURE_DIR,
    read_op_raw,
    run_vacask,
    vacask_available,
)

pytestmark = [
    pytest.mark.skipif(not _BOSDI_AVAILABLE, reason="bosdi package not available"),
    pytest.mark.skipif(not vacask_available(), reason="VACASK binary not installed"),
]


# ──────────────────────────────────────────────────────────────────────
# Helpers shared across tests
# ──────────────────────────────────────────────────────────────────────


def _circulax_nmos_id(vds: float, vgs: float, vbs: float = 0.0, w: float = 10e-6,
                      length: float = 1e-6) -> float:
    """Solve a single NMOS in DC and return the drain current in Amps.

    Mirrors ``tests/test_psp103_device.py::_compile_single_nmos`` but with
    configurable biases and returns the current through the drain vsource.
    """
    from fixtures.psp103_models import geom_settings, make_psp103_descriptors
    from circulax import compile_netlist
    from circulax.components.electronic import VoltageSource
    from circulax.solvers import analyze_circuit

    psp103n, _ = make_psp103_descriptors()
    netlist = {
        "instances": {
            "GND": {"component": "ground"},
            "Vds": {"component": "vsrc", "settings": {"V": vds}},
            "Vgs": {"component": "vsrc", "settings": {"V": vgs}},
            "Vbs": {"component": "vsrc", "settings": {"V": vbs}},
            "M":   {"component": "nmos", "settings": geom_settings(w, length)},
        },
        "connections": {
            "Vds,p1": "d,p1", "Vds,p2": "GND,p1",
            "Vgs,p1": "g,p1", "Vgs,p2": "GND,p1",
            "Vbs,p1": "b,p1", "Vbs,p2": "GND,p1",
            "M,D": "d,p1",
            "M,G": "g,p1",
            "M,S": "GND,p1",
            "M,B": "b,p1",
        },
    }
    models = {"nmos": psp103n, "vsrc": VoltageSource}
    groups, sys_size, port_map = compile_netlist(netlist, models)
    solver = analyze_circuit(groups, sys_size, backend="klu_split")

    G_HOM = 1e-2
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, G_HOM)
    y0 = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y = solver.solve_dc_gmin(groups, y0, g_start=G_HOM, n_steps=30)
    assert jnp.all(jnp.isfinite(y)), "circulax DC solve produced non-finite values"
    return float(y[port_map["Vds,i_src"]])


# ──────────────────────────────────────────────────────────────────────
# Stage 3 — single NMOS DC op-point, bit-for-bit Id comparison
# ──────────────────────────────────────────────────────────────────────


# Bias points listed in (sim_file, analysis_name, out_file, vds, vgs, label).
# Each sim file is a ready-to-run VACASK netlist under tests/fixtures/vacask/.
_NMOS_OP_CASES = [
    # sim_file              analysis out_name            Vds  Vgs  label
    ("nmos_op.sim",         "op1",   "nmos_op.raw",      1.0, 0.7, "saturation: Vds=1.0, Vgs=0.7"),
    ("nmos_op_vgs12.sim",   "op1",   "nmos_op_vgs12.raw", 0.6, 1.2, "strong inv: Vds=0.6, Vgs=1.2"),
    ("nmos_op_sub.sim",     "op1",   "nmos_op_sub.raw",  0.6, 0.3, "near subthr: Vds=0.6, Vgs=0.3"),
]


@pytest.mark.parametrize(("sim_file", "analysis", "out_name", "vds", "vgs", "label"),
                         _NMOS_OP_CASES)
def test_nmos_dc_id_matches_vacask(sim_file, analysis, out_name, vds, vgs, label):
    """|Id| from circulax matches VACASK's vds branch-flow current.

    VACASK reports ``vds:flow(br)`` which is current flowing into the +
    terminal of the drain vsource — for an NMOS with S=GND and the drain
    sinking current through the channel, that branch carries −I_d.
    Compare magnitudes.
    """
    raw_path = run_vacask(sim_file, analysis_name=analysis, out_name=out_name)
    op = read_op_raw(raw_path)

    assert "vds:flow(br)" in op, f"VACASK op-point missing vds:flow(br); got {list(op)}"
    id_vacask = abs(op["vds:flow(br)"])
    id_circulax = abs(_circulax_nmos_id(vds=vds, vgs=vgs))

    print(f"\n[{label}]  VACASK |Id| = {id_vacask:.6e} A  "
          f"circulax |Id| = {id_circulax:.6e} A  "
          f"ratio = {id_circulax / max(id_vacask, 1e-30):.4f}")

    # 1% tolerance — the DC device eval should be bit-for-bit since both
    # simulators call the same OSDI binary with the same parameter card.
    # 1% absorbs the last few digits of Newton convergence residual.
    assert id_circulax == pytest.approx(id_vacask, rel=1e-2, abs=1e-12), (
        f"{label}: |Id| circulax {id_circulax:.6e} A vs VACASK {id_vacask:.6e} A"
    )


def test_pmos_dc_id_matches_vacask():
    """|Id| for PMOS (W=20µm) at VDD=1.2V, Vsg=1.0, Vsd=0.6 matches VACASK."""
    raw_path = run_vacask("pmos_op.sim", analysis_name="op1", out_name="pmos_op.raw")
    op = read_op_raw(raw_path)
    assert "vd:flow(br)" in op
    id_vacask = abs(op["vd:flow(br)"])

    from fixtures.psp103_models import geom_settings, make_psp103_descriptors
    from circulax import compile_netlist
    from circulax.components.electronic import VoltageSource
    from circulax.solvers import analyze_circuit

    _, psp103p = make_psp103_descriptors()
    netlist = {
        "instances": {
            "GND": {"component": "ground"},
            "Vdd": {"component": "vsrc", "settings": {"V": 1.2}},
            "Vd":  {"component": "vsrc", "settings": {"V": 0.6}},
            "Vg":  {"component": "vsrc", "settings": {"V": 0.2}},
            "M":   {"component": "pmos",
                    "settings": geom_settings(20e-6, 1e-6)},
        },
        "connections": {
            "Vdd,p1": "vdd,p1", "Vdd,p2": "GND,p1",
            "Vd,p1":  "d,p1",   "Vd,p2":  "GND,p1",
            "Vg,p1":  "g,p1",   "Vg,p2":  "GND,p1",
            "M,D": "d,p1",
            "M,G": "g,p1",
            "M,S": "vdd,p1",
            "M,B": "vdd,p1",
        },
    }
    models = {"pmos": psp103p, "vsrc": VoltageSource}
    groups, sys_size, port_map = compile_netlist(netlist, models)
    solver = analyze_circuit(groups, sys_size, backend="klu_split")
    G_HOM = 1e-2
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, G_HOM)
    y0 = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y = solver.solve_dc_gmin(groups, y0, g_start=G_HOM, n_steps=30)
    id_circulax = abs(float(y[port_map["Vd,i_src"]]))

    print(f"\n[PMOS W=20µm, Vsg=1.0, Vsd=0.6]  VACASK |Id| = {id_vacask:.6e} A  "
          f"circulax |Id| = {id_circulax:.6e} A  ratio = {id_circulax/max(id_vacask,1e-30):.4f}")

    assert id_circulax == pytest.approx(id_vacask, rel=1e-2, abs=1e-12)


# ──────────────────────────────────────────────────────────────────────
# Stage 4 — CMOS inverter DC transfer curve (7 Vin points)
# ──────────────────────────────────────────────────────────────────────


_INVERTER_VTC_CASES = [
    ("inverter_vin00.sim", "op1", "inverter_vin00.raw", 0.0),
    ("inverter_vin02.sim", "op1", "inverter_vin02.raw", 0.2),
    ("inverter_vin04.sim", "op1", "inverter_vin04.raw", 0.4),
    ("inverter_vin06.sim", "op1", "inverter_vin06.raw", 0.6),
    ("inverter_vin08.sim", "op1", "inverter_vin08.raw", 0.8),
    ("inverter_vin10.sim", "op1", "inverter_vin10.raw", 1.0),
    ("inverter_vin12.sim", "op1", "inverter_vin12.raw", 1.2),
]


@pytest.mark.parametrize(("sim_file", "analysis", "out_name", "vin"), _INVERTER_VTC_CASES)
def test_inverter_vtc_matches_vacask(sim_file, analysis, out_name, vin):
    """CMOS-inverter Vout(Vin) matches VACASK at each transfer-curve point.

    PSP103 model card from VACASK ring benchmark; W_n=10 µm, W_p=20 µm,
    L=1 µm, VDD=1.2 V — same sizing as the ring oscillator.
    """
    from fixtures.psp103_models import geom_settings, make_psp103_descriptors
    from circulax import compile_netlist
    from circulax.components.electronic import VoltageSource
    from circulax.solvers import analyze_circuit

    raw_path = run_vacask(sim_file, analysis_name=analysis, out_name=out_name)
    op = read_op_raw(raw_path)
    vout_vacask = op["out"]

    psp103n, psp103p = make_psp103_descriptors()
    netlist = {
        "instances": {
            "GND": {"component": "ground"},
            "Vdd": {"component": "vsrc", "settings": {"V": 1.2}},
            "Vin": {"component": "vsrc", "settings": {"V": vin}},
            "MN":  {"component": "nmos", "settings": geom_settings(10e-6, 1e-6)},
            "MP":  {"component": "pmos", "settings": geom_settings(20e-6, 1e-6)},
        },
        "connections": {
            "Vdd,p1": "vdd,p1", "Vdd,p2": "GND,p1",
            "Vin,p1": "in,p1",  "Vin,p2": "GND,p1",
            "MN,D": "out,p1", "MN,G": "in,p1", "MN,S": "GND,p1", "MN,B": "GND,p1",
            "MP,D": "out,p1", "MP,G": "in,p1", "MP,S": "vdd,p1", "MP,B": "vdd,p1",
        },
    }
    models = {"nmos": psp103n, "pmos": psp103p, "vsrc": VoltageSource}
    groups, sys_size, port_map = compile_netlist(netlist, models)
    solver = analyze_circuit(groups, sys_size, backend="klu_split")
    G_HOM = 1e-2
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, G_HOM)
    y0 = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y = solver.solve_dc_gmin(groups, y0, g_start=G_HOM, n_steps=30)
    vout_circulax = float(y[port_map["out,p1"]])

    print(f"\n[Vin={vin:.2f} V]  VACASK Vout={vout_vacask:.6f}  "
          f"circulax Vout={vout_circulax:.6f}  "
          f"diff={1e3*(vout_circulax - vout_vacask):+.3f} mV")

    # 5 mV tolerance: VTC sits in a high-gain region so small Newton
    # residual translates to mV-level Vout differences in the transition.
    assert vout_circulax == pytest.approx(vout_vacask, abs=5e-3), (
        f"Vin={vin}: Vout circulax {vout_circulax:.6f} V vs VACASK {vout_vacask:.6f} V"
    )


# ──────────────────────────────────────────────────────────────────────
# Stage 5 — 3-stage inverter chain DC
# ──────────────────────────────────────────────────────────────────────


_CHAIN3_CASES = [
    ("chain3_vin00.sim", "op1", "chain3_vin00.raw", 0.0),
    ("chain3_vin12.sim", "op1", "chain3_vin12.raw", 1.2),
]


@pytest.mark.parametrize(("sim_file", "analysis", "out_name", "vin"), _CHAIN3_CASES)
def test_chain3_dc_matches_vacask(sim_file, analysis, out_name, vin):
    """3-stage inverter chain node voltages match VACASK at both polarities."""
    from fixtures.psp103_models import geom_settings, make_psp103_descriptors
    from circulax import compile_netlist
    from circulax.components.electronic import VoltageSource
    from circulax.solvers import analyze_circuit

    raw_path = run_vacask(sim_file, analysis_name=analysis, out_name=out_name)
    op = read_op_raw(raw_path)

    psp103n, psp103p = make_psp103_descriptors()
    instances: dict = {
        "GND": {"component": "ground"},
        "Vdd": {"component": "vsrc", "settings": {"V": 1.2}},
        "Vin": {"component": "vsrc", "settings": {"V": vin}},
    }
    connections: dict = {
        "Vdd,p1": "vdd,p1", "Vdd,p2": "GND,p1",
        "Vin,p1": "n0,p1",  "Vin,p2": "GND,p1",
    }
    for stage in range(3):
        in_node, out_node = f"n{stage}", f"n{stage + 1}"
        mn, mp = f"mn{stage}", f"mp{stage}"
        instances[mn] = {"component": "nmos", "settings": geom_settings(10e-6, 1e-6)}
        instances[mp] = {"component": "pmos", "settings": geom_settings(20e-6, 1e-6)}
        connections[f"{mn},D"] = f"{out_node},p1"
        connections[f"{mn},G"] = f"{in_node},p1"
        connections[f"{mn},S"] = "GND,p1"
        connections[f"{mn},B"] = "GND,p1"
        connections[f"{mp},D"] = f"{out_node},p1"
        connections[f"{mp},G"] = f"{in_node},p1"
        connections[f"{mp},S"] = "vdd,p1"
        connections[f"{mp},B"] = "vdd,p1"
    models = {"nmos": psp103n, "pmos": psp103p, "vsrc": VoltageSource}
    groups, sys_size, port_map = compile_netlist(
        {"instances": instances, "connections": connections}, models
    )
    solver = analyze_circuit(groups, sys_size, backend="klu_split")
    G_HOM = 1e-2
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, G_HOM)
    y0 = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y = solver.solve_dc_gmin(groups, y0, g_start=G_HOM, n_steps=30)

    print(f"\n[3-stage chain Vin={vin:.2f} V]")
    for k in ("n1", "n2", "n3"):
        v_v = op[k]
        v_c = float(y[port_map[f"{k},p1"]])
        print(f"  {k}: VACASK={v_v:.6f}  circulax={v_c:.6f}  diff={1e3*(v_c - v_v):+.3f} mV")
        assert v_c == pytest.approx(v_v, abs=5e-3), (
            f"{k} at Vin={vin}: circulax {v_c:.6f} V vs VACASK {v_v:.6f} V"
        )
