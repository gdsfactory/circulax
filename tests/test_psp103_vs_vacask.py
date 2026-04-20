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
import numpy as np
import pytest

from circulax.components.osdi import _BOSDI_AVAILABLE

_TESTS = Path(__file__).resolve().parent
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from fixtures.vacask_runner import (  # noqa: E402
    FIXTURE_DIR,
    read_op_raw,
    read_time_raw,
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


# ──────────────────────────────────────────────────────────────────────
# Stage 4 (transient) — CMOS inverter step response
# ──────────────────────────────────────────────────────────────────────


def _crossing_time(t: np.ndarray, x: np.ndarray, level: float, *,
                   direction: str = "any") -> float:
    """Linear-interp the time at which ``x`` first crosses ``level``.

    direction in {"rising", "falling", "any"}.
    """
    diffs = np.diff(np.sign(x - level))
    if direction == "rising":
        crossings = np.where(diffs > 0)[0]
    elif direction == "falling":
        crossings = np.where(diffs < 0)[0]
    else:
        crossings = np.where(diffs != 0)[0]
    if not len(crossings):
        return float("nan")
    i = int(crossings[0])
    x0, x1 = float(x[i]), float(x[i + 1])
    t0, t1 = float(t[i]), float(t[i + 1])
    return t0 + (level - x0) * (t1 - t0) / (x1 - x0)


def _run_circulax_inverter_step(integrator_name: str, *, dt0: float = 5e-12,
                                t1: float = 5e-9, n_save: int = 1000):
    import diffrax

    from fixtures.psp103_models import geom_settings, make_psp103_descriptors
    from circulax import compile_netlist
    from circulax.components.electronic import (
        PulseVoltageSource,
        VoltageSource,
    )
    from circulax.solvers import analyze_circuit, setup_transient
    from circulax.solvers.transient import (
        BDF2FactorizedTransientSolver,
        FactorizedTransientSolver,
        SDIRK3FactorizedTransientSolver,
    )

    integrators = {
        "BE":     FactorizedTransientSolver,
        "BDF2":   BDF2FactorizedTransientSolver,
        "SDIRK3": SDIRK3FactorizedTransientSolver,
    }
    icls = integrators[integrator_name]

    psp103n, psp103p = make_psp103_descriptors()
    netlist = {
        "instances": {
            "GND": {"component": "ground"},
            "Vdd": {"component": "vsrc",  "settings": {"V": 1.2}},
            "Vin": {"component": "pulse",
                    "settings": {"v1": 0.0, "v2": 1.2, "td": 1e-9,
                                 "tr": 100e-12, "tf": 100e-12,
                                 "pw": 10e-9, "per": 20e-9}},
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
    models = {
        "nmos": psp103n, "pmos": psp103p,
        "vsrc": VoltageSource, "pulse": PulseVoltageSource,
    }
    groups, sys_size, port_map = compile_netlist(netlist, models)
    solver = analyze_circuit(groups, sys_size, backend="klu_split")

    G_HOM = 1e-2
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, G_HOM)
    y0 = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y0 = solver.solve_dc_gmin(groups, y0, g_start=G_HOM, n_steps=30)
    assert jnp.all(jnp.isfinite(y0))

    run = setup_transient(groups, solver, transient_solver=icls)
    sol = run(
        t0=0.0, t1=t1, dt0=dt0, y0=y0,
        saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t1, n_save)),
        max_steps=2_000_000,
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    import numpy as _np
    t = _np.asarray(sol.ts)
    vin_trace = _np.asarray(sol.ys)[:, port_map["in,p1"]]
    vout_trace = _np.asarray(sol.ys)[:, port_map["out,p1"]]
    return t, vin_trace, vout_trace


@pytest.mark.parametrize("integrator", ["BE", "BDF2", "SDIRK3"])
def test_inverter_step_tphl_vs_vacask(integrator):
    """Inverter step-response propagation delay (Vin→Vout, 50 % crossing).

    VACASK reference uses trapezoidal at 5 ps fixed step (matches the OSDI
    PSP103 binary).  circulax runs the same Vin pulse with the listed
    integrator at 5 ps fixed step.  Reports the delay for each integrator
    so we can quantify the ring-osc 6.7× gap as a single-stage effect.

    Tolerance on assertion is wide on purpose — this test is a *measurement*
    of the integrator-induced gap, not a regression gate.  We only assert
    finite output and that Vout cleanly transitions high→low.
    """
    raw_path = run_vacask("inverter_step.sim", analysis_name="tran1",
                          out_name="inverter_step.raw")
    ref = read_time_raw(raw_path)

    t_ref = ref["time"]
    vin_ref = ref["in"]
    vout_ref = ref["out"]
    tpHL_vacask = (
        _crossing_time(t_ref, vout_ref, level=0.6, direction="falling")
        - _crossing_time(t_ref, vin_ref, level=0.6, direction="rising")
    )

    t_c, vin_c, vout_c = _run_circulax_inverter_step(integrator)
    tpHL_circulax = (
        _crossing_time(t_c, vout_c, level=0.6, direction="falling")
        - _crossing_time(t_c, vin_c, level=0.6, direction="rising")
    )

    print(
        f"\n[{integrator}]  VACASK tpHL = {tpHL_vacask*1e12:.1f} ps  "
        f"circulax tpHL = {tpHL_circulax*1e12:.1f} ps  "
        f"ratio = {tpHL_circulax / tpHL_vacask:.2f}"
    )

    # Sanity: the inverter must transition.
    assert vout_c[0] > 0.9 * 1.2, f"Vout(0) = {vout_c[0]:.3f}, expected ~VDD"
    assert vout_c[-1] < 0.1 * 1.2, f"Vout(end) = {vout_c[-1]:.3f}, expected ~0"
    assert np.isfinite(tpHL_circulax), f"circulax {integrator} did not produce a clean falling edge"


# ──────────────────────────────────────────────────────────────────────
# Stage 5 (transient) — 3-stage chain step response (loading + chain delay)
# ──────────────────────────────────────────────────────────────────────


def _run_circulax_chain3_step(integrator_name: str, *, dt0: float = 5e-12,
                              t1: float = 5e-9, n_save: int = 1000):
    import diffrax

    from fixtures.psp103_models import geom_settings, make_psp103_descriptors
    from circulax import compile_netlist
    from circulax.components.electronic import (
        PulseVoltageSource,
        VoltageSource,
    )
    from circulax.solvers import analyze_circuit, setup_transient
    from circulax.solvers.transient import (
        BDF2FactorizedTransientSolver,
        FactorizedTransientSolver,
        SDIRK3FactorizedTransientSolver,
    )

    integrators = {
        "BE":     FactorizedTransientSolver,
        "BDF2":   BDF2FactorizedTransientSolver,
        "SDIRK3": SDIRK3FactorizedTransientSolver,
    }
    icls = integrators[integrator_name]

    psp103n, psp103p = make_psp103_descriptors()
    instances: dict = {
        "GND": {"component": "ground"},
        "Vdd": {"component": "vsrc", "settings": {"V": 1.2}},
        "Vin": {"component": "pulse",
                "settings": {"v1": 0.0, "v2": 1.2, "td": 1e-9,
                             "tr": 100e-12, "tf": 100e-12,
                             "pw": 10e-9, "per": 20e-9}},
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
    models = {
        "nmos": psp103n, "pmos": psp103p,
        "vsrc": VoltageSource, "pulse": PulseVoltageSource,
    }
    groups, sys_size, port_map = compile_netlist(
        {"instances": instances, "connections": connections}, models,
    )
    solver = analyze_circuit(groups, sys_size, backend="klu_split")

    G_HOM = 1e-2
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, G_HOM)
    y0 = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y0 = solver.solve_dc_gmin(groups, y0, g_start=G_HOM, n_steps=30)

    run = setup_transient(groups, solver, transient_solver=icls)
    sol = run(
        t0=0.0, t1=t1, dt0=dt0, y0=y0,
        saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t1, n_save)),
        max_steps=2_000_000,
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    t = np.asarray(sol.ts)
    ys = np.asarray(sol.ys)
    return t, {
        f"n{i}": ys[:, port_map[f"n{i},p1"]] for i in range(4)
    }


@pytest.mark.parametrize("integrator", ["BE", "BDF2"])
def test_chain3_step_per_stage_delay_vs_vacask(integrator):
    """3-stage chain stage-by-stage propagation delay vs VACASK.

    Each stage in the chain drives the next stage's gate, so the per-stage
    delay sees the full intrinsic load (~170 fF C[int0,int0] of the next
    transistor pair).  This is the in-between case between the standalone
    inverter (no load) and the 9-stage ring (full feedback).  The test
    reports per-stage delays for VACASK and circulax and prints both so
    the gap is visible at this scale.
    """
    raw_path = run_vacask("chain3_step.sim", analysis_name="tran1",
                          out_name="chain3_step.raw")
    ref = read_time_raw(raw_path)
    t_ref = ref["time"]

    def stage_delays(t_arr, signal_dict):
        # Stage k delay = time(stage-output cross 50%) - time(stage-input cross 50%).
        # Stages alternate rising/falling edges (inverters).
        delays = []
        for k in range(3):
            in_sig = signal_dict[f"n{k}"]
            out_sig = signal_dict[f"n{k + 1}"]
            in_dir  = "rising" if k % 2 == 0 else "falling"
            out_dir = "falling" if k % 2 == 0 else "rising"
            t_in  = _crossing_time(t_arr, in_sig,  level=0.6, direction=in_dir)
            t_out = _crossing_time(t_arr, out_sig, level=0.6, direction=out_dir)
            delays.append(t_out - t_in)
        return delays

    vac_signals = {f"n{i}": ref[f"n{i}"] for i in range(4)}
    vac_delays = stage_delays(t_ref, vac_signals)

    t_c, c_signals = _run_circulax_chain3_step(integrator)
    c_delays = stage_delays(t_c, c_signals)

    print(f"\n[{integrator}] chain-3 per-stage delays (Vin step 0→VDD at t=1ns):")
    print(f"  stage   VACASK [ps]   circulax [ps]   ratio")
    for k, (v, c) in enumerate(zip(vac_delays, c_delays)):
        print(f"    {k+1:>3d}    {v*1e12:>10.1f}     {c*1e12:>10.1f}     {c/v:.2f}")

    for k in range(3):
        assert np.isfinite(c_delays[k]), (
            f"circulax {integrator} stage {k+1} has no clean transition"
        )
