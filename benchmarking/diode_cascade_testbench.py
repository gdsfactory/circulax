"""Diode cascade (Cockcroft-Walton ×2) testbench.

Circuit
-------
  vs  a  0   SIN(0, 50V, 100kHz)
  r1  a  1   0.01Ω
  c1  1  2   100nF  /  c3  1  2  100nF  (parallel)
  d1  0  1   D1N4007  (anode=0, cathode=1)
  c2  0  10  100nF
  d2  1  10  D1N4007
  d3  10 2   D1N4007
  c4  10 20  100nF
  d4  2  20  D1N4007

Nodes compared: v(1), v(10), v(20)  — 4× peak voltage at v(20)

Usage
-----
  pixi run -e benchmark python benchmarking/diode_cascade_testbench.py [--plot]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

KLUJAX_SRC = pathlib.Path("/home/chris/code/klujax/klujax")
if KLUJAX_SRC.exists() and str(KLUJAX_SRC) not in sys.path:
    sys.path.insert(0, str(KLUJAX_SRC))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import diffrax  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from bench_utils.ngspice import run_ngspice  # noqa: E402
from bench_utils.plotting import plot_comparison  # noqa: E402
from bench_utils.runner import SolverFn, SolverResult, run_benchmark  # noqa: E402

from circulax.compiler import compile_netlist  # noqa: E402
from circulax.components.base_component import PhysicsReturn, Signals, States, component  # noqa: E402
from circulax.components.electronic import (  # noqa: E402
    Capacitor, Resistor, VoltageSourceAC, _junction_charge,
)
from circulax.solvers import analyze_circuit, setup_transient, BDF2RefactoringTransientSolver  # noqa: E402

# ---------------------------------------------------------------------------
# DiodeLimited — clamped Shockley + SPICE junction capacitance
# ---------------------------------------------------------------------------

EXP_CAP = 40.0


@component(ports=("p1", "p2"))
def DiodeLimited(
    signals: Signals, s: States,
    Is: float = 1e-12, n: float = 1.0, Vt: float = 25.85e-3,
    Cj0: float = 0.0, Vj: float = 1.0, m: float = 0.5,
) -> PhysicsReturn:
    """Shockley diode with clamped exponent and SPICE junction capacitance."""
    vd = signals.p1 - signals.p2
    vd_norm = jnp.clip(vd / (n * Vt), -80.0, EXP_CAP)
    i = Is * jnp.expm1(vd_norm)
    q = _junction_charge(vd, Cj0, Vj, m)
    return {"p1": i, "p2": -i}, {"p1": q, "p2": -q}


# ---------------------------------------------------------------------------
# Circuit parameters — must match circuits/diode_cascade.cir exactly
# ---------------------------------------------------------------------------
V_AMP = 50.0
F_SRC = 100e3
R_SER = 0.01
RS_D  = 0.042
C_VAL = 100e-9
IS_D  = 76.9e-12
N_D   = 1.45
CJ0_D = 26.5e-12
VJ_D  = 1.0
M_D   = 0.333

T_END = 5e-3
DT = 10e-9
N_STEPS = int(T_END / DT)
WARMUP_STEPS = 2
WARMUP_T_END = WARMUP_STEPS * DT

STEP_CONTROLLER = diffrax.PIDController(
    rtol=1e-3, atol=1e-4,
    pcoeff=0.2, icoeff=0.5, dcoeff=0.0,
    force_dtmin=True, dtmin=1E-6*DT, dtmax=DT,
    error_order=2,
)

HERE        = pathlib.Path(__file__).parent
CIR_FILE    = HERE / "circuits" / "diode_cascade.cir"
NG_OUTPUT   = pathlib.Path("/tmp/ngspice_diode_cascade.dat")
PLOT_OUTPUT = HERE / "diode_cascade_comparison.png"

NODES = ["v(1)", "v(10)", "v(20)"]


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

def solver_ngspice() -> SolverResult:
    run_ngspice(CIR_FILE, NODES, output_path=NG_OUTPUT)  # warmup
    t0 = time.perf_counter()
    ng_time, ng_v = run_ngspice(CIR_FILE, NODES, output_path=NG_OUTPUT)
    elapsed = time.perf_counter() - t0
    return SolverResult(
        name="ngspice",
        time=ng_time,
        signals=ng_v,
        elapsed=elapsed,
        n_steps=len(ng_time),
    )


def solver_circulax(n_save: int = 10_001) -> SolverResult:
    d_settings = {"Is": IS_D, "n": N_D, "Cj0": CJ0_D, "Vj": VJ_D, "m": M_D}

    t0 = time.perf_counter()
    net_dict = {
        "instances": {
            "GND":  {"component": "ground"},
            "VS":   {"component": "ac_source",  "settings": {"V": V_AMP, "freq": F_SRC}},
            "R1":   {"component": "resistor",   "settings": {"R": R_SER}},
            "C1":   {"component": "capacitor",  "settings": {"C": C_VAL}},
            "C2":   {"component": "capacitor",  "settings": {"C": C_VAL}},
            "C3":   {"component": "capacitor",  "settings": {"C": C_VAL}},
            "C4":   {"component": "capacitor",  "settings": {"C": C_VAL}},
            "RS1":  {"component": "resistor",   "settings": {"R": RS_D}},
            "D1":   {"component": "diode",      "settings": d_settings},
            "RS2":  {"component": "resistor",   "settings": {"R": RS_D}},
            "D2":   {"component": "diode",      "settings": d_settings},
            "RS3":  {"component": "resistor",   "settings": {"R": RS_D}},
            "D3":   {"component": "diode",      "settings": d_settings},
            "RS4":  {"component": "resistor",   "settings": {"R": RS_D}},
            "D4":   {"component": "diode",      "settings": d_settings},
        },
        "connections": {
            "GND,p1":  ("VS,p2", "RS1,p1", "C2,p1"),
            "VS,p1":   "R1,p1",
            "R1,p2":   ("C1,p1", "D1,p2", "RS2,p1", "C3,p1"),
            "RS1,p2":  "D1,p1",
            "C1,p2":   ("C3,p2", "D3,p2", "RS4,p1"),
            "C2,p2":   ("D2,p2", "RS3,p1", "C4,p1"),
            "RS2,p2":  "D2,p1",
            "RS3,p2":  "D3,p1",
            "RS4,p2":  "D4,p1",
            "C4,p2":   "D4,p2",
        },
    }
    models_map = {
        "ground":    lambda: 0,
        "ac_source": VoltageSourceAC,
        "resistor":  Resistor,
        "capacitor": Capacitor,
        "diode":     DiodeLimited,
    }
    groups, sys_size, port_map = compile_netlist(net_dict, models_map)
    linear_strategy = analyze_circuit(groups, sys_size, is_complex=False, backend="klu_split")
    y_op = linear_strategy.solve_dc(groups, jnp.zeros(sys_size))
    transient_sim = setup_transient(groups=groups, linear_strategy=linear_strategy,
                                    transient_solver=BDF2RefactoringTransientSolver)
    compile_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    saveat_w = diffrax.SaveAt(ts=jnp.linspace(0.0, WARMUP_T_END, 101))
    transient_sim(t0=0.0, t1=WARMUP_T_END, dt0=DT, y0=y_op,
                  saveat=saveat_w, max_steps=WARMUP_STEPS + 100,
                  stepsize_controller=STEP_CONTROLLER).ys.block_until_ready()
    warmup_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, T_END, n_save))
    sol = transient_sim(t0=0.0, t1=T_END, dt0=DT, y0=y_op,
                        saveat=saveat, max_steps=N_STEPS * 10,
                        stepsize_controller=STEP_CONTROLLER)
    sol.ys.block_until_ready()
    elapsed = time.perf_counter() - t0

    cx_time = np.asarray(sol.ts)
    return SolverResult(
        name="circulax",
        time=cx_time,
        signals={
            "v(1)":  np.asarray(sol.ys[:, port_map["R1,p2"]]),
            "v(10)": np.asarray(sol.ys[:, port_map["C2,p2"]]),
            "v(20)": np.asarray(sol.ys[:, port_map["C4,p2"]]),
        },
        elapsed=elapsed,
        n_steps=int(sol.stats["num_steps"]),
        compile_time=compile_time,
        warmup_time=warmup_time,
    )


# ---------------------------------------------------------------------------
# Solver registry — add new solvers here
# ---------------------------------------------------------------------------

SOLVERS: dict[str, SolverFn] = {
    "ngspice":  solver_ngspice,
    "circulax": solver_circulax,
}
REFERENCE = "ngspice"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Diode cascade testbench")
    parser.add_argument("--plot",   action="store_true")
    parser.add_argument("--n-save", type=int, default=10_001)
    args = parser.parse_args()

    solvers = dict(SOLVERS)
    solvers["circulax"] = lambda: solver_circulax(n_save=args.n_save)

    results = run_benchmark(
        solvers=solvers,
        reference=REFERENCE,
        nodes=NODES,
        title=(
            f"Diode Cascade Testbench (Cockcroft-Walton ×2)  "
            f"VS={V_AMP}V {F_SRC/1e3:.0f}kHz  "
            f"C={C_VAL*1e9:.0f}nF  T_end={T_END*1e3:.0f}ms  dt={DT*1e9:.0f}ns"
        ),
    )

    if args.plot:
        ref = results[REFERENCE]
        cx = results["circulax"]
        panels = []
        for node, title in [("v(1)", "V(1) — input node"),
                             ("v(10)", "V(10) — mid-stage"),
                             ("v(20)", "V(20) — output (4× multiplier)")]:
            err = cx.signals[node] - np.interp(cx.time, ref.time, ref.signals[node])
            panels += [
                {"title": title,
                 "ref_time": ref.time, "ref_signal": ref.signals[node],
                 "test_time": cx.time, "test_signal": cx.signals[node]},
                {"title": f"{title} — error",
                 "ref_time": cx.time, "ref_signal": err,
                 "test_time": cx.time, "test_signal": err,
                 "show_error": True},
            ]
        plot_comparison(
            ref_label="NGSpice (full D1N4007)",
            test_label="Circulax (DiodeLimited)",
            time_scale=1e3, time_unit="ms",
            panels=panels,
            output_path=PLOT_OUTPUT,
        )


if __name__ == "__main__":
    main()
