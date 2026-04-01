"""RC pulse-train testbench.

Circuit
-------
  vs 1 0  PULSE(0 1 1u 1u 1u 1m 2m)   — 0→1V, TD=TR=TF=1µs, PW=1ms, PER=2ms
  R1 1 2  1kΩ
  C1 2 0  1µF
  τ = RC = 1ms

Nodes compared: v(1) [source], v(2) [capacitor]

Usage
-----
  pixi run -e benchmark python benchmarking/rc_pulse_testbench.py [--plot]
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
from circulax.components.electronic import Capacitor, PulseVoltageSource, Resistor  # noqa: E402
from circulax.solvers import BDF2RefactoringTransientSolver, analyze_circuit, setup_transient  # noqa: E402

# ---------------------------------------------------------------------------
# Circuit parameters — must match circuits/rc_pulse.cir exactly
# ---------------------------------------------------------------------------
R = 1e3
C = 1e-6
V1, V2 = 0.0, 1.0
TD, TR, TF, PW, PER = 1e-6, 1e-6, 1e-6, 1e-3, 2e-3

T_END = 1.0
DT = 1e-6
N_STEPS = int(T_END / DT)
WARMUP_STEPS = 2
WARMUP_T_END = WARMUP_STEPS * DT

STEP_CONTROLLER = diffrax.PIDController(
    rtol=1e-3, atol=1e-4,
    pcoeff=0.2, icoeff=0.5, dcoeff=0.0,
    force_dtmin=True, dtmin=1E-6*DT, dtmax=DT,
    error_order=2,
)

HERE = pathlib.Path(__file__).parent
CIR_FILE = HERE / "circuits" / "rc_pulse.cir"
NG_OUTPUT = pathlib.Path("/tmp/ngspice_rc_pulse.dat")
PLOT_OUTPUT = HERE / "rc_pulse_comparison.png"

NODES = ["v(1)", "v(2)"]


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
    t0 = time.perf_counter()
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "VS": {"component": "pulse_source", "settings": {
                "v1": V1, "v2": V2, "td": TD, "tr": TR, "tf": TF, "pw": PW, "per": PER,
            }},
            "R1": {"component": "resistor",  "settings": {"R": R}},
            "C1": {"component": "capacitor", "settings": {"C": C}},
        },
        "connections": {
            "GND,p1": ("VS,p2", "C1,p2"),
            "VS,p1":  "R1,p1",
            "R1,p2":  "C1,p1",
        },
    }
    models_map = {
        "ground":       lambda: 0,
        "pulse_source": PulseVoltageSource,
        "resistor":     Resistor,
        "capacitor":    Capacitor,
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
            "v(1)": np.asarray(sol.ys[:, port_map["VS,p1"]]),
            "v(2)": np.asarray(sol.ys[:, port_map["C1,p1"]]),
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
    parser = argparse.ArgumentParser(description="RC pulse testbench")
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
            f"RC Pulse Testbench  "
            f"R={R:.0f}Ω  C={C*1e6:.0f}µF  τ={R*C*1e3:.1f}ms  "
            f"T_end={T_END}s  dt={DT*1e6:.0f}µs"
        ),
    )

    if args.plot:
        ref = results[REFERENCE]
        cx = results["circulax"]
        err_v2 = cx.signals["v(2)"] - np.interp(cx.time, ref.time, ref.signals["v(2)"])
        plot_comparison(
            ref_label="NGSpice",
            test_label="Circulax (klu-split)",
            time_scale=1e3,
            time_unit="ms",
            panels=[
                {"title": "Source voltage V(1)",
                 "ref_time": ref.time, "ref_signal": ref.signals["v(1)"],
                 "test_time": cx.time, "test_signal": cx.signals["v(1)"]},
                {"title": "Capacitor voltage V(2)",
                 "ref_time": ref.time, "ref_signal": ref.signals["v(2)"],
                 "test_time": cx.time, "test_signal": cx.signals["v(2)"]},
                {"title": "Circulax − NGSpice  [V(2)]",
                 "ref_time": cx.time, "ref_signal": err_v2,
                 "test_time": cx.time, "test_signal": err_v2,
                 "show_error": True},
            ],
            output_path=PLOT_OUTPUT,
        )


if __name__ == "__main__":
    main()
