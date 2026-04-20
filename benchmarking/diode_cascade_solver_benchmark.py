"""Diode ladder internal solver comparison benchmark.

Circuits
--------
Two circuits are compared:

1. ``diode_cascade`` — fixed Cockcroft-Walton ×2 multiplier (4 diodes, ~13 nodes).
   Same topology as diode_cascade_testbench.py.

2. ``diode_ladder`` — scalable N-stage shunt rectifier ladder.
   Each stage adds one node, one diode (D1N4007 + series R), and one shunt cap.
   System size ≈ N + 4 nodes.  During the positive half-cycle all N diodes
   transition simultaneously, creating an N×N nonlinear Newton system.

       VS(50V, 1kHz) → R_s(50Ω) → n_0 → [R(50Ω) → n_i → D_i↓GND  C_i↓GND] × N → R_load(50Ω) → GND

Purpose
-------
Show where ``refactor`` (quadratic Newton) beats ``factor`` (frozen-Jacobian) and vice versa.
*Factor* wins when Newton converges in 1–2 iterations per step (linear and mildly nonlinear).
*Refactor* wins when Newton needs ≥3 iterations per step (large circuits, tight tolerances,
strong nonlinearity).

Controls tightened vs the ngspice comparison testbench:
  - PID step controller:  rtol=1e-5, atol=1e-6  (was rtol=1e-3, atol=1e-4)
  - dtmax lifted to 1µs (was 10ns) so the adaptive solver takes larger steps
  - Newton tolerances:    rtol=1e-8, atol=1e-8  (was rtol=1e-5, atol=1e-5)

Usage
-----
  pixi run -e benchmark python benchmarking/diode_cascade_solver_benchmark.py
  pixi run -e benchmark python benchmarking/diode_cascade_solver_benchmark.py \\
      --circuit ladder --stages 10 50 200
  pixi run -e benchmark python benchmarking/diode_cascade_solver_benchmark.py \\
      --backends klu_split_factor klu_split_refactor klu_rs_split_factor klu_rs_split_refactor
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

from circulax.compiler import compile_netlist  # noqa: E402
from circulax.components.base_component import PhysicsReturn, Signals, States, component  # noqa: E402
from circulax.components.electronic import (  # noqa: E402
    Capacitor,
    Resistor,
    VoltageSourceAC,
    _junction_charge,
)
from circulax.solvers import analyze_circuit, setup_transient  # noqa: E402
from circulax.solvers.transient import (  # noqa: E402
    BDF2FactorizedTransientSolver,
    BDF2RefactoringTransientSolver,
    BDF2VectorizedTransientSolver,
)

# ---------------------------------------------------------------------------
# DiodeLimited — clamped Shockley + SPICE junction capacitance
# ---------------------------------------------------------------------------

EXP_CAP = 40.0


@component(ports=("p1", "p2"))
def DiodeLimited(
    signals: Signals,
    s: States,
    Is: float = 1e-12,
    n: float = 1.0,
    Vt: float = 25.85e-3,
    Cj0: float = 0.0,
    Vj: float = 1.0,
    m: float = 0.5,
) -> PhysicsReturn:
    """Shockley diode with clamped exponent and SPICE junction capacitance."""
    vd = signals.p1 - signals.p2
    vd_norm = jnp.clip(vd / (n * Vt), -80.0, EXP_CAP)
    i = Is * jnp.expm1(vd_norm)
    q = _junction_charge(vd, Cj0, Vj, m)
    return {"p1": i, "p2": -i}, {"p1": q, "p2": -q}


# ---------------------------------------------------------------------------
# Circuit parameters — D1N4007 model
# ---------------------------------------------------------------------------
V_AMP = 50.0
F_SRC_CASCADE = 100e3   # 100 kHz — matches ngspice testbench
F_SRC_LADDER = 1e3      # 1 kHz — longer period → larger adaptive steps → more Newton iters
R_SER = 0.01
RS_D = 0.042
C_VAL = 100e-9
IS_D = 76.9e-12
N_D = 1.45
CJ0_D = 26.5e-12
VJ_D = 1.0
M_D = 0.333

R_LADDER_SERIES = 50.0  # series resistor between ladder nodes
R_LOAD = 50.0           # terminating load

T_END_CASCADE = 5e-3    # 5 ms (500 cycles at 100 kHz)
T_END_LADDER = 50e-3    # 50 ms (50 cycles at 1 kHz)
DT0 = 10e-9             # initial timestep guess

NEWTON_RTOL = 1e-8
NEWTON_ATOL = 1e-8

# Tighter step controller: larger dtmax forces bigger steps → more Newton iters
STEP_CONTROLLER = diffrax.PIDController(
    rtol=1e-5,
    atol=1e-6,
    pcoeff=0.2,
    icoeff=0.5,
    dcoeff=0.0,
    force_dtmin=True,
    dtmin=1e-14,
    dtmax=1e-6,      # 1µs — 10-100× larger than old 10ns cap
    error_order=2,
)

MODELS_MAP = {
    "ground": lambda: 0,
    "ac_source": VoltageSourceAC,
    "resistor": Resistor,
    "capacitor": Capacitor,
    "diode": DiodeLimited,
}

BACKENDS_TO_COMPARE = [
    "klu_rs_split_refactor",
    "klu_rs_split_factor",
    "klu_split_factor",
    "klu_split_refactor",
    "klu_rs_split",
]

REFERENCE_BACKEND = "klu_rs_split_refactor"


# ---------------------------------------------------------------------------
# Netlists
# ---------------------------------------------------------------------------


def build_cascade_netlist() -> tuple[dict, dict[str, str]]:
    """Cockcroft-Walton ×2 (4 diodes, ~13 nodes)."""
    d_settings = {"Is": IS_D, "n": N_D, "Cj0": CJ0_D, "Vj": VJ_D, "m": M_D}
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "VS": {"component": "ac_source", "settings": {"V": V_AMP, "freq": F_SRC_CASCADE}},
            "R1": {"component": "resistor", "settings": {"R": R_SER}},
            "C1": {"component": "capacitor", "settings": {"C": C_VAL}},
            "C2": {"component": "capacitor", "settings": {"C": C_VAL}},
            "C3": {"component": "capacitor", "settings": {"C": C_VAL}},
            "C4": {"component": "capacitor", "settings": {"C": C_VAL}},
            "RS1": {"component": "resistor", "settings": {"R": RS_D}},
            "D1": {"component": "diode", "settings": d_settings},
            "RS2": {"component": "resistor", "settings": {"R": RS_D}},
            "D2": {"component": "diode", "settings": d_settings},
            "RS3": {"component": "resistor", "settings": {"R": RS_D}},
            "D3": {"component": "diode", "settings": d_settings},
            "RS4": {"component": "resistor", "settings": {"R": RS_D}},
            "D4": {"component": "diode", "settings": d_settings},
        },
        "connections": {
            "GND,p1": ("VS,p2", "RS1,p1", "C2,p1"),
            "VS,p1": "R1,p1",
            "R1,p2": ("C1,p1", "D1,p2", "RS2,p1", "C3,p1"),
            "RS1,p2": "D1,p1",
            "C1,p2": ("C3,p2", "D3,p2", "RS4,p1"),
            "C2,p2": ("D2,p2", "RS3,p1", "C4,p1"),
            "RS2,p2": "D2,p1",
            "RS3,p2": "D3,p1",
            "RS4,p2": "D4,p1",
            "C4,p2": "D4,p2",
        },
    }
    # Nodes to monitor
    node_map = {"v(1)": "R1,p2", "v(10)": "C2,p2", "v(20)": "C4,p2"}
    return net, node_map


def build_ladder_netlist(n_stages: int) -> tuple[dict, dict[str, str]]:
    """N-stage shunt diode rectifier ladder.

    Each stage i adds: R_i (series, 50Ω), C_i (shunt cap, 100nF), RS_i + D_i (shunt diode).
    Node n_0 is after the source resistor; node n_i is after stage i's series resistor.
    All diodes switch simultaneously on each positive half-cycle, creating a coupled
    N-variable nonlinear system per Newton step.
    """
    d_settings = {"Is": IS_D, "n": N_D, "Cj0": CJ0_D, "Vj": VJ_D, "m": M_D}
    instances: dict = {
        "GND": {"component": "ground"},
        "VS": {"component": "ac_source", "settings": {"V": V_AMP, "freq": F_SRC_LADDER}},
        "Rs": {"component": "resistor", "settings": {"R": R_SER}},
    }
    connections: dict = {
        "GND,p1": ["VS,p2"],
        "VS,p1": "Rs,p1",
    }

    prev = "Rs,p2"
    for i in range(1, n_stages + 1):
        instances[f"Rser{i}"] = {"component": "resistor", "settings": {"R": R_LADDER_SERIES}}
        instances[f"C{i}"] = {"component": "capacitor", "settings": {"C": C_VAL}}
        instances[f"RSd{i}"] = {"component": "resistor", "settings": {"R": RS_D}}
        instances[f"D{i}"] = {"component": "diode", "settings": d_settings}
        connections[f"Rser{i},p1"] = prev
        # node n_i: Rser output, C shunt, RSd+D shunt
        node_i = f"Rser{i},p2"
        connections[f"C{i},p1"] = node_i
        connections[f"RSd{i},p1"] = node_i
        connections[f"RSd{i},p2"] = f"D{i},p1"
        connections[f"C{i},p2"] = "GND,p1"  # adds C to GND connection tuple
        connections[f"D{i},p2"] = "GND,p1"
        prev = node_i

    # Terminating load
    instances["Rload"] = {"component": "resistor", "settings": {"R": R_LOAD}}
    connections["Rload,p1"] = prev
    connections["Rload,p2"] = "GND,p1"

    # Collect all GND connections
    gnd_ports: list[str] = ["VS,p2"]
    for i in range(1, n_stages + 1):
        gnd_ports += [f"C{i},p2", f"D{i},p2"]
    gnd_ports.append("Rload,p2")
    connections["GND,p1"] = tuple(gnd_ports)

    net = {"instances": instances, "connections": connections}
    node_map = {
        "v_in": "Rs,p2",
        "v_out": f"Rser{n_stages},p2",
    }
    return net, node_map


# ---------------------------------------------------------------------------
# Per-backend runner
# ---------------------------------------------------------------------------


def run_one(net: dict, node_map: dict[str, str], t_end: float, backend: str) -> dict:
    """Compile, warm up, and time a circuit for one backend."""
    max_steps = max(1_000_000, int(t_end / 1e-9))
    warmup_t_end = 2 * (1.0 / F_SRC_LADDER)   # 2 low-freq cycles

    t0 = time.perf_counter()
    groups, sys_size, port_map = compile_netlist(net, MODELS_MAP)
    linear_strategy = analyze_circuit(groups, sys_size, is_complex=False, backend=backend)
    y_op = linear_strategy.solve_dc(groups, jnp.zeros(sys_size))

    # Pick the right transient solver class, then inject tight Newton tolerances
    if hasattr(linear_strategy, "refactor_jacobian"):
        tsolver_cls = BDF2RefactoringTransientSolver
    elif hasattr(linear_strategy, "factor_jacobian"):
        tsolver_cls = BDF2FactorizedTransientSolver
    else:
        tsolver_cls = BDF2VectorizedTransientSolver
    transient_sim = setup_transient(
        groups=groups,
        linear_strategy=linear_strategy,
        transient_solver=tsolver_cls(
            linear_solver=linear_strategy,
            newton_rtol=NEWTON_RTOL,
            newton_atol=NEWTON_ATOL,
        ),
    )
    compile_time = time.perf_counter() - t0

    node_indices = jnp.array([port_map[key] for key in node_map.values()])

    @jax.jit
    def save_nodes(t, y, args):  # noqa: ANN001, ANN202
        return y[node_indices]

    n_warmup_save = 201
    saveat_w = diffrax.SaveAt(ts=jnp.linspace(0.0, warmup_t_end, n_warmup_save), fn=save_nodes)
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, t_end, 10_001), fn=save_nodes)

    # ── Warmup ───────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    transient_sim(
        t0=0.0,
        t1=warmup_t_end,
        dt0=DT0,
        y0=y_op,
        saveat=saveat_w,
        max_steps=100_000,
        stepsize_controller=STEP_CONTROLLER,
    ).ys.block_until_ready()
    warmup_time = time.perf_counter() - t0

    # ── Timed run ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    sol = transient_sim(
        t0=0.0,
        t1=t_end,
        dt0=DT0,
        y0=y_op,
        saveat=saveat,
        max_steps=max_steps,
        stepsize_controller=STEP_CONTROLLER,
    )
    sol.ys.block_until_ready()
    elapsed = time.perf_counter() - t0

    n_steps = int(sol.stats["num_steps"])
    converged = sol.result == diffrax.RESULTS.successful

    ts = np.asarray(sol.ts)
    ys = np.asarray(sol.ys)  # (10_001, n_nodes)
    signals = {name: ys[:, i] for i, name in enumerate(node_map)}

    linear_strategy.cleanup()

    return {
        "backend": backend,
        "sys_size": sys_size,
        "compile_time": compile_time,
        "warmup_time": warmup_time,
        "elapsed": elapsed,
        "n_steps": n_steps,
        "us_per_step": elapsed / max(n_steps, 1) * 1e6,
        "converged": converged,
        "ts": ts,
        "signals": signals,
    }


# ---------------------------------------------------------------------------
# Accuracy vs reference
# ---------------------------------------------------------------------------


def _accuracy_vs_ref(result: dict, ref: dict, steady_start: float) -> dict[str, dict[str, float]]:
    """RMS and max absolute error in steady state for each monitored node."""
    metrics: dict[str, dict[str, float]] = {}
    mask = result["ts"] >= steady_start
    ref_mask = ref["ts"] >= steady_start
    for node in result["signals"]:
        v_test = result["signals"][node][mask]
        v_ref = np.interp(result["ts"][mask], ref["ts"][ref_mask], ref["signals"][node][ref_mask])
        err = v_test - v_ref
        metrics[node] = {
            "max_abs_mv": float(np.max(np.abs(err)) * 1e3),
            "rms_mv": float(np.sqrt(np.mean(err**2)) * 1e3),
        }
    return metrics


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_run_summary(results: dict[str, dict], backends: list[str], label: str) -> None:
    print(f"\n── Timing: {label} ──────────────────────────────────────────────────")  # noqa: T201
    hdr = f"  {'backend':<24}  {'sys':>5}  {'compile':>8}  {'warmup':>7}  {'sim':>7}  {'steps':>8}  {'µs/step':>9}  {'status':>6}"
    print(hdr)  # noqa: T201
    print("  " + "─" * (len(hdr) - 2))  # noqa: T201
    for b in backends:
        r = results.get(b, {})
        if "error" in r:
            print(f"  {b:<24}  ERROR: {r['error']}")  # noqa: T201
            continue
        status = "OK" if r.get("converged") else "FAIL"
        print(  # noqa: T201
            f"  {b:<24}  "
            f"{r['sys_size']:>5,}  "
            f"{r['compile_time']:>7.3f}s  "
            f"{r['warmup_time']:>6.3f}s  "
            f"{r['elapsed']:>6.2f}s  "
            f"{r['n_steps']:>8,}  "
            f"{r['us_per_step']:>8.2f}µs  "
            f"{status:>6}"
        )


def _print_accuracy(results: dict[str, dict], backends: list[str], ref_backend: str, steady_start: float) -> None:
    if ref_backend not in results or "error" in results[ref_backend]:
        print(f"\n  [accuracy skipped — reference '{ref_backend}' failed]")  # noqa: T201
        return
    ref = results[ref_backend]
    nodes = list(ref.get("signals", {}).keys())
    if not nodes:
        return
    print(f"\n── Accuracy vs {ref_backend} (steady-state window) ──────────────────")  # noqa: T201
    col = "  max|err|       RMS"
    hdr = f"  {'backend':<24}" + "".join(f"  {n:>{len(col)}}" for n in nodes)
    print(hdr)  # noqa: T201
    sub = f"  {'':24}" + "".join(f"  {col}" for _ in nodes)
    print(sub)  # noqa: T201
    print("  " + "─" * (len(sub) - 2))  # noqa: T201
    for b in backends:
        if b == ref_backend:
            print(f"  {b:<24}  [reference]")  # noqa: T201
            continue
        r = results.get(b, {})
        if "error" in r or not r.get("converged"):
            print(f"  {b:<24}  [no data]")  # noqa: T201
            continue
        metrics = _accuracy_vs_ref(r, ref, steady_start)
        row = f"  {b:<24}"
        for node in nodes:
            m = metrics[node]
            row += f"  {m['max_abs_mv']:>9.3f}mV  {m['rms_mv']:>7.3f}mV"
        print(row)  # noqa: T201


def _run_circuit(label: str, net: dict, node_map: dict, t_end: float, steady_start: float, backends: list[str], ref: str) -> None:
    """Run all backends on one circuit and print results."""
    print(f"\n{'═' * 70}")  # noqa: T201
    print(f"  {label}")  # noqa: T201
    print(f"{'═' * 70}")  # noqa: T201
    results: dict[str, dict] = {}
    for b in backends:
        backends_to_run = backends if ref in backends else [ref, *backends]
        _ = backends_to_run  # reference handled outside loop if needed
        print(f"\n  [{b}]  running...", flush=True)  # noqa: T201
        try:
            r = run_one(net, node_map, t_end, b)
            results[b] = r
            status = "✓" if r["converged"] else "✗ FAILED"
            print(  # noqa: T201
                f"    compile={r['compile_time']:.3f}s  warmup={r['warmup_time']:.3f}s  "
                f"sim={r['elapsed']:.3f}s  steps={r['n_steps']:,}  "
                f"{r['us_per_step']:.2f}µs/step  {status}"
            )
        except Exception as exc:  # noqa: BLE001
            results[b] = {"error": str(exc), "converged": False}
            print(f"    ERROR: {exc}")  # noqa: T201
    _print_run_summary(results, backends, label)
    _print_accuracy(results, backends, ref, steady_start)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run the solver comparison benchmark."""
    parser = argparse.ArgumentParser(description="Diode solver comparison benchmark")
    parser.add_argument(
        "--circuit",
        choices=["cascade", "ladder", "both"],
        default="both",
        help="Which circuit to benchmark (default: both)",
    )
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        default=[10, 50, 200],
        metavar="N",
        help="Ladder stages to sweep (default: 10 50 200)",
    )
    parser.add_argument(
        "--backends",
        type=str,
        nargs="+",
        default=BACKENDS_TO_COMPARE,
        metavar="BACKEND",
        help=f"Backends to compare (default: {' '.join(BACKENDS_TO_COMPARE)})",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=REFERENCE_BACKEND,
        help=f"Reference backend for accuracy (default: {REFERENCE_BACKEND})",
    )
    args = parser.parse_args()

    backends = args.backends
    ref = args.reference
    if ref not in backends:
        backends = [ref, *backends]

    print("=" * 70)  # noqa: T201
    print("Diode Solver Benchmark  (tightened controls)")  # noqa: T201
    print(f"  Newton: rtol={NEWTON_RTOL:.0e}  atol={NEWTON_ATOL:.0e}")  # noqa: T201
    print("  Step: rtol=1e-5  atol=1e-6  dtmax=1µs")  # noqa: T201
    print(f"  Backends: {backends}  Reference: {ref}")  # noqa: T201
    print("=" * 70)  # noqa: T201

    run_cascade = args.circuit in ("cascade", "both")
    run_ladder = args.circuit in ("ladder", "both")

    if run_cascade:
        net, node_map = build_cascade_netlist()
        _run_circuit(
            label=f"Cockcroft-Walton ×2  ({V_AMP}V, {F_SRC_CASCADE / 1e3:.0f}kHz, 4 diodes)",
            net=net,
            node_map=node_map,
            t_end=T_END_CASCADE,
            steady_start=T_END_CASCADE - 2.0 / F_SRC_CASCADE,
            backends=backends,
            ref=ref,
        )

    if run_ladder:
        for n_stages in sorted(set(args.stages)):
            net, node_map = build_ladder_netlist(n_stages)
            _run_circuit(
                label=(
                    f"Diode ladder  N={n_stages} stages  "
                    f"({V_AMP}V, {F_SRC_LADDER / 1e3:.0f}kHz)  sys≈{n_stages + 4} nodes"
                ),
                net=net,
                node_map=node_map,
                t_end=T_END_LADDER,
                steady_start=T_END_LADDER - 2.0 / F_SRC_LADDER,
                backends=backends,
                ref=ref,
            )


if __name__ == "__main__":
    main()
