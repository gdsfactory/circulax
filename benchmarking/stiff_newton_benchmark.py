"""Stiff Newton benchmark: expose the refactor vs factor crossover.

When Newton converges in 1 iteration per step (small dt, mildly nonlinear),
``factor`` always wins — refactor adds overhead with no benefit.  The advantage
of ``refactor`` (quadratic Newton) appears when Newton needs ≥3 iterations per
step, which requires:

  1. **Large timesteps** relative to circuit dynamics (frozen Jacobian at the
     predicted state is far from the converged Jacobian).
  2. **Strong nonlinearity** (Jacobian changes significantly between iterates).

This benchmark creates that regime using an **LC+diode chain**:

    VS(50V, 1kHz) ─ Rs ─ n0 ─ L1 ─ n1 ─ L2 ─ ... ─ LN ─ nN ─ Rload ─ GND
                              |           |                   |
                              C1(100nF)   C2(100nF)           CN(100nF)
                              RSd1+D1↓    RSd2+D2↓            RSdN+DN↓ (all to GND)

LC resonance: f_r = 1/(2π√LC) ≈ 50 kHz (period ≈ 20 µs).
Driven at 1 kHz → stiffness ratio ≈ 50.

The key control is ``ConstantStepSize(dt)``.  With the PID controller from the
diode_cascade benchmark (dtmax=1µs ≪ LC period) Newton converges in 1 iteration.
Here we fix dt = 5–40% of the LC period, forcing the solver to bridge large state
changes in a single step.  The frozen Jacobian (factor) then requires many more
Newton iterations than the fresh Jacobian (refactor), and the crossover becomes
visible.

Usage::

  pixi run -e benchmark python benchmarking/stiff_newton_benchmark.py
  pixi run -e benchmark python benchmarking/stiff_newton_benchmark.py \\
      --stages 5 10 20 --dt 1e-6 5e-6 10e-6 20e-6 \\
      --backends klu_rs_split_factor klu_rs_split_refactor klu_split_factor klu_split_refactor
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

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
    Inductor,
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
# Circuit parameters
# ---------------------------------------------------------------------------

V_AMP = 50.0        # AC source amplitude (V)
F_SRC = 1e3         # driving frequency (Hz)
L_VAL = 100e-6      # inductor per stage (H)   — f_r ≈ 50 kHz
C_VAL = 100e-9      # cap per stage (F)
import math
F_R = 1.0 / (2.0 * math.pi * math.sqrt(L_VAL * C_VAL))  # ≈ 50.3 kHz

R_SRC = 10.0        # source series resistance (Ω)
R_LOAD = 50.0       # terminating load (Ω)
RS_D = 0.042        # diode series resistance (Ω)
IS_D = 76.9e-12     # diode saturation current (A)
N_D = 1.45          # diode ideality factor
CJ0_D = 26.5e-12    # junction capacitance (F)
VJ_D = 1.0          # junction potential (V)
M_D = 0.333         # grading coefficient

T_END = 5e-3        # 5 ms = 5 source cycles at 1 kHz; ≈ 250 LC periods

NEWTON_RTOL = 1e-8
NEWTON_ATOL = 1e-8
# Allow more Newton iterations so factor can converge even with large dt.
# Refactor needs fewer iterations (quadratic), factor may need many (linear).
NEWTON_MAX_STEPS = 100

BACKENDS_TO_COMPARE = [
    "klu_rs_split_factor",
    "klu_rs_split_refactor",
    "klu_split_factor",
    "klu_split_refactor",
]
REFERENCE_BACKEND = "klu_rs_split_refactor"

DEFAULT_DT_VALUES = [1e-6, 5e-6, 10e-6, 20e-6]   # 5%, 25%, 50%, 100% of LC period

MODELS_MAP = {
    "ground": lambda: 0,
    "ac_source": VoltageSourceAC,
    "resistor": Resistor,
    "capacitor": Capacitor,
    "inductor": Inductor,
    "diode": DiodeLimited,
}


# ---------------------------------------------------------------------------
# Netlist builder
# ---------------------------------------------------------------------------


def build_lc_diode_chain(n_stages: int) -> tuple[dict, dict[str, str]]:
    """N-stage series-inductor / shunt-cap / shunt-diode chain.

    Each stage replaces the resistor of the diode ladder with an inductor,
    creating stiff LC dynamics (f_r ≈ 50 kHz) on top of the diode nonlinearity.
    During each AC cycle the inductors ring at 50 kHz while the diodes clamp
    negative node voltages — a regime where the Jacobian changes dramatically
    between Newton iterates when the step size is a significant fraction of the
    LC period.
    """
    d_settings = {"Is": IS_D, "n": N_D, "Cj0": CJ0_D, "Vj": VJ_D, "m": M_D}
    instances: dict = {
        "GND": {"component": "ground"},
        "VS": {"component": "ac_source", "settings": {"V": V_AMP, "freq": F_SRC}},
        "Rs": {"component": "resistor", "settings": {"R": R_SRC}},
    }
    connections: dict = {
        "GND,p1": ["VS,p2"],
        "VS,p1": "Rs,p1",
    }

    prev = "Rs,p2"
    for i in range(1, n_stages + 1):
        instances[f"L{i}"] = {"component": "inductor", "settings": {"L": L_VAL}}
        instances[f"C{i}"] = {"component": "capacitor", "settings": {"C": C_VAL}}
        instances[f"RSd{i}"] = {"component": "resistor", "settings": {"R": RS_D}}
        instances[f"D{i}"] = {"component": "diode", "settings": d_settings}
        connections[f"L{i},p1"] = prev
        node_i = f"L{i},p2"
        connections[f"C{i},p1"] = node_i
        connections[f"RSd{i},p1"] = node_i
        connections[f"RSd{i},p2"] = f"D{i},p1"
        connections[f"C{i},p2"] = "GND,p1"
        connections[f"D{i},p2"] = "GND,p1"
        prev = node_i

    instances["Rload"] = {"component": "resistor", "settings": {"R": R_LOAD}}
    connections["Rload,p1"] = prev
    connections["Rload,p2"] = "GND,p1"

    gnd_ports: list[str] = ["VS,p2"]
    for i in range(1, n_stages + 1):
        gnd_ports += [f"C{i},p2", f"D{i},p2"]
    gnd_ports.append("Rload,p2")
    connections["GND,p1"] = tuple(gnd_ports)

    net = {"instances": instances, "connections": connections}
    node_map = {"v_in": "Rs,p2", "v_out": f"L{n_stages},p2"}
    return net, node_map


# ---------------------------------------------------------------------------
# Per-backend-per-dt runner
# ---------------------------------------------------------------------------


def _pick_tsolver_cls(linear_strategy):
    if hasattr(linear_strategy, "refactor_jacobian"):
        return BDF2RefactoringTransientSolver
    if hasattr(linear_strategy, "factor_jacobian"):
        return BDF2FactorizedTransientSolver
    return BDF2VectorizedTransientSolver


def run_one(net: dict, node_map: dict[str, str], t_end: float, backend: str, dt: float) -> dict:
    """Compile, warm up, and time one backend at a fixed step size."""
    n_steps_run = int(t_end / dt)
    warmup_t_end = 2.0 / F_SRC  # 2 source cycles as warmup
    n_steps_warmup = int(warmup_t_end / dt)

    t0 = time.perf_counter()
    groups, sys_size, port_map = compile_netlist(net, MODELS_MAP)
    linear_strategy = analyze_circuit(groups, sys_size, is_complex=False, backend=backend)
    y_op = linear_strategy.solve_dc(groups, jnp.zeros(sys_size))
    tsolver_cls = _pick_tsolver_cls(linear_strategy)
    transient_sim = setup_transient(
        groups=groups,
        linear_strategy=linear_strategy,
        transient_solver=tsolver_cls(
            linear_solver=linear_strategy,
            newton_rtol=NEWTON_RTOL,
            newton_atol=NEWTON_ATOL,
            newton_max_steps=NEWTON_MAX_STEPS,
        ),
    )
    compile_time = time.perf_counter() - t0

    node_indices = jnp.array([port_map[key] for key in node_map.values()])

    @jax.jit
    def save_nodes(t, y, args):  # noqa: ANN001, ANN202
        return y[node_indices]

    saveat_w = diffrax.SaveAt(ts=jnp.linspace(0.0, warmup_t_end, min(201, n_steps_warmup + 1)), fn=save_nodes)
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, t_end, min(10_001, n_steps_run + 1)), fn=save_nodes)
    step_ctrl = diffrax.ConstantStepSize()

    # ── Warmup (compiles JIT) ────────────────────────────────────────────
    t0 = time.perf_counter()
    transient_sim(
        t0=0.0, t1=warmup_t_end, dt0=dt, y0=y_op,
        saveat=saveat_w, max_steps=n_steps_warmup + 10,
        stepsize_controller=step_ctrl,
    ).ys.block_until_ready()
    warmup_time = time.perf_counter() - t0

    # ── Timed run ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    sol = transient_sim(
        t0=0.0, t1=t_end, dt0=dt, y0=y_op,
        saveat=saveat, max_steps=n_steps_run + 10,
        stepsize_controller=step_ctrl,
    )
    sol.ys.block_until_ready()
    elapsed = time.perf_counter() - t0

    n_steps_actual = int(sol.stats["num_steps"])
    converged = sol.result == diffrax.RESULTS.successful

    linear_strategy.cleanup()

    return {
        "backend": backend,
        "dt": dt,
        "sys_size": sys_size,
        "compile_time": compile_time,
        "warmup_time": warmup_time,
        "elapsed": elapsed,
        "n_steps": n_steps_actual,
        "us_per_step": elapsed / max(n_steps_actual, 1) * 1e6,
        "converged": converged,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _fmt_cell(result: dict | None) -> str:
    if result is None:
        return "       n/a"
    if "error" in result:
        return "     ERROR"
    if not result["converged"]:
        return "      FAIL"
    return f"{result['us_per_step']:>9.2f}µs"


def _print_crossover_table(
    results: dict[tuple[str, float], dict],
    backends: list[str],
    dt_values: list[float],
    label: str,
) -> None:
    """Print backends as rows, dt values as columns."""
    dt_labels = [f"dt={dt * 1e6:.0f}µs" for dt in dt_values]
    col_w = 12
    hdr_dt = "".join(f"  {lbl:>{col_w}}" for lbl in dt_labels)
    print(f"\n── {label} ──")  # noqa: T201
    print(f"  {'backend':<28}{hdr_dt}  compile  warmup")  # noqa: T201
    print("  " + "─" * (28 + col_w * len(dt_values) + 2 + 16))  # noqa: T201
    for b in backends:
        row = f"  {b:<28}"
        compile_t = "  n/a"
        warmup_t = "  n/a"
        for dt in dt_values:
            r = results.get((b, dt))
            row += f"  {_fmt_cell(r):>{col_w}}"
            if r and "error" not in r:
                compile_t = f"{r['compile_time']:6.2f}s"
                warmup_t = f"{r['warmup_time']:5.2f}s"
        print(f"{row}  {compile_t}  {warmup_t}")  # noqa: T201


def _print_legend(dt_values: list[float]) -> None:
    lc_period_us = 1e6 / F_R
    print(f"\n  LC period ≈ {lc_period_us:.1f} µs  (f_r ≈ {F_R / 1e3:.1f} kHz)")  # noqa: T201
    fracs = [f"dt={dt * 1e6:.0f}µs = {dt * F_R * 100:.0f}% of LC period" for dt in dt_values]
    for f in fracs:
        print(f"    {f}")  # noqa: T201
    print()  # noqa: T201


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901
    """Parse CLI arguments and run the stiff Newton benchmark."""
    parser = argparse.ArgumentParser(description="Stiff Newton benchmark (LC+diode chain, fixed dt)")
    parser.add_argument("--stages", type=int, nargs="+", default=[5, 10, 20], metavar="N")
    parser.add_argument("--dt", type=float, nargs="+", default=DEFAULT_DT_VALUES, metavar="DT",
                        help="Fixed step sizes in seconds (default: 1µs 5µs 10µs 20µs)")
    parser.add_argument("--backends", type=str, nargs="+", default=BACKENDS_TO_COMPARE)
    parser.add_argument("--reference", type=str, default=REFERENCE_BACKEND)
    args = parser.parse_args()

    backends = args.backends
    if args.reference not in backends:
        backends = [args.reference, *backends]
    dt_values = sorted(args.dt)

    lc_period_us = 1e6 / F_R
    print("=" * 70)  # noqa: T201
    print("Stiff Newton Benchmark  —  LC+diode chain, ConstantStepSize")  # noqa: T201
    print(f"  Circuit: L={L_VAL * 1e6:.0f}µH  C={C_VAL * 1e9:.0f}nF  f_r≈{F_R / 1e3:.1f}kHz  (period≈{lc_period_us:.1f}µs)")  # noqa: T201
    print(f"  Source:  {V_AMP}V  {F_SRC / 1e3:.1f}kHz  →  stiffness ratio ≈ {F_R / F_SRC:.0f}×")  # noqa: T201
    print(f"  Newton:  rtol={NEWTON_RTOL:.0e}  atol={NEWTON_ATOL:.0e}  max_steps={NEWTON_MAX_STEPS}")  # noqa: T201
    print(f"  dt:      {[f'{dt*1e6:.0f}µs' for dt in dt_values]}  ({[f'{dt*F_R*100:.0f}%' for dt in dt_values]} of LC period)")  # noqa: T201
    print(f"  Backends: {backends}")  # noqa: T201
    print("=" * 70)  # noqa: T201
    _print_legend(dt_values)

    for n_stages in sorted(set(args.stages)):
        net, node_map = build_lc_diode_chain(n_stages)
        # Estimate sys_size: 1 per node (≈ 2*N+3) + 1 per inductor state (N) + diode/cap contributions
        sys_est = 3 * n_stages + 5
        label = f"LC+diode chain  N={n_stages} stages  sys≈{sys_est} vars"
        print(f"\n{'═' * 70}")  # noqa: T201
        print(f"  {label}")  # noqa: T201
        print(f"{'═' * 70}")  # noqa: T201

        results: dict[tuple[str, float], dict] = {}
        for b in backends:
            for dt in dt_values:
                key = (b, dt)
                print(f"\n  [{b}  dt={dt * 1e6:.0f}µs]  running...", flush=True)  # noqa: T201
                try:
                    r = run_one(net, node_map, T_END, b, dt)
                    results[key] = r
                    status = "✓" if r["converged"] else "✗ DIVERGED"
                    print(  # noqa: T201
                        f"    compile={r['compile_time']:.3f}s  warmup={r['warmup_time']:.3f}s  "
                        f"sim={r['elapsed']:.3f}s  steps={r['n_steps']:,}  "
                        f"{r['us_per_step']:.2f}µs/step  {status}"
                    )
                except Exception as exc:  # noqa: BLE001
                    results[key] = {"error": str(exc), "converged": False}
                    print(f"    ERROR: {exc}")  # noqa: T201

        _print_crossover_table(results, backends, dt_values, label)


if __name__ == "__main__":
    main()
