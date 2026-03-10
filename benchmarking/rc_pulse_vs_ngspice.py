"""RC pulse-train benchmark: Circulax vs NGSpice.

Circuit
-------
  vs 1 0  PULSE(0 1 1u 1u 1u 1m 2m)   — 0→1V, TD=TR=TF=1µs, PW=1ms, PER=2ms
  R1 1 2  1kΩ
  C1 2 0  1µF
  τ = RC = 1ms

Comparison metric: V(2) — capacitor voltage vs time.

Usage
-----
  pixi run -e benchmark bench_rc
  # or directly:
  pixi run -e benchmark python benchmarking/rc_pulse_vs_ngspice.py [--plot]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

# ---------------------------------------------------------------------------
# klujax (split solver) — must be on path before importing circulax
# ---------------------------------------------------------------------------
KLUJAX_SRC = pathlib.Path("/home/chris/code/klujax/klujax")
if KLUJAX_SRC.exists() and str(KLUJAX_SRC) not in sys.path:
    sys.path.insert(0, str(KLUJAX_SRC))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import diffrax  # noqa: E402 (after jax config)

# Make bench_utils importable when run directly from repo root
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from bench_utils.metrics import compare_waveforms  # noqa: E402
from bench_utils.ngspice import run_ngspice  # noqa: E402
from bench_utils.plotting import plot_comparison  # noqa: E402

from circulax.compiler import compile_netlist  # noqa: E402
from circulax.components.electronic import Capacitor, PulseVoltageSource, Resistor  # noqa: E402
from circulax.solvers import analyze_circuit, setup_transient  # noqa: E402

# ---------------------------------------------------------------------------
# Circuit parameters — must match circuits/rc_pulse.cir exactly
# ---------------------------------------------------------------------------
R = 1e3       # Ω
C = 1e-6      # F
V1 = 0.0      # low voltage
V2 = 1.0      # high voltage
TD = 1e-6     # delay
TR = 1e-6     # rise time
TF = 1e-6     # fall time
PW = 1e-3     # pulse width
PER = 2e-3    # period

T_END = 1.0   # total simulation time (s)
DT = 1e-6     # fixed timestep (matches NGSpice tran step)
N_STEPS = int(T_END / DT)   # 1_000_000 steps

WARMUP_STEPS = 1_000         # steps used for JIT-warmup run
WARMUP_T_END = WARMUP_STEPS * DT

HERE = pathlib.Path(__file__).parent
CIR_FILE = HERE / "circuits" / "rc_pulse.cir"
NG_OUTPUT = pathlib.Path("/tmp/ngspice_rc_pulse.dat")
PLOT_OUTPUT = HERE / "rc_pulse_comparison.png"


# ---------------------------------------------------------------------------
# Circulax runner — returns compiled objects so warmup and timed run share them
# ---------------------------------------------------------------------------

def build_circulax():
    """Compile netlist and return (transient_sim, y_op, port_map) ready to run."""
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "VS": {
                "component": "pulse_source",
                "settings": {"v1": V1, "v2": V2, "td": TD, "tr": TR, "tf": TF, "pw": PW, "per": PER},
            },
            "R1": {"component": "resistor", "settings": {"R": R}},
            "C1": {"component": "capacitor", "settings": {"C": C}},
        },
        "connections": {
            "GND,p1": ("VS,p2", "C1,p2"),
            "VS,p1": "R1,p1",
            "R1,p2": "C1,p1",
        },
    }
    models_map = {
        "resistor": Resistor,
        "capacitor": Capacitor,
        "pulse_source": PulseVoltageSource,
        "ground": lambda: 0,
    }
    groups, sys_size, port_map = compile_netlist(net_dict, models_map)
    linear_strategy = analyze_circuit(groups, sys_size, is_complex=False, backend="klu-split")
    y_op = linear_strategy.solve_dc(groups, jnp.zeros(sys_size))
    transient_sim = setup_transient(groups=groups, linear_strategy=linear_strategy)
    return transient_sim, y_op, port_map


def run_transient(transient_sim, y_op, t_end: float, n_save: int):
    """Execute one transient run and block until complete."""
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, t_end, n_save))
    sol = transient_sim(
        t0=0.0,
        t1=t_end,
        dt0=DT,
        y0=y_op,
        saveat=saveat,
        max_steps=int(jnp.ceil(t_end / DT)),
    )
    # Block on JAX async dispatch
    sol.ys.block_until_ready()
    return sol


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RC pulse benchmark: Circulax vs NGSpice")
    parser.add_argument("--plot", action="store_true", help="Save comparison plot")
    parser.add_argument(
        "--n-save",
        type=int,
        default=10_001,
        help="Circulax save points for the timed run (default: 10001 → 100µs spacing)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RC Pulse Benchmark: Circulax vs NGSpice")
    print(f"  R={R:.0f}Ω  C={C*1e6:.0f}µF  τ={R*C*1e3:.1f}ms")
    print(f"  PULSE: {V1}→{V2}V  TD={TD*1e6:.0f}µs  TR={TR*1e6:.0f}µs  "
          f"TF={TF*1e6:.0f}µs  PW={PW*1e3:.1f}ms  PER={PER*1e3:.1f}ms")
    print(f"  T_end={T_END}s  ({T_END/PER:.0f} periods)  dt={DT*1e6:.0f}µs  "
          f"N_steps={N_STEPS:,}")
    print(f"  klujax src: {KLUJAX_SRC}  (exists={KLUJAX_SRC.exists()})")
    print("=" * 60)

    # ── NGSpice ──────────────────────────────────────────────────────────────
    print("\n[NGSpice] Warmup run...")
    run_ngspice(CIR_FILE, ["v(1)", "v(2)"], output_path=NG_OUTPUT)

    print("[NGSpice] Timed run...")
    t0 = time.perf_counter()
    ng_time, ng_voltages = run_ngspice(CIR_FILE, ["v(1)", "v(2)"], output_path=NG_OUTPUT)
    t_ng = time.perf_counter() - t0
    ng_steps = len(ng_time)
    t_ng_per_step_us = t_ng / ng_steps * 1e6
    print(f"          {t_ng:.3f}s  ({ng_steps:,} timepoints  {t_ng_per_step_us:.3f} µs/step)")

    # ── Circulax ─────────────────────────────────────────────────────────────
    print("\n[Circulax] Compiling netlist + DC solve...")
    t0 = time.perf_counter()
    transient_sim, y_op, port_map = build_circulax()
    transient_sim_jit = jax.jit(transient_sim)
    t_compile = time.perf_counter() - t0
    print(f"           {t_compile:.3f}s")

    print(f"[Circulax] Warmup run ({WARMUP_STEPS:,} steps → {WARMUP_T_END*1e3:.1f}ms)...")
    t0 = time.perf_counter()
    run_transient(transient_sim_jit, y_op, WARMUP_T_END, n_save=101)
    t_warmup = time.perf_counter() - t0
    print(f"           {t_warmup:.3f}s  (JIT + {WARMUP_STEPS:,} steps)")

    print(f"[Circulax] Timed run ({N_STEPS:,} steps, n_save={args.n_save:,})...")
    t0 = time.perf_counter()
    sol = run_transient(transient_sim_jit, y_op, T_END, n_save=args.n_save)
    t_cx = time.perf_counter() - t0
    t_cx_per_step_us = t_cx / N_STEPS * 1e6
    print(f"           {t_cx:.3f}s  ({t_cx_per_step_us:.3f} µs/step)")

    cx_time = np.asarray(sol.ts)
    cx_v_src = np.asarray(sol.ys[:, port_map["VS,p1"]])
    cx_v_cap = np.asarray(sol.ys[:, port_map["C1,p1"]])

    # ── Results ──────────────────────────────────────────────────────────────
    print("\n── Timing ────────────────────────────────────────────────────")
    print(f"  NGSpice  compile : n/a (interpreter)")
    print(f"  Circulax compile : {t_compile:.3f}s  (netlist + DC)")
    print(f"  NGSpice  warmup  : 1 run (process startup)")
    print(f"  Circulax warmup  : {t_warmup:.3f}s  (JIT + {WARMUP_STEPS:,} steps)")
    print(f"  NGSpice  timed   : {t_ng:.3f}s  →  {t_ng_per_step_us:.3f} µs/step")
    print(f"  Circulax timed   : {t_cx:.3f}s  →  {t_cx_per_step_us:.3f} µs/step")
    print(f"  Speed ratio (timed): {t_cx/t_ng:.2f}x  ({'faster' if t_cx < t_ng else 'slower'} than NGSpice)")

    print("\n── Accuracy ──────────────────────────────────────────────────")
    cmp_v1 = compare_waveforms(ng_time, ng_voltages["v(1)"], cx_time, cx_v_src, node="v(1)")
    cmp_v2 = compare_waveforms(ng_time, ng_voltages["v(2)"], cx_time, cx_v_cap, node="v(2)")
    cmp_v1.print()
    cmp_v2.print()

    if args.plot:
        err_v2 = cx_v_cap - np.interp(cx_time, ng_time, ng_voltages["v(2)"])
        plot_comparison(
            ref_label="NGSpice",
            test_label="Circulax (klu_split)",
            time_scale=1e3,
            time_unit="ms",
            panels=[
                {
                    "title": "Source voltage V(1)",
                    "ref_time": ng_time, "ref_signal": ng_voltages["v(1)"],
                    "test_time": cx_time, "test_signal": cx_v_src,
                },
                {
                    "title": "Capacitor voltage V(2)",
                    "ref_time": ng_time, "ref_signal": ng_voltages["v(2)"],
                    "test_time": cx_time, "test_signal": cx_v_cap,
                },
                {
                    "title": "Circulax − NGSpice  [V(2)]",
                    "ref_time": cx_time, "ref_signal": err_v2,
                    "test_time": cx_time, "test_signal": err_v2,
                    "show_error": True,
                },
            ],
            output_path=PLOT_OUTPUT,
        )


if __name__ == "__main__":
    main()
