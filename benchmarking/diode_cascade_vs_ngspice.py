"""Diode cascade (Cockcroft-Walton ×2) benchmark: Circulax vs NGSpice.

Circuit
-------
  vs  a  0   SIN(0, 50V, 100kHz)
  r1  a  1   0.01Ω
  c1  1  2   100nF
  d1  0  1   D1N4007  (anode=0, cathode=1)
  c2  0  10  100nF
  d2  1  10  D1N4007
  c3  1  2   100nF    (parallel with c1)
  d3  10 2   D1N4007
  c4  10 20  100nF
  d4  2  20  D1N4007

Nodes: a, 0(GND), 1, 2, 10, 20
Output: V(20) — 4× peak voltage multiplier

Diode model:  D1N4007
  NGSpice:    IS=76.9p, RS=42mΩ, N=1.45, CJO=26.5pF, M=0.333, BV=1kV
  Circulax:   IS=76.9p, RS=42mΩ, N=1.45, CJO=26.5pF, M=0.333  (RS explicit, BV omitted)

Usage
-----
  pixi run -e benchmark python benchmarking/diode_cascade_vs_ngspice.py [--plot]
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

from bench_utils.metrics import compare_waveforms  # noqa: E402
from bench_utils.ngspice import run_ngspice  # noqa: E402
from bench_utils.plotting import plot_comparison  # noqa: E402

from circulax.compiler import compile_netlist  # noqa: E402
from circulax.components.base_component import component  # noqa: E402
from circulax.components.electronic import (  # noqa: E402
    Capacitor,
    Resistor,
    VoltageSourceAC,
    _junction_charge,
)
from circulax.components.base_component import Signals, States, PhysicsReturn  # noqa: E402
from circulax.solvers import analyze_circuit, setup_transient  # noqa: E402


# ---------------------------------------------------------------------------
# Diode with argument-clamped Shockley equation + SPICE junction capacitance.
#
# The stock Diode clips Vd at ±5V, which still allows exp(133) ≈ 10^57 A —
# large enough to make the Newton Jacobian singular on high-voltage circuits.
# Here we clip the normalised exponent at EXP_CAP (default 40), giving a
# max current of IS * exp(40) ≈ 18 kA — physically enormous but numerically
# tractable (Jacobian element ≈ 18e3 / (n*Vt) ~ 480 kA/V rather than 10^49).
#
# Junction capacitance uses _junction_charge from electronic.py (fixed to
# return Q with dQ/dvd = +Cj > 0 and a non-diverging linear extrapolation).
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
    """Shockley diode with clamped exponent and SPICE junction capacitance.

    Parameters
    ----------
    Is  : saturation current (A)
    n   : ideality factor
    Vt  : thermal voltage (V)
    Cj0 : zero-bias junction capacitance (F); 0 disables charge term
    Vj  : built-in junction potential (V)
    m   : junction grading coefficient
    """
    vd = signals.p1 - signals.p2
    vd_norm = jnp.clip(vd / (n * Vt), -80.0, EXP_CAP)
    i = Is * jnp.expm1(vd_norm)
    q = _junction_charge(vd, Cj0, Vj, m)
    return {"p1": i, "p2": -i}, {"p1": q, "p2": -q}

# ---------------------------------------------------------------------------
# Circuit parameters — must match circuits/diode_cascade.cir exactly
# ---------------------------------------------------------------------------
V_AMP   = 50.0      # V  (sine amplitude)
F_SRC   = 100e3     # Hz (source frequency)
R_SER   = 0.01      # Ω  (series resistance, SPICE r1)
RS_D    = 0.042     # Ω  (D1N4007 RS — added in series to each diode)
C_VAL   = 100e-9    # F  (all capacitors)
IS_D    = 76.9e-12  # A  (D1N4007 saturation current)
N_D     = 1.45      #    (D1N4007 ideality factor)
CJ0_D   = 26.5e-12  # F  (D1N4007 zero-bias junction capacitance)
VJ_D    = 1.0       # V  (built-in potential, SPICE default)
M_D     = 0.333     #    (D1N4007 junction grading coefficient)

T_END   = 5e-3      # s  (5ms — 500 cycles at 100kHz)
DT      = 10e-9     # s  (10ns step, matching NGSpice)
N_STEPS = int(T_END / DT)   # 500_000

WARMUP_STEPS = 1_000
WARMUP_T_END = WARMUP_STEPS * DT

HERE       = pathlib.Path(__file__).parent
CIR_FILE   = HERE / "circuits" / "diode_cascade.cir"
NG_OUTPUT  = pathlib.Path("/tmp/ngspice_diode_cascade.dat")
PLOT_OUTPUT = HERE / "diode_cascade_comparison.png"

NG_NODES = ["v(1)", "v(10)", "v(20)"]


# ---------------------------------------------------------------------------
# Circulax
# ---------------------------------------------------------------------------

def build_circulax():
    """Compile netlist; return (transient_sim, y_op, port_map).

    Each diode is modelled as RS (series resistor) + DiodeLimited so that:
    - RS matches the D1N4007 bulk resistance (42 mΩ)
    - DiodeLimited clamps the exponential argument to prevent Jacobian blow-up
    Nodes RS_Dx are internal intermediate nodes between RS and D junction.
    """
    net_dict = {
        "instances": {
            "GND":   {"component": "ground"},
            "VS":    {"component": "ac_source",  "settings": {"V": V_AMP, "freq": F_SRC}},
            "R1":    {"component": "resistor",   "settings": {"R": R_SER}},
            "C1":    {"component": "capacitor",  "settings": {"C": C_VAL}},
            "C2":    {"component": "capacitor",  "settings": {"C": C_VAL}},
            "C3":    {"component": "capacitor",  "settings": {"C": C_VAL}},
            "C4":    {"component": "capacitor",  "settings": {"C": C_VAL}},
            # Each diode split into RS resistor + ideal junction:
            #   D1: GND(0) → RS1 → j1 → node1   (anode=0, cathode=1)
            "RS1":   {"component": "resistor",   "settings": {"R": RS_D}},
            "D1":    {"component": "diode",      "settings": {"Is": IS_D, "n": N_D, "Cj0": CJ0_D, "Vj": VJ_D, "m": M_D}},
            #   D2: node1 → RS2 → j2 → node10
            "RS2":   {"component": "resistor",   "settings": {"R": RS_D}},
            "D2":    {"component": "diode",      "settings": {"Is": IS_D, "n": N_D, "Cj0": CJ0_D, "Vj": VJ_D, "m": M_D}},
            #   D3: node10 → RS3 → j3 → node2
            "RS3":   {"component": "resistor",   "settings": {"R": RS_D}},
            "D3":    {"component": "diode",      "settings": {"Is": IS_D, "n": N_D, "Cj0": CJ0_D, "Vj": VJ_D, "m": M_D}},
            #   D4: node2 → RS4 → j4 → node20
            "RS4":   {"component": "resistor",   "settings": {"R": RS_D}},
            "D4":    {"component": "diode",      "settings": {"Is": IS_D, "n": N_D, "Cj0": CJ0_D, "Vj": VJ_D, "m": M_D}},
        },
        "connections": {
            # node 0 (GND) — D1 anode side and C2
            "GND,p1":  ("VS,p2", "RS1,p1", "C2,p1"),
            # node a (source output)
            "VS,p1":   "R1,p1",
            # node 1
            "R1,p2":   ("C1,p1", "D1,p2", "RS2,p1", "C3,p1"),
            # intermediate node between RS1 and D1
            "RS1,p2":  "D1,p1",
            # node 2
            "C1,p2":   ("C3,p2", "D3,p2", "RS4,p1"),
            # node 10
            "C2,p2":   ("D2,p2", "RS3,p1", "C4,p1"),
            # intermediate nodes for RS2/RS3/RS4
            "RS2,p2":  "D2,p1",
            "RS3,p2":  "D3,p1",
            "RS4,p2":  "D4,p1",
            # node 20
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
    linear_strategy = analyze_circuit(groups, sys_size, is_complex=False, backend="dense")
    y_op = linear_strategy.solve_dc(groups, jnp.zeros(sys_size))
    transient_sim = setup_transient(groups=groups, linear_strategy=linear_strategy)
    return transient_sim, y_op, port_map


def run_transient(transient_sim, y_op, t_end: float, n_save: int):
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, t_end, n_save))
    sol = transient_sim(
        t0=0.0,
        t1=t_end,
        dt0=DT,
        y0=y_op,
        saveat=saveat,
        max_steps=int(t_end / DT) + 10,
    )
    sol.ys.block_until_ready()
    return sol


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Diode cascade benchmark: Circulax vs NGSpice")
    parser.add_argument("--plot", action="store_true", help="Save comparison plot")
    parser.add_argument("--n-save", type=int, default=10_001,
                        help="Circulax save points (default: 10001)")
    args = parser.parse_args()

    print("=" * 60)
    print("Diode Cascade Benchmark: Circulax vs NGSpice")
    print(f"  VS: {V_AMP}V  {F_SRC/1e3:.0f}kHz sine")
    print(f"  R={R_SER}Ω  C={C_VAL*1e9:.0f}nF × 4  D1N4007 × 4")
    print(f"  Circulax model: IS={IS_D:.2e}, N={N_D}, RS={RS_D}Ω, CJO={CJ0_D*1e12:.1f}pF, M={M_D}")
    print(f"  NGSpice model:  IS={IS_D:.2e}, N={N_D}, RS={RS_D}Ω, CJO={CJ0_D*1e12:.1f}pF, M={M_D}, BV=1kV")
    print(f"  T_end={T_END*1e3:.0f}ms  dt={DT*1e9:.0f}ns  N_steps={N_STEPS:,}")
    print(f"  klujax src: {KLUJAX_SRC}  (exists={KLUJAX_SRC.exists()})")
    print("=" * 60)

    # ── NGSpice ──────────────────────────────────────────────────────────────
    print("\n[NGSpice] Warmup run...")
    run_ngspice(CIR_FILE, NG_NODES, output_path=NG_OUTPUT)

    print("[NGSpice] Timed run...")
    t0 = time.perf_counter()
    ng_time, ng_v = run_ngspice(CIR_FILE, NG_NODES, output_path=NG_OUTPUT)
    t_ng = time.perf_counter() - t0
    ng_steps = len(ng_time)
    t_ng_per_step_us = t_ng / ng_steps * 1e6
    print(f"          {t_ng:.3f}s  ({ng_steps:,} timepoints  {t_ng_per_step_us:.3f} µs/step)")

    # ── Circulax ─────────────────────────────────────────────────────────────
    print("\n[Circulax] Compiling netlist + DC solve...")
    t0 = time.perf_counter()
    transient_sim, y_op, port_map = build_circulax()
    t_compile = time.perf_counter() - t0
    print(f"           {t_compile:.3f}s")

    print(f"[Circulax] Warmup run ({WARMUP_STEPS:,} steps → {WARMUP_T_END*1e6:.1f}µs)...")
    t0 = time.perf_counter()
    run_transient(transient_sim, y_op, WARMUP_T_END, n_save=101)
    t_warmup = time.perf_counter() - t0
    print(f"           {t_warmup:.3f}s")

    print(f"[Circulax] Timed run ({N_STEPS:,} steps, n_save={args.n_save:,})...")
    t0 = time.perf_counter()
    sol = run_transient(transient_sim, y_op, T_END, n_save=args.n_save)
    t_cx = time.perf_counter() - t0
    t_cx_per_step_us = t_cx / N_STEPS * 1e6
    print(f"           {t_cx:.3f}s  ({t_cx_per_step_us:.3f} µs/step)")

    cx_time  = np.asarray(sol.ts)
    cx_v1    = np.asarray(sol.ys[:, port_map["R1,p2"]])    # node 1
    cx_v10   = np.asarray(sol.ys[:, port_map["C2,p2"]])    # node 10
    cx_v20   = np.asarray(sol.ys[:, port_map["C4,p2"]])    # node 20

    # ── Results ──────────────────────────────────────────────────────────────
    print("\n── Timing ────────────────────────────────────────────────────")
    print(f"  NGSpice  compile : n/a")
    print(f"  Circulax compile : {t_compile:.3f}s  (netlist + DC)")
    print(f"  NGSpice  warmup  : 1 run")
    print(f"  Circulax warmup  : {t_warmup:.3f}s  (JIT + {WARMUP_STEPS:,} steps)")
    print(f"  NGSpice  timed   : {t_ng:.3f}s  →  {t_ng_per_step_us:.3f} µs/step")
    print(f"  Circulax timed   : {t_cx:.3f}s  →  {t_cx_per_step_us:.3f} µs/step")
    print(f"  Speed ratio      : {t_cx/t_ng:.2f}x  ({'faster' if t_cx < t_ng else 'slower'} than NGSpice)")

    print("\n── Final V(20) peak (last 2 cycles) ─────────────────────────")
    last_2cyc = cx_time > (T_END - 2 / F_SRC)
    ng_last   = ng_time  > (T_END - 2 / F_SRC)
    print(f"  NGSpice  V(20) peak : {ng_v['v(20)'][ng_last].max():.3f} V")
    print(f"  Circulax V(20) peak : {cx_v20[last_2cyc].max():.3f} V")
    print(f"  (theoretical 4×peak: {4 * V_AMP:.1f} V)")

    print("\n── Accuracy (note: models differ — RS and CJO not in Circulax) ──")
    for ng_node, cx_arr, label in [
        ("v(1)",  cx_v1,  "v(1)  [node 1]"),
        ("v(10)", cx_v10, "v(10) [node 10]"),
        ("v(20)", cx_v20, "v(20) [output]"),
    ]:
        cmp = compare_waveforms(ng_time, ng_v[ng_node], cx_time, cx_arr, node=label)
        cmp.print()

    if args.plot:
        panels = []
        for ng_node, cx_arr, title in [
            ("v(1)",  cx_v1,  "V(1) — input node"),
            ("v(10)", cx_v10, "V(10) — mid-stage"),
            ("v(20)", cx_v20, "V(20) — output (4× multiplier)"),
        ]:
            err = cx_arr - np.interp(cx_time, ng_time, ng_v[ng_node])
            panels += [
                {
                    "title": title,
                    "ref_time": ng_time, "ref_signal": ng_v[ng_node],
                    "test_time": cx_time, "test_signal": cx_arr,
                },
                {
                    "title": f"{title} — error",
                    "ref_time": cx_time, "ref_signal": err,
                    "test_time": cx_time, "test_signal": err,
                    "show_error": True,
                },
            ]
        plot_comparison(
            ref_label="NGSpice (full D1N4007)",
            test_label="Circulax (Shockley only)",
            time_scale=1e3,
            time_unit="ms",
            panels=panels,
            output_path=PLOT_OUTPUT,
        )


if __name__ == "__main__":
    main()
