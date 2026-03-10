"""Full-wave bridge rectifier benchmark: Circulax vs NGSpice.

Circuit
-------
  vs   inp  inn  SIN(0, 20V, 50Hz)   — floating source, grounded via RGND1/RGND2
  d1   inp  outp  D1N4007            — (anode=inp,  cathode=outp)
  d2   outn inp   D1N4007            — (anode=outn, cathode=inp)
  d3   inn  outp  D1N4007            — (anode=inn,  cathode=outp)
  d4   outn inn   D1N4007            — (anode=outn, cathode=inn)
  cl   outp outn  100µF              — smoothing capacitor
  rl   outp outn  1kΩ               — load resistor
  rgnd1 inn  0   1MΩ                — DC-bias reference
  rgnd2 outn 0   1MΩ

Diode model:  D1N4007
  NGSpice:    IS=76.9p, RS=42mΩ, N=1.45, CJO=26.5pF, M=0.333, BV=1kV
  Circulax:   IS=76.9p, RS=42mΩ, N=1.45, CJO=26.5pF, M=0.333  (RS explicit, BV omitted)

Output: V(outp) − V(outn) — smoothed rectified DC.

Usage
-----
  pixi run -e benchmark python benchmarking/fullwave_rect_vs_ngspice.py [--plot]
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
# Diode with clamped Shockley + SPICE junction capacitance (fixed sign).
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
# Circuit parameters — must match circuits/fullwave_rect.cir exactly
# ---------------------------------------------------------------------------
V_AMP   = 20.0      # V  (sine amplitude)
F_SRC   = 50.0      # Hz (mains frequency)
RS_D    = 0.042     # Ω  (D1N4007 series resistance — modelled explicitly)
IS_D    = 76.9e-12  # A
N_D     = 1.45
CJ0_D   = 26.5e-12  # F
VJ_D    = 1.0       # V
M_D     = 0.333
C_LOAD  = 100e-6    # F
R_LOAD  = 1e3       # Ω
R_GND   = 1e6       # Ω  (bias resistors to GND)

T_END   = 1.0       # s  (1 s = 50 mains cycles)
DT      = 1e-6      # s  (1µs, matching NGSpice tran step)
N_STEPS = int(T_END / DT)   # 1_000_000

WARMUP_STEPS = 1_000
WARMUP_T_END = WARMUP_STEPS * DT

HERE        = pathlib.Path(__file__).parent
CIR_FILE    = HERE / "circuits" / "fullwave_rect.cir"
NG_OUTPUT   = pathlib.Path("/tmp/ngspice_fullwave_rect.dat")
PLOT_OUTPUT = HERE / "fullwave_rect_comparison.png"

NG_NODES = ["v(outp)", "v(outn)"]


# ---------------------------------------------------------------------------
# Circulax
# ---------------------------------------------------------------------------

def build_circulax():
    """Compile netlist and return (transient_sim, y_op, port_map).

    Bridge topology:
      VS between inp (p1) and inn (p2).
      Each SPICE diode split into RS (bulk resistance) + DiodeLimited junction.

      D1: inp  → RS1 → RS1j → D1 → outp   (anode=inp,  cathode=outp)
      D2: outn → RS2 → RS2j → D2 → inp    (anode=outn, cathode=inp)
      D3: inn  → RS3 → RS3j → D3 → outp   (anode=inn,  cathode=outp)
      D4: outn → RS4 → RS4j → D4 → inn    (anode=outn, cathode=inn)
    """
    d_settings = {"Is": IS_D, "n": N_D, "Cj0": CJ0_D, "Vj": VJ_D, "m": M_D}

    net_dict = {
        "instances": {
            "GND":   {"component": "ground"},
            "VS":    {"component": "ac_source", "settings": {"V": V_AMP, "freq": F_SRC}},
            # D1: inp → RS1 → D1 → outp
            "RS1":   {"component": "resistor",  "settings": {"R": RS_D}},
            "D1":    {"component": "diode",     "settings": d_settings},
            # D2: outn → RS2 → D2 → inp
            "RS2":   {"component": "resistor",  "settings": {"R": RS_D}},
            "D2":    {"component": "diode",     "settings": d_settings},
            # D3: inn → RS3 → D3 → outp
            "RS3":   {"component": "resistor",  "settings": {"R": RS_D}},
            "D3":    {"component": "diode",     "settings": d_settings},
            # D4: outn → RS4 → D4 → inn
            "RS4":   {"component": "resistor",  "settings": {"R": RS_D}},
            "D4":    {"component": "diode",     "settings": d_settings},
            # Load and grounding
            # NOTE: resistor names must NOT contain "GND" as a substring —
            # build_net_map uses substring matching to identify the ground node.
            "CL":    {"component": "capacitor", "settings": {"C": C_LOAD}},
            "RL":    {"component": "resistor",  "settings": {"R": R_LOAD}},
            "RBIAS1": {"component": "resistor",  "settings": {"R": R_GND}},  # inn → GND
            "RBIAS2": {"component": "resistor",  "settings": {"R": R_GND}},  # outn → GND
        },
        "connections": {
            # GND (node 0)
            "GND,p1":   ("RBIAS1,p2", "RBIAS2,p2"),
            # inp: VS+, D1 anode side, D2 cathode
            "VS,p1":    ("RS1,p1", "D2,p2"),
            # inn: VS-, D3 anode side, D4 cathode, RBIAS1
            "VS,p2":    ("RS3,p1", "D4,p2", "RBIAS1,p1"),
            # outp: D1 cathode, D3 cathode, CL+, RL+
            "D1,p2":    ("D3,p2", "CL,p1", "RL,p1"),
            # outn: D2 anode side, D4 anode side, CL-, RL-, RBIAS2
            "RS2,p1":   ("RS4,p1", "CL,p2", "RL,p2", "RBIAS2,p1"),
            # RS-D intermediate nodes
            "RS1,p2":   "D1,p1",
            "RS2,p2":   "D2,p1",
            "RS3,p2":   "D3,p1",
            "RS4,p2":   "D4,p1",
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
        max_steps=int(jnp.ceil(t_end / DT)),
    )
    sol.ys.block_until_ready()
    return sol


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Full-wave rectifier benchmark: Circulax vs NGSpice")
    parser.add_argument("--plot", action="store_true", help="Save comparison plot")
    parser.add_argument("--n-save", type=int, default=10_001,
                        help="Circulax save points (default: 10001)")
    args = parser.parse_args()

    print("=" * 60)
    print("Full-Wave Rectifier Benchmark: Circulax vs NGSpice")
    print(f"  VS: {V_AMP}V  {F_SRC:.0f}Hz sine  (mains bridge rectifier)")
    print(f"  4× D1N4007  CL={C_LOAD*1e6:.0f}µF  RL={R_LOAD/1e3:.0f}kΩ")
    print(f"  Circulax model: IS={IS_D:.2e}, N={N_D}, RS={RS_D}Ω, CJO={CJ0_D*1e12:.1f}pF, M={M_D}")
    print(f"  T_end={T_END}s ({T_END*F_SRC:.0f} cycles)  dt={DT*1e6:.0f}µs  N_steps={N_STEPS:,}")
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
    ng_vdiff = ng_v["v(outp)"] - ng_v["v(outn)"]
    print(f"          {t_ng:.3f}s  ({ng_steps:,} timepoints  {t_ng_per_step_us:.3f} µs/step)")

    # ── Circulax ─────────────────────────────────────────────────────────────
    print("\n[Circulax] Compiling netlist + DC solve...")
    t0 = time.perf_counter()
    transient_sim, y_op, port_map = build_circulax()
    t_compile = time.perf_counter() - t0
    print(f"           {t_compile:.3f}s")

    print(f"[Circulax] Timed run ({N_STEPS:,} steps, n_save={args.n_save:,})...")
    t0 = time.perf_counter()
    sol = run_transient(transient_sim, y_op, T_END, n_save=args.n_save)
    t_cx = time.perf_counter() - t0
    t_cx_per_step_us = t_cx / N_STEPS * 1e6
    print(f"           {t_cx:.3f}s  ({t_cx_per_step_us:.3f} µs/step)")

    cx_time   = np.asarray(sol.ts)
    cx_voutp  = np.asarray(sol.ys[:, port_map["D1,p2"]])     # outp
    cx_voutn  = np.asarray(sol.ys[:, port_map["RS2,p1"]])    # outn
    cx_vdiff  = cx_voutp - cx_voutn

    # ── Results ──────────────────────────────────────────────────────────────
    print("\n── Timing ────────────────────────────────────────────────────")
    print(f"  NGSpice  compile : n/a")
    print(f"  Circulax compile : {t_compile:.3f}s  (netlist + DC)")
    print(f"  NGSpice  timed   : {t_ng:.3f}s  →  {t_ng_per_step_us:.3f} µs/step")
    print(f"  Circulax timed   : {t_cx:.3f}s  →  {t_cx_per_step_us:.3f} µs/step")
    print(f"  Speed ratio      : {t_cx/t_ng:.2f}x  ({'faster' if t_cx < t_ng else 'slower'} than NGSpice)")

    print("\n── Steady-state DC output (last 2 cycles) ────────────────────")
    last_2cyc  = cx_time  > (T_END - 2.0 / F_SRC)
    ng_last    = ng_time  > (T_END - 2.0 / F_SRC)
    cx_dc_mean = cx_vdiff[last_2cyc].mean()
    ng_dc_mean = ng_vdiff[ng_last].mean()
    cx_ripple  = cx_vdiff[last_2cyc].max() - cx_vdiff[last_2cyc].min()
    ng_ripple  = ng_vdiff[ng_last].max()   - ng_vdiff[ng_last].min()
    print(f"  NGSpice  mean V(outp,outn) : {ng_dc_mean:.4f} V   ripple: {ng_ripple*1e3:.2f} mV")
    print(f"  Circulax mean V(outp,outn) : {cx_dc_mean:.4f} V   ripple: {cx_ripple*1e3:.2f} mV")
    print(f"  Ideal no-drop DC           : {V_AMP * 2**0.5 / 3.14159 * 2:.4f} V  (2Vpeak/π × 2 half-waves)")

    print("\n── Accuracy ──────────────────────────────────────────────────")
    for ng_node, cx_arr, label in [
        ("v(outp)", cx_voutp, "v(outp)"),
        ("v(outn)", cx_voutn, "v(outn)"),
    ]:
        cmp = compare_waveforms(ng_time, ng_v[ng_node], cx_time, cx_arr, node=label)
        cmp.print()
    cmp_diff = compare_waveforms(ng_time, ng_vdiff, cx_time, cx_vdiff, node="v(outp,outn)")
    cmp_diff.print()

    if args.plot:
        err_diff = cx_vdiff - np.interp(cx_time, ng_time, ng_vdiff)
        panels = [
            {
                "title": "V(outp) — positive rail",
                "ref_time": ng_time, "ref_signal": ng_v["v(outp)"],
                "test_time": cx_time, "test_signal": cx_voutp,
            },
            {
                "title": "V(outp, outn) — rectified output",
                "ref_time": ng_time, "ref_signal": ng_vdiff,
                "test_time": cx_time, "test_signal": cx_vdiff,
            },
            {
                "title": "Circulax − NGSpice  [V(outp,outn)]",
                "ref_time": cx_time, "ref_signal": err_diff,
                "test_time": cx_time, "test_signal": err_diff,
                "show_error": True,
            },
        ]
        plot_comparison(
            ref_label="NGSpice (full D1N4007)",
            test_label="Circulax (DiodeLimited)",
            time_scale=1.0,
            time_unit="s",
            panels=panels,
            output_path=PLOT_OUTPUT,
        )


if __name__ == "__main__":
    main()
