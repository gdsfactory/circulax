"""LC ladder scalability testbench.

Circuit
-------
  Vin  → Rs(50Ω) → [L(10nH)─C(4pF)] × N → Rl(50Ω) → GND

  L=10nH, C=4pF  →  Z₀ = √(L/C) = 50Ω  (matched),  τ/stage = √(LC) = 200ps
  Source: SmoothPulse step, delay=2ns, tr=10ps
  T_MAX = 3 × N × 0.5ns  (3× total propagation delay)

Sweep
-----
  N_SECTIONS ∈ SWEEP_SECTIONS  (default: 100, 1000, 10000, 50000)

This is a scalability benchmark — no reference solver, just Circulax timing
vs system size.  SaveAt uses only [t0, T_MAX] with a 2-node projection
(Rs input, Rl output) to avoid O(N) memory at each save point.

Usage
-----
  pixi run -e benchmark python benchmarking/lc_ladder_testbench.py
  pixi run -e benchmark python benchmarking/lc_ladder_testbench.py --sections 100 1000
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
from circulax.components.electronic import (  # noqa: E402
    Capacitor, Inductor, Resistor, SmoothPulse,
)
from circulax.solvers import analyze_circuit, setup_transient  # noqa: E402

# ---------------------------------------------------------------------------
# Circuit parameters
# ---------------------------------------------------------------------------
L_VAL    = 10e-9   # H  — inductor per section
C_VAL    = 4e-12   # F  — capacitor per section
R_SOURCE = 50.0    # Ω  — source impedance (matched)
R_LOAD   = 50.0    # Ω  — load impedance (matched)
V_STEP   = 1.0     # V  — step amplitude
DELAY    = 2e-9    # s  — step delay
T_RISE   = 1e-11   # s  — step rise time

STEP_DELAY_PER_SECTION = jnp.sqrt(L_VAL * C_VAL)   # 200ps per section

# PID adaptive step controller (matched to notebook)
STEP_CONTROLLER = diffrax.PIDController(
    rtol=1e-3, atol=1e-4,
    pcoeff=0.2, icoeff=0.5, dcoeff=0.0,
    force_dtmin=True, dtmin=1e-14, dtmax=1e-9,
    error_order=2,
)

WARMUP_STEPS = 2
MODELS_MAP = {
    "ground":         lambda: 0,
    "voltage_source": SmoothPulse,
    "resistor":       Resistor,
    "capacitor":      Capacitor,
    "inductor":       Inductor,
}

# Default sweep — override with --sections
SWEEP_SECTIONS = [100, 1000, 10000, 50000]


# ---------------------------------------------------------------------------
# Netlist builder
# ---------------------------------------------------------------------------

def build_netlist(n_sections: int) -> dict:
    """Build an N-section LC ladder netlist."""
    net: dict = {
        "instances": {
            "GND": {"component": "ground"},
            "Vin": {"component": "voltage_source",
                    "settings": {"V": V_STEP, "delay": DELAY, "tr": T_RISE}},
            "Rs":  {"component": "resistor", "settings": {"R": R_SOURCE}},
            "Rl":  {"component": "resistor", "settings": {"R": R_LOAD}},
        },
        "connections": {
            "GND,p1": ("Vin,p2", "Rl,p2"),
            "Vin,p1": "Rs,p1",
        },
    }
    prev = "Rs,p2"
    for i in range(n_sections):
        net["instances"][f"L{i}"] = {"component": "inductor",  "settings": {"L": L_VAL}}
        net["instances"][f"C{i}"] = {"component": "capacitor", "settings": {"C": C_VAL}}
        net["connections"][f"L{i},p1"] = prev
        net["connections"][f"L{i},p2"] = f"C{i},p1"
        gnd_conn = net["connections"]["GND,p1"]
        net["connections"]["GND,p1"] = (*gnd_conn, f"C{i},p2")
        prev = f"L{i},p2"
    net["connections"]["Rl,p1"] = prev
    return net


# ---------------------------------------------------------------------------
# Per-size solver
# ---------------------------------------------------------------------------

def run_one(n_sections: int) -> dict:
    """Compile, warm up, and time one LC ladder of *n_sections* sections."""
    t_max = float(3 * n_sections * 0.5e-9)
    max_steps = max(1_000_000, int(t_max / 1e-10))

    # ── Compile ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    net_dict = build_netlist(n_sections)
    groups, sys_size, port_map = compile_netlist(net_dict, MODELS_MAP)
    linear_strategy = analyze_circuit(groups, sys_size, is_complex=False,
                                      backend="klu_split")
    y0 = linear_strategy.solve_dc(groups, jnp.zeros(sys_size))
    transient_sim = setup_transient(groups=groups, linear_strategy=linear_strategy)
    compile_time = time.perf_counter() - t0

    # SaveAt: only t0 and T_MAX, saving only the 2 I/O node voltages.
    # Using fn= avoids storing the full O(N) state vector at each save point.
    io_indices = jnp.array([port_map["Rs,p2"], port_map["Rl,p1"]])

    @jax.jit
    def save_io(t, y, args):
        return y[io_indices]

    saveat_w = diffrax.SaveAt(ts=jnp.array([0.0, float(WARMUP_STEPS * 1e-9)]),
                              fn=save_io)
    saveat   = diffrax.SaveAt(ts=jnp.array([0.0, t_max]), fn=save_io)

    # ── Warmup ───────────────────────────────────────────────────────────────
    # Must use the same stepsize_controller so the warmup JIT-compiles the
    # exact same code path as the timed run.
    t0 = time.perf_counter()
    transient_sim(
        t0=0.0, t1=float(WARMUP_STEPS * 1e-9), dt0=1e-11, y0=y0,
        stepsize_controller=STEP_CONTROLLER,
        saveat=saveat_w, max_steps=100,
    ).ys.block_until_ready()
    warmup_time = time.perf_counter() - t0

    # ── Timed run ────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    sol = transient_sim(
        t0=0.0, t1=t_max, dt0=1e-11, y0=y0,
        stepsize_controller=STEP_CONTROLLER,
        saveat=saveat,
        max_steps=max_steps,
    )
    sol.ys.block_until_ready()
    elapsed = time.perf_counter() - t0

    n_steps   = int(sol.stats["num_steps"])
    converged = sol.result == diffrax.RESULTS.successful

    # Extract I/O voltages at t=T_MAX (index 1)
    v_in_final  = float(sol.ys[1, 0])
    v_out_final = float(sol.ys[1, 1])
    theory_delay_ns = float(n_sections * STEP_DELAY_PER_SECTION * 1e9)

    linear_strategy.cleanup()

    return {
        "n_sections":    n_sections,
        "sys_size":      sys_size,
        "t_max_ns":      t_max * 1e9,
        "compile_time":  compile_time,
        "warmup_time":   warmup_time,
        "elapsed":       elapsed,
        "n_steps":       n_steps,
        "us_per_step":   elapsed / max(n_steps, 1) * 1e6,
        "converged":     converged,
        "v_in_final":    v_in_final,
        "v_out_final":   v_out_final,
        "theory_delay_ns": theory_delay_ns,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LC ladder scalability testbench")
    parser.add_argument(
        "--sections", type=int, nargs="+", default=SWEEP_SECTIONS,
        metavar="N", help="Number of LC sections to sweep (default: 100 1000 10000 50000)",
    )
    args = parser.parse_args()

    sections = sorted(set(args.sections))

    print("=" * 70)
    print("LC Ladder Scalability Testbench  (klu_split backend)")
    print(f"  L={L_VAL*1e9:.0f}nH  C={C_VAL*1e12:.0f}pF  Z₀=50Ω  τ/stage=200ps")
    print(f"  Sections to sweep: {sections}")
    print("=" * 70)

    rows = []
    for n in sections:
        print(f"\n[N={n:>6}]  sys_size≈{2*n+2}  T_MAX={3*n*0.5:.1f}ns  running...")
        r = run_one(n)
        rows.append(r)
        status = "✓" if r["converged"] else "✗ FAILED"
        print(
            f"          compile={r['compile_time']:.3f}s  "
            f"warmup={r['warmup_time']:.3f}s  "
            f"sim={r['elapsed']:.3f}s  "
            f"steps={r['n_steps']:,}  "
            f"{r['us_per_step']:.2f}µs/step  {status}"
        )

    print("\n── Scaling Summary ──────────────────────────────────────────────────")
    hdr = (f"  {'N':>7}  {'sys_size':>9}  {'compile':>8}  {'warmup':>7}  "
           f"{'sim':>7}  {'steps':>7}  {'µs/step':>9}  {'V_out@T_MAX':>12}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        status = "" if r["converged"] else "  FAILED"
        print(
            f"  {r['n_sections']:>7,}  "
            f"{r['sys_size']:>9,}  "
            f"{r['compile_time']:>7.3f}s  "
            f"{r['warmup_time']:>6.3f}s  "
            f"{r['elapsed']:>6.2f}s  "
            f"{r['n_steps']:>7,}  "
            f"{r['us_per_step']:>8.2f}µs  "
            f"{r['v_out_final']:>11.4f}V"
            f"{status}"
        )


if __name__ == "__main__":
    main()
