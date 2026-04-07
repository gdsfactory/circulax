"""Diode half-wave clipper — Harmonic Balance vs transient testbench.

Circuit
-------
  vs  inp  0    SIN(0, 2V, 1kHz)
  rs  inp  mid  1kΩ   (series)
  d1  mid  out  D1N4007
  rl  out  0    10kΩ  (load)

Benchmark hypothesis
--------------------
NGSpice runs a full 20-cycle transient (20 ms) before reaching periodic steady
state.  Circulax Harmonic Balance (HB) finds the same steady state directly in
O(N_harmonics) Newton steps — no need to time-march through transient.

A key advantage of HB is the frequency sweep use case: ``jax.vmap`` runs all
frequencies in a single compiled XLA call, while NGSpice requires one serial
transient per frequency.

Modes
-----
  default  Single-frequency comparison + harmonic spectrum extraction from transient
  --sweep  100-frequency sweep: NGSpice serial vs Circulax HB vmapped

Usage
-----
  pixi run -e benchmark python benchmarking/diode_clipper_hb_testbench.py
  pixi run -e benchmark python benchmarking/diode_clipper_hb_testbench.py --plot
  pixi run -e benchmark python benchmarking/diode_clipper_hb_testbench.py --sweep
  pixi run -e benchmark python benchmarking/diode_clipper_hb_testbench.py --sweep --plot
  pixi run -e benchmark python benchmarking/diode_clipper_hb_testbench.py --n-harmonics 20
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import tempfile
import time

import equinox as eqx
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
    Resistor,
    VoltageSourceAC,
    _junction_charge,
)
from circulax.solvers import (  # noqa: E402
    BDF2RefactoringTransientSolver,
    analyze_circuit,
    setup_harmonic_balance,
    setup_transient,
)

# ---------------------------------------------------------------------------
# DiodeLimited — clamped Shockley + SPICE junction capacitance (D1N4007)
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
# Circuit parameters — must match circuits/diode_clipper.cir exactly
# ---------------------------------------------------------------------------

F_SRC = 1e3  # 1 kHz drive frequency
V_AMP = 2.0  # V peak
R_SER = 1e3  # Ω  series resistor
R_LOAD = 10e3  # Ω  load resistor

# D1N4007 SPICE parameters (same as diode_cascade_testbench)
IS_D = 76.9e-12
N_D = 1.45
VT_D = 25.85e-3
CJ0_D = 26.5e-12
VJ_D = 1.0
M_D = 0.333

N_CYCLES = 20  # cycles NGSpice (and Circulax transient) simulate
T_END = N_CYCLES / F_SRC  # 20 ms
DT = 1e-6  # 1 µs time step
N_HARMONICS = 10  # HB harmonics

STEP_CONTROLLER = diffrax.PIDController(
    rtol=1e-3,
    atol=1e-4,
    pcoeff=0.2,
    icoeff=0.5,
    dcoeff=0.0,
    force_dtmin=True,
    dtmin=1e-6 * DT,
    dtmax=DT,
    error_order=2,
)

HERE = pathlib.Path(__file__).parent
CIR_FILE = HERE / "circuits" / "diode_clipper.cir"
NG_OUTPUT = pathlib.Path("/tmp/ngspice_diode_clipper.dat")
PLOT_OUTPUT = HERE / "diode_clipper_hb_comparison.png"

NODES = ["v(inp)", "v(out)"]


# ---------------------------------------------------------------------------
# Shared netlist builder
# ---------------------------------------------------------------------------


def _build_netlist():
    d_settings = {"Is": IS_D, "n": N_D, "Vt": VT_D, "Cj0": CJ0_D, "Vj": VJ_D, "m": M_D}
    net_dict = {
        "instances": {
            "Vs": {"component": "ac_source", "settings": {"V": V_AMP, "freq": F_SRC}},
            "Rs": {"component": "resistor", "settings": {"R": R_SER}},
            "D1": {"component": "diode", "settings": d_settings},
            "RL": {"component": "resistor", "settings": {"R": R_LOAD}},
        },
        "connections": {
            "Vs,p1": "Rs,p1",
            "Rs,p2": "D1,p1",
            "D1,p2": "RL,p1",
            "RL,p2": "GND,p1",
            "Vs,p2": "GND,p1",
        },
    }
    models_map = {
        "ac_source": VoltageSourceAC,
        "resistor": Resistor,
        "diode": DiodeLimited,
    }
    return net_dict, models_map


def _last_cycle(time_arr: np.ndarray, signals: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    """Extract the last complete period and shift its time axis to start at 0.

    Uses searchsorted so it works regardless of whether the time array is
    uniformly spaced (Circulax SaveAt) or not (NGSpice adaptive output).
    """
    T = 1.0 / F_SRC
    t_start = time_arr[-1] - T
    idx = int(np.searchsorted(time_arr, t_start))
    t_last = time_arr[idx:] - time_arr[idx]
    sig_last = {k: v[idx:] for k, v in signals.items()}
    return t_last, sig_last


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------


def solver_ngspice() -> SolverResult:
    run_ngspice(CIR_FILE, NODES, output_path=NG_OUTPUT)  # warmup
    t0 = time.perf_counter()
    ng_time, ng_v = run_ngspice(CIR_FILE, NODES, output_path=NG_OUTPUT)
    elapsed = time.perf_counter() - t0

    # Return the last cycle as the periodic steady state reference
    t_ss, sig_ss = _last_cycle(ng_time, ng_v)
    return SolverResult(
        name="ngspice",
        time=t_ss,
        signals=sig_ss,
        elapsed=elapsed,
        n_steps=len(ng_time),
        metadata={"note": f"last cycle of {N_CYCLES}-cycle transient"},
    )


def solver_circulax_hb(n_harmonics: int = N_HARMONICS) -> SolverResult:
    net_dict, models_map = _build_netlist()

    # --- compile + DC ---
    t0 = time.perf_counter()
    groups, sys_size, port_map = compile_netlist(net_dict, models_map)
    dc_solver = analyze_circuit(groups, sys_size, backend="dense")
    y_dc = dc_solver.solve_dc(groups, jnp.zeros(sys_size))
    run_hb = setup_harmonic_balance(groups, sys_size, freq=F_SRC, num_harmonics=n_harmonics)
    compile_time = time.perf_counter() - t0

    # --- JIT warmup ---
    t0 = time.perf_counter()
    y_time_w, _ = run_hb(y_dc)
    y_time_w.block_until_ready()
    warmup_time = time.perf_counter() - t0

    # --- timed solve ---
    t0 = time.perf_counter()
    y_time, _ = run_hb(y_dc)
    y_time.block_until_ready()
    elapsed = time.perf_counter() - t0

    K = 2 * n_harmonics + 1
    t_hb = np.linspace(0.0, 1.0 / F_SRC, K, endpoint=False)

    y_time_np = np.asarray(y_time)
    return SolverResult(
        name="circulax-hb",
        time=t_hb,
        signals={
            "v(inp)": y_time_np[:, port_map["Vs,p1"]],
            "v(out)": y_time_np[:, port_map["RL,p1"]],
        },
        elapsed=elapsed,
        n_steps=K,  # Newton iterations ≈ K (each solves a K×N system)
        compile_time=compile_time,
        warmup_time=warmup_time,
        metadata={"harmonics": n_harmonics, "K": K},
    )


def solver_circulax_transient(n_save: int = 2001) -> SolverResult:
    net_dict, models_map = _build_netlist()

    # --- compile + setup ---
    t0 = time.perf_counter()
    groups, sys_size, port_map = compile_netlist(net_dict, models_map)
    linear_strategy = analyze_circuit(groups, sys_size, is_complex=False, backend="klu_split")
    y_op = linear_strategy.solve_dc(groups, jnp.zeros(sys_size))
    transient_sim = setup_transient(
        groups=groups,
        linear_strategy=linear_strategy,
        transient_solver=BDF2RefactoringTransientSolver,
    )
    compile_time = time.perf_counter() - t0

    # --- JIT warmup (2 steps) ---
    warmup_t_end = 2 * DT
    t0 = time.perf_counter()
    transient_sim(
        t0=0.0,
        t1=warmup_t_end,
        dt0=DT,
        y0=y_op,
        saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, warmup_t_end, 11)),
        max_steps=100,
        stepsize_controller=STEP_CONTROLLER,
    ).ys.block_until_ready()
    warmup_time = time.perf_counter() - t0

    # --- timed full run ---
    t0 = time.perf_counter()
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, T_END, n_save))
    sol = transient_sim(
        t0=0.0,
        t1=T_END,
        dt0=DT,
        y0=y_op,
        saveat=saveat,
        max_steps=int(T_END / DT) * 10,
        stepsize_controller=STEP_CONTROLLER,
    )
    sol.ys.block_until_ready()
    elapsed = time.perf_counter() - t0

    cx_time = np.asarray(sol.ts)
    cx_sigs = {
        "v(inp)": np.asarray(sol.ys[:, port_map["Vs,p1"]]),
        "v(out)": np.asarray(sol.ys[:, port_map["RL,p1"]]),
    }
    t_ss, sig_ss = _last_cycle(cx_time, cx_sigs)

    return SolverResult(
        name="circulax-transient",
        time=t_ss,
        signals=sig_ss,
        elapsed=elapsed,
        n_steps=int(sol.stats["num_steps"]),
        compile_time=compile_time,
        warmup_time=warmup_time,
        metadata={"note": f"last cycle of {N_CYCLES}-cycle transient"},
    )


# ---------------------------------------------------------------------------
# Harmonic extraction from a transient waveform (FFT of the last M cycles)
# ---------------------------------------------------------------------------


def extract_harmonics_fft(
    time_arr: np.ndarray,
    signal: np.ndarray,
    freq: float,
    n_harmonics: int,
    n_cycles: int = 5,
) -> np.ndarray:
    """Return harmonic amplitudes via FFT of the last *n_cycles* of a transient.

    Parameters
    ----------
    time_arr, signal:
        Uniformly-sampled time-domain waveform (or close to it).
    freq:
        Fundamental frequency in Hz.
    n_harmonics:
        Number of harmonics to return (excluding DC at index 0).
    n_cycles:
        How many complete periods from the end of the waveform to use.
        More cycles → better frequency resolution but same harmonic peaks.

    Returns
    -------
    amps : np.ndarray, shape (n_harmonics+1,)
        Two-sided amplitudes at DC, f0, 2f0, ..., n_harmonics*f0.

    """
    T = 1.0 / freq
    t_start = time_arr[-1] - n_cycles * T
    idx = int(np.searchsorted(time_arr, t_start))
    seg = signal[idx:]
    N = len(seg)

    fft_vals = np.fft.rfft(seg) / N
    fft_freqs = np.fft.rfftfreq(N, d=float(time_arr[1] - time_arr[idx + 1]) if N > 1 else 1.0)

    # Use median spacing for non-uniform arrays
    dt_med = float(np.median(np.diff(time_arr[idx:])))
    fft_freqs = np.fft.rfftfreq(N, d=dt_med)

    amps = np.zeros(n_harmonics + 1)
    for k in range(n_harmonics + 1):
        f_k = k * freq
        i_k = int(np.argmin(np.abs(fft_freqs - f_k)))
        scale = 1.0 if k == 0 else 2.0  # fold negative-frequency twin
        amps[k] = scale * np.abs(fft_vals[i_k])
    return amps


def print_harmonic_comparison(
    hb_y_freq: np.ndarray,
    transient_sol,
    port_map: dict,
    n_harmonics: int,
) -> None:
    """Compare harmonic amplitudes from HB y_freq vs FFT of the transient."""
    vout_idx = port_map["RL,p1"]
    vinp_idx = port_map["Vs,p1"]

    # HB amplitudes: 2*|y_freq[k]| for k>=1
    scale = np.where(np.arange(n_harmonics + 1) == 0, 1.0, 2.0)
    hb_vout = scale * np.abs(np.asarray(hb_y_freq[: n_harmonics + 1, vout_idx]))
    hb_vinp = scale * np.abs(np.asarray(hb_y_freq[: n_harmonics + 1, vinp_idx]))

    # Transient FFT amplitudes (last 5 cycles, dense SaveAt)
    t_arr = np.asarray(transient_sol.ts)
    vout_arr = np.asarray(transient_sol.ys[:, vout_idx])
    vinp_arr = np.asarray(transient_sol.ys[:, vinp_idx])
    fft_vout = extract_harmonics_fft(t_arr, vout_arr, F_SRC, n_harmonics, n_cycles=5)
    fft_vinp = extract_harmonics_fft(t_arr, vinp_arr, F_SRC, n_harmonics, n_cycles=5)

    print("\n── Harmonic spectrum comparison (V_out) ─────────────────────────")
    print(f"  {'Harmonic':>12s}  {'HB (V)':>10s}  {'Trans FFT (V)':>14s}  {'|Δ| (mV)':>10s}")
    print(f"  {'-' * 12}  {'-' * 10}  {'-' * 14}  {'-' * 10}")
    for k in range(n_harmonics + 1):
        label = "DC" if k == 0 else f"{k}f₀"
        diff_mv = abs(hb_vout[k] - fft_vout[k]) * 1e3
        print(f"  {label:>12s}  {hb_vout[k]:>10.4f}  {fft_vout[k]:>14.4f}  {diff_mv:>10.3f}")

    print("\n── Harmonic spectrum comparison (V_inp) ─────────────────────────")
    print(f"  {'Harmonic':>12s}  {'HB (V)':>10s}  {'Trans FFT (V)':>14s}  {'|Δ| (mV)':>10s}")
    print(f"  {'-' * 12}  {'-' * 10}  {'-' * 14}  {'-' * 10}")
    for k in range(n_harmonics + 1):
        label = "DC" if k == 0 else f"{k}f₀"
        diff_mv = abs(hb_vinp[k] - fft_vinp[k]) * 1e3
        print(f"  {label:>12s}  {hb_vinp[k]:>10.4f}  {fft_vinp[k]:>14.4f}  {diff_mv:>10.3f}")


# ---------------------------------------------------------------------------
# Frequency sweep benchmark
# ---------------------------------------------------------------------------

N_SWEEP = 100
SWEEP_FREQS = np.geomspace(100.0, 100e3, N_SWEEP)  # 100 Hz → 100 kHz
N_CYCLES_SWEEP = 20  # cycles per frequency

# NGSpice .cir template — frequency and timing are injected per run
_CIR_SWEEP_TEMPLATE = """\
* Diode half-wave clipper — parametric sweep run
.model D1N4007 D IS=76.9p RS=42.0m BV=1.00k IBV=5.00u CJO=26.5p M=0.333 N=1.45
vs inp 0 dc=0 sin 0 {v_amp} {freq}
rs inp mid {r_ser}
d1 mid out D1N4007
rl out 0 {r_load}
.options klu method=gear maxord=2
.control
  tran {dt_save} {t_end} 0 {dt_save}
  wrdata {output_path} v(inp) v(out)
  set noaskquit
  quit
.endc
.end
"""


def _ngspice_one_freq(freq: float) -> tuple[float, float]:
    """Run NGSpice at one frequency; return (fund amp of v(inp), v(out))."""
    t_end = N_CYCLES_SWEEP / freq
    dt_save = t_end / (N_CYCLES_SWEEP * 100)  # 100 pts/cycle
    output_path = f"/tmp/ngspice_clipper_sweep_{freq:.2f}.dat"

    cir_text = _CIR_SWEEP_TEMPLATE.format(
        v_amp=V_AMP,
        freq=f"{freq:.6g}",
        r_ser=R_SER,
        r_load=R_LOAD,
        dt_save=f"{dt_save:.6e}",
        t_end=f"{t_end:.6e}",
        output_path=output_path,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".cir", delete=False) as f:
        f.write(cir_text)
        cir_path = f.name
    try:
        ng_time, ng_v = run_ngspice(cir_path, ["v(inp)", "v(out)"], output_path=output_path)
    finally:
        os.unlink(cir_path)

    # FFT of last cycle → fundamental amplitude
    amp_inp = extract_harmonics_fft(ng_time, ng_v["v(inp)"], freq, n_harmonics=1)[1]
    amp_out = extract_harmonics_fft(ng_time, ng_v["v(out)"], freq, n_harmonics=1)[1]
    return amp_inp, amp_out


def run_sweep_benchmark(sweep_freqs: np.ndarray, n_harmonics: int = N_HARMONICS) -> None:
    """Compare NGSpice serial sweep vs Circulax HB vmapped sweep.

    Prints timing for both approaches and the v(out) fundamental amplitude
    at every frequency point.
    """
    n_freq = len(sweep_freqs)
    print(f"\n{'=' * 68}")
    print(f"Frequency Sweep  ({n_freq} points, {sweep_freqs[0]:.0f} Hz – {sweep_freqs[-1] / 1e3:.0f} kHz)")
    print(f"{'=' * 68}")

    # ------------------------------------------------------------------
    # NGSpice — serial run per frequency
    # ------------------------------------------------------------------
    print(f"\n[ngspice] serial ({n_freq} runs)  ...", end=" ", flush=True)

    # warmup
    _ngspice_one_freq(float(sweep_freqs[0]))

    ng_amps_inp = np.zeros(n_freq)
    ng_amps_out = np.zeros(n_freq)
    t0 = time.perf_counter()
    for i, f in enumerate(sweep_freqs):
        ng_amps_inp[i], ng_amps_out[i] = _ngspice_one_freq(float(f))
    ng_elapsed = time.perf_counter() - t0
    print(f"done  {ng_elapsed:.3f}s  ({ng_elapsed / n_freq * 1e3:.1f} ms/freq)")

    # ------------------------------------------------------------------
    # Circulax HB — vmapped over all frequencies simultaneously
    # ------------------------------------------------------------------
    print(f"[circulax-hb] vmapped ({n_freq} freqs)  ...", end=" ", flush=True)

    net_dict, models_map = _build_netlist()
    groups, sys_size, port_map = compile_netlist(net_dict, models_map)
    dc_solver = analyze_circuit(groups, sys_size, backend="dense")
    y_dc = dc_solver.solve_dc(groups, jnp.zeros(sys_size))

    vout_idx = port_map["RL,p1"]
    vinp_idx = port_map["Vs,p1"]

    def hb_solve_freq(sweep_freq):
        # Update the source frequency to match the HB sweep frequency.
        # VoltageSourceAC bakes freq into its params at compile time; we override
        # it here so the source drives at sweep_freq, not the original F_SRC.
        updated_groups = eqx.tree_at(
            lambda g: g["ac_source"].params.freq,
            groups,
            jnp.asarray([sweep_freq]),
        )
        run_hb = setup_harmonic_balance(updated_groups, sys_size, freq=sweep_freq, num_harmonics=n_harmonics)
        _, y_freq = run_hb(y_dc)
        # Two-sided fundamental amplitude at k=1
        amp_inp = 2.0 * jnp.abs(y_freq[1, vinp_idx])
        amp_out = 2.0 * jnp.abs(y_freq[1, vout_idx])
        return amp_inp, amp_out

    hb_sweep_jit = jax.jit(jax.vmap(hb_solve_freq))

    # warmup / compile
    t0 = time.perf_counter()
    hb_amps_inp_w, hb_amps_out_w = hb_sweep_jit(jnp.array(sweep_freqs))
    hb_amps_out_w.block_until_ready()
    hb_compile = time.perf_counter() - t0

    # timed run
    t0 = time.perf_counter()
    hb_amps_inp, hb_amps_out = hb_sweep_jit(jnp.array(sweep_freqs))
    hb_amps_out.block_until_ready()
    hb_elapsed = time.perf_counter() - t0

    hb_amps_inp = np.asarray(hb_amps_inp)
    hb_amps_out = np.asarray(hb_amps_out)
    print(f"done  compile={hb_compile:.3f}s  timed={hb_elapsed:.3f}s  ({hb_elapsed / n_freq * 1e3:.2f} ms/freq)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n── Sweep Timing ─────────────────────────────────────────────────")
    print(f"  ngspice (serial)    {ng_elapsed:.3f}s  ({ng_elapsed/n_freq*1e3:.1f} ms/freq)")
    print(f"  circulax-hb (vmap)  compile={hb_compile:.3f}s  timed={hb_elapsed:.3f}s  "
          f"({hb_elapsed/n_freq*1e3:.2f} ms/freq)")
    speedup = ng_elapsed / hb_elapsed
    total_cx = hb_compile + hb_elapsed
    total_speedup = ng_elapsed / total_cx
    print(f"  speedup (timed only):   {speedup:.1f}×")
    print(f"  speedup (incl compile): {total_speedup:.1f}×")

    rms_err = np.sqrt(np.mean((ng_amps_out - hb_amps_out) ** 2)) * 1e3
    print("\n── Sweep Accuracy vs NGSpice ────────────────────────────────────")
    print(f"  V(out) fundamental RMS error over sweep: {rms_err:.2f} mV")

    print("\n── V(out) Fundamental Amplitude (selected points) ───────────────")
    print(f"  {'freq (Hz)':>12s}  {'NGSpice (V)':>12s}  {'HB (V)':>10s}  {'err (mV)':>10s}")
    print(f"  {'-' * 12}  {'-' * 12}  {'-' * 10}  {'-' * 10}")
    indices = np.round(np.linspace(0, n_freq - 1, 12)).astype(int)
    for i in indices:
        err_mv = (hb_amps_out[i] - ng_amps_out[i]) * 1e3
        print(f"  {sweep_freqs[i]:>12.1f}  {ng_amps_out[i]:>12.4f}  {hb_amps_out[i]:>10.4f}  {err_mv:>+10.3f}")

    return ng_elapsed, hb_elapsed, hb_compile, sweep_freqs, ng_amps_out, hb_amps_out


# ---------------------------------------------------------------------------
# Solver registry
# ---------------------------------------------------------------------------

SOLVERS: dict[str, SolverFn] = {
    "ngspice": solver_ngspice,
    "circulax-hb": solver_circulax_hb,
    "circulax-transient": solver_circulax_transient,
}
REFERENCE = "ngspice"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Diode clipper HB testbench")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--sweep", action="store_true", help="Run 100-frequency sweep benchmark instead of single-freq")
    parser.add_argument("--n-harmonics", type=int, default=N_HARMONICS)
    parser.add_argument("--n-save", type=int, default=2001, help="SaveAt points for Circulax transient (default 2001)")
    parser.add_argument("--no-transient", action="store_true", help="Skip Circulax transient solver in single-freq mode")
    parser.add_argument("--n-sweep", type=int, default=N_SWEEP, help="Number of frequency points in sweep (default 100)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Mode A: frequency sweep
    # ------------------------------------------------------------------
    if args.sweep:
        sweep_freqs = np.geomspace(100.0, 100e3, args.n_sweep)
        sweep_results = run_sweep_benchmark(sweep_freqs, n_harmonics=args.n_harmonics)

        if args.plot:
            ng_elapsed, hb_elapsed, hb_compile, freqs, ng_amps, hb_amps = sweep_results
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                print("matplotlib not available — skipping sweep plot")
                return
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

            ax = axes[0]
            ax.semilogx(freqs, ng_amps, "C0-", lw=2, label="NGSpice (serial)")
            ax.semilogx(freqs, hb_amps, "C1--", lw=2, label=f"Circulax HB vmap ({args.n_harmonics} harmonics)")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("V(out) fundamental amplitude (V)")
            ax.set_title("Frequency Sweep — V(out) fundamental")
            ax.legend()
            ax.grid(True, which="both", alpha=0.3)

            ax = axes[1]
            err_mv = (hb_amps - ng_amps) * 1e3
            ax.semilogx(freqs, err_mv, "C2-", lw=1.5)
            ax.axhline(0, color="gray", lw=0.8, ls="--")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("HB − NGSpice (mV)")
            ax.set_title("Sweep amplitude error")
            ax.grid(True, which="both", alpha=0.3)

            plt.suptitle(
                f"Diode Clipper Sweep  NGSpice {ng_elapsed:.2f}s total  |  "
                f"HB vmap compile={hb_compile:.2f}s timed={hb_elapsed:.3f}s",
                fontsize=10,
            )
            plt.tight_layout()
            sweep_plot = HERE / "diode_clipper_sweep_comparison.png"
            plt.savefig(sweep_plot, dpi=150)
            print(f"Sweep plot saved → {sweep_plot}")
        return

    # ------------------------------------------------------------------
    # Mode B: single-frequency comparison + harmonic analysis
    # ------------------------------------------------------------------

    # Dense transient for harmonic FFT (more points per cycle than default)
    n_save_harmonics = max(args.n_save, N_CYCLES * 200 + 1)  # ≥200 pts/cycle

    solvers = dict(SOLVERS)
    solvers["circulax-hb"] = lambda: solver_circulax_hb(n_harmonics=args.n_harmonics)
    solvers["circulax-transient"] = lambda: solver_circulax_transient(n_save=args.n_save)
    if args.no_transient:
        del solvers["circulax-transient"]

    results = run_benchmark(
        solvers=solvers,
        reference=REFERENCE,
        nodes=NODES,
        title=(
            f"Diode Clipper — HB vs Transient  "
            f"VS={V_AMP}V {F_SRC / 1e3:.0f}kHz  "
            f"Rs={R_SER / 1e3:.0f}kΩ  RL={R_LOAD / 1e3:.0f}kΩ  "
            f"HB harmonics={args.n_harmonics}  "
            f"NGSpice/CX-transient: {N_CYCLES} cycles"
        ),
    )

    # ------------------------------------------------------------------
    # Harmonic comparison: HB y_freq vs transient FFT
    # ------------------------------------------------------------------
    # Re-run HB to get y_freq directly (solver_circulax_hb returns SolverResult
    # not y_freq, so we run setup_harmonic_balance again for the spectrum).
    net_dict, models_map = _build_netlist()
    groups, sys_size, port_map = compile_netlist(net_dict, models_map)
    dc_solver = analyze_circuit(groups, sys_size, backend="dense")
    y_dc = dc_solver.solve_dc(groups, jnp.zeros(sys_size))
    run_hb = setup_harmonic_balance(groups, sys_size, freq=F_SRC, num_harmonics=args.n_harmonics)
    _, hb_y_freq = run_hb(y_dc)

    # Dense transient for accurate FFT
    print(f"\n[transient-dense] running (n_save={n_save_harmonics} for FFT) ...", end=" ", flush=True)
    linear_strategy = analyze_circuit(groups, sys_size, is_complex=False, backend="klu_split")
    y_op = linear_strategy.solve_dc(groups, jnp.zeros(sys_size))
    transient_sim = setup_transient(groups=groups, linear_strategy=linear_strategy, transient_solver=BDF2RefactoringTransientSolver)
    # warmup
    transient_sim(
        t0=0.0,
        t1=2 * DT,
        dt0=DT,
        y0=y_op,
        saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, 2 * DT, 11)),
        max_steps=100,
        stepsize_controller=STEP_CONTROLLER,
    ).ys.block_until_ready()
    dense_sol = transient_sim(
        t0=0.0,
        t1=T_END,
        dt0=DT,
        y0=y_op,
        saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, T_END, n_save_harmonics)),
        max_steps=int(T_END / DT) * 10,
        stepsize_controller=STEP_CONTROLLER,
    )
    dense_sol.ys.block_until_ready()
    print("done")

    print_harmonic_comparison(hb_y_freq, dense_sol, port_map, args.n_harmonics)

    # ------------------------------------------------------------------
    # Waveform plot
    # ------------------------------------------------------------------
    if args.plot:
        ref = results[REFERENCE]
        panels = []
        t_scale = 1e3
        t_unit = "ms"
        for node, title in [("v(inp)", "Input V(inp)"), ("v(out)", "Output V(out) — clipped")]:
            panel = {
                "title": title,
                "ref_time": ref.time,
                "ref_signal": ref.signals[node],
            }
            for name, res in results.items():
                if name == REFERENCE:
                    continue
                panel["test_time"] = res.time
                panel["test_signal"] = res.signals[node]
            panels.append(panel)

        for name in [k for k in results if k != REFERENCE]:
            res = results[name]
            err = res.signals["v(out)"] - np.interp(res.time, ref.time, ref.signals["v(out)"])
            panels.append(
                {
                    "title": f"V(out) error — {name} − NGSpice",
                    "ref_time": res.time,
                    "ref_signal": err,
                    "test_time": res.time,
                    "test_signal": err,
                    "show_error": True,
                }
            )

        plot_comparison(
            ref_label="NGSpice (last cycle)",
            test_label="Circulax",
            time_scale=t_scale,
            time_unit=t_unit,
            panels=panels,
            output_path=PLOT_OUTPUT,
        )
        print(f"Waveform plot saved → {PLOT_OUTPUT}")


if __name__ == "__main__":
    main()
