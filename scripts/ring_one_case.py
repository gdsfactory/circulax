"""Run ONE ring-osc case, append result to reports/ring_sweep.csv.

Each invocation does exactly one transient run with the given c_load / gmin.
Designed to be invoked in a tight shell loop so a VS Code / OOM crash only
kills the current case and leaves the earlier results intact.

Usage:
    pixi run python scripts/ring_one_case.py  <c_load_fF>  <gmin>
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "tests"))
sys.path.insert(0, "/home/cdaunt/code/bosdi/src")

CSV_PATH = _REPO / "reports" / "ring_sweep.csv"


def build_netlist(c_load: float, n_stages: int = 9):
    """Build an N-stage CMOS ring oscillator.  ``n_stages`` must be odd."""
    if n_stages < 3 or n_stages % 2 == 0:
        msg = f"n_stages must be odd and ≥ 3; got {n_stages}"
        raise ValueError(msg)
    from fixtures.psp103_models import geom_settings, make_psp103_descriptors
    from circulax import compile_netlist
    from circulax.components.electronic import (
        Capacitor, Resistor, SmoothPulse, VoltageSource,
    )

    psp103n, psp103p = make_psp103_descriptors()
    mos_n = geom_settings(10e-6, 1e-6)
    mos_p = geom_settings(20e-6, 1e-6)

    instances = {
        "Vvdd":  {"component": "vsrc", "settings": {"V": 1.2}},
        "Vkick": {"component": "kick", "settings": {"V": 1.0, "delay": 1e-9, "tr": 1e-9}},
        "Rkick": {"component": "r_kick", "settings": {"R": 1e5}},
    }
    connections = {
        "Vvdd,p1": "vdd,p1",   "Vvdd,p2": "GND,p1",
        "Vkick,p1": "kick_n,p1", "Vkick,p2": "GND,p1",
        "Rkick,p1": "kick_n,p1", "Rkick,p2": "n1,p1",
    }
    for stage in range(1, n_stages + 1):
        in_n, out_n = f"n{stage}", f"n{stage % n_stages + 1}"
        mn, mp = f"mn{stage}", f"mp{stage}"
        instances[mn] = {"component": "nmos", "settings": mos_n}
        instances[mp] = {"component": "pmos", "settings": mos_p}
        connections[f"{mn},D"] = f"{out_n},p1"
        connections[f"{mn},G"] = f"{in_n},p1"
        connections[f"{mn},S"] = "GND,p1"
        connections[f"{mn},B"] = "GND,p1"
        connections[f"{mp},D"] = f"{out_n},p1"
        connections[f"{mp},G"] = f"{in_n},p1"
        connections[f"{mp},S"] = "vdd,p1"
        connections[f"{mp},B"] = "vdd,p1"
        if c_load > 0:
            cl = f"CL{stage}"
            instances[cl] = {"component": "cload", "settings": {"C": c_load}}
            connections[f"{cl},p1"] = f"{out_n},p1"
            connections[f"{cl},p2"] = "GND,p1"

    models = {
        "nmos": psp103n, "pmos": psp103p,
        "vsrc": VoltageSource, "kick": SmoothPulse,
        "r_kick": Resistor, "cload": Capacitor,
    }
    return compile_netlist(
        {"instances": instances, "connections": connections, "ports": {"out": "n1,p1"}},
        models,
    )


def dom_freq(t: np.ndarray, x: np.ndarray) -> float:
    x = x - x.mean()
    dt = float(t[1] - t[0])
    freqs = np.fft.rfftfreq(len(x), d=dt)
    power = np.abs(np.fft.rfft(x))
    power[0] = 0.0
    return float(freqs[int(np.argmax(power))])


def append_csv(row: dict) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["c_load_fF", "gmin", "status", "freq_MHz", "period_ns", "swing_V", "wall_s"]
    exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def run_once(c_load: float, gmin: float, dt0: float = 1e-12, t1: float = 50e-9,
             n_save: int = 1000):
    from circulax.solvers import analyze_circuit, setup_transient
    from circulax.solvers.transient import SDIRK3VectorizedTransientSolver

    t_start = time.time()
    groups, sys_size, port_map = build_netlist(c_load)
    solver_0 = analyze_circuit(groups, sys_size)
    solver = eqx.tree_at(lambda s: s.g_leak, solver_0, gmin)

    # DC init with default gmin.
    G_HOM = 1e-2
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver_0, G_HOM)
    y_src = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y0 = solver_0.solve_dc_gmin(groups, y_src, g_start=G_HOM, n_steps=30)
    if not bool(jnp.all(jnp.isfinite(y0))):
        return {"c_load_fF": c_load * 1e15, "gmin": gmin, "status": "DC_diverged",
                "wall_s": time.time() - t_start}

    n1 = port_map["n1,p1"]
    run = setup_transient(groups, solver, transient_solver=SDIRK3VectorizedTransientSolver)
    try:
        sol = run(
            t0=0.0, t1=t1, dt0=dt0, y0=y0,
            saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t1, n_save)),
            max_steps=2_000_000,
            stepsize_controller=diffrax.PIDController(
                rtol=1e-4, atol=1e-6, dtmin=1e-14, dtmax=1e-10,
            ),
        )
    except Exception as e:
        return {"c_load_fF": c_load * 1e15, "gmin": gmin,
                "status": f"ERROR_{type(e).__name__}",
                "wall_s": time.time() - t_start}

    ys = np.asarray(sol.ys)
    if not np.all(np.isfinite(ys)):
        return {"c_load_fF": c_load * 1e15, "gmin": gmin, "status": "NaN",
                "wall_s": time.time() - t_start}
    v = ys[:, n1]
    freq = dom_freq(np.asarray(sol.ts), v)
    return {
        "c_load_fF": c_load * 1e15, "gmin": gmin, "status": "ok",
        "freq_MHz": freq / 1e6, "period_ns": 1e9 / freq if freq > 0 else float("inf"),
        "swing_V": float(v.max() - v.min()), "wall_s": time.time() - t_start,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("c_load_fF", type=float, help="external load cap per stage, fF (0 = none)")
    p.add_argument("gmin", type=float, help="runtime g_leak")
    p.add_argument("--t1-ns", type=float, default=50.0, help="sim end time in ns")
    p.add_argument("--dt0-ps", type=float, default=1.0, help="initial step in ps")
    args = p.parse_args()

    c_load = args.c_load_fF * 1e-15
    row = run_once(c_load, args.gmin, dt0=args.dt0_ps * 1e-12, t1=args.t1_ns * 1e-9)
    append_csv(row)
    status = row.get("status", "?")
    if status == "ok":
        print(
            f"c_load={row['c_load_fF']:>6.1f}fF gmin={row['gmin']:.1e} "
            f"freq={row['freq_MHz']:>7.2f}MHz period={row['period_ns']:>6.3f}ns "
            f"swing={row['swing_V']:.3f}V  ({row['wall_s']:.1f}s)"
        )
    else:
        print(
            f"c_load={row['c_load_fF']:>6.1f}fF gmin={row['gmin']:.1e} "
            f"STATUS={status}  ({row.get('wall_s', 0):.1f}s)"
        )


if __name__ == "__main__":
    main()
