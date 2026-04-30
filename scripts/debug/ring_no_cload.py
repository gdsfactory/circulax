"""Ring oscillator with C_load=0: crosscheck bosdi author's candidate (2).

The author's hypothesis: circulax's C_load >= alpha * |G_ii| * dt stability
heuristic oversizes the external lumped load because |G_ii| captures the
95.5 mS gds coupling through the constraint row, which isn't the actual
"free" node admittance seen by the integrator. If we can converge with
C_load=0 and just a larger gmin, the heuristic is the culprit and removing
or tightening it should close most of the 6.7x frequency gap.

This script runs the 9-stage ring with C_load swept down toward zero at
a few gmin levels, reports whether the transient produces finite output
and what frequency/swing we get. A positive result (finite output,
frequency approaching 937 MHz at the smallest C_load) nails candidate (2).
"""

from __future__ import annotations

import sys
from pathlib import Path

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_TESTS = Path(__file__).resolve().parents[1] / "tests"
sys.path.insert(0, str(_TESTS))
sys.path.insert(0, "/home/cdaunt/code/bosdi/src")

from fixtures.psp103_models import geom_settings, make_psp103_descriptors

from circulax import compile_netlist
from circulax.components.electronic import (
    Capacitor,
    Resistor,
    SmoothPulse,
    VoltageSource,
)
from circulax.solvers import analyze_circuit, setup_transient
from circulax.solvers.transient import SDIRK3VectorizedTransientSolver


def build_netlist(c_load: float):
    psp103n, psp103p = make_psp103_descriptors()
    mos_n = geom_settings(10e-6, 1e-6)
    mos_p = geom_settings(20e-6, 1e-6)

    instances: dict = {
        "Vvdd":  {"component": "vsrc", "settings": {"V": 1.2}},
        "Vkick": {"component": "kick", "settings": {"V": 1.0, "delay": 1e-9, "tr": 1e-9}},
        "Rkick": {"component": "r_kick", "settings": {"R": 1e5}},
    }
    connections: dict = {
        "Vvdd,p1": "vdd,p1",   "Vvdd,p2": "GND,p1",
        "Vkick,p1": "kick_n,p1", "Vkick,p2": "GND,p1",
        "Rkick,p1": "kick_n,p1", "Rkick,p2": "n1,p1",
    }
    for stage in range(1, 10):
        in_n, out_n = f"n{stage}", f"n{stage % 9 + 1}"
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


def dc_init(groups, sys_size, solver, g_homotopy=1e-2):
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, g_homotopy)
    y_src = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    return solver.solve_dc_gmin(groups, y_src, g_start=g_homotopy, n_steps=30)


def dom_freq(t: np.ndarray, x: np.ndarray) -> float:
    x = x - x.mean()
    dt = float(t[1] - t[0])
    freqs = np.fft.rfftfreq(len(x), d=dt)
    power = np.abs(np.fft.rfft(x))
    power[0] = 0.0
    return float(freqs[int(np.argmax(power))])


def run_case(c_load: float, gmin: float, dt0: float = 1e-12, t1: float = 100e-9,
             n_save: int = 2000) -> dict:
    groups, sys_size, port_map = build_netlist(c_load)
    solver_0 = analyze_circuit(groups, sys_size)
    # Override the runtime gmin (g_leak) everywhere.
    solver = eqx.tree_at(lambda s: s.g_leak, solver_0, gmin)

    y0 = dc_init(groups, sys_size, solver_0)  # DC init with default gmin
    if not bool(jnp.all(jnp.isfinite(y0))):
        return {"c_load": c_load, "gmin": gmin, "status": "DC diverged"}

    n1 = port_map["n1,p1"]
    run = setup_transient(groups, solver, transient_solver=SDIRK3VectorizedTransientSolver)
    try:
        sol = run(
            t0=0.0, t1=t1, dt0=dt0, y0=y0,
            saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t1, n_save)),
            max_steps=2_000_000,
            stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6, dtmin=1e-14, dtmax=1e-10),
        )
    except Exception as e:
        return {"c_load": c_load, "gmin": gmin, "status": f"ERROR: {type(e).__name__}"}

    t = np.asarray(sol.ts)
    ys = np.asarray(sol.ys)
    if not np.all(np.isfinite(ys)):
        return {"c_load": c_load, "gmin": gmin, "status": "NaN in transient"}

    v = ys[:, n1]
    swing = float(v.max() - v.min())
    freq = dom_freq(t, v)
    return {
        "c_load": c_load, "gmin": gmin, "status": "ok",
        "freq_MHz": freq / 1e6, "period_ns": 1e9 / freq if freq > 0 else float("inf"),
        "swing_V": swing,
    }


def main() -> None:
    print(f"{'C_load':>10} {'gmin':>10} {'freq MHz':>10} {'period ns':>11} {'swing V':>10}  status")
    print("-" * 80)
    for c_load in (50e-15, 20e-15, 10e-15, 5e-15, 0.0):
        for gmin in (1e-9, 1e-6, 1e-4, 1e-3, 1e-2):
            r = run_case(c_load, gmin)
            if r["status"] == "ok":
                print(f"{c_load * 1e15:>8.1f}fF {gmin:>10.1e} "
                      f"{r['freq_MHz']:>10.2f} {r['period_ns']:>11.3f} "
                      f"{r['swing_V']:>10.3f}  {r['status']}")
            else:
                print(f"{c_load * 1e15:>8.1f}fF {gmin:>10.1e} {'':>33}  {r['status']}")


if __name__ == "__main__":
    main()
