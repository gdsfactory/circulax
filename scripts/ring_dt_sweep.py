"""Ring oscillator dt sweep: fixed-step vs adaptive, various dt, measure freq.

Purpose: is circulax's 6.7× slow ring a transient-solver artefact (too-coarse
dt) or a Jacobian/capacitance-content issue independent of stepping?
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


def build_netlist(c_load: float = 50e-15):
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
        in_n  = f"n{stage}"
        out_n = f"n{stage % 9 + 1}"
        mn, mp, cl = f"mn{stage}", f"mp{stage}", f"CL{stage}"
        instances[mn] = {"component": "nmos", "settings": mos_n}
        instances[mp] = {"component": "pmos", "settings": mos_p}
        instances[cl] = {"component": "cload", "settings": {"C": c_load}}
        connections[f"{mn},D"] = f"{out_n},p1"
        connections[f"{mn},G"] = f"{in_n},p1"
        connections[f"{mn},S"] = "GND,p1"
        connections[f"{mn},B"] = "GND,p1"
        connections[f"{mp},D"] = f"{out_n},p1"
        connections[f"{mp},G"] = f"{in_n},p1"
        connections[f"{mp},S"] = "vdd,p1"
        connections[f"{mp},B"] = "vdd,p1"
        connections[f"{cl},p1"] = f"{out_n},p1"
        connections[f"{cl},p2"] = "GND,p1"

    models = {
        "nmos": psp103n, "pmos": psp103p,
        "vsrc": VoltageSource, "kick": SmoothPulse,
        "r_kick": Resistor, "cload": Capacitor,
    }
    return compile_netlist({"instances": instances, "connections": connections, "ports": {"out": "n1,p1"}}, models)


def dc_init(groups, sys_size, solver):
    G_HOM = 1e-2
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, G_HOM)
    y_src = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    return solver.solve_dc_gmin(groups, y_src, g_start=G_HOM, n_steps=30)


def run_transient(groups, sys_size, solver, y0, *, t1, dt0, n_save, stepsize_controller):
    run = setup_transient(groups, solver, transient_solver=SDIRK3VectorizedTransientSolver)
    sol = run(
        t0=0.0, t1=t1, dt0=dt0, y0=y0,
        saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t1, n_save)),
        max_steps=2_000_000,
        stepsize_controller=stepsize_controller,
    )
    return np.asarray(sol.ts), np.asarray(sol.ys)


def dom_freq(t: np.ndarray, x: np.ndarray) -> float:
    x = x - x.mean()
    dt = float(t[1] - t[0])
    freqs = np.fft.rfftfreq(len(x), d=dt)
    power = np.abs(np.fft.rfft(x))
    power[0] = 0.0
    return float(freqs[int(np.argmax(power))])


def main() -> None:
    groups, sys_size, port_map = build_netlist(c_load=50e-15)
    solver = analyze_circuit(groups, sys_size)
    y0 = dc_init(groups, sys_size, solver)
    n1 = port_map["n1,p1"]

    print(f"\n{'case':<28} {'saved pts':>10} {'freq [MHz]':>13} {'period [ns]':>13} {'swing [V]':>10}")
    print("-" * 78)

    cases: list[tuple[str, dict]] = [
        ("fixed dt=10 ps, t=100ns",   dict(dt0=1.0e-11, t1=100e-9, n_save=2000,
                                            stepsize_controller=diffrax.ConstantStepSize())),
        ("fixed dt=5 ps, t=100ns",    dict(dt0=5.0e-12, t1=100e-9, n_save=2000,
                                            stepsize_controller=diffrax.ConstantStepSize())),
        ("fixed dt=2 ps, t=40ns",     dict(dt0=2.0e-12, t1=40e-9, n_save=2000,
                                            stepsize_controller=diffrax.ConstantStepSize())),
        ("fixed dt=1 ps, t=20ns",     dict(dt0=1.0e-12, t1=20e-9, n_save=2000,
                                            stepsize_controller=diffrax.ConstantStepSize())),
        ("PID rtol=1e-4 atol=1e-6",   dict(dt0=1.0e-12, t1=100e-9, n_save=2000,
                                            stepsize_controller=diffrax.PIDController(
                                                rtol=1e-4, atol=1e-6, dtmin=1e-14, dtmax=1e-10))),
        ("PID rtol=1e-6 atol=1e-8",   dict(dt0=1.0e-12, t1=100e-9, n_save=2000,
                                            stepsize_controller=diffrax.PIDController(
                                                rtol=1e-6, atol=1e-8, dtmin=1e-14, dtmax=1e-10))),
    ]

    for label, kwargs in cases:
        try:
            t, ys = run_transient(groups, sys_size, solver, y0, **kwargs)
            v = ys[:, n1]
            if not np.all(np.isfinite(v)):
                print(f"{label:<28} {'NaN in transient':>40}")
                continue
            f = dom_freq(t, v)
            period_ns = 1e9 / f if f > 0 else float("inf")
            swing = float(v.max() - v.min())
            print(f"{label:<28} {len(t):>10d} {f / 1e6:>13.2f} {period_ns:>13.2f} {swing:>10.3f}")
        except Exception as e:
            print(f"{label:<28} FAILED: {type(e).__name__}: {str(e)[:60]}")


if __name__ == "__main__":
    main()
