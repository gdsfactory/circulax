"""Ring oscillator benchmark using a pure-JAX MOSFET.

Builds the same 9-stage CMOS ring as scripts/ring_one_case.py but
substitutes the simplified JAX-native MOSFET from
tests/fixtures/mosfet_simple.py for PSP103.  No bosdi, no FFI, no
OSDI — everything lives inside the XLA graph.

Purpose: measure wall time and per-step cost to compare against the
bosdi+OSDI path and VACASK.  Isolates whether the FFI boundary is
the dominant per-step overhead in circulax.

Reports:

- Steady-state wall and µs/step (post-JIT).
- JIT compile time (first call warmup).
- Oscillation frequency — if the simplified model's Id is off by
  ~30 %, frequency will shift but the ring should still oscillate.

Usage:
    pixi run python scripts/bench_jax_native_ring.py
"""

from __future__ import annotations

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
sys.path.insert(0, str(_REPO / "scripts"))

from fixtures.mosfet_simple import MosfetSimple
from circulax import compile_netlist
from circulax.components.electronic import (
    Capacitor,
    Resistor,
    SmoothPulse,
    VoltageSource,
)
from circulax.solvers import analyze_circuit, setup_transient
from circulax.solvers.transient import TrapRefactoringTransientSolver


def dom_freq(t: np.ndarray, x: np.ndarray) -> float:
    """Rising mid-supply zero-crossing period → frequency (Hz)."""
    centered = x - x.mean()
    rising = np.where(np.diff(np.sign(centered)) > 0)[0]
    if len(rising) < 3:
        return float("nan")
    rising = rising[1:]
    times = []
    for i in rising:
        x0, x1 = float(centered[i]), float(centered[i + 1])
        t0, t1 = float(t[i]), float(t[i + 1])
        times.append(t0 - x0 * (t1 - t0) / (x1 - x0))
    if len(times) < 2:
        return float("nan")
    return float(1.0 / np.median(np.diff(np.asarray(times))))


def build_ring_netlist(n_stages: int = 9, c_load: float = 0.0) -> dict:
    """9-stage CMOS ring with the simplified JAX-native MOSFET."""
    if n_stages < 3 or n_stages % 2 == 0:
        raise ValueError(f"n_stages must be odd and ≥ 3; got {n_stages}")

    # NMOS: W = 10 µm, L = 1 µm.  PMOS: W = 20 µm, L = 1 µm, type = −1.
    # Same sizing as the OSDI path.  KP fit separately per polarity against
    # PSP103 bias points (see scripts/fit_mosfet_params.py).
    nmos_settings = {"type":  1.0, "W": 10e-6, "L": 1e-6,
                     "Vt": 0.171, "KP": 530e-6}
    pmos_settings = {"type": -1.0, "W": 20e-6, "L": 1e-6,
                     "Vt": 0.169, "KP": 590e-6}

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

    for stage in range(1, n_stages + 1):
        in_n = f"n{stage}"
        out_n = f"n{stage % n_stages + 1}"
        mn, mp = f"mn{stage}", f"mp{stage}"
        instances[mn] = {"component": "nmos", "settings": nmos_settings}
        instances[mp] = {"component": "pmos", "settings": pmos_settings}
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

    return {
        "instances": instances,
        "connections": connections,
        "ports": {"out": "n1,p1"},
    }


def main() -> None:
    netlist = build_ring_netlist(n_stages=9, c_load=0.0)
    models = {
        "nmos": MosfetSimple,
        "pmos": MosfetSimple,
        "vsrc": VoltageSource,
        "kick": SmoothPulse,
        "r_kick": Resistor,
        "cload": Capacitor,
    }

    # Compile: this is where circulax builds the ComponentGroup and the
    # sparse Jacobian structure.  For a pure-JAX device, all the physics
    # is a Python function that gets traced — no FFI.
    t0 = time.time()
    groups, sys_size, port_map = compile_netlist(netlist, models)
    compile_netlist_wall = time.time() - t0

    solver = analyze_circuit(groups, sys_size, backend="klu_rs_split")

    # DC homotopy init.
    t0 = time.time()
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, 1e-2)
    y_src = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y0 = solver.solve_dc_gmin(groups, y_src, g_start=1e-2, n_steps=30)
    dc_wall = time.time() - t0
    if not bool(jnp.all(jnp.isfinite(y0))):
        print("DC solve diverged — simplified model may be too out-of-spec.")
        return

    run = setup_transient(groups, solver, transient_solver=TrapRefactoringTransientSolver)

    T1 = 1e-6
    DT = 5e-11
    N_SAVE = 4000
    n_steps = int(T1 / DT)
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, T1, N_SAVE))
    controller = diffrax.ConstantStepSize()

    # JIT warmup.
    t0 = time.time()
    _ = run(
        t0=0.0, t1=10 * DT, dt0=DT, y0=y0,
        saveat=diffrax.SaveAt(ts=jnp.linspace(0, 10 * DT, 5)),
        max_steps=50, stepsize_controller=controller,
    ).ys.block_until_ready()
    compile_wall = time.time() - t0

    # Timed run.
    t0 = time.time()
    sol = run(
        t0=0.0, t1=T1, dt0=DT, y0=y0, saveat=saveat,
        max_steps=int(2 * n_steps), stepsize_controller=controller,
    )
    sol.ys.block_until_ready()
    wall = time.time() - t0

    ys = np.asarray(sol.ys)
    t = np.asarray(sol.ts)
    if not np.all(np.isfinite(ys)):
        print("Transient contains NaN — simplified model may be too aggressive.")
        return

    v = ys[:, port_map["n1,p1"]]
    mask = t > 100e-9
    freq = dom_freq(t[mask], v[mask])
    us_per_step = wall / n_steps * 1e6
    swing = float(v.max() - v.min())

    print("=" * 78)
    print("Ring oscillator with JAX-native MOSFET (no OSDI, no FFI)")
    print("=" * 78)
    print(f"  Circuit        : 9-stage CMOS ring, W=10/20 µm, L=1 µm, VDD=1.2 V")
    print(f"  Device model   : MosfetSimple (~40-line level-1 + Meyer cap)")
    print(f"  System size    : {sys_size} unknowns")
    print(f"  Sim window     : {T1 * 1e9:.0f} ns, dt = {DT * 1e12:.0f} ps, {n_steps} fixed steps")
    print()
    print(f"  compile_netlist wall : {compile_netlist_wall:.3f} s")
    print(f"  DC homotopy wall     : {dc_wall:.3f} s")
    print(f"  JIT compile (warmup) : {compile_wall:.2f} s   (one-time)")
    print()
    print(f"  Transient wall       : {wall:.3f} s")
    print(f"  µs / step            : {us_per_step:.1f}")
    print(f"  Oscillation freq     : {freq / 1e6:.2f} MHz  (ref PSP103/OSDI path: 289 MHz)")
    print(f"  Swing on n1          : {swing:.3f} V  (VDD = 1.2 V)")
    print()
    print("  Comparison:")
    print(f"    circulax + OSDI (bosdi FFI, post-Tier-3) : 7.53 s / 377 µs/step / 289 MHz")
    print(f"    circulax + JAX-native (this run)         : {wall:.2f} s / {us_per_step:.0f} µs/step / {freq / 1e6:.0f} MHz")
    print(f"    VACASK (OSDI C++)                        : 1.19 s / 48 µs/step / 289 MHz")


if __name__ == "__main__":
    main()
