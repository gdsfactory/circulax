"""Circulax ring oscillator for the benchmark sweep.

Wraps the N-stage CMOS ring from scripts/ring_one_case.py so a single
``run(n_stages=...)`` call produces the wall / µs-per-step numbers the
harness in ``run.py`` needs.  Matches the VACASK ring template
(W=10 µm NMOS, W=20 µm PMOS, L=1 µm, VDD=1.2 V, 1 µs tran at 50 ps
fixed step → 20 000 steps).

Models: PSP103 via the OSDI binary that ships with bosdi.  Use the
``jax-native`` variant to swap in the simplified JAX-native MOSFET
from ``tests/fixtures/mosfet_simple.py`` instead (no OSDI / FFI at all).
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

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "tests"))
sys.path.insert(0, str(_REPO / "scripts"))

# Upstream settings, exactly.
T1 = 1e-6        # 1 µs
DT = 5e-11       # 50 ps
N_STEPS = int(T1 / DT)
N_SAVE = 4000


def _build_ring(n_stages: int, variant: str):
    """Return (groups, sys_size, port_map) for an N-stage CMOS ring."""
    if variant == "osdi":
        from ring_one_case import build_netlist as _bn
        return _bn(c_load=0.0, n_stages=n_stages)
    if variant == "jax-native":
        from fixtures.mosfet_simple import MosfetSimple
        from circulax import compile_netlist
        from circulax.components.electronic import (
            Capacitor, Resistor, SmoothPulse, VoltageSource,
        )

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
        models = {
            "nmos": MosfetSimple, "pmos": MosfetSimple,
            "vsrc": VoltageSource, "kick": SmoothPulse,
            "r_kick": Resistor,
        }
        return compile_netlist(
            {"instances": instances, "connections": connections,
             "ports": {"out": "n1,p1"}}, models,
        )
    raise ValueError(f"unknown variant: {variant!r}")


def _dom_freq(t: np.ndarray, x: np.ndarray) -> float:
    """Period → frequency from rising mid-level zero-crossings."""
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


def run(n_stages: int = 9, variant: str = "osdi") -> dict:
    """Build, DC-init, JIT-warm, time one 1 µs transient."""
    from circulax.solvers import analyze_circuit, setup_transient
    from circulax.solvers.transient import TrapFactorizedTransientSolver

    groups, sys_size, port_map = _build_ring(n_stages, variant)
    solver = analyze_circuit(groups, sys_size, backend="klu_rs_split")

    t_dc = time.perf_counter()
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, 1e-2)
    y_src = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y0 = solver.solve_dc_gmin(groups, y_src, g_start=1e-2, n_steps=30)
    dc_s = time.perf_counter() - t_dc
    if not bool(jnp.all(jnp.isfinite(y0))):
        return {"status": "dc_diverged", "n_stages": n_stages, "variant": variant}

    run_fn = setup_transient(
        groups, solver, transient_solver=TrapFactorizedTransientSolver,
    )
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, T1, N_SAVE))
    controller = diffrax.ConstantStepSize()
    max_steps = int(2 * N_STEPS)

    # Warmup must share the real call's *static* JAX shapes so the
    # timed run hits the JIT cache.  diffrax bakes both max_steps and
    # SaveAt.ts-length into the compiled function — if the warmup uses
    # max_steps=50 with a 5-point saveat, the timed call recompiles
    # from scratch and you charge yourself the compile time twice.
    # Keep ts-length and max_steps identical; shrink the ts *values*
    # and t1 so integration exits early and warmup stays cheap.
    saveat_warmup = diffrax.SaveAt(ts=jnp.linspace(0.0, 2 * DT, N_SAVE))
    t_compile = time.perf_counter()
    _ = run_fn(
        t0=0.0, t1=2 * DT, dt0=DT, y0=y0, saveat=saveat_warmup,
        max_steps=max_steps, stepsize_controller=controller,
    ).ys.block_until_ready()
    compile_s = time.perf_counter() - t_compile

    t0 = time.perf_counter()
    sol = run_fn(
        t0=0.0, t1=T1, dt0=DT, y0=y0, saveat=saveat,
        max_steps=max_steps, stepsize_controller=controller,
    )
    sol.ys.block_until_ready()
    wall = time.perf_counter() - t0

    ys = np.asarray(sol.ys)
    t = np.asarray(sol.ts)
    if not np.all(np.isfinite(ys)):
        return {"status": "tran_nan", "n_stages": n_stages, "variant": variant,
                "wall_s": wall, "compile_s": compile_s, "dc_s": dc_s,
                "n_steps": N_STEPS}

    v = ys[:, port_map["n1,p1"]]
    mask = t > 100e-9
    freq = _dom_freq(t[mask], v[mask]) if mask.any() else float("nan")

    return {
        "status": "ok", "n_stages": n_stages, "variant": variant,
        "wall_s": wall, "compile_s": compile_s, "dc_s": dc_s,
        "n_steps": N_STEPS, "us_per_step": wall / N_STEPS * 1e6,
        "sys_size": sys_size, "freq_MHz": freq / 1e6 if np.isfinite(freq) else None,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-stages", type=int, default=9)
    p.add_argument("--variant", choices=("osdi", "jax-native"), default="osdi")
    args = p.parse_args()
    r = run(args.n_stages, args.variant)
    print(r)
