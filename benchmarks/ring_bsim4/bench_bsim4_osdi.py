"""BSIM4 ring oscillator via the OSDI path.

circulax-VA's lowering doesn't currently support BSIM4's MIR (it uses
unnamed branch-current probes for the substrate-resistance network,
which falls outside what circulax/va/lowering.py models).  But the
precompiled ``circulax/components/osdi/bsim4v8.osdi`` binary exists
and goes through circulax's OSDI integration — same solver / assembly
/ component path as PSP103-OSDI.  This script verifies the ring
oscillator framework works end-to-end with a different MOSFET model.
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

REPO = Path("/home/cdaunt/code/circulax/circulax-va")
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "scripts"))

from circulax import compile_netlist  # noqa: E402
from circulax.components.electronic import (  # noqa: E402
    Capacitor, Resistor, SmoothPulse, VoltageSource,
)
from circulax.components.osdi import osdi_component  # noqa: E402
from circulax.solvers import analyze_circuit, setup_transient  # noqa: E402
from circulax.solvers.transient import TrapFactorizedTransientSolver  # noqa: E402
from circulax.va.va_defaults import parse_va_defaults_expanded  # noqa: E402

BSIM4_OSDI = REPO / "circulax" / "components" / "osdi" / "compiled" / "bsim4v8.osdi"
BSIM4_VA = REPO / "tests" / "data" / "va" / "bsim4v8.va"


def make_bsim4_descriptors():
    """Build BSIM4 NMOS + PMOS osdi_component descriptors with realistic
    defaults parsed from the .va source.
    """
    defaults_spec = parse_va_defaults_expanded(BSIM4_VA)
    # ParamSpec carries (default, type) pairs; pull out numeric defaults.
    defaults = {}
    for name, spec in defaults_spec.items():
        v = getattr(spec, "default", None)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            defaults[name] = float(v)
        elif isinstance(v, str):
            try:
                defaults[name] = float(v)
            except ValueError:
                pass
    # type=1 → NMOS, type=-1 → PMOS
    nmos = dict(defaults)
    nmos["type"] = 1.0
    pmos = dict(defaults)
    pmos["type"] = -1.0
    print(f"BSIM4: {len(defaults)} numeric default params parsed from .va")

    # ports follow BSIM4's declaration: d, g, s, b
    n_dev = osdi_component(osdi_path=str(BSIM4_OSDI), ports=("d", "g", "s", "b"),
                           default_params=nmos)
    p_dev = osdi_component(osdi_path=str(BSIM4_OSDI), ports=("d", "g", "s", "b"),
                           default_params=pmos)
    return n_dev, p_dev


def geom(W: float, L: float):
    """Per-instance geometry — keep it minimal; BSIM4 has its own AS/AD/PS/PD logic."""
    return {"w": W, "l": L}


def build_ring(n_stages: int = 9):
    if n_stages < 3 or n_stages % 2 == 0:
        raise ValueError(f"n_stages must be odd and ≥ 3; got {n_stages}")
    n_dev, p_dev = make_bsim4_descriptors()

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
    g_n = geom(10e-6, 1e-6)
    g_p = geom(20e-6, 1e-6)
    for stage in range(1, n_stages + 1):
        in_n, out_n = f"n{stage}", f"n{stage % n_stages + 1}"
        mn, mp = f"mn{stage}", f"mp{stage}"
        instances[mn] = {"component": "nmos", "settings": g_n}
        instances[mp] = {"component": "pmos", "settings": g_p}
        connections[f"{mn},d"] = f"{out_n},p1"
        connections[f"{mn},g"] = f"{in_n},p1"
        connections[f"{mn},s"] = "GND,p1"
        connections[f"{mn},b"] = "GND,p1"
        connections[f"{mp},d"] = f"{out_n},p1"
        connections[f"{mp},g"] = f"{in_n},p1"
        connections[f"{mp},s"] = "vdd,p1"
        connections[f"{mp},b"] = "vdd,p1"

    models = {"nmos": n_dev, "pmos": p_dev,
              "vsrc": VoltageSource, "kick": SmoothPulse,
              "r_kick": Resistor, "cload": Capacitor}
    return compile_netlist(
        {"instances": instances, "connections": connections, "ports": {"out": "n1,p1"}},
        models,
    )


def main():
    T1 = 1e-6
    DT = 5e-11
    N_STEPS = int(T1 / DT)
    N_SAVE = 4000

    print(f"Building BSIM4 ring oscillator (N=9, T1={T1*1e6}µs, dt={DT*1e12}ps)…")
    groups, sys_size, port_map = build_ring(9)
    print(f"  sys_size: {sys_size}")

    solver = analyze_circuit(groups, sys_size, backend="klu_rs_split")

    print("DC homotopy…")
    t0 = time.perf_counter()
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, 1e-2)
    y_src = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y0 = solver.solve_dc_gmin(groups, y_src, g_start=1e-2, n_steps=30)
    dc_s = time.perf_counter() - t0
    print(f"  DC: {dc_s:.1f}s, y0 finite: {bool(jnp.all(jnp.isfinite(y0)))}")

    if not bool(jnp.all(jnp.isfinite(y0))):
        print("  DC diverged — bailing")
        return

    print("Transient (warmup + timed)…")
    run_fn = setup_transient(groups, solver,
                             transient_solver=TrapFactorizedTransientSolver)
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, T1, N_SAVE))
    saveat_warm = diffrax.SaveAt(ts=jnp.linspace(0.0, 2*DT, N_SAVE))
    controller = diffrax.ConstantStepSize()
    max_steps = int(2 * N_STEPS)

    t_w = time.perf_counter()
    _ = run_fn(t0=0.0, t1=2*DT, dt0=DT, y0=y0, saveat=saveat_warm,
               max_steps=max_steps, stepsize_controller=controller).ys.block_until_ready()
    warmup_s = time.perf_counter() - t_w

    t_r = time.perf_counter()
    sol = run_fn(t0=0.0, t1=T1, dt0=DT, y0=y0, saveat=saveat,
                 max_steps=max_steps, stepsize_controller=controller)
    sol.ys.block_until_ready()
    wall = time.perf_counter() - t_r

    ys = np.asarray(sol.ys)
    ts = np.asarray(sol.ts)
    finite = bool(np.all(np.isfinite(ys)))
    us_per_step = wall / N_STEPS * 1e6

    print(f"  warmup (compile): {warmup_s:.1f}s")
    print(f"  transient wall: {wall:.2f}s ({us_per_step:.1f} µs/step)")
    print(f"  finite: {finite}")

    if finite:
        v_n1 = ys[:, port_map["n1,p1"]]
        mask = ts > 100e-9
        if mask.sum() > 3:
            x = v_n1[mask] - v_n1[mask].mean()
            zc = np.where(np.diff(np.sign(x)) > 0)[0]
            if len(zc) >= 3:
                period = np.median(np.diff(ts[mask][zc[1:]]))
                freq = 1.0 / period if period > 0 else float("nan")
                print(f"  oscillation freq: {freq/1e6:.2f} MHz")


if __name__ == "__main__":
    main()
