"""BDF2 with and without load cap and Schur; find the smallest-remaining gap."""

from __future__ import annotations

import sys
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
sys.path.insert(0, str(_REPO / "scripts"))

from benchmarks.ring.circulax.ring_one_case import build_netlist, dom_freq


def run(label: str, c_load: float, integrator_name: str = "BDF2", use_schur: bool = False,
        t1: float = 30e-9):
    from circulax.solvers import analyze_circuit, setup_transient
    from circulax.solvers.transient import (
        BDF2FactorizedTransientSolver,
        SDIRK3FactorizedTransientSolver,
        VectorizedTransientSolver,
        FactorizedTransientSolver,
    )
    integrators = {
        "BE":    FactorizedTransientSolver,
        "BDF2":  BDF2FactorizedTransientSolver,
        "SDIRK3": SDIRK3FactorizedTransientSolver,
    }
    integrator_cls = integrators[integrator_name]

    if use_schur:
        from fixtures.psp103_models import (
            PSP103_OSDI, PSP103N_DEFAULTS, PSP103P_DEFAULTS, geom_settings,
        )
        from circulax import compile_netlist
        from circulax.components.osdi import osdi_component
        from circulax.components.electronic import (
            Capacitor, Resistor, SmoothPulse, VoltageSource,
        )
        nmos = osdi_component(osdi_path=PSP103_OSDI, ports=("D","G","S","B"),
                              default_params=PSP103N_DEFAULTS, use_schur_reduction=True)
        pmos = osdi_component(osdi_path=PSP103_OSDI, ports=("D","G","S","B"),
                              default_params=PSP103P_DEFAULTS, use_schur_reduction=True)
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
        models = {"nmos": nmos, "pmos": pmos, "vsrc": VoltageSource,
                  "kick": SmoothPulse, "r_kick": Resistor, "cload": Capacitor}
        groups, sys_size, port_map = compile_netlist(
            {"instances": instances, "connections": connections, "ports": {"out": "n1,p1"}}, models)
    else:
        groups, sys_size, port_map = build_netlist(c_load)

    solver = analyze_circuit(groups, sys_size, backend="klu_split")

    G_HOM = 1e-2
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, G_HOM)
    y_src = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y0 = solver.solve_dc_gmin(groups, y_src, g_start=G_HOM, n_steps=30)
    if not bool(jnp.all(jnp.isfinite(y0))):
        print(f"  {label:<40s}  DC diverged")
        return

    n1 = port_map["n1,p1"]
    run_fn = setup_transient(groups, solver, transient_solver=integrator_cls)
    try:
        sol = run_fn(
            t0=0.0, t1=t1, dt0=1e-12, y0=y0,
            saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t1, 500)),
            max_steps=2_000_000,
            stepsize_controller=diffrax.PIDController(
                rtol=1e-4, atol=1e-6, dtmin=1e-14, dtmax=1e-10,
            ),
        )
    except Exception as e:
        print(f"  {label:<40s}  FAILED: {type(e).__name__}: {str(e)[:50]}")
        return

    ys = np.asarray(sol.ys)
    t = np.asarray(sol.ts)
    if not np.all(np.isfinite(ys)):
        print(f"  {label:<40s}  NaN")
        return
    v = ys[:, n1]
    freq = dom_freq(t, v)
    print(f"  {label:<40s}  period={1e9/freq:>6.3f} ns  "
          f"freq={freq/1e6:>7.2f} MHz  swing={v.max() - v.min():.3f} V")


if __name__ == "__main__":
    print("[full 6x6 assembly]")
    run("BDF2 + C_load=50fF",             c_load=50e-15, integrator_name="BDF2")
    run("BDF2 + C_load=0",                c_load=0,      integrator_name="BDF2")
    run("BE + C_load=0",                  c_load=0,      integrator_name="BE")
    print("[Schur-reduced assembly]")
    run("BDF2 + Schur + C_load=50fF",     c_load=50e-15, integrator_name="BDF2", use_schur=True)
    run("BDF2 + Schur + C_load=0",        c_load=0,      integrator_name="BDF2", use_schur=True)
    run("BE + Schur + C_load=0",          c_load=0,      integrator_name="BE",   use_schur=True)
