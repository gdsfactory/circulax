"""Run the PSP103 9-stage ring oscillator with Schur-reduced OSDI assembly.

Builds the same ring as tests/test_psp103_ring_oscillator.py but flips
``use_schur_reduction=True`` on the NMOS and PMOS descriptors.  Internal
OSDI nodes (di, si) are eliminated from the per-device stamp at the
integrator's effective coefficient (alpha/dt); the global Newton sees a
4x4 terminal-only stamp per device.

Decisive experiment to isolate candidate (1)/(3) in
docs/bosdi_psp103_ring_oscillator_issue.md:

- If the Schur-reduced ring produces ~937 MHz, the 6.7x slowdown lives in
  circulax's handling of internal-node unknowns (constraint-row dQ/dt
  routing, SDIRK3-stage coefficient treatment, or the Newton residual
  assembly for OSDI internal rows).
- If it still produces ~160 MHz, the gap is on the terminal side or in
  osdi_eval itself and needs a different investigation.
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
sys.path.insert(0, "/home/cdaunt/code/bosdi/src")


def make_psp103_descriptors_schur():
    """Same defaults as fixtures.psp103_models, but with Schur reduction on."""
    from circulax.components.osdi import osdi_component
    from fixtures.psp103_models import (
        PSP103_OSDI,
        PSP103N_DEFAULTS,
        PSP103P_DEFAULTS,
    )
    psp103n = osdi_component(
        osdi_path=PSP103_OSDI,
        ports=("D", "G", "S", "B"),
        default_params=PSP103N_DEFAULTS,
        use_schur_reduction=True,
    )
    psp103p = osdi_component(
        osdi_path=PSP103_OSDI,
        ports=("D", "G", "S", "B"),
        default_params=PSP103P_DEFAULTS,
        use_schur_reduction=True,
    )
    return psp103n, psp103p


def build_ring(c_load: float):
    from fixtures.psp103_models import geom_settings
    from circulax import compile_netlist
    from circulax.components.electronic import (
        Capacitor, Resistor, SmoothPulse, VoltageSource,
    )

    psp103n, psp103p = make_psp103_descriptors_schur()
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


def run_schur_ring(c_load: float = 50e-15, t1: float = 50e-9, dt0: float = 1e-12,
                   n_save: int = 1000, backend: str = "klu_split"):
    from circulax.solvers import analyze_circuit, setup_transient
    from circulax.solvers.transient import SDIRK3VectorizedTransientSolver

    t_start = time.time()
    groups, sys_size, port_map = build_ring(c_load)
    solver = analyze_circuit(groups, sys_size, backend=backend)

    G_HOM = 1e-2
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, G_HOM)
    y_src = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y0 = solver.solve_dc_gmin(groups, y_src, g_start=G_HOM, n_steps=30)
    if not bool(jnp.all(jnp.isfinite(y0))):
        print("DC solve returned non-finite values — aborting.")
        return

    print(f"  DC solve ok in {time.time() - t_start:.1f}s; y0 stats: "
          f"min={float(jnp.min(y0)):.3e}, max={float(jnp.max(y0)):.3e}")

    n1 = port_map["n1,p1"]
    run = setup_transient(groups, solver, transient_solver=SDIRK3VectorizedTransientSolver)
    t_tran = time.time()
    sol = run(
        t0=0.0, t1=t1, dt0=dt0, y0=y0,
        saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t1, n_save)),
        max_steps=2_000_000,
        stepsize_controller=diffrax.PIDController(
            rtol=1e-4, atol=1e-6, dtmin=1e-14, dtmax=1e-10,
        ),
    )
    ys = np.asarray(sol.ys)
    t = np.asarray(sol.ts)
    print(f"  transient done in {time.time() - t_tran:.1f}s")

    if not np.all(np.isfinite(ys)):
        print("  NaN in transient output — solver diverged.")
        return

    v = ys[:, n1]
    freq = dom_freq(t, v)
    print(f"\n  === SCHUR-REDUCED ring oscillator  (backend={backend}) ===")
    print(f"  c_load = {c_load*1e15:.1f} fF")
    print(f"  period = {1e9/freq:.3f} ns  ({freq/1e6:.2f} MHz)")
    print(f"  swing  = {v.max() - v.min():.3f} V")
    print(f"  VACASK reference: 1.07 ns (937 MHz) — ratio = {freq / 936.81e6:.3f}")
    print(f"  circulax (full 6x6, klu_split_factor): 6.26 ns (160 MHz)")


if __name__ == "__main__":
    import sys as _sys
    backend = _sys.argv[1] if len(_sys.argv) > 1 else "klu_split"
    run_schur_ring(c_load=50e-15, t1=50e-9, backend=backend)
