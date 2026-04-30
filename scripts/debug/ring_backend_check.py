"""Run the (non-Schur) ring with klu_split (refactor/quadratic) backend.

Quick isolation: does the refactor backend alone shift the frequency away
from 159 MHz, independent of the Schur experiment? If yes, the solver
choice matters; if no, the gap is elsewhere in the assembly.
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

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "tests"))
sys.path.insert(0, "/home/cdaunt/code/bosdi/src")
sys.path.insert(0, str(_REPO / "scripts"))

from benchmarks.ring.circulax.ring_one_case import build_netlist, dom_freq


def run(c_load: float, backend: str, t1: float = 50e-9, dt0: float = 1e-12,
        n_save: int = 1000):
    from circulax.solvers import analyze_circuit, setup_transient
    from circulax.solvers.transient import SDIRK3VectorizedTransientSolver

    groups, sys_size, port_map = build_netlist(c_load)
    solver_0 = analyze_circuit(groups, sys_size, backend=backend)

    G_HOM = 1e-2
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver_0, G_HOM)
    y_src = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y0 = solver_0.solve_dc_gmin(groups, y_src, g_start=G_HOM, n_steps=30)
    assert bool(jnp.all(jnp.isfinite(y0)))

    n1 = port_map["n1,p1"]
    run_fn = setup_transient(groups, solver_0, transient_solver=SDIRK3VectorizedTransientSolver)
    sol = run_fn(
        t0=0.0, t1=t1, dt0=dt0, y0=y0,
        saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t1, n_save)),
        max_steps=2_000_000,
        stepsize_controller=diffrax.PIDController(
            rtol=1e-4, atol=1e-6, dtmin=1e-14, dtmax=1e-10,
        ),
    )
    ys = np.asarray(sol.ys)
    t = np.asarray(sol.ts)
    v = ys[:, n1]
    freq = dom_freq(t, v)
    print(
        f"backend={backend:>22s}  c_load={c_load*1e15:>5.1f}fF  "
        f"period={1e9/freq:>6.3f} ns  freq={freq/1e6:>7.2f} MHz  "
        f"swing={v.max() - v.min():.3f} V"
    )


if __name__ == "__main__":
    for backend in ("klu", "klu_split_linear", "klu_split", "klu_split_refactor"):
        try:
            run(c_load=50e-15, backend=backend, t1=30e-9, n_save=500)
        except Exception as e:
            print(f"backend={backend}: FAILED {type(e).__name__}: {str(e)[:60]}")
