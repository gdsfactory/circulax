"""Verify whether our vmap benchmark is fully inside a JIT boundary.

The bosdi author flagged that osdi_eval's vmap rule does
``jnp.broadcast_to(x[None], ...)`` on non-batched args; under jit this
is a free XLA view, in eager mode it can force a params-copy per call.

Our scripts/bench_one.py::run_vmap already wraps the vmap'd function
in ``jax.jit``, but let's confirm with a direct A/B:

  A. sim_batch = jax.jit(jax.vmap(sim_one))              (current)
  B. sim_batch = jax.jit(jax.vmap(jax.jit(sim_one)))     (inner JIT too)

If A and B give the same wall, we were already fully staged.  If B is
faster, there was eager dispatch escaping and we need to keep the
inner jit.
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
sys.path.insert(0, "/home/cdaunt/code/bosdi/src")

from ring_one_case import build_netlist
from circulax.solvers import analyze_circuit, setup_transient
from circulax.solvers.transient import TrapRefactoringTransientSolver


def _build():
    groups, sys_size, port_map = build_netlist(c_load=0)
    solver = analyze_circuit(groups, sys_size, backend="klu_rs_split")
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, 1e-2)
    y_src = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y0 = solver.solve_dc_gmin(groups, y_src, g_start=1e-2, n_steps=30)
    y0 = jnp.asarray(y0)
    run = setup_transient(groups, solver, transient_solver=TrapRefactoringTransientSolver)
    return sys_size, port_map, run, y0


def run_variant(label: str, inner_jit: bool, batch: int = 8):
    sys_size, port_map, run, y0 = _build()
    t1 = 200e-9
    dt = 5e-11
    n1 = port_map["n1,p1"]
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, t1, 200))
    controller = diffrax.ConstantStepSize()

    def sim_one(y_init):
        sol = run(t0=0.0, t1=t1, dt0=dt, y0=y_init, saveat=saveat,
                  max_steps=int(2 * t1 / dt), stepsize_controller=controller)
        return sol.ys[:, n1]

    target = jax.jit(sim_one) if inner_jit else sim_one
    sim_batch = jax.jit(jax.vmap(target))

    rng = np.random.default_rng(0)
    perturb = jnp.asarray(rng.normal(scale=1e-4, size=(batch, sys_size)))
    y0_batch = jnp.broadcast_to(y0, (batch, sys_size)) + perturb

    # Warmup (JIT compile).
    t0 = time.time()
    _ = sim_batch(y0_batch).block_until_ready()
    compile_wall = time.time() - t0

    # Timed run.
    t0 = time.time()
    out = sim_batch(y0_batch); out.block_until_ready()
    wall = time.time() - t0

    us_per_ring_step = wall / batch / int(t1 / dt) * 1e6
    print(
        f"  {label:<40s}  compile={compile_wall:>5.2f}s  wall={wall:>6.2f}s  "
        f"µs/ring/step={us_per_ring_step:>6.1f}"
    )


if __name__ == "__main__":
    print("9-stage PSP103 ring, vmap batch=8, 200 ns at dt=50 ps\n")
    run_variant("A. jax.jit(jax.vmap(sim_one))            ", inner_jit=False, batch=8)
    run_variant("B. jax.jit(jax.vmap(jax.jit(sim_one)))   ", inner_jit=True,  batch=8)
    print()
    run_variant("A. batch=32, outer jit only              ", inner_jit=False, batch=32)
    run_variant("B. batch=32, outer + inner jit           ", inner_jit=True,  batch=32)
