"""Profile Newton iterations per timestep for the VA ring oscillator.

Runs an N=3 ring oscillator with VA PSP103 models for a short transient
(200 steps at 50 ps) and counts Newton iterations per step using a
jax.debug.callback inside the Newton loop.

Usage:
    pixi run python scripts/va_newton_profile.py [--n-stages N] [--n-steps N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

jax.config.update("jax_enable_x64", True)

# Persistent compilation cache.
_JAX_CACHE = Path.home() / ".cache" / "jax" / "circulax_ring"
_JAX_CACHE.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_JAX_CACHE))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "tests"))
sys.path.insert(0, str(_REPO / "scripts"))

# -----------------------------------------------------------------------
# Instrumented solver subclass
# -----------------------------------------------------------------------
from circulax.solvers.assembly import (
    assemble_residual_only_real,
    assemble_system_real,
)
from circulax.solvers.linear import DAMPING_EPS, DAMPING_FACTOR, GROUND_STIFFNESS
from circulax.solvers.transient import TrapFactorizedTransientSolver, _compute_history_fq, _trap_preamble, free_numeric

# Accumulator list (host side); filled via callback.
_newton_counts: list[int] = []


def _record_newton_count(n: int) -> None:
    """Host callback: appends the iteration count to the global list."""
    _newton_counts.append(int(n))


class InstrumentedTrapFactorized(TrapFactorizedTransientSolver):
    """TrapFactorizedTransientSolver that records Newton iteration counts
    via jax.debug.callback after each step.
    """

    def step(self, terms, t0, t1, y0, args, solver_state, options):  # noqa: D102
        component_groups, num_vars = args
        h_n = t1 - t0
        is_complex = getattr(self.linear_solver, "is_complex", False)

        y_pred, alpha, make_residual = _trap_preamble(
            y0, t0, h_n, solver_state, component_groups, num_vars, is_complex,
        )

        # Only real path is needed for the ring oscillator.
        _, _, frozen_jac_vals = assemble_system_real(
            y_pred, component_groups, t1, h_n, alpha=alpha,
        )
        ground_indices = [0]

        numeric_handle = self.linear_solver.factor_jacobian(frozen_jac_vals)

        def newton_update_step(y: jax.Array, _: Any) -> jax.Array:
            total_f, total_q = assemble_residual_only_real(y, component_groups, t1, h_n)
            residual = make_residual(total_f, total_q)
            for idx in ground_indices:
                residual = residual.at[idx].add(GROUND_STIFFNESS * y[idx])

            sol = self.linear_solver.solve_with_frozen_jacobian(-residual, numeric_handle)
            delta = sol.value
            max_change = jnp.max(jnp.abs(delta))
            damping = jnp.minimum(1.0, DAMPING_FACTOR / (max_change + DAMPING_EPS))
            return y + delta * damping

        fpi = optx.FixedPointIteration(rtol=self.newton_rtol, atol=self.newton_atol)
        sol = optx.fixed_point(
            newton_update_step, fpi, y_pred,
            max_steps=self.newton_max_steps, throw=False,
        )
        free_numeric(numeric_handle)

        # Emit the iteration count back to the host.
        jax.debug.callback(_record_newton_count, sol.stats["num_steps"])

        y_next = sol.value
        y_error = y_next - y_pred

        # Cache f/q at the converged solution for the next step's preamble.
        f_new, q_new = _compute_history_fq(component_groups, y_next, t1, num_vars, is_complex)
        new_state = (y0, h_n, f_new, q_new)

        result = jax.lax.cond(
            sol.result == optx.RESULTS.successful,
            lambda _: diffrax.RESULTS.successful,
            lambda _: diffrax.RESULTS.nonlinear_divergence,
            None,
        )
        return y_next, y_error, {"y0": y0, "y1": y_next}, new_state, result


# -----------------------------------------------------------------------
# Ring oscillator build (VA variant, copied from bench_circulax.py)
# -----------------------------------------------------------------------

def _build_va_ring(n_stages: int):
    import dataclasses
    import importlib.util
    import tempfile

    from circulax.va.emitter import emit_source
    from circulax.va.va_defaults import parse_va_defaults_expanded
    from fixtures.psp103_models import PSP103N_DEFAULTS, PSP103P_DEFAULTS, geom_settings

    from circulax import compile_netlist
    from circulax.components.electronic import Capacitor, Resistor, SmoothPulse, VoltageSource
    from circulax.va import compile_va_unopt_with_split, lower

    _PSP103_VA = _REPO / "tests" / "data" / "va" / "psp103v4" / "psp103.va"
    print(f"[build] Compiling PSP103 VA from {_PSP103_VA}...", flush=True)
    t0 = time.perf_counter()
    dump = compile_va_unopt_with_split(str(_PSP103_VA))
    defaults = parse_va_defaults_expanded(_PSP103_VA)
    print(f"[build] VA compile done in {time.perf_counter()-t0:.1f}s", flush=True)

    int_static_n = {
        name: int(spec.default)
        for name, spec in defaults.items()
        if spec.type_ == "int"
    }
    int_static_n["TYPE"] = 1
    int_static_p = {**int_static_n, "TYPE": -1}

    dev_n = lower(dump.modules[0], va_defaults=defaults, collapse_nodes=True,
                  static_params=int_static_n,
                  differentiable_params=(),
                  class_name="PSP103N")
    dev_p = lower(dump.modules[0], va_defaults=defaults, collapse_nodes=True,
                  static_params=int_static_p,
                  differentiable_params=(),
                  class_name="PSP103P")
    tmp = tempfile.mkdtemp()
    out = Path(tmp) / "psp103_va_bench.py"
    out.write_text(emit_source([dev_n, dev_p]))
    spec = importlib.util.spec_from_file_location("psp103_va_bench", out)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls_n = mod.PSP103N
    cls_p = mod.PSP103P

    def _fields(cls):
        return {f.name: f for f in dataclasses.fields(cls) if f._field_type.name == "_FIELD"}

    def _coerce(fields_by_name, name, value):
        ft = fields_by_name[name].type
        return int(float(value)) if ft == "int" else float(value)

    fn = _fields(cls_n)
    nmos_p = {k: _coerce(fn, k, v) for k, v in PSP103N_DEFAULTS.items() if k in fn}
    nmos_p.update({k: _coerce(fn, k, v) for k, v in geom_settings(10e-6, 1e-6).items() if k in fn})
    fp = _fields(cls_p)
    pmos_p = {k: _coerce(fp, k, v) for k, v in PSP103P_DEFAULTS.items() if k in fp}
    pmos_p.update({k: _coerce(fp, k, v) for k, v in geom_settings(20e-6, 1e-6).items() if k in fp})

    instances: dict = {
        "Vvdd":  {"component": "vsrc",  "settings": {"V": 1.2}},
        "Vkick": {"component": "kick",  "settings": {"V": 1.0, "delay": 1e-9, "tr": 1e-9}},
        "Rkick": {"component": "r_kick","settings": {"R": 1e5}},
    }
    connections: dict = {
        "Vvdd,p1": "vdd,p1",   "Vvdd,p2": "GND,p1",
        "Vkick,p1": "kick_n,p1", "Vkick,p2": "GND,p1",
        "Rkick,p1": "kick_n,p1", "Rkick,p2": "n1,p1",
    }
    for stage in range(1, n_stages + 1):
        in_n  = f"n{stage}"
        out_n = f"n{stage % n_stages + 1}"
        mn, mp = f"mn{stage}", f"mp{stage}"
        instances[mn] = {"component": "nmos", "settings": nmos_p}
        instances[mp] = {"component": "pmos", "settings": pmos_p}
        connections[f"{mn},D"] = f"{out_n},p1"
        connections[f"{mn},G"] = f"{in_n},p1"
        connections[f"{mn},S"] = "GND,p1"
        connections[f"{mn},B"] = "GND,p1"
        connections[f"{mp},D"] = f"{out_n},p1"
        connections[f"{mp},G"] = f"{in_n},p1"
        connections[f"{mp},S"] = "vdd,p1"
        connections[f"{mp},B"] = "vdd,p1"
    models = {"nmos": cls_n, "pmos": cls_p, "vsrc": VoltageSource,
              "kick": SmoothPulse, "r_kick": Resistor, "cload": Capacitor}
    return compile_netlist(
        {"instances": instances, "connections": connections, "ports": {"out": "n1,p1"}},
        models,
    )


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Profile Newton iterations per step (VA ring).")
    ap.add_argument("--n-stages", type=int, default=3,
                    help="Number of ring inverter stages (default 3; must be odd)")
    ap.add_argument("--n-steps", type=int, default=200,
                    help="Number of timesteps to profile (default 200)")
    ap.add_argument("--dt", type=float, default=50e-12,
                    help="Timestep in seconds (default 50ps)")
    args = ap.parse_args()

    n_stages = args.n_stages
    n_steps  = args.n_steps
    dt       = args.dt
    T1       = n_steps * dt

    print(f"=== VA Newton Profiler: N={n_stages} stages, {n_steps} steps at {dt*1e12:.0f} ps ===",
          flush=True)

    from circulax.solvers import analyze_circuit, setup_transient

    print("[setup] Building netlist...", flush=True)
    t0 = time.perf_counter()
    groups, sys_size, port_map = _build_va_ring(n_stages)
    print(f"[setup] Netlist done in {time.perf_counter()-t0:.1f}s", flush=True)

    solver = analyze_circuit(groups, sys_size, backend="klu_split")

    # DC init (matches bench_circulax.py VA path).
    ring_nodes = {f"n{s},p1" for s in range(1, n_stages + 1)}
    y_init = jnp.zeros(sys_size)
    for key, idx in port_map.items():
        if key in ring_nodes:
            y_init = y_init.at[idx].set(0.6)
        elif key == "vdd,p1":
            y_init = y_init.at[idx].set(1.2)

    print("[dc] Running DC gmin stepping...", flush=True)
    t0 = time.perf_counter()
    y0 = solver.solve_dc_gmin(groups, y_init, g_start=1e-2, n_steps=30)
    print(f"[dc] Done in {time.perf_counter()-t0:.1f}s, y0_max={float(jnp.max(jnp.abs(y0))):.3f}",
          flush=True)

    if not bool(jnp.all(jnp.isfinite(y0))) or float(jnp.max(jnp.abs(y0))) > 10.0:
        print("ERROR: DC solve diverged — check DC initialisation.")
        sys.exit(1)

    # Instrumented transient solver.
    instrumented_solver = InstrumentedTrapFactorized(linear_solver=solver)
    run_fn = setup_transient(groups, solver,
                             transient_solver=instrumented_solver)

    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, T1, min(n_steps, 200)))
    controller = diffrax.ConstantStepSize()
    max_steps = int(2 * n_steps)

    # Warmup (2 steps) — must share the same static JAX shapes.
    saveat_warmup = diffrax.SaveAt(ts=jnp.linspace(0.0, 2 * dt, min(n_steps, 200)))
    print("[jit] Warming up (2 steps, shared shapes)...", flush=True)
    t_compile = time.perf_counter()
    _newton_counts.clear()
    _ = run_fn(
        t0=0.0, t1=2 * dt, dt0=dt, y0=y0,
        saveat=saveat_warmup, max_steps=max_steps,
        stepsize_controller=controller,
    ).ys.block_until_ready()
    compile_s = time.perf_counter() - t_compile
    warmup_counts = list(_newton_counts)
    print(f"[jit] Done in {compile_s:.1f}s  (warmup Newton counts: {warmup_counts})", flush=True)

    # Timed profiling run.
    _newton_counts.clear()
    print(f"[run] Profiling {n_steps} steps...", flush=True)
    t_run = time.perf_counter()
    sol = run_fn(
        t0=0.0, t1=T1, dt0=dt, y0=y0,
        saveat=saveat, max_steps=max_steps,
        stepsize_controller=controller,
    )
    sol.ys.block_until_ready()
    wall = time.perf_counter() - t_run

    counts = list(_newton_counts)

    if not counts:
        print("WARNING: No Newton counts collected — callback may not have fired.")
        print(f"  sol.result = {sol.result}")
        sys.exit(1)

    arr = np.array(counts, dtype=np.int32)
    avg = arr.mean()
    mx  = arr.max()
    mn  = arr.min()
    p50 = np.percentile(arr, 50)
    p90 = np.percentile(arr, 90)
    p99 = np.percentile(arr, 99)
    n_collected = len(arr)

    print()
    print("=" * 60)
    print("Newton iteration statistics per timestep")
    print("=" * 60)
    print(f"  Steps collected : {n_collected}  (requested {n_steps})")
    print(f"  Average         : {avg:.2f}")
    print(f"  Min             : {mn}")
    print(f"  Max             : {mx}")
    print(f"  P50 (median)    : {p50:.1f}")
    print(f"  P90             : {p90:.1f}")
    print(f"  P99             : {p99:.1f}")
    print()

    # Histogram (ASCII, 10 bins).
    counts_unique, bin_counts = np.unique(arr, return_counts=True)
    print("  Histogram (iterations : count):")
    for val, cnt in zip(counts_unique, bin_counts):
        bar = "#" * min(int(cnt / max(bin_counts) * 40), 40)
        pct = 100.0 * cnt / n_collected
        print(f"    {val:3d} : {bar:<40s} {cnt:5d}  ({pct:5.1f}%)")
    print()
    print(f"  Wall time for {n_steps} steps : {wall:.2f}s  ({wall/n_steps*1e6:.1f} us/step)")
    print()

    # Key question: is average > 10?
    if avg > 10:
        print(f"  VERDICT: avg={avg:.1f} > 10 -> Task 6 (tolerance relaxation / predictor) IS needed.")
    else:
        print(f"  VERDICT: avg={avg:.1f} <= 10 -> Newton convergence is acceptable, Task 6 not required.")
    print("=" * 60)


if __name__ == "__main__":
    main()
