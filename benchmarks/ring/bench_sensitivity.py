"""Production workflow benchmark: OSDI forward + adjoint parameter sensitivity.

Demonstrates the two-API approach for differentiable circuit simulation:

  1. Forward simulation  — OSDI (compiled PSP103 binary) via bosdi FFI.
     Fast because no JAX tracing through the device physics.

  2. Parameter sensitivity — discrete adjoint via
     ``transient_parameter_sensitivity`` (or DC fallback).
     Avoids autodiff through the OSDI XLA FFI; uses finite-difference
     ∂F/∂p per parameter with one KLU adjoint solve per time step.

Key design point: OSDI and the adjoint *are* compatible.  The forward
trajectory is computed through OSDI (fast), the backward adjoint sweep
rebuilds J_eff at each checkpoint using the same OSDI residual evaluation,
then solves the adjoint linear system.

Usage:
    pixi run python benchmarks/ring/bench_sensitivity.py [--n-stages N]
                                                         [--n-steps N]
                                                         [--mode dc|transient]
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

# Persistent JAX compilation cache (avoids re-JIT across runs)
_JAX_CACHE = Path.home() / ".cache" / "jax" / "circulax_ring"
_JAX_CACHE.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_JAX_CACHE))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "tests"))
sys.path.insert(0, "/home/cdaunt/code/bosdi/src")

# Simulation parameters
T1 = 200e-12      # 200 ps (short enough for fast bench, enough for oscillation)
DT = 50e-12       # 50 ps fixed step
N_STEPS_DEFAULT = 4  # 4 steps × 50 ps = 200 ps (JIT warmup dominates anyway)


# ---------------------------------------------------------------------------
# Circuit build
# ---------------------------------------------------------------------------


def build_ring_osdi(n_stages: int = 3):
    """Build N-stage CMOS ring oscillator with PSP103 OSDI model.

    Returns (groups, sys_size, port_map, psp103n_descriptor, psp103p_descriptor).
    """
    from fixtures.psp103_models import geom_settings, make_psp103_descriptors

    from circulax import compile_netlist
    from circulax.components.electronic import Capacitor, Resistor, SmoothPulse, VoltageSource

    psp103n, psp103p = make_psp103_descriptors()
    mos_n = geom_settings(10e-6, 1e-6)
    mos_p = geom_settings(20e-6, 1e-6)

    instances = {
        "Vvdd":  {"component": "vsrc",   "settings": {"V": 1.2}},
        "Vkick": {"component": "kick",   "settings": {"V": 1.0, "delay": 1e-9, "tr": 1e-9}},
        "Rkick": {"component": "r_kick", "settings": {"R": 1e5}},
    }
    connections = {
        "Vvdd,p1": "vdd,p1",      "Vvdd,p2": "GND,p1",
        "Vkick,p1": "kick_n,p1",  "Vkick,p2": "GND,p1",
        "Rkick,p1": "kick_n,p1",  "Rkick,p2": "n1,p1",
    }
    for stage in range(1, n_stages + 1):
        in_n, out_n = f"n{stage}", f"n{stage % n_stages + 1}"
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

    models = {
        "nmos": psp103n, "pmos": psp103p,
        "vsrc": VoltageSource, "kick": SmoothPulse,
        "r_kick": Resistor, "cload": Capacitor,
    }
    groups, sys_size, port_map = compile_netlist(
        {"instances": instances, "connections": connections, "ports": {"out": "n1,p1"}},
        models,
    )
    return groups, sys_size, port_map, psp103n, psp103p


# ---------------------------------------------------------------------------
# DC initialisation
# ---------------------------------------------------------------------------


def dc_init(groups, sys_size, solver):
    """Run gmin-stepping DC initialisation.  Returns y0 (DC operating point)."""
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, 1e-2)
    y_src = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    return solver.solve_dc_gmin(groups, y_src, g_start=1e-2, n_steps=30)


# ---------------------------------------------------------------------------
# Forward transient simulation
# ---------------------------------------------------------------------------


def run_forward_transient(groups, sys_size, solver, y0, t1=T1, dt=DT, n_save=50):
    """Run transient simulation and return (sol, wall_s, compile_s).

    Uses a short warmup run (same max_steps/saveat shape, tiny t1) to pay the
    JIT cost before the timed run.
    """
    from circulax.solvers import setup_transient
    from circulax.solvers.transient import TrapFactorizedTransientSolver

    run_fn = setup_transient(groups, solver, transient_solver=TrapFactorizedTransientSolver)
    n_steps = int(t1 / dt)
    max_steps = int(2 * n_steps)
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, t1, n_save))
    controller = diffrax.ConstantStepSize()

    # Warmup: same shape parameters, tiny t1 so execution exits quickly.
    saveat_warmup = diffrax.SaveAt(ts=jnp.linspace(0.0, 2 * dt, n_save))
    t_compile = time.perf_counter()
    _ = run_fn(
        t0=0.0, t1=2 * dt, dt0=dt, y0=y0, saveat=saveat_warmup,
        max_steps=max_steps, stepsize_controller=controller,
    ).ys.block_until_ready()
    compile_s = time.perf_counter() - t_compile

    t_wall = time.perf_counter()
    sol = run_fn(
        t0=0.0, t1=t1, dt0=dt, y0=y0, saveat=saveat,
        max_steps=max_steps, stepsize_controller=controller,
    )
    sol.ys.block_until_ready()
    wall_s = time.perf_counter() - t_wall

    return sol, wall_s, compile_s


# ---------------------------------------------------------------------------
# OSDI group key resolution
# ---------------------------------------------------------------------------


def find_osdi_keys(groups: dict) -> list[str]:
    """Return keys of all OsdiComponentGroup instances in the group dict."""
    try:
        from bosdi.circulax import OsdiComponentGroup
    except ImportError:
        return []
    return [k for k, g in groups.items() if isinstance(g, OsdiComponentGroup)]


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def main(n_stages: int = 3, mode: str = "transient", n_forward_steps: int = 50) -> None:
    """Run the sensitivity benchmark.

    Args:
        n_stages: Number of ring oscillator stages (must be odd, >= 3).
        mode: ``"transient"`` for full transient adjoint,
              ``"dc"`` for DC-only adjoint (faster, simpler).
        n_forward_steps: Number of time points saved in the trajectory.
            More points = more adjoint steps = more accurate but slower.

    """
    from circulax.solvers import analyze_circuit
    from circulax.solvers.adjoint import transient_parameter_sensitivity
    from circulax.solvers.sensitivity import dc_parameter_sensitivity

    sep = "-" * 62
    print(sep)
    print("  Circulax OSDI Forward + Adjoint Sensitivity Benchmark")
    print(sep)
    print(f"  Ring stages    : {n_stages}")
    print(f"  Sensitivity    : {mode}")
    print(f"  Forward saves  : {n_forward_steps}")
    print()

    # -----------------------------------------------------------------------
    # Step 1: Build circuit
    # -----------------------------------------------------------------------
    t0 = time.perf_counter()
    groups, sys_size, port_map, psp103n_desc, psp103p_desc = build_ring_osdi(n_stages)
    build_s = time.perf_counter() - t0
    print(f"  [build]  groups compiled   : {build_s:.2f} s  (sys_size={sys_size})")

    osdi_keys = find_osdi_keys(groups)
    if not osdi_keys:
        print("  ERROR: No OSDI groups found.  bosdi not available?")
        return
    print(f"  [build]  OSDI group keys   : {osdi_keys}")

    # Separate NMOS and PMOS group keys
    # By convention the compiler assigns group keys by component type.
    # PSP103 NMOS and PMOS share the same .osdi file so they both become
    # OsdiComponentGroup but with different default params.
    nmos_key = osdi_keys[0]  # first OSDI group = NMOS (alphabetical or insertion order)
    print()

    # -----------------------------------------------------------------------
    # Step 2: DC operating point
    # -----------------------------------------------------------------------
    solver = analyze_circuit(groups, sys_size, backend="klu_split")
    t0 = time.perf_counter()
    y0 = dc_init(groups, sys_size, solver)
    dc_s = time.perf_counter() - t0
    if not bool(jnp.all(jnp.isfinite(y0))):
        print("  ERROR: DC solve diverged — check circuit topology.")
        return
    y_max = float(jnp.max(jnp.abs(y0)))
    if y_max > 10.0:
        print(f"  ERROR: DC solution magnitude {y_max:.2e} V — likely diverged.")
        return
    print(f"  [DC]     operating point   : {dc_s:.2f} s  (|y|_max={y_max:.3f} V)")

    # -----------------------------------------------------------------------
    # Step 3: Forward transient simulation
    # -----------------------------------------------------------------------
    print()
    print(f"  [fwd]    running OSDI transient (T={T1*1e12:.0f} ps, dt={DT*1e12:.0f} ps) ...")
    sol, fwd_wall, fwd_compile = run_forward_transient(
        groups, sys_size, solver, y0, t1=T1, dt=DT, n_save=n_forward_steps
    )
    y_traj = sol.ys  # shape: (n_forward_steps, sys_size)
    ts = sol.ts      # shape: (n_forward_steps,)

    n_actual_steps = int(T1 / DT)
    us_per_step = fwd_wall / n_actual_steps * 1e6

    if not np.all(np.isfinite(np.asarray(y_traj))):
        print("  ERROR: Transient returned NaN.  Try shorter T1 or larger dt.")
        return

    print(f"  [fwd]    JIT/compile       : {fwd_compile:.2f} s")
    print(f"  [fwd]    forward wall time : {fwd_wall:.4f} s")
    print(f"  [fwd]    steps simulated   : {n_actual_steps}")
    print(f"  [fwd]    us/step (OSDI)    : {us_per_step:.1f} us")

    # -----------------------------------------------------------------------
    # Step 4: Parameter sensitivity
    #
    # Parameters chosen for physical relevance to ring oscillator performance:
    #   delvto  — threshold voltage shift (NMOS; directly shifts VT, affects
    #             drive current and hence ring oscillation frequency).
    #   vfbo    — flat-band voltage offset (shifts inversion onset, affects
    #             sub-threshold leakage and switching speed).
    #
    # Both are PSP103 runtime process parameters with non-trivial gradients.
    # Note: tox/nf are compile-time constants in this OSDI build and give
    # zero gradient — use delvto/vfbo instead.
    # -----------------------------------------------------------------------
    param_names = ["delvto", "vfbo"]

    # Build loss function: mean squared output voltage over last half of trajectory
    n1_idx = port_map["n1,p1"]
    half = n_forward_steps // 2

    def loss_fn_traj(yt, ts_unused):
        """Sum of squared node-1 voltage over final half of trajectory."""
        return jnp.sum(yt[half:, n1_idx] ** 2)

    def loss_fn_final(y_final):
        """Squared output voltage at final time step."""
        return y_final[n1_idx] ** 2

    print()
    print(f"  [sens]   mode = '{mode}', params = {param_names}")

    t0_sens = time.perf_counter()

    if mode == "dc":
        # -------------------------------------------------------------------
        # DC adjoint: single linear solve, then FD per param per device
        # Fastest; validates the adjoint API without paying transient cost.
        # -------------------------------------------------------------------
        grads = dc_parameter_sensitivity(
            groups,
            solver,
            y0,
            lambda y: y[n1_idx] ** 2,
            osdi_group_key=nmos_key,
            param_names=param_names,
            model_descriptor=psp103n_desc,
        )

    elif mode == "transient":
        # -------------------------------------------------------------------
        # Transient adjoint: backward sweep over all n_forward_steps checkpoints.
        # Uses KLU solver (same solver object from forward pass).
        # -------------------------------------------------------------------
        try:
            grads = transient_parameter_sensitivity(
                groups,
                solver,
                y_traj,
                ts,
                loss_fn_traj,
                osdi_group_key=nmos_key,
                param_names=param_names,
                model_descriptor=psp103n_desc,
            )
        except TypeError as e:
            # DenseSolver fallback if KLU attributes not present
            from circulax.solvers.adjoint import transient_parameter_sensitivity_dense
            print(f"  [sens]   KLU adjoint failed ({e!r}), falling back to dense ...")
            grads = transient_parameter_sensitivity_dense(
                groups,
                y_traj,
                ts,
                loss_fn_traj,
                osdi_group_key=nmos_key,
                param_names=param_names,
                model_descriptor=psp103n_desc,
            )
    else:
        print(f"  ERROR: unknown mode {mode!r}.  Use 'dc' or 'transient'.")
        return

    sens_s = time.perf_counter() - t0_sens

    # -----------------------------------------------------------------------
    # Step 5: Report
    # -----------------------------------------------------------------------
    print()
    print(sep)
    print("  Results")
    print(sep)
    print(f"  Build + compile  : {build_s:.2f} s")
    print(f"  DC init          : {dc_s:.2f} s")
    print(f"  Forward (JIT)    : {fwd_compile:.2f} s")
    print(f"  Forward (run)    : {fwd_wall:.4f} s   ({us_per_step:.1f} us/step)")
    print(f"  Gradient ({mode:>9s}) : {sens_s:.2f} s   ({len(param_names)} params, {groups[nmos_key].params.shape[0]} NMOS devices)")
    print()
    print("  Gradients (∂loss/∂p per NMOS device):")
    for pname, g in grads.items():
        arr = np.asarray(g)
        print(f"    {pname:>10s}: {arr}  (shape {arr.shape})")
    print()
    print("  Loss function    : sum(V_n1[t>T/2]^2) for transient, V_n1(DC)^2 for dc")
    print()
    print("  Note: OSDI forward path used for all simulation.")
    print("        Adjoint differentiates through OSDI residual via finite difference.")
    print("        No autodiff through the XLA FFI is required.")
    print("        tox/nf are compile-time OSDI constants → zero gradient (by design).")
    print("        delvto/vfbo are runtime params → non-zero gradient.")
    if mode == "transient":
        n_adj_steps = n_forward_steps - 1
        print(f"        Adjoint performed {n_adj_steps} backward linear solves (one per checkpoint gap).")
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="OSDI forward + adjoint sensitivity benchmark for ring oscillator"
    )
    p.add_argument("--n-stages", type=int, default=3,
                   help="Number of ring stages (odd, >=3; default 3 for speed)")
    p.add_argument("--mode", choices=("dc", "transient"), default="transient",
                   help="Sensitivity mode: 'dc' (fast, DC only) or 'transient' (full adjoint)")
    p.add_argument("--n-forward-steps", type=int, default=50,
                   help="Number of trajectory checkpoints saved (more = more adjoint steps)")
    args = p.parse_args()

    main(n_stages=args.n_stages, mode=args.mode, n_forward_steps=args.n_forward_steps)
