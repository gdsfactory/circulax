"""Circulax ring oscillator for the benchmark sweep.

Wraps the N-stage CMOS ring from scripts/ring_one_case.py so a single
``run(n_stages=...)`` call produces the wall / µs-per-step numbers the
harness in ``run.py`` needs.  Matches the VACASK ring template
(W=10 µm NMOS, W=20 µm PMOS, L=1 µm, VDD=1.2 V, 1 µs tran at 50 ps
fixed step → 20 000 steps).

Models:
  - ``osdi``       — PSP103 via the bosdi FFI path to the compiled .osdi binary.
  - ``va``         — PSP103 compiled from Verilog-A via MIR→XLA lowering.
                     First-call JIT is ~320 s per function (DC + transient).
                     Requires PYTHONPATH with local klujax/klujax_rs builds
                     (see /home/cdaunt/code/circulax/circulax/.env).
  - ``jax-native`` — simplified square-law MOSFET (no OSDI / no VA), ~20% off PSP103.
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

# Persistent compilation cache: avoids re-compiling the same XLA program
# across runs. With the larger setup function from
# ``compile_va_unopt_with_split`` (PSP103: 2086 cache slots vs 608 from
# the older ``compile_va_unopt`` path), first-time JIT can take 15+ min;
# subsequent runs hit the cache and skip the cost entirely.  Each
# (device-class, sys_size) pair has its own cache entry, so different
# N stages don't share but a re-bench at the same N is near-free.
_JAX_CACHE = Path.home() / ".cache" / "jax" / "circulax_ring"
_JAX_CACHE.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_JAX_CACHE))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "tests"))
sys.path.insert(0, str(_REPO / "scripts"))

# Upstream settings, exactly.
T1 = 1e-6        # 1 µs
DT = 5e-11       # 50 ps
N_STEPS = int(T1 / DT)
N_SAVE = 4000


def _build_ring(n_stages: int, variant: str, differentiable: bool = False):
    """Return (groups, sys_size, port_map) for an N-stage CMOS ring.

    ``differentiable=True`` keeps float process parameters as JAX-traced
    leaves so ``jax.grad`` works through them.  Integer switch parameters
    (TYPE, SWGEO, SWIGATE, …) are always folded as SCCP constants regardless.
    """
    if variant == "osdi":
        from ring_one_case import build_netlist as _bn
        return _bn(c_load=0.0, n_stages=n_stages)
    if variant == "va":
        import dataclasses
        import importlib.util
        import subprocess
        import tempfile
        from circulax import compile_netlist
        from circulax.components.electronic import Capacitor, Resistor, SmoothPulse, VoltageSource
        from circulax.va import compile_va_unopt_with_split, lower
        from circulax.va.emitter import emit_source
        from circulax.va.va_defaults import parse_va_defaults_expanded
        from fixtures.psp103_models import PSP103N_DEFAULTS, PSP103P_DEFAULTS, geom_settings

        _PSP103_VA = _REPO / "tests" / "data" / "va" / "psp103v4" / "psp103.va"
        # unopt-MIR-with-split: correct phi structure (nested conditionals
        # round-trip cleanly) AND init/eval split for proper per-instance
        # caching. Closes the lowering-time vs runtime trade-off that the
        # plain ``compile_va_unopt`` path had.
        dump = compile_va_unopt_with_split(str(_PSP103_VA))
        defaults = parse_va_defaults_expanded(_PSP103_VA)

        # Bake all integer switch parameters as SCCP constants so the lowering
        # can eliminate dead branches (SWGIDL=0 → no GIDL block, etc.) before
        # emitting Python.  This shrinks the XLA graph significantly compared
        # to TYPE-only lowering.  Float process params stay as class-level
        # static fields (always concrete inside JIT), or as JAX leaves when
        # differentiable=True is requested.
        int_static_n = {
            name: int(spec.default)
            for name, spec in defaults.items()
            if spec.type_ == "int"
        }
        int_static_n["TYPE"] = 1
        int_static_p = {**int_static_n, "TYPE": -1}

        diff_params = None if differentiable else ()
        dev_n = lower(dump.modules[0], va_defaults=defaults, collapse_nodes=True,
                      static_params=int_static_n,
                      differentiable_params=diff_params,
                      class_name="PSP103N")
        dev_p = lower(dump.modules[0], va_defaults=defaults, collapse_nodes=True,
                      static_params=int_static_p,
                      differentiable_params=diff_params,
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

        instances = {
            "Vvdd":  {"component": "vsrc",  "settings": {"V": 1.2}},
            "Vkick": {"component": "kick",  "settings": {"V": 1.0, "delay": 1e-9, "tr": 1e-9}},
            "Rkick": {"component": "r_kick","settings": {"R": 1e5}},
        }
        connections = {
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


def run(n_stages: int = 9, variant: str = "osdi", differentiable: bool = False,
        backend: str = "klu_split", profile_dir: str | None = None) -> dict:
    """Build, DC-init, JIT-warm, time one 1 µs transient.

    When ``profile_dir`` is supplied, ``jax.profiler.start_trace`` wraps
    the timed transient run and writes a perfetto trace to that path.
    Open with ``perfetto.dev`` or ``tensorboard --logdir=<profile_dir>``.
    """
    from circulax.solvers import analyze_circuit, setup_transient
    from circulax.solvers.transient import TrapFactorizedTransientSolver

    groups, sys_size, port_map = _build_ring(n_stages, variant, differentiable=differentiable)
    solver = analyze_circuit(groups, sys_size, backend=backend)

    t_dc = time.perf_counter()
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, 1e-2)
    if variant == "va":
        # VA PSP103 noise states (v_NOI, i_NOII) stagnate Newton from y=0
        # (delta_max = VDD/g_leak = 120 every step; damping collapses update
        # to zero).  Initialise ring nodes at VDD/2 where the Jacobian is
        # well-conditioned, then use gmin stepping only.
        ring_nodes = {f"n{s},p1" for s in range(1, n_stages + 1)}
        y_init = jnp.zeros(sys_size)
        for key, idx in port_map.items():
            if key in ring_nodes:
                y_init = y_init.at[idx].set(0.6)
            elif key == "vdd,p1":
                y_init = y_init.at[idx].set(1.2)
        y0 = solver.solve_dc_gmin(groups, y_init, g_start=1e-2, n_steps=30)
    else:
        y_src = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
        y0 = solver.solve_dc_gmin(groups, y_src, g_start=1e-2, n_steps=30)
    dc_s = time.perf_counter() - t_dc
    if not bool(jnp.all(jnp.isfinite(y0))):
        return {"status": "dc_diverged", "n_stages": n_stages, "variant": variant}
    # Magnitude sanity: jax.lax.scan-based homotopy can return finite-but-
    # absurd values when an inner Newton solve fails to converge (the
    # FixedPointIteration result is silently discarded by ``solve_dc_gmin``).
    # This circuit is CMOS at VDD=1.2 V — anything above ~10 V is a
    # divergence in disguise. TODO: have ``solve_dc_gmin`` propagate the
    # converged flag instead of relying on the caller for this check.
    y_max = float(jnp.max(jnp.abs(y0)))
    if y_max > 10.0:
        return {"status": "dc_diverged", "n_stages": n_stages, "variant": variant,
                "notes": f"|y0|_max={y_max:.2e}"}

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

    if profile_dir is not None:
        Path(profile_dir).mkdir(parents=True, exist_ok=True)
        jax.profiler.start_trace(profile_dir)

    t0 = time.perf_counter()
    sol = run_fn(
        t0=0.0, t1=T1, dt0=DT, y0=y0, saveat=saveat,
        max_steps=max_steps, stepsize_controller=controller,
    )
    sol.ys.block_until_ready()
    wall = time.perf_counter() - t0

    if profile_dir is not None:
        jax.profiler.stop_trace()

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
        "status": "ok", "n_stages": n_stages, "variant": variant, "backend": backend,
        "wall_s": wall, "compile_s": compile_s, "dc_s": dc_s,
        "n_steps": N_STEPS, "us_per_step": wall / N_STEPS * 1e6,
        "sys_size": sys_size, "freq_MHz": freq / 1e6 if np.isfinite(freq) else None,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-stages", type=int, default=9)
    p.add_argument("--variant", choices=("osdi", "va", "jax-native"), default="osdi")
    p.add_argument("--backend", default="klu_split",
                   help="Linear solver backend (klu_split, klu_split_refactor, klu, dense)")
    p.add_argument("--differentiable", action="store_true",
                   help="Keep float params as JAX leaves for jax.grad support")
    p.add_argument("--profile", default=None,
                   help="Directory to dump a jax.profiler trace into")
    args = p.parse_args()
    r = run(args.n_stages, args.variant, differentiable=args.differentiable,
            backend=args.backend, profile_dir=args.profile)
    print(r)
