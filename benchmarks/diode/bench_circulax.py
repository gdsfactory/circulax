"""Single-diode rectifier benchmark for circulax — VA vs analytical.

Cheap iteration vehicle (sub-second JIT) to isolate whether the per-step
gap to vajax/VACASK on the PSP103 ring is integration-level or
model-size-dependent. Same harness shape as benchmarks/ring/bench_circulax.py.

Variants:
  - ``analytical`` — circulax.components.electronic.Diode (Shockley, no
    internal node, no charge).  Goes through the regular @component path
    and ``_primal_and_jac_real`` in assembly.py.  Reference for "ideal"
    integration cost at small model size.
  - ``va``         — circulax.va lowering of tests/data/va/diode.va, an
    SPICE-like diode with internal node ``CI``, ohmic branch, junction
    charge, breakdown, temperature, area scaling. Wrapped in
    ``@va_component(jacobian_fn=...)`` with the custom JVP.

Circuit:
  Vsrc(sine, 1V, 1MHz) → R(1k) → diode → GND
  1 µs transient at 50 ps fixed step (20 000 points) — matches the ring
  benchmark's dt so per-step numbers are directly comparable.

Usage:
    PYTHONPATH="/home/cdaunt/code/gdsfactory/klujax/:/home/cdaunt/code/klujax_rs-static:/home/cdaunt/code/bosdi/src" \
      pixi run python benchmarks/diode/bench_circulax.py --variant va
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

T1 = 1e-6        # 1 µs
DT = 5e-11       # 50 ps
N_STEPS = int(T1 / DT)
N_SAVE = 4000


def _build_circuit(variant: str):
    """Return (groups, sys_size, port_map) for the rectifier."""
    from circulax import compile_netlist
    from circulax.components.electronic import Diode, Resistor, VoltageSourceAC

    if variant == "analytical":
        diode_cls = Diode
        diode_settings = {"Is": 1e-12, "n": 1.0, "Vt": 25.85e-3}
        diode_ports = {"p1": "A", "p2": "C"}  # for connection mapping
    elif variant == "va":
        import importlib.util
        import tempfile
        from circulax.va import compile_va, lower
        from circulax.va.emitter import emit_source
        from circulax.va.va_defaults import parse_va_defaults_expanded

        _DIODE_VA = _REPO / "tests" / "data" / "va" / "diode.va"
        dump = compile_va(str(_DIODE_VA))
        defaults = parse_va_defaults_expanded(_DIODE_VA)
        dev = lower(dump.modules[0], va_defaults=defaults, class_name="Diode")
        tmp = tempfile.mkdtemp()
        out = Path(tmp) / "diode_va_bench.py"
        out.write_text(emit_source([dev]))
        spec = importlib.util.spec_from_file_location("diode_va_bench", out)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        diode_cls = mod.Diode

        # Read defaults from the .va so we can pass concrete values. The
        # generated class also expects sim-supplied params (_temperature,
        # _mfactor, _simparam_gmin) which are Equinox fields with defaults.
        diode_settings = {
            "Is": 1e-12, "N": 1.0, "Rs": 1.0, "BV": 1e20, "IBV": 1e-10,
            "XTI": 3.0, "EG": 1.12, "Tnom": 26.85,
            "Cjo": 0.0, "Vj": 1.0, "M": 0.5, "FC": 0.5, "TT": 0.0,
            "area": 1.0,
            "_temperature": 300.0, "_mfactor": 1.0, "_simparam_gmin": 1e-12,
        }
        diode_ports = {"A": "A", "C": "C"}
    else:
        raise ValueError(f"unknown variant: {variant!r}")

    instances = {
        "Vs": {"component": "vsrc", "settings": {"V": 1.0, "freq": 1e6}},
        "R1": {"component": "res", "settings": {"R": 1e3}},
        "D1": {"component": "diode", "settings": diode_settings},
    }
    if variant == "analytical":
        connections = {
            "Vs,p1": "n_in,p1",   "Vs,p2": "GND,p1",
            "R1,p1": "n_in,p1",   "R1,p2": "n_d,p1",
            "D1,p1": "n_d,p1",    "D1,p2": "GND,p1",
        }
    else:
        connections = {
            "Vs,p1": "n_in,p1",   "Vs,p2": "GND,p1",
            "R1,p1": "n_in,p1",   "R1,p2": "n_d,p1",
            "D1,A":  "n_d,p1",    "D1,C":  "GND,p1",
        }
    netlist = {
        "instances": instances,
        "connections": connections,
        "ports": {"out": "n_d,p1"},
    }
    models = {"vsrc": VoltageSourceAC, "res": Resistor, "diode": diode_cls}
    return compile_netlist(netlist, models)


def run(variant: str = "analytical") -> dict:
    """Build, DC-init, JIT-warm, time one 1 µs transient."""
    from circulax.solvers import analyze_circuit, setup_transient
    from circulax.solvers.transient import TrapFactorizedTransientSolver

    groups, sys_size, port_map = _build_circuit(variant)
    solver = analyze_circuit(groups, sys_size, backend="klu_split")

    t_dc = time.perf_counter()
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, 1e-2)
    y_src = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=10)
    y0 = solver.solve_dc_gmin(groups, y_src, g_start=1e-2, n_steps=20)
    dc_s = time.perf_counter() - t_dc
    if not bool(jnp.all(jnp.isfinite(y0))):
        return {"status": "dc_diverged", "variant": variant}

    run_fn = setup_transient(
        groups, solver, transient_solver=TrapFactorizedTransientSolver,
    )
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, T1, N_SAVE))
    controller = diffrax.ConstantStepSize()
    max_steps = int(2 * N_STEPS)

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
    if not np.all(np.isfinite(ys)):
        return {"status": "tran_nan", "variant": variant,
                "wall_s": wall, "compile_s": compile_s, "dc_s": dc_s,
                "n_steps": N_STEPS}

    return {
        "status": "ok", "variant": variant,
        "wall_s": wall, "compile_s": compile_s, "dc_s": dc_s,
        "n_steps": N_STEPS, "us_per_step": wall / N_STEPS * 1e6,
        "sys_size": sys_size,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=("analytical", "va"), default="analytical")
    args = p.parse_args()
    r = run(args.variant)
    print(r)
