"""Circulax version of the VACASK rc benchmark.

Same circuit as /home/cdaunt/code/vacask/VACASK/benchmark/rc/*/runme.sim:

    Vpulse(0→1V, rise=1µs, fall=1µs, width=1ms, period=2ms)
         │
        ─┴─  R = 1 kΩ
         │
        ─┴─  C = 1 µF
         │
        GND

Transient 0–1 s at 1 µs fixed step (1 000 000 steps — same as upstream).
"""

from __future__ import annotations

import time

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# Upstream settings, exactly.
T1 = 1.0       # 1 s
DT = 1e-6      # 1 µs fixed step
N_STEPS = int(T1 / DT)
N_SAVE = 2000  # coarse saveat; we just need wall time


def build_netlist():
    return {
        "instances": {
            "GND": {"component": "ground"},
            "Vs": {
                "component": "pulse",
                "settings": {
                    "v1": 0.0, "v2": 1.0,
                    "td": 1e-6, "tr": 1e-6, "tf": 1e-6,
                    "pw": 1e-3, "per": 2e-3,
                },
            },
            "R1": {"component": "res", "settings": {"R": 1e3}},
            "C1": {"component": "cap", "settings": {"C": 1e-6}},
        },
        "connections": {
            "Vs,p1": "n1,p1",  "Vs,p2": "GND,p1",
            "R1,p1": "n1,p1",  "R1,p2": "n2,p1",
            "C1,p1": "n2,p1",  "C1,p2": "GND,p1",
        },
        "ports": {"out": "n2,p1"},
    }


def run() -> dict:
    """Run the rc transient.  Returns a dict with wall_s, compile_s, us_per_step."""
    from circulax import compile_netlist
    from circulax.components.electronic import (
        Capacitor, PulseVoltageSource, Resistor,
    )
    from circulax.solvers import analyze_circuit, setup_transient
    from circulax.solvers.transient import TrapFactorizedTransientSolver

    models = {
        "res": Resistor, "cap": Capacitor, "pulse": PulseVoltageSource,
    }

    groups, sys_size, port_map = compile_netlist(build_netlist(), models)
    solver = analyze_circuit(groups, sys_size, backend="klu_split")
    y0 = jnp.zeros(sys_size)
    run_fn = setup_transient(
        groups, solver, transient_solver=TrapFactorizedTransientSolver,
    )
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, T1, N_SAVE))
    controller = diffrax.ConstantStepSize()

    # Warmup: same graph shape, tiny window.
    t_compile = time.perf_counter()
    _ = run_fn(
        t0=0.0, t1=10 * DT, dt0=DT, y0=y0,
        saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, 10 * DT, 5)),
        max_steps=50, stepsize_controller=controller,
    ).ys.block_until_ready()
    compile_s = time.perf_counter() - t_compile

    # Timed run.
    t0 = time.perf_counter()
    sol = run_fn(
        t0=0.0, t1=T1, dt0=DT, y0=y0, saveat=saveat,
        max_steps=int(2 * N_STEPS), stepsize_controller=controller,
    )
    sol.ys.block_until_ready()
    wall = time.perf_counter() - t0

    return {
        "wall_s": wall,
        "compile_s": compile_s,
        "n_steps": N_STEPS,
        "us_per_step": wall / N_STEPS * 1e6,
        "sys_size": sys_size,
    }


if __name__ == "__main__":
    r = run()
    print(f"circulax rc: wall={r['wall_s']:.3f}s  "
          f"compile={r['compile_s']:.2f}s  "
          f"µs/step={r['us_per_step']:.2f}  sys_size={r['sys_size']}")
