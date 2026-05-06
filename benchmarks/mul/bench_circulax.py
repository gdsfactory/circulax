"""Circulax version of the VACASK mul benchmark.

4-stage diode voltage multiplier driven by 50 V / 100 kHz sine.  Same
circuit as /home/cdaunt/code/vacask/VACASK/benchmark/mul/*/runme.sim.

Diode: D1N4007 (IS=76.9p, N=1.45, RS=42mΩ, CJO=26.5p, M=0.333).  We
build a local @component for it that includes the junction capacitance,
which dominates the transient — plain Shockley underestimates the AC
charge the multiplier has to push around.

Transient 0–5 ms at 10 ns fixed step (500 000 steps), matching upstream.
"""

from __future__ import annotations

import time

import diffrax
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


T1 = 5e-3       # 5 ms
DT = 1e-8       # 10 ns fixed step
N_STEPS = int(T1 / DT)
N_SAVE = 2000


def _make_d1n4007():
    """Build a D1N4007 @component (Shockley + junction cap, no RS).

    RS=42 mΩ in the upstream model is negligible at these currents; skipping
    it keeps the system size identical to a plain diode.  Junction cap is
    the load-bearing part for a multiplier transient.
    """
    from circulax.components.base_component import component

    @component(ports=("p1", "p2"))
    def D1N4007(signals, s,
                Is: float = 76.9e-12,
                N: float = 1.45,
                Vt: float = 25.85e-3,
                Cjo: float = 26.5e-12,
                M: float = 0.333,
                Vj: float = 0.7,
                Fc: float = 0.5):
        vd = signals.p1 - signals.p2
        vd_safe = jnp.clip(vd, -5.0, 5.0)
        i = Is * (jnp.exp(vd_safe / (N * Vt)) - 1.0)

        # Junction charge — SPICE-style, smooth through the Fc*Vj crossover.
        # Reverse/weak-forward:  Q = Cjo * Vj * (1 - (1 - V/Vj)^(1-M)) / (1-M)
        # Strong forward:        linearised past Fc*Vj to stay well-behaved.
        one_m_M = 1.0 - M
        v_low = jnp.minimum(vd, Fc * Vj)
        q_low = Cjo * Vj * (1.0 - (1.0 - v_low / Vj) ** one_m_M) / one_m_M
        # Linear extrapolation beyond Fc*Vj using value + slope at the joint.
        c_joint = Cjo * (1.0 - Fc) ** (-M)
        q_joint = Cjo * Vj * (1.0 - (1.0 - Fc) ** one_m_M) / one_m_M
        q_high = q_joint + c_joint * (vd - Fc * Vj)
        q = jnp.where(vd < Fc * Vj, q_low, q_high)

        return {"p1": i, "p2": -i}, {"p1": q, "p2": -q}

    return D1N4007


def build_netlist():
    C = 100e-9
    return {
        "instances": {
            "GND": {"component": "ground"},
            "Vs": {"component": "sine",
                   "settings": {"V": 50.0, "freq": 100e3}},
            "R1": {"component": "res", "settings": {"R": 0.01}},
            "C1": {"component": "cap", "settings": {"C": C}},
            "C2": {"component": "cap", "settings": {"C": C}},
            "C3": {"component": "cap", "settings": {"C": C}},
            "C4": {"component": "cap", "settings": {"C": C}},
            "D1": {"component": "diode", "settings": {}},
            "D2": {"component": "diode", "settings": {}},
            "D3": {"component": "diode", "settings": {}},
            "D4": {"component": "diode", "settings": {}},
        },
        # Node names match upstream: a (source), 1, 2, 10, 20.
        "connections": {
            "Vs,p1": "a,p1",   "Vs,p2": "GND,p1",
            "R1,p1": "a,p1",   "R1,p2": "n1,p1",
            "C1,p1": "n1,p1",  "C1,p2": "n2,p1",
            "D1,p1": "GND,p1", "D1,p2": "n1,p1",   # d1 0 1
            "C2,p1": "GND,p1", "C2,p2": "n10,p1",
            "D2,p1": "n1,p1",  "D2,p2": "n10,p1",  # d2 1 10
            "C3,p1": "n1,p1",  "C3,p2": "n2,p1",
            "D3,p1": "n10,p1", "D3,p2": "n2,p1",   # d3 10 2
            "C4,p1": "n10,p1", "C4,p2": "n20,p1",
            "D4,p1": "n2,p1",  "D4,p2": "n20,p1",  # d4 2 20
        },
        "ports": {"out": "n20,p1"},
    }


def run() -> dict:
    from circulax import compile_netlist
    from circulax.components.electronic import (
        Capacitor, Resistor, VoltageSourceAC,
    )
    from circulax.solvers import analyze_circuit, setup_transient
    from circulax.solvers.transient import TrapFactorizedTransientSolver

    models = {
        "res": Resistor, "cap": Capacitor,
        "sine": VoltageSourceAC, "diode": _make_d1n4007(),
    }

    groups, sys_size, port_map = compile_netlist(build_netlist(), models)
    solver = analyze_circuit(groups, sys_size, backend="klu_split")
    y0 = jnp.zeros(sys_size)
    run_fn = setup_transient(
        groups, solver, transient_solver=TrapFactorizedTransientSolver,
    )
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, T1, N_SAVE))
    controller = diffrax.ConstantStepSize()

    t_compile = time.perf_counter()
    _ = run_fn(
        t0=0.0, t1=10 * DT, dt0=DT, y0=y0,
        saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, 10 * DT, 5)),
        max_steps=50, stepsize_controller=controller,
    ).ys.block_until_ready()
    compile_s = time.perf_counter() - t_compile

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
    print(f"circulax mul: wall={r['wall_s']:.3f}s  "
          f"compile={r['compile_s']:.2f}s  "
          f"µs/step={r['us_per_step']:.2f}  sys_size={r['sys_size']}")
