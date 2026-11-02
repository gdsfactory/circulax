# Copyright 2026 Circulax contributors

"""Accuracy regressions for SDIRK3 charge histories (issue #96)."""

import math

import diffrax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax.compiler import compile_netlist
from circulax.components.base_component import PhysicsReturn, Signals, component
from circulax.components.electronic import Capacitor, Resistor
from circulax.solvers.linear import analyze_circuit
from circulax.solvers.transient import (
    SDIRK3FactorizedTransientSolver,
    SDIRK3RefactoringTransientSolver,
    SDIRK3VectorizedTransientSolver,
)

_SOLVERS = [
    (SDIRK3VectorizedTransientSolver, "dense"),
    (SDIRK3FactorizedTransientSolver, "klu_split_linear"),
    (SDIRK3RefactoringTransientSolver, "klu_split_refactor"),
]


@component(ports=("p1", "p2"), holomorphic=True)
def nonlinear_capacitor(signals: Signals) -> PhysicsReturn:
    v = signals.p1 - signals.p2
    q = v + 0.2 * v**3
    return {}, {"p1": q, "p2": -q}


def _decay_circuit(capacitor: type = Capacitor) -> tuple:
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "R1": {"component": "resistor", "settings": {"R": 1.0}},
            "C1": {"component": "capacitor", "settings": {"C": 1.0} if capacitor is Capacitor else {}},
        },
        "connections": {"R1,p1": "C1,p1", "GND,p1": ("R1,p2", "C1,p2")},
    }
    return compile_netlist(net, {"ground": lambda: 0, "resistor": Resistor, "capacitor": capacitor})


def _solve(solver: diffrax.AbstractSolver, groups: dict, y0: jnp.ndarray, t1: float, h: float) -> jnp.ndarray:
    term = diffrax.ODETerm(lambda _t, y, _args: jnp.zeros_like(y))
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t1,
        dt0=h,
        y0=y0,
        args=(groups, solver.linear_solver.sys_size // (2 if solver.linear_solver.is_complex else 1)),
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=diffrax.SaveAt(t1=True),
        max_steps=1000,
    )
    assert sol.result == diffrax.RESULTS.successful
    return sol.ys[-1]


@pytest.mark.parametrize(("solver_cls", "backend"), _SOLVERS)
@pytest.mark.parametrize("is_complex", [False, True])
def test_rc_decay_third_order(solver_cls: type, backend: str, *, is_complex: bool) -> None:
    groups, size, ports = _decay_circuit()
    node = ports["C1,p1"]
    linear = analyze_circuit(groups, size, backend=backend, is_complex=is_complex, g_leak=0.0)
    solver = solver_cls(linear_solver=linear, newton_rtol=1e-13, newton_atol=1e-13, newton_max_steps=50)
    y0 = jnp.zeros(2 * size if is_complex else size).at[node].set(1.0)
    if is_complex:
        y0 = y0.at[size + node].set(0.5)
    errors = []
    for h in (0.1, 0.05, 0.025):
        y = _solve(solver, groups, y0, 1.0, h)
        expected = math.exp(-1.0) * np.asarray(y0)
        errors.append(float(jnp.max(jnp.abs(y - expected))))
    assert errors[-1] < 2e-7
    # Third order gives a factor of eight; allow finite-step asymptotic error.
    assert np.all(np.asarray(errors[:-1]) / errors[1:] > 7.5), errors


def _nonlinear_reference(h: float) -> float:
    """Independent RK charge-rate stages: K_i = -F(Y_i), H_i = Qn + h sum(a_ij K_j)."""
    gamma = 0.4358665215084589
    a21 = (1 - gamma) / 2
    b1 = gamma * (gamma - 2) / (gamma - 1)
    b2 = 1 - gamma - b1
    qn = 1.2  # Q(1), where Q(v) = v + 0.2 v**3.

    def stage(history: float) -> float:
        y = 1.0
        for _ in range(20):
            y -= (y + 0.2 * y**3 + gamma * h * y - history) / (1 + 0.6 * y**2 + gamma * h)
        return y

    k1 = -stage(qn)
    k2 = -stage(qn + h * a21 * k1)
    return stage(qn + h * (b1 * k1 + b2 * k2))


@pytest.mark.parametrize(("solver_cls", "backend"), _SOLVERS)
def test_nonlinear_charge_matches_rate_form(solver_cls: type, backend: str) -> None:
    groups, size, ports = _decay_circuit(nonlinear_capacitor)
    node = ports["C1,p1"]
    linear = analyze_circuit(groups, size, backend=backend, g_leak=0.0)
    solver = solver_cls(linear_solver=linear, newton_rtol=1e-13, newton_atol=1e-13, newton_max_steps=50)
    y0 = jnp.zeros(size).at[node].set(1.0)
    h = 0.2
    y = _solve(solver, groups, y0, h, h)
    assert float(y[node]) == pytest.approx(_nonlinear_reference(h), rel=0.0, abs=2e-11)
