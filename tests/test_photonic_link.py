# Copyright (c) 2026 Circulax contributors
"""End-to-end loaded EO/OE link against coherent independent formulas."""

import diffrax
import jax.numpy as jnp
import numpy as np

from circulax.solvers.transient import BDF2FactorizedTransientSolver
from examples.photonic.library_link import build_link


def test_loaded_link_finite_drive_charging_transient() -> None:
    circuit = build_link(voltage=0.8)
    times = jnp.linspace(0, 6e-10, 121)
    errors = []
    cold = circuit.dc(params={"DRIVE.dc": 0.0})
    for step in (4e-12, 2e-12):
        solution = circuit.transient(
            t0=0.0,
            t1=6e-10,
            dt0=step,
            saveat=diffrax.SaveAt(ts=times),
            y0=cold,
            transient_solver=BDF2FactorizedTransientSolver,
            stepsize_controller=diffrax.ConstantStepSize(),
            max_steps=500,
        )
        output = np.asarray(circuit.port(solution.ys, "output")).real
        # DC optical boundary is algebraic; ignore the explicitly supplied all-zero t=0 sample.
        junction = 0.8 * (1 - np.exp(-np.asarray(times) / 1e-10))
        reference = 50 * 0.8 * 0.002 * (1 + np.sin(0.7 * junction)) / 2
        errors.append(np.max(abs(output[1:] - reference[1:])))
    assert errors[1] < 4e-4
    assert errors[1] < 0.7 * errors[0]
