"""Sequential delay separation measurements; run as a module from the repo root."""

# ruff: noqa: T201
import json
import time

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

from circulax import Circuit, compile_circuit
from circulax.components.electronic import Resistor, VoltageSourceAC
from circulax.fitting import ModelCoefficients, ModelFitOptions, circuit_from_coefficients, fit_model


def main() -> None:
    """Report order, errors, realization size and separately warmed solver costs."""
    f = np.linspace(0, 10, 151)
    source = ModelCoefficients(
        np.array([-5.0]), np.array([[[0.0], [1.0]], [[1.0], [0.0]]]), np.array([[0.0, 0.2], [0.2, 0.0]]), port_delays=[0.03, 0.03]
    )
    data = source.evaluate(f)
    for mode in ("none", "supplied"):
        options = ModelFitOptions(
            delay_mode=mode, port_delays=(0.03, 0.03) if mode == "supplied" else None, normalized_rmse=1e-4, max_absolute_error=1e-3
        )
        coefficients = fit_model(data, f, options=options)
        start = time.perf_counter()
        model = circuit_from_coefficients(coefficients)
        conversion = time.perf_counter() - start
        net = {
            "instances": {
                "dut": {"component": "dut"},
                "v": {"component": "v", "settings": {"V": 1.0, "freq": 1.0}},
                "rs": {"component": "r", "settings": {"R": 50.0}},
                "rl": {"component": "r", "settings": {"R": 50.0}},
                "g": {"component": "ground"},
            },
            "connections": {"g,p1": ("v,p2", "rl,p2"), "v,p1": "rs,p1", "rs,p2": "dut,p1", "dut,p2": "rl,p1"},
        }
        start = time.perf_counter()
        circuit = compile_circuit(net, {"dut": model, "v": VoltageSourceAC, "r": Resistor, "ground": lambda: 0}, g_leak=0)
        compilation = time.perf_counter() - start
        coefficients.evaluate(f)
        start = time.perf_counter()
        for _ in range(100):
            coefficients.evaluate(f)
        evaluation = (time.perf_counter() - start) / 100
        ts = jnp.linspace(0, 1, 101)

        def transient(circuit: Circuit = circuit, ts: jax.Array = ts) -> diffrax.Solution:
            return circuit.transient(t0=0, t1=1, dt0=0.005, saveat=ts, max_steps=1000, throw=True)

        transient = jax.jit(transient)
        start = time.perf_counter()
        jax.block_until_ready(transient().ys)
        cold = time.perf_counter() - start
        start = time.perf_counter()
        solution = transient()
        jax.block_until_ready(solution.ys)
        warm = time.perf_counter() - start
        width = circuit.sys_size * (2 if circuit.solver.is_complex else 1)
        print(
            json.dumps(
                {
                    "mode": mode,
                    "poles": len(coefficients.poles),
                    "core_states": 2 * len(coefficients.poles),
                    "circuit_unknowns": circuit.sys_size,
                    "real_solver_unknowns": width,
                    "line_algebraic_unknowns": coefficients.metadata.get("line_algebraic_unknowns", 0),
                    "history_state_bytes_per_sample": 8 * width if mode == "supplied" else 0,
                    "history_buffer_bytes_at_max_steps_1000": 1001 * 8 * (width + 1) if mode == "supplied" else 0,
                    "training_nrmse": coefficients.metadata["training_nrmse"],
                    "sampled_core_peak_singular_value": float(
                        np.max(np.linalg.svd(coefficients.evaluate_core(f), compute_uv=False))
                    ),
                    "fit_seconds": coefficients.metadata["fitting_seconds"],
                    "enforcement_seconds": coefficients.metadata["enforcement_seconds"],
                    "conversion_seconds": conversion,
                    "parent_compilation_seconds": compilation,
                    "warm_evaluate_seconds": evaluation,
                    "cold_transient_seconds": cold,
                    "warm_transient_seconds": warm,
                    "transient_steps": int(solution.stats["num_steps"]),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
