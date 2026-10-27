"""Issue #94: synchronized original/current lookup and native MZM benchmarks.

Run from the repository root, with the same affinity/thread limits per variant:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0-3 .venv/bin/python \
      benchmarks/delay_history/run.py --variant current --output /tmp/delay-current
Use --variant original to load the solver and interpolation from --reference.
The MZM mode requires the motivating tutorial revision and companion helper
identified by SHA256 in results.json (these were uncommitted local inputs and
are deliberately not bundled with this solver patch). It executes that notebook
in memory, never rewriting it. Lookup, oracle and cache modes run standalone.
The default reference is the published pre-fix development commit; its solver
sources match the f73ecab snapshot used for the recorded measurements.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples/photonic"))

import jax
import jax.numpy as jnp
import numpy as np

from circulax.solvers import assembly, transient
from circulax.solvers.circuit_diffeq import circuit_diffeqsolve

jax.config.update("jax_enable_x64", True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--variant", choices=["original", "current"], required=True)
parser.add_argument("--mode", choices=["lookup", "mzm", "oracle", "cache", "adaptive"], default="lookup")
parser.add_argument("--reference", default="54620a09e4873387d567f74ec7b75870b9dd577a")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--compare", type=Path, help="Original MZM output directory; assert complete trace agreement.")
opts = parser.parse_args()
if opts.mode == "adaptive" and opts.variant != "current":
    parser.error("--mode adaptive compares inferred and explicit capacities on the current solver.")
if opts.mode == "oracle" and opts.variant != "original":
    parser.error("--mode oracle requires --variant original to load the numerical reference.")
if opts.mode == "mzm" and not (ROOT / "examples/photonic/traveling_wave_mzm_tutorial.py").is_file():
    parser.error("MZM mode requires the uncommitted tutorial/helper revision identified in results.json.")
opts.output.mkdir(parents=True, exist_ok=True)


def original_module(name, filename, *, ordinary_history=False, expose_history=False):
    source = subprocess.check_output(["git", "show", f"{opts.reference}:{filename}"], cwd=ROOT, text=True)
    if ordinary_history:
        source = source.replace(" + [state.hist_t, state.hist_y]", "")
    if expose_history:
        source = source.replace("controller_state=None,", "controller_state=(final_state.hist_t, final_state.num_accepted_steps),")
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, filename, "exec"), module.__dict__)
    return module


if opts.variant == "original":
    original_assembly = original_module("original_assembly", "circulax/solvers/assembly.py")
    if opts.mode != "oracle":
        assembly._interp_delayed = original_assembly._interp_delayed
    original_solve = original_module(
        "original_diffeq", "circulax/solvers/circuit_diffeq.py", expose_history=opts.mode == "oracle"
    ).circuit_diffeqsolve
    ordinary_solve = original_module(
        "original_ordinary_diffeq", "circulax/solvers/circuit_diffeq.py", ordinary_history=True
    ).circuit_diffeqsolve

    def dispatch(**kwargs):
        kwargs.pop("history_capacity", None)
        kwargs.pop("max_delay", None)
        return original_solve(**kwargs)
else:
    dispatch = circuit_diffeqsolve

results = {
    "variant": opts.variant,
    "jax": jax.__version__,
    "backend": jax.default_backend(),
    "affinity": sorted(os.sched_getaffinity(0)),
    "threads": os.environ.get("OPENBLAS_NUM_THREADS"),
    "reference": opts.reference,
}

if opts.mode == "adaptive":
    spec = importlib.util.spec_from_file_location("delay_regressions", ROOT / "tests/test_delay_history.py")
    regression = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = regression
    spec.loader.exec_module(regression)
    parameters = jnp.array([0.23, 0.07, 0.9, 0.12, -0.2])
    results["adaptive_capacity"] = []
    reference = None
    for label, capacity in [("inferred", None), ("previous_fallback", 1024)]:
        started = time.perf_counter()
        solution = regression.sized_feedback(parameters, capacity=capacity)
        solution.ys.block_until_ready()
        cold_seconds = time.perf_counter() - started
        if reference is None:
            reference = np.asarray(solution.ys)
        actual = np.asarray(solution.ys)
        np.testing.assert_allclose(actual, reference, rtol=2e-13, atol=2e-13)
        started = time.perf_counter()
        for _ in range(20):
            regression.sized_feedback(parameters, capacity=capacity).ys.block_until_ready()
        record = {
            "case": label,
            "cold_seconds": cold_seconds,
            "warm_seconds": (time.perf_counter() - started) / 20,
            "capacity": int(solution.stats["history_capacity"]),
            "history_bytes": int(solution.stats["history_bytes"]),
            "accepted": int(solution.stats["num_accepted_steps"]),
            "rejected": int(solution.stats["num_rejected_steps"]),
            "trace_max_abs_error": float(np.max(np.abs(actual - reference))),
        }
        results["adaptive_capacity"].append(record)
        print(record, flush=True)
elif opts.mode == "cache":
    import diffrax

    from circulax import compile_circuit
    from circulax.components.electronic import Resistor, WaveformVoltageSource

    circuit = compile_circuit(
        {
            "instances": {"GND": {"component": "ground"}, "V": {"component": "V"}, "R": {"component": "R"}},
            "connections": {"GND,p1": ("V,p2", "R,p2"), "V,p1": "R,p1"},
        },
        {"ground": lambda: 0, "V": WaveformVoltageSource, "R": Resistor},
        backend="dense",
    )
    setup = transient.setup_transient
    if opts.variant == "original":
        original_transient = original_module("original_transient", "circulax/solvers/transient.py")
        original_transient.circuit_diffeqsolve = original_solve
        setup = original_transient.setup_transient
    results["cache_calls_seconds"] = []
    for index in range(2):
        print("DEFAULT TERM CALL", index, flush=True)
        run = setup(circuit.groups, circuit.solver)
        started = time.perf_counter()
        run(
            t0=0.0,
            t1=1.0,
            dt0=0.1,
            y0=jnp.zeros(circuit.solver.sys_size),
            saveat=diffrax.SaveAt(ts=jnp.linspace(0, 1, 11)),
            max_steps=12,
        ).ys.block_until_ready()
        results["cache_calls_seconds"].append(time.perf_counter() - started)
elif opts.mode == "oracle":
    import diffrax

    spec = importlib.util.spec_from_file_location("delay_regressions", ROOT / "tests/test_delay_history.py")
    regression = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = regression
    spec.loader.exec_module(regression)
    parameters = jnp.array([0.23, 0.07, 0.9, 0.12, -0.2])
    results["oracle"] = []
    for adaptive in [False, True]:

        def old_trace(p, solve=original_solve, solution=False):
            sol = solve(
                regression.TERM,
                regression.FeedbackSolver(oracle=True),
                0.0,
                4.0,
                0.8 if adaptive else 0.1,
                p[3:],
                args=(p[:2], p[2]),
                saveat=diffrax.SaveAt(ts=jnp.concatenate([jnp.array([0.0]), jnp.linspace(0.013, 4 - 0.013, 79), jnp.array([4.0])])),
                record_history=True,
                max_steps=512,
                checkpoints=2,
                stepsize_controller=diffrax.PIDController(rtol=0.03, atol=0.01) if adaptive else diffrax.ConstantStepSize(),
            )
            return sol if solution else sol.ys

        def new_trace(p):
            return regression.solve(p, 12 if adaptive else 6, adaptive=adaptive, checkpoints=2).ys

        old, new = old_trace(parameters), new_trace(parameters)
        old_grad = jax.jacrev(old_trace)(parameters)
        ordinary_grad = jax.jacrev(lambda p: old_trace(p, ordinary_solve))(parameters)
        # Diffrax differentiates the discrete solve with its accepted mesh
        # fixed, not the PID controller's step-selection decisions. Obtain
        # that mesh from a diagnostic-only full-history Solution field.
        mesh, count = old_trace(parameters, solution=True).controller_state
        mesh = mesh[: int(count) + 1]

        class FixedMesh(diffrax.StepTo):
            def init(self, terms, t0, t1, y0, dt0, args, func, error_order):
                return super().init(terms, t0, t1, y0, None, args, func, error_order)

        def frozen_trace(p):
            return circuit_diffeqsolve(
                regression.TERM,
                regression.FeedbackSolver(),
                0.0,
                4.0,
                0.1,
                p[3:],
                args=(p[:2], p[2]),
                saveat=diffrax.SaveAt(ts=jnp.concatenate([jnp.array([0.0]), jnp.linspace(0.013, 4 - 0.013, 79), jnp.array([4.0])])),
                record_history=True,
                max_delay=jnp.max(p[:2]),
                history_capacity=12 if adaptive else 6,
                max_steps=512,
                checkpoints=2,
                stepsize_controller=FixedMesh(ts=mesh),
            ).ys

        epsilon = 1e-5
        finite_difference = jnp.stack(
            [
                (frozen_trace(parameters.at[k].add(epsilon)) - frozen_trace(parameters.at[k].add(-epsilon))) / (2 * epsilon)
                for k in range(parameters.size)
            ],
            axis=-1,
        )
        new_grad = jax.jacrev(new_trace)(parameters)
        jax.block_until_ready((old, new, old_grad, new_grad))
        np.savez(
            opts.output / f"oracle-{adaptive}.npz",
            old=old,
            new=new,
            old_grad=old_grad,
            new_grad=new_grad,
            ordinary_grad=ordinary_grad,
            finite_difference=finite_difference,
        )
        record = {
            "adaptive": adaptive,
            "trace_max_abs": float(jnp.max(jnp.abs(old - new))),
            "original_buffer_jacobian_difference": float(jnp.max(jnp.abs(old_grad - new_grad))),
            "ordinary_history_jacobian_difference": float(jnp.max(jnp.abs(ordinary_grad - new_grad))),
            "current_vs_finite_difference": float(jnp.max(jnp.abs(finite_difference - new_grad))),
            "original_vs_finite_difference": float(jnp.max(jnp.abs(finite_difference - old_grad))),
        }
        np.testing.assert_allclose(new, old, atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(new_grad, ordinary_grad, atol=2e-11, rtol=2e-11)
        results["oracle"].append(record)
        print(record, flush=True)
elif opts.mode == "lookup":
    read = jax.jit(assembly._interp_delayed)
    rng = np.random.default_rng(6)
    results["lookup"] = []
    for n in [12, 257, 9217]:
        accepted = n // 2
        times = jnp.asarray(np.r_[np.arange(accepted) * 0.3, np.full(n - accepted, np.inf)])
        values = jnp.asarray(rng.normal(size=(n, 96)))
        args = times, values, jnp.full(12, 2.3), jnp.arange(96).reshape(12, 8), n * 0.15, jnp.ones((12, 8))
        started = time.perf_counter()
        compiled = read.lower(*args).compile()
        compile_seconds = time.perf_counter() - started
        compiled(*args).block_until_ready()
        started = time.perf_counter()
        for _ in range(100):
            compiled(*args).block_until_ready()
        record = {"slots": n, "compile_s": compile_seconds, "warm_ms": 10 * (time.perf_counter() - started)}
        results["lookup"].append(record)
        print(record, flush=True)
else:
    import diffrax
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    notebook = json.loads((ROOT / "examples/photonic/traveling_wave_mzm_reference.ipynb").read_text())
    namespace = {"__name__": "__benchmark__"}
    # Capture complete real/unrolled state traces for numerical comparisons.
    records = []

    def measured(**kwargs):
        index = len(records)
        started = time.perf_counter()
        sol = dispatch(**kwargs)
        sol.ys.block_until_ready()
        record = {
            "call": index,
            "seconds": time.perf_counter() - started,
            "steps": int(sol.stats["num_steps"]),
            "history_capacity": int(sol.stats.get("history_capacity", kwargs["max_steps"] + 1)),
            "history_bytes": int(sol.stats.get("history_bytes", (kwargs["max_steps"] + 1) * (kwargs["y0"].size + 1) * 8)),
        }
        actual = np.asarray(sol.ys)
        if opts.compare is not None:
            expected = np.load(opts.compare / f"trace-{index}.npy")
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
            record["trace_max_abs_error"] = float(np.max(np.abs(actual - expected)))
        records.append(record)
        np.save(opts.output / f"trace-{index}.npy", actual)
        print("SOLVE", record, flush=True)
        return sol

    transient.circuit_diffeqsolve = measured
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        print("CELL", index, flush=True)
        exec(compile("".join(cell["source"]), f"mzm-cell-{index}", "exec"), namespace)
        if index == 22:
            # Same native-delay circuit and history budget as the reported
            # 2048-step benchmark; a stable term isolates warm execution.
            from circulax.solvers import setup_transient

            circuit = namespace["transient_link"]
            y0 = circuit.dc()
            run = setup_transient(circuit.groups, circuit.solver)

            def zero(t, y, args):
                return jnp.zeros_like(y)

            term = diffrax.ODETerm(zero)
            for label in ["cold", "warm"]:
                print("SHORT", label, flush=True)
                run(
                    t0=0.0,
                    t1=32 * namespace["ui"],
                    dt0=namespace["integration_dt"],
                    y0=y0,
                    saveat=diffrax.SaveAt(ts=jnp.arange(1025) * namespace["sample_dt"]),
                    max_steps=9236,
                    term=term,
                    throw=True,
                ).ys.block_until_ready()
        if index in [25, 28]:
            plt.gcf().savefig(opts.output / f"figure-{index}.png", dpi=120)
        plt.close("all")
    results["solves"] = records

(opts.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
