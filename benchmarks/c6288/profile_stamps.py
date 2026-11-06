"""Isolate raw evaluation, assembly, and numeric KLU cost at the zero initial iterate.

These microbenchmarks explain structure costs; they are not a decomposition of
an entire nonlinear transient with changing biases and iteration counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.c6288.bench_circulax import prepare_netlist
from circulax.netlist_io import Library
from circulax.osdi_topology import gather
from circulax.solvers.assembly import assemble_system_real
from osdi_jax import osdi_eval_with_handle


def measure(function, arguments, repeats=3, iterations=10):
    warm = function(*arguments)
    jax.block_until_ready(warm)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            value = function(*arguments)
            jax.block_until_ready(value)
        timings.append((time.perf_counter() - start) / iterations)
    return {"seconds_per_call": timings, "median_s": float(np.median(timings)), "iterations_per_repeat": iterations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--vacask-root", type=Path, required=True)
    parser.add_argument("--representation", choices=("full", "reduced"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    baseline = json.loads(args.baseline.read_text())
    modules = []
    for module in baseline["modules"]:
        path = Path(module["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == module["sha256"]
        modules.append(path)
    with tempfile.TemporaryDirectory() as temporary:
        netlist = Path(temporary) / "input.cir"
        prepare_netlist(args.vacask_root, netlist, baseline["stop_s"])
        resolved = Library.from_file(netlist, dialect="ngspice").resolve()
    names = [f"{kind}{bit}" for kind, count in (("a", 16), ("b", 16), ("p", 32)) for bit in range(count)]
    resolved.ports = {name: name for name in names}
    circuit = resolved.compile(
        osdi_modules=tuple(modules),
        backend="klu_split",
        analysis="tran",
        state_policy="limiting_only",
        osdi_reduction=args.representation == "reduced",
    )
    y = jnp.zeros(circuit.sys_size)
    group = circuit.groups["psp103va"]
    raw = gather(group, y)
    native = jax.jit(lambda v: osdi_eval_with_handle(group.handle, v, group.states))
    assembly = eqx.filter_jit(lambda y, groups: assemble_system_real(y, groups, jnp.asarray(0.0), 2e-12, alpha=1.5))
    f, q, values = assembly(y, circuit.groups)
    jax.block_until_ready((f, q, values))
    solve = eqx.filter_jit(lambda values, rhs: circuit.solver._solve_impl(values, rhs).value)
    raw_outputs = native(raw)
    numeric_output = solve(values, -f)
    jax.block_until_ready((raw_outputs, numeric_output))
    finite = all(np.isfinite(np.asarray(value)).all() for value in (*raw_outputs, numeric_output))
    if not finite:
        raise RuntimeError("Non-finite zero-iterate kernel output; do not interpret its timings")
    result = {
        "finite_kernel_outputs": bool(finite),
        "raw_output_shapes": [list(value.shape) for value in raw_outputs],
        "representation": args.representation,
        "unknowns": circuit.sys_size,
        "unique_coo": circuit.solver.n_unique,
        "raw_voltage_shape": list(raw.shape),
        "raw_model_eval": measure(native, (raw,)),
        "full_system_assembly_including_eval": measure(assembly, (y, circuit.groups)),
        "coalesce_numeric_klu_factor_and_solve": measure(solve, (values, -f)),
        "scope": "Zero initial iterate, alpha=1.5, dt=2 ps. Native/assembly/solve measured separately after own JIT warmup; numeric solve includes COO coalescing; not a whole-transient decomposition.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
