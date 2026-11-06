"""Batched OSDI stress test using the original VACASK c6288 cards and topology."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import math
import platform
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any


def abi_version(path: Path) -> str:
    """Inspect exported version symbols before calling BOSDI."""
    library = ctypes.CDLL(str(path.resolve()))
    return ".".join(str(ctypes.c_uint32.in_dll(library, f"OSDI_VERSION_{part}").value) for part in ("MAJOR", "MINOR"))


def prepare_netlist(root: Path, target: Path, stop: float) -> list[Path]:
    """Keep the upstream SPICE cards, topology, and multiplier test vector.

    VACASK's driver omits width/period/fall. Make them explicit for the native
    pulse source; the falling edge remains beyond the simulated interval.
    """
    source = root / "benchmark/c6288/ngspice"
    files = [source / name for name in ("runme.sim", "models.inc", "multiplier.inc")]
    text = files[0].read_text().split("\n.control\n", 1)[0].split("\n", 1)[1]
    for name in ("models.inc", "multiplier.inc"):
        text = text.replace(f'.include "{name}"', f'.include "{source / name}"')
    original = "pulse 0 1.2 0.1n 0.1n"
    if text.count(original) != 1:
        msg = "Upstream driver changed; review its pulse defaults"
        raise ValueError(msg)
    text = text.replace(original, f"PULSE(0 1.2 0.1n 0.1n 0.1n {stop:.17g} {2 * stop:.17g})")
    text = "\n".join(line for line in text.splitlines() if not line.startswith(".options"))
    target.write_text(text + "\n")
    return files


def check_product(voltages: Any, expected: int, vdd: float = 1.2) -> dict:
    """Require valid logic levels, not just a thresholded integer match."""
    import numpy as np

    values = np.asarray(voltages)
    bits = [(expected >> bit) & 1 for bit in range(32)]
    valid = np.isfinite(values) & (values >= -0.1 * vdd) & (values <= 1.1 * vdd)
    valid &= np.where(bits, values >= 0.8 * vdd, values <= 0.2 * vdd)
    observed = sum(int(value > vdd / 2) << bit for bit, value in enumerate(values))
    return {
        "expected": f"0x{expected:08x}",
        "observed": f"0x{observed:08x}",
        "valid_logic": bool(valid.all()),
        "invalid_bits": np.flatnonzero(~valid).tolist(),
    }


def run(args: argparse.Namespace, record: Callable[[dict], None]) -> dict:  # noqa: C901, PLR0915 -- benchmark phases and JIT audit
    """Prepare, warm up with identical shapes, time, and validate the transient."""
    import diffrax
    import jax
    import jax.numpy as jnp
    import numpy as np

    from circulax.netlist_io import Library
    from circulax.netlist_io.osdi import compile_va
    from circulax.solvers import setup_transient
    from circulax.solvers.transient import (
        BDF2FactorizedTransientSolver,
        BDF2RefactoringTransientSolver,
        BDF2VectorizedTransientSolver,
        FactorizedTransientSolver,
        RefactoringTransientSolver,
        TrapFactorizedTransientSolver,
        TrapRefactoringTransientSolver,
        TrapVectorizedTransientSolver,
        VectorizedTransientSolver,
    )

    jax.config.update("jax_enable_x64", True)  # noqa: FBT003 -- JAX configuration API
    if jax.default_backend() != "cpu":
        msg = "BOSDI requires CPU; set JAX_PLATFORMS=cpu"
        raise ValueError(msg)
    started = time.perf_counter()
    result = {
        "status": "preparing",
        "representation": args.representation,
        "stop_s": args.stop,
        "step_s": args.step,
        "backend": args.backend,
        "solver": args.solver,
        "jax": jax.__version__,
        "platform": platform.platform(),
        "initial_conditions": "uic_zero",
        "newton": args.newton,
        "max_newton": args.max_newton,
        "newton_rtol": 1e-6,
        "newton_atol": 1e-8,
        "timing_method": "dynamic_times_shared_save_callback",
    }
    result["bosdi"] = importlib.metadata.version("bosdi")
    import osdi_jax
    import osdi_shim_nb

    result["runtime_modules"] = [
        {"path": module.__file__, "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
        for module in (osdi_jax, osdi_shim_nb)
    ]

    result["compiler"] = (
        subprocess.check_output([args.compiler, "--version"], text=True).strip()  # noqa: S603 -- explicit compiler
        if not (args.psp_osdi and args.resistor_osdi)
        else "precompiled"
    )
    record(result)
    modules = [
        args.psp_osdi or compile_va(args.vacask_root / "devices/psp103v4/psp103.va", compiler=args.compiler),
        args.resistor_osdi or compile_va(args.vacask_root / "devices/spice/resistor.va", compiler=args.compiler),
    ]
    result["modules"] = [
        {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "abi": abi_version(p)} for p in modules
    ]
    record(result)
    if any(m["abi"] != "0.4" for m in result["modules"]):
        msg = (
            "Circulax Library currently requires OSDI 0.4. Use a 0.4 OpenVAF compiler or compatible "
            "--psp-osdi/--resistor-osdi; BOSDI experimental 0.5 support alone is insufficient."
        )
        raise ValueError(msg)
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary) / "c6288.cir"
        files = prepare_netlist(args.vacask_root, target, args.stop)
        result["sources"] = [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
        resolved = Library.from_file(target, dialect="ngspice").resolve()
    result["leaf_counts"] = dict(Counter(inst.module for inst in resolved.instances))
    result["status"] = "compiling_topology"
    record(result)
    # Public aliases select only 64 signals for saved output (not every internal node).
    names = [f"{kind}{bit}" for kind, count in (("a", 16), ("b", 16), ("p", 32)) for bit in range(count)]
    resolved.ports = {name: name for name in names}
    topology_start = time.perf_counter()
    circuit = resolved.compile(
        osdi_modules=tuple(modules),
        backend=args.backend,
        analysis="tran",
        state_policy="limiting_only",
        osdi_reduction=args.representation == "reduced",
    )
    result["topology_s"] = time.perf_counter() - topology_start
    topology_summary = {}
    for name, group in circuit.groups.items():
        handle = getattr(group, "handle", None)
        metadata = getattr(handle, "topology", None)
        if metadata is None:
            continue
        mapping = metadata.raw_to_representative
        patterns, counts = np.unique(mapping, axis=0, return_counts=True)
        descriptor = circuit.source_models[name]
        topology_summary[name] = {
            "temperature_K": descriptor.model.temperature,
            "analysis": descriptor.model.analysis,
            "simparams": dict(descriptor.model.simparams),
            "params_sha256": hashlib.sha256(np.asarray(group.params).tobytes()).hexdigest(),
            "raw_width": group.num_nodes,
            "num_terminals": group.num_pins,
            "instances": len(mapping),
            "surviving_internal_slots": int(np.sum(mapping[:, group.num_pins :] == np.arange(group.num_pins, group.num_nodes))),
            "patterns": [
                {"representatives": pattern.tolist(), "instances": int(count)}
                for pattern, count in zip(patterns, counts, strict=True)
            ],
            "global_coo_entries": int(group.jac_rows.size),
        }
    result["configured_topology"] = topology_summary
    result["linear_structure"] = {
        "unique_coo_entries": int(circuit.solver.n_unique),
        "device_coo_entries": sum(int(group.jac_rows.size) for group in circuit.groups.values()),
    }
    indices = jnp.asarray([circuit.port_map[name] for name in names])
    result.update(
        sys_size=circuit.sys_size,
        setup_s=time.perf_counter() - started,
        batches={name: len(group.var_indices) for name, group in circuit.groups.items()},
        status="prepared",
    )
    record(result)
    if args.prepare_only:
        return result
    solvers = {"trap": TrapFactorizedTransientSolver, "bdf2": BDF2FactorizedTransientSolver, "euler": FactorizedTransientSolver}
    if args.newton == "full":
        solvers = {"bdf2": BDF2VectorizedTransientSolver, "trap": TrapVectorizedTransientSolver, "euler": VectorizedTransientSolver}
    elif args.newton == "full-refactor":
        # Full Newton with one fresh LU per step and a fused KLU refactor per iteration.
        solvers = {
            "bdf2": BDF2RefactoringTransientSolver,
            "trap": TrapRefactoringTransientSolver,
            "euler": RefactoringTransientSolver,
        }
    integrator = solvers[args.solver](
        linear_solver=circuit.solver, newton_rtol=1e-6, newton_atol=1e-8, newton_max_steps=args.max_newton
    )
    simulate = setup_transient(circuit.groups, circuit.solver, transient_solver=integrator)
    n_steps = math.ceil(args.stop / args.step)
    count = min(n_steps + 1, 1001)
    y0 = jnp.zeros(circuit.sys_size)

    def save_signals(_t: Any, y: Any, _args: Any) -> Any:
        return y[indices]

    def execute(stop: float) -> diffrax.Solution:
        return simulate(
            t0=jnp.asarray(0.0, dtype=jnp.float64),
            t1=jnp.asarray(stop, dtype=jnp.float64),
            dt0=jnp.asarray(args.step, dtype=jnp.float64),
            y0=y0,
            saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, stop, count), fn=save_signals),
            max_steps=n_steps + 10,
            stepsize_controller=diffrax.ConstantStepSize(),
            throw=False,
        )

    result["status"] = "warming_up"
    record(result)
    start = time.perf_counter()
    warm = execute(min(2 * args.step, args.stop))
    warm.ys.block_until_ready()
    result["compile_s"] = time.perf_counter() - start
    if warm.result != diffrax.RESULTS.successful:
        result.update(status="warmup_failed", solver_result=str(warm.result), warmup_steps=int(warm.stats["num_steps"]))
        record(result)
        return result
    if args.audit_only:
        durations = []
        for _ in range(2):
            start = time.perf_counter()
            audit = execute(min(4 * args.step, args.stop))
            audit.ys.block_until_ready()
            if audit.result != diffrax.RESULTS.successful:
                result.update(status="audit_failed", solver_result=str(audit.result))
                record(result)
                return result
            durations.append(time.perf_counter() - start)
        result.update(status="jit_audited", audit_call_s=durations)
        record(result)
        return result
    result["status"] = "running"
    record(result)
    start = time.perf_counter()
    sol = execute(args.stop)
    sol.ys.block_until_ready()
    wall = time.perf_counter() - start
    values = np.asarray(sol.ys)
    completed = sol.result == diffrax.RESULTS.successful and bool(np.isfinite(values).all())
    result.update(
        wall_s=wall,
        n_steps=int(sol.stats["num_steps"]),
        solver_result=str(sol.result),
        us_per_step=wall / max(int(sol.stats["num_steps"]), 1) * 1e6,
        finite=bool(np.isfinite(values).all()),
        status="transient_failed",
    )
    if completed:
        result["initial_product"] = check_product(values[0, 32:], 0)
        result["final_product"] = check_product(values[-1, 32:], 0xFFFE0001)
        result["inputs_high"] = bool(np.all((values[-1, :32] >= 0.96) & (values[-1, :32] <= 1.32)))
        result["status"] = "ok" if result["final_product"]["valid_logic"] and result["inputs_high"] else "logic_mismatch"
    np.savez_compressed(args.output.with_suffix(".npz"), time=np.asarray(sol.ts), voltages=values, names=names)
    result["waveforms"] = args.output.with_suffix(".npz").name
    record(result)
    return result
