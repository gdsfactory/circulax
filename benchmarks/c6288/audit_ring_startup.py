"""Audit ring homotopy residuals and compare raw-consistent common UIC startup."""

import json
import os
from pathlib import Path

os.environ.setdefault("CIRCULAX_RING_CACHE", str(Path(__file__).parent / "ring-jax-cache"))
import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.ring.bench_circulax import _build_ring, _dom_freq, DT, T1, N_SAVE, N_STEPS
from circulax.solvers import analyze_circuit, setup_transient
from circulax.solvers.assembly import assemble_residual_only_real
from circulax.solvers.transient import TrapFactorizedTransientSolver

ROOT = Path(__file__).parent


def main():
    result = {}
    waves = {}
    for representation in ("full", "reduced"):
        groups, size, ports = _build_ring(15, "osdi", osdi_reduction=representation == "reduced")
        solver = analyze_circuit(groups, size, backend="klu_split")
        high = eqx.tree_at(lambda s: s.g_leak, solver, 1e-2)
        source = high.solve_dc_source(groups, jnp.zeros(size), n_steps=20)
        initial = solver.solve_dc_gmin(groups, source, g_start=1e-2, n_steps=30)
        f, _ = assemble_residual_only_real(initial, groups, 0.0, 1e18)
        refined, converged = solver.solve_dc_checked(groups, initial, rtol=1e-11, atol=1e-13, max_steps=500)
        rf, _ = assemble_residual_only_real(refined, groups, 0.0, 1e18)
        record = {
            "unknowns": size,
            "initial_output_V": float(initial[ports["n1,p1"]]),
            "homotopy_max_residual_excluding_ground": float(jnp.max(jnp.abs(f[1:]))),
            "refinement_converged": bool(converged),
            "refined_output_V": float(refined[ports["n1,p1"]]),
            "refined_max_residual_excluding_ground": float(jnp.max(jnp.abs(rf[1:]))),
        }
        y0 = jnp.zeros(size)
        for stage in range(1, 16):
            y0 = y0.at[ports[f"n{stage},p1"]].set(0.6)
        y0 = y0.at[ports["vdd,p1"]].set(1.2)
        # Expand aliases in the full representation using the actual setup map;
        # leave genuine noise auxiliaries and source-current initial values zero.
        for group in groups.values():
            if not hasattr(group, "handle") or group.handle is None:
                continue
            mapping = group.handle.topology.raw_to_representative
            indices = np.asarray(group.var_indices)
            for device, row in enumerate(mapping):
                for raw, rep in enumerate(row):
                    if rep != raw:
                        y0 = y0.at[indices[device, raw]].set(0.0 if rep < 0 else y0[indices[device, rep]])
        run = setup_transient(groups, solver, transient_solver=TrapFactorizedTransientSolver)

        def execute(stop):
            return run(
                t0=jnp.asarray(0.0),
                t1=jnp.asarray(stop),
                dt0=jnp.asarray(DT),
                y0=y0,
                saveat=diffrax.SaveAt(ts=jnp.linspace(0, stop, N_SAVE)),
                max_steps=2 * N_STEPS,
                stepsize_controller=diffrax.ConstantStepSize(),
            )

        execute(2 * DT).ys.block_until_ready()
        solution = execute(T1)
        solution.ys.block_until_ready()
        values = np.asarray(solution.ys)
        times = np.asarray(solution.ts)
        volts = values[:, ports["n1,p1"]]
        record.update(
            common_uic_success=bool(solution.result == diffrax.RESULTS.successful),
            common_uic_finite=bool(np.isfinite(values).all()),
            common_uic_freq_MHz=_dom_freq(times[times > 1e-7], volts[times > 1e-7]) / 1e6,
        )
        np.savez_compressed(ROOT / f"ring-15-common-{representation}.npz", time=times, voltages=volts)
        result[representation] = record
        waves[representation] = volts
        (ROOT / "ring-startup-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    delta = waves["full"] - waves["reduced"]
    result["common_uic_comparison"] = {"max_abs_V": float(abs(delta).max()), "rms_V": float(np.sqrt(np.mean(delta**2)))}
    (ROOT / "ring-startup-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
