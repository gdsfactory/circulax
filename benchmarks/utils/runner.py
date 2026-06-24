"""Common benchmark runner.

Usage in a testbench
--------------------
::

    SOLVERS: dict[str, SolverFn] = {
        "ngspice": solver_ngspice,
        "circulax": solver_circulax,
    }

    results = run_benchmark(
        solvers=SOLVERS,
        reference="ngspice",
        nodes=["v(out)", "v(in)"],
        title="My circuit testbench",
    )

Adding a new solver is a one-liner in the ``SOLVERS`` dict.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .metrics import compare_waveforms


@dataclass
class SolverResult:
    """All outputs produced by one solver for a single benchmark run."""

    name: str
    time: np.ndarray  # shape (N,)  time points (s)
    signals: dict[str, np.ndarray]  # node_label → voltage array (N,)
    elapsed: float  # timed-run wall time (s)
    n_steps: int  # number of integration steps
    compile_time: float = 0.0  # netlist compile / setup time (s)
    warmup_time: float | None = None  # JIT/tracing warmup time (s), if any
    metadata: dict = field(default_factory=dict)  # solver-specific extras


# A solver callable takes no arguments and returns a SolverResult.
# Bind circuit-specific parameters (n_save, dt, …) into the closure before
# adding it to SOLVERS.
SolverFn = Callable[[], SolverResult]


def run_benchmark(
    solvers: dict[str, SolverFn],
    reference: str,
    nodes: list[str],
    *,
    title: str = "Benchmark",
) -> dict[str, SolverResult]:
    """Run all solvers, print a timing + accuracy report, return results.

    Parameters
    ----------
    solvers:
        Ordered ``{name: callable}`` mapping.  Each callable is called once
        and must return a :class:`SolverResult`.  Add new solvers here.
    reference:
        Key of the reference solver.  All other solvers are compared against
        its waveforms.
    nodes:
        Node labels to compare.  Every ``SolverResult.signals`` dict must
        contain each label as a key.
    title:
        Header text printed above the report.

    """
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)

    results: dict[str, SolverResult] = {}
    for name, fn in solvers.items():
        print(f"\n[{name}] running...")
        results[name] = fn()

    ref = results[reference]

    # ── Timing ───────────────────────────────────────────────────────────────
    print("\n── Timing ────────────────────────────────────────────────────────")
    col_w = max(len(n) for n in results) + 2
    for name, r in results.items():
        per_us = r.elapsed / max(r.n_steps, 1) * 1e6
        parts: list[str] = [f"  {name:<{col_w}}"]
        if r.compile_time:
            parts.append(f"compile={r.compile_time:.3f}s")
        if r.warmup_time is not None:
            parts.append(f"warmup={r.warmup_time:.3f}s")
        parts.append(f"timed={r.elapsed:.3f}s ({per_us:.3f} µs/step)")
        if name != reference:
            ratio = r.elapsed / ref.elapsed
            tag = "faster" if ratio < 1 else "slower"
            parts.append(f"[{ratio:.2f}× {tag} than {reference}]")
        print("  ".join(parts))

    # ── Accuracy ─────────────────────────────────────────────────────────────
    print(f"\n── Accuracy (reference: {reference}) ────────────────────────────")
    for name, r in results.items():
        if name == reference:
            continue
        print(f"  [{name}]")
        for node in nodes:
            cmp = compare_waveforms(
                ref.time,
                ref.signals[node],
                r.time,
                r.signals[node],
                node=node,
            )
            cmp.print()

    return results
