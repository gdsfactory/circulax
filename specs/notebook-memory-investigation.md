# Notebook Memory Investigation (Resolved)

## Summary

The traveling-wave MZM notebooks exhausted a 16 GB CI runner because every step
of a factorized KLU transient leaked one native numeric factorization. The leak
came from klujax 0.5.1/0.5.2, not from JAX compilation. klujax 0.5.3 restores a
jit-safe free, the transient solvers pass the required dependency (#108), and
circulax requires `klujax>=0.5.3,<0.6` (#109).

## Symptom

[CI job 127576787653](https://github.com/gdsfactory/circulax/actions/runs/42122089917/job/127576787653)
(PR #104, `26a9028`) ran the TFLN notebook's main transient. The kernel's RSS
grew from 1.2 GiB to 14.7 GiB within about 30 s and all swap was used; the
runner then shut down. Shortened CI records, output-only saving and heatmap
eyes reduced the symptom but did not remove it.

## Root cause

klujax 0.5.1 (gdsfactory/klujax#32) replaced its free primitive with C++ RAII
owners. `free_numeric` became a deprecated no-op. Outside `jit`, `factor`
returns a `KLUNumeric` owner whose destructor frees the factorization. Inside
`jit`, `factor` returns a raw `uint64` handle, which no owner wraps.

Every factorized and refactoring transient solver factors once per step inside
the jitted integration loop and then calls `free_numeric`. With klujax
0.5.1/0.5.2 that call did nothing, so each accepted step leaked one numeric
factorization. Circulax pinned klujax 0.5.2 on 2026-09-14.

## Evidence

Phase-resolved RSS of the TFLN notebook, CI record (6,144 steps), klujax 0.5.2:

| Phase | RSS |
|---|---|
| Trace and lower | 1,513 → 1,562 MiB |
| XLA backend compile | → 1,693 MiB |
| Execute | → 9,682 MiB, retained after return |

Compilation added 131 MiB. Execution added 8.0 GiB, about 1.3 MiB per step,
which matches one leaked factorization per step. The full record (18,304
steps) would need roughly 24 GiB.

A standalone check factored a 6,744-unknown banded matrix 1,000 times inside
`jax.lax.fori_loop`. RSS grew by 1,689 MiB without a free and by 4 MiB with
`free_numeric(numeric, dependency=x)`, with zero live handles.

## Fix

- **klujax 0.5.3** (gdsfactory/klujax#37, #38) adds a jit-safe free primitive.
  `free_numeric(raw, dependency=results)` orders the free after every consumer
  passed in `dependency`. Allocation and free are side effects, so XLA cannot
  remove or hoist them. Without `dependency`, a raw free emits a
  `FutureWarning` and skips the free; klujax 0.6.0 will raise instead.
  `refactor` now returns the refactor output for raw handles, and autodiff
  through raw-handle solves raises a clear error.
- **#108** passes the step's final Newton iterate (`sol.value`, or `Y3` for
  SDIRK3) as the dependency at all eight call sites. It also uses the fused
  `refactor_and_solve`, fixing refactors that XLA removed as dead code (#106).
- **#109** requires `klujax>=0.5.3,<0.6`.

The TFLN notebook's `TrapVectorizedTransientSolver` workaround and the
`CIRCULAX_NOTEBOOK_CI` short records are no longer needed and were removed.

## Verification

Each notebook ran start to finish in a fresh process with klujax 0.5.3, sampling
RSS every 20 ms:

| Notebook | Record | Factorizations | Live handles at end | Peak RSS |
|---|---|---|---|---|
| TFLN MZM, default factorized solver | 6,144 steps | 6,144 | 0 | 1.73 GiB |
| TFLN MZM, default factorized solver | 18,304 steps | 18,304 | 0 | 1.76 GiB |
| Traveling-wave PN MZM | short NRZ + PAM4 | 9,216 | 0 | 5.47 GiB |
| Traveling-wave PN MZM | full NRZ + PAM4 | 27,584 | 0 | 5.48 GiB |

Peak RSS no longer depends on record length. The TFLN circuit still matches the
convolution reference (RMS 2.2e-4 of input power on the full record), and the
default factorized solver is faster than the removed workaround (69 s versus
86 s for the whole notebook).

## Constraints

- Raw numeric handles are forward-only. Gradients through KLU transients were
  never supported (`factor` has no JVP rule); use `backend="dense"` for
  transient gradients. DC solves and `dc_parameter_sensitivity` use the fused
  `solve_with_symbol`/`tsolve_with_symbol` and remain differentiable on every
  backend.
- Code that factors inside `jit` must free each raw handle exactly once with
  every consumer result in `dependency`, and must use the handle returned by
  `refactor`.

## Open items

- The traveling-wave notebook peaks at about 5.5 GiB for both record lengths
  and ends at 2.6 GiB. It is a one-off cost, probably compilation or a large
  intermediate, rather than a leak. It exceeds the provisional 4 GiB notebook
  budget but fits the 16 GB runner.
- The TFLN eye fold queries about 0.7 UI past the saved record for its last
  trace; `np.interp` holds the endpoint there. Exclude incomplete traces before
  using the eye quantitatively.
- A circulax regression test should assert that a short KLU transient returns
  the live numeric count to its baseline once klujax exposes the counter
  publicly.
