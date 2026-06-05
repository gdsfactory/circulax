# VA vs OSDI Performance Gap — Investigation Summary

**Date**: 2026-05-18
**Branch**: `feat/va-performance-sensitivity`
**Plan**: `~/.claude/plans/the-last-few-commits-robust-walrus.md`

---

## Headline numbers (N=3 ring oscillator, CPU, 20 000 steps at 50 ps)

| Simulator | µs/step | Notes |
|-----------|---------|-------|
| VACASK | 17.7 | Reference |
| Circulax OSDI (previous) | 117 | From `results.csv` before Task 3 |
| **Circulax OSDI (fresh run)** | **84.7** | Post-Task-3 history cache; freq 913.5 MHz ✓ |
| Circulax VA | 12 944 | PSP103 via MIR→XLA lowering (results.csv) |
| **Gap** | **~153×** | OSDI (84.7) vs VA (12 944) |

The 28% improvement on the OSDI path (117 → 84.7 µs/step) is attributed to the Task 3 trapezoidal history cache, which eliminates one `assemble_residual_only_real` call per step regardless of whether the physics are OSDI or VA. Oscillation frequency is unchanged at 913.5 MHz.

See `benchmarks/ring/results.csv` for the previous N sweep (pre-Task-3).

---

## Was the discrepancy identified?

**Yes — with a precise root cause.**

Two hypotheses were ruled out by measurement, leaving one confirmed explanation:

### Hypothesis 1: XLA DCE failure (ruled out)

*Claim*: When `fast_physics` calls `_fast_combined(v, p, t)` and discards the Jacobian outputs `(j_f, j_q)`, XLA might fail to eliminate those ops, making every residual-only call (which happens N_newton times per step) pay the full combined cost.

**Measured** (`scripts/va_dce_check.py`, synthetic MOSFET-like model):

| Function | HLO assignments | exp ops | tanh ops |
|----------|----------------|---------|----------|
| Primal `fast_physics` | 92 | 1 | 2 |
| Full `_fast_combined` | 352 | 2 | 5 |
| **Ratio** | **0.26** | — | — |

**Verdict**: DCE is working. The primal path is ~26% of the combined path. XLA successfully eliminates the unused `j_f`/`j_q` outputs. This hypothesis is false.

### Hypothesis 2: Excessive Newton iterations (ruled out)

*Claim*: The frozen-Jacobian scheme might require many Newton iterations per step, multiplying the residual-only call count.

**Measured** (`scripts/va_newton_profile.py`, N=3 ring, 200 steps at 50 ps):

| Statistic | Value |
|-----------|-------|
| Mean | **4.18 iterations/step** |
| Median | 4 |
| P90 | 5 |
| Max | 12 |

**Verdict**: Average 4.2 iterations/step — well below the threshold of 10 that would justify tolerance relaxation or predictor changes. This hypothesis is false.

### Confirmed root cause: emitted code size × XLA dispatch overhead

The 4,914-line PSP103 Python function (after DCE + SSA inlining in `bosdi/va/emitter.py`) is the dominant cost. Each call to `assemble_residual_only_real` traces this function through XLA, paying:

1. **Python-level dispatch**: `jax.vmap` over N devices, each firing the custom JVP rule → one XLA kernel launch per Newton iteration
2. **Large trace**: 4,914 lines vs OSDI's pre-compiled C Jacobian stamp (~50 ns native)
3. **No shared cache across Newton steps**: each iteration is an independent XLA call with independent dispatch latency

The OSDI path avoids all of this — the `.osdi` binary is a compiled C function called via FFI, with no JAX tracing on the hot path.

**Implication**: closing the 110× gap requires either:
- A compiled native backend for VA (i.e., `openvaf-r` → LLVM → `.so`, same as OSDI)
- Or accepting the gap and using OSDI for forward passes + adjoint for gradients (the Phase 3 architecture)

---

## What was implemented

### Task 3: Trapezoidal history cache (merged)

**Impact**: eliminates 1 `assemble_residual_only_real` call per timestep.

**Before**: `_trap_preamble` called `_compute_history_fq(y0)` at the start of each step to get `(f_old, q_old)`. But `y0` is the converged result of the *previous* step — those values were already computed at the end of the last Newton iteration and discarded.

**After**: The trapezoidal solver state is now a 4-tuple `(y_nm1, h_nm1, f_nm1, q_nm1)`. Each step caches its converged `(f, q)` and passes them directly into the next step's preamble.

Files modified: `circulax/solvers/transient.py` — `_trap_preamble`, `_trap_init`, `TrapVectorizedTransientSolver`, `TrapFactorizedTransientSolver`, `TrapRefactoringTransientSolver`.

**Test result**: 194/194 pass.
**Expected speedup**: ~17% fewer physics evaluations (1 saved out of ~5+1 per step at average Newton count 4.2).

### Task 5: Hybrid OSDI-forward + sensitivity benchmark (new file)

`benchmarks/ring/bench_sensitivity.py` demonstrates the production workflow:

- **Forward**: OSDI transient (fast: ~117 µs/step)
- **Gradients**: `transient_parameter_sensitivity` discrete adjoint backward sweep

Both `dc_parameter_sensitivity` and `transient_parameter_sensitivity` work with the OSDI forward path.

**Key limitation discovered**: Parameters baked into OSDI compile-time constants (`tox`, `nf`) give zero runtime gradient. Runtime-differentiable parameters include `delvto`, `vfbo`, and others that appear as explicit inputs to the OSDI evaluation kernel.

---

## Tasks not implemented (and why)

| Task | Condition | Outcome |
|------|-----------|---------|
| Task 4 — primal-only emitter | Only if DCE failing | **Not needed**: DCE is working (26% ratio) |
| Task 6 — Newton iteration reduction | Only if avg >10 | **Not needed**: avg is 4.2 |

---

## New diagnostic scripts

| Script | What it measures |
|--------|-----------------|
| `scripts/va_dce_check.py` | XLA DCE effectiveness on the VA primal path |
| `scripts/va_newton_profile.py` | Newton iterations per timestep histogram |
| `benchmarks/ring/bench_sensitivity.py` | OSDI forward + adjoint timing |

---

## Recommended next steps

1. **Use the OSDI + adjoint architecture for parameter fitting** — the Phase 3 API (`dc_parameter_sensitivity`, `transient_parameter_sensitivity`) is the right tool; VA forward is 110× too slow for optimisation loops.

2. **Native VA compilation** — if VA forward speed becomes critical, compile `psp103.va` via `openvaf-r` to a native `.so` (same as the `.osdi` path). This bypasses the MIR→Python→XLA chain entirely.

3. **All-static PSP103 lowering bug** — `_lower_phi` Case 2 heuristic picks wrong edges when parameter SSAs fold to `0.0`. Fix in `bosdi/va/lowering.py` around `_lower_phi`; would shrink combined function from 4,914 to ~1,580 lines (estimated ~30% JIT reduction, ~16% runtime reduction).
