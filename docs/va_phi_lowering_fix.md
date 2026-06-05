# VA Phi Lowering Fix: Case 1.5 SCCP Live-Edge Shortcut

**Date**: 2026-05-18
**Repo**: `bosdi` (`src/bosdi/va/lowering.py`)
**Branch context**: `circulax/feat/va-performance-sensitivity`

---

## Background: The 12,944 µs/step Performance Gap

Before this work, a 6-ring PSP103 oscillator simulation ran at ~12,944 µs/step instead of the
~2,000 µs/step projected in an earlier estimate. The gap was a measurement artifact: the estimate
confused **Python source lines** with **XLA operations**.

### SSA inlining does not reduce XLA ops

`_inline_single_use_ssas` in `emitter.py` substitutes single-use SSA temporaries into their
consumer expressions. This reduces the number of Python local variables, but JAX traces the same
operations regardless:

```python
# Before inlining (2 lines, 2 XLA ops):
v123 = jnp.exp(x)
v456 = v123 + y

# After inlining (1 line, still 2 XLA ops):
v456 = jnp.exp(x) + y
```

Both produce identical XLA HLO graphs. The earlier projection linearly extrapolated from line count
to runtime — a ~6× error.

### Actual XLA op counts

| Stage | Python lines | XLA ops | Change |
|-------|-------------|---------|--------|
| Before any optimization | 34,538 (init+eval) | ~22,800 | — |
| After DCE | 22,117 (eval only) | ~22,100 | −705 (3.2%) |
| After SSA inlining | ~4,900 | ~22,100 | 0 removed |

Only DCE removes real XLA operations. SSA inlining is cosmetic for cached execution.

### Why 12,944 µs/step is correct

- ~22,000 XLA ops × ~16 ns/op (CPU dispatch) ≈ 350 µs per device evaluation
- 6 MOSFETs × 350 µs = ~2,100 µs per residual assembly
- ~6.2 evaluations/step (4.2 Newton iters + Jacobian + history) × 2,100 µs ≈ **13,000 µs/step** ✓

---

## The All-Static Lowering Bug

### What "all-static" means

When `static_params={all params}`, SCCP (Sparse Conditional Constant Propagation) can fold
every parameter-dependent conditional branch to a compile-time constant, eliminating dead branches
entirely. For PSP103 (~22,000 XLA ops with TYPE-only static), this should shrink the combined
function significantly.

### The bug: wrong phi edge selection

`_lower_phi` in `lowering.py` resolves phi nodes (merge points in the control-flow graph) into
Python/JAX expressions. It has two resolution paths:

- **Case 1**: Diamond detection — emits `jnp.where(cond, true_expr, false_expr)` when a 2-edge
  or 3-edge diamond pattern is found.
- **Case 2**: Heuristic fallback — resolves all edges and picks the "most informative" one
  (voltage-referencing > non-literal > non-zero literal > first resolved).

Under all-static lowering, SCCP folds branch conditions to constants, so many 2-edge phis no
longer match a diamond (the condition SSA is a literal `True`/`False`, not a computation). These
fall through to Case 2, where the heuristic could pick a sentinel-valued constant path (e.g.,
`NFA = 8e22`) instead of the SCCP-preferred live edge.

### Root cause

`SccpResult.live_phi_value(phi_edges, block_label)` already knows which phi edge is executable
when SCCP has eliminated all but one predecessor. But Case 2 never consulted it — it ran the
priority heuristic blindly.

---

## The Fix: Case 1.5

**Location**: `bosdi/src/bosdi/va/lowering.py`, after line 2320 (end of Case 1b)

```python
# Case 1.5: SCCP live-edge shortcut.
# When SCCP has eliminated all but one predecessor edge (e.g. a
# parameter-dependent condition folds to a constant), resolve only that
# live edge.  This fires after diamond detection so we never silently
# drop voltage-dependent branches that Case 1 would have preserved via
# ``jnp.where`` — by this point neither diamond detector matched, so
# there is no runtime condition to preserve.
from .sccp import SccpResult as _SccpResult  # noqa: PLC0415

if isinstance(sccp, _SccpResult):
    phi_block = sccp.block_of(inst.result)
    if phi_block is not None:
        live_ssa = sccp.live_phi_value(inst.phi_edges, phi_block)
        if live_ssa is not None:
            live = resolve_edge(live_ssa)
            if live is not None:
                return live
```

### Why this placement is safe

The original code had a "Case 0" SCCP shortcut (before diamond detection) that was removed because
it preempted Case 1 and silently dropped voltage-dependent branches that should have become
`jnp.where` expressions. Case 1.5 avoids this by sitting **after both diamond detectors**: if we
reach Case 1.5, neither detector matched, so there is no runtime condition to preserve. Using
SCCP's single live edge is then strictly correct.

### API used

- `sccp.block_of(inst.result)` → `str | None` — the block label containing this phi's def
  (from `SccpResult.ssa_block`, populated during SCCP traversal)
- `sccp.live_phi_value(phi_edges, block_label)` → `str | None` — the SSA name of the unique
  executable predecessor edge, or `None` if zero or multiple edges are live

---

## Validation Results

| Metric | Value |
|--------|-------|
| All-static sentinel leaks before fix | 0 (Case 2 heuristic already mitigated) |
| All-static sentinel leaks after fix | 0 |
| TYPE-only emitted source size | 975,539 chars |
| All-static emitted source size | 489,331 chars (−50%) |
| Test suite | 194 passed, 16 skipped |

The 50% source size reduction for all-static confirms SCCP constant-folding is eliminating
dead branches at lowering time. Whether this translates to XLA op reduction (and thus runtime
improvement) depends on whether JAX's HLO optimizer also sees through the folded constants —
that measurement is deferred to the benchmark phase.

---

## What Remains

The 12,944 µs/step figure is the correct baseline for TYPE-only static lowering. To achieve a
real runtime improvement, the path forward is reducing **XLA ops**, not Python source lines:

1. **All-static XLA op count**: measure how many HLO ops the all-static combined function
   produces vs TYPE-only (~22,000). If SCCP folding eliminates ~8,000 ops, that's a genuine
   ~37% speedup.
2. **Benchmark the all-static path**: run the ring oscillator with the all-static PSP103 model
   and measure µs/step.
3. **JUNCAP200 STI expll bug**: the STI segment's `expll` phi still misclassifies at V=0
   (~1e5× error), tracked separately.
