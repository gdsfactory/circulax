# VA Performance: PHI Node Batching + Verification Plan

**Date**: 2026-05-18
**Executed**: 2026-05-18
**Repos**: `bosdi` (`/home/cdaunt/code/bosdi`), `circulax` (`/home/cdaunt/code/circulax/circulax`)
**Branch**: `circulax/feat/va-performance-sensitivity`
**Reference**: vajax at `/home/cdaunt/code/vajax/` implements all three techniques

---

## Status of Three Optimizations

| Optimization | Status | Impact |
|---|---|---|
| Init/eval cache split | Already implemented | XLA constant-folds setup; `compile_va_unopt_with_split` produces split |
| Analytical Jacobian from MIR | Already implemented | `@va_component(jacobian_fn=..., combined_fn=...)` bypasses `jax.jacfwd` |
| **PHI node batching** | **✅ Implemented** | 57 `tree_map` batches replace 110 `jnp.where` calls; <2% per-step runtime improvement |

---

## Task 1: Verify Analytical Jacobian Is Firing

**Goal**: Confirm the emitted PSP103 source uses `@va_component` (not `@component` fallback) and that the ring benchmark actually takes the `combined_func` assembly path.

**Files**:
- `bosdi/src/bosdi/va/emitter.py` — `_has_jacobian()` at line 94
- `bosdi/src/bosdi/va/lowering.py` — Jacobian validation at lines 1255-1298
- `circulax/circulax/solvers/assembly.py` — `combined_func` path at lines 290-298

**Steps**:

1. Run the lowering for PSP103 with int-only static params and inspect the emitted source:
   ```python
   from circulax.va import compile_va_unopt_with_split, lower
   from circulax.va.emitter import emit_source
   from circulax.va.va_defaults import parse_va_defaults_expanded

   va = "tests/data/va/psp103v4/psp103.va"
   dump = compile_va_unopt_with_split(va)
   defaults = parse_va_defaults_expanded(va)
   int_static = {n: int(s.default) for n, s in defaults.items() if s.type_ == "int"}
   int_static["TYPE"] = 1
   dev = lower(dump.modules[0], va_defaults=defaults, collapse_nodes=True,
               static_params=int_static, class_name="PSP103N")
   src = emit_source([dev])
   ```
   Check the emitted source for:
   - `@va_component(` (not `@component(`) — confirms Jacobian was generated
   - `jacobian_fn=_PSP103N_jacobian` — confirms Jacobian wrapper exists
   - `combined_fn=_PSP103N_combined` — confirms combined function exists
   - Count `jnp.where(` occurrences — baseline for PHI batching improvement

2. If `@component(` is found instead: the Jacobian SSA resolution failed silently.
   - Check `dev.jacobian_resist` and `dev.jacobian_react` — if empty, SSAs were DCE'd
   - Add logging to `lowering.py:1262-1279` to report which SSAs failed

**Deliverable**: Report confirming Jacobian is active (or identifying the failure point).

---

## Task 2: PHI Node Batching in bosdi Emitter

**Goal**: Group PHI nodes that share the same branch condition and emit a single batched `jnp.where` via `jax.tree_util.tree_map` instead of N separate `jnp.where` calls.

**Reference implementation**: vajax at `/home/cdaunt/code/vajax/openvaf_jax/codegen/function_builder.py` lines 133-258 (`_emit_batched_phis`).

### Background

bosdi's `_lower_phi()` in `lowering.py` resolves each PHI node independently:
- **Case 1 (diamond detection)**: Finds 2-edge diamond pattern, emits `jnp.where(cond, true, false)`
- **Case 1b (nested 3-edge)**: PSP103 `expll` macro, emits nested `jnp.where`
- **Case 1.5 (SCCP shortcut)**: Single live edge, no `jnp.where` at all
- **Case 2 (fallback)**: Picks best edge by heuristic, no `jnp.where`

Each resolved PHI becomes a separate line in the emitted source: `v_N = jnp.where(cond, true_expr, false_expr)`.

PSP103 produces **200-400 `jnp.where` calls**. Many of these share the same condition (e.g., all PHIs at the merge point after `if (SWGIDL > 0)` share the SWGIDL condition).

### Architecture

The change spans two files in bosdi:

#### 2A. `lowering.py` — Structured PHI Resolution

Currently `_lower_phi()` returns an `Expr` (string expression). Refactor to also return structured resolution metadata when a diamond is detected.

**Current flow**:
```
_lower_phi(inst) → Expr("jnp.where(cond, true, false)")
  → stored in cse_hoists as (ssa_name, expr_text)
  → emitted as one line per PHI
```

**New flow**:
```
_lower_phi(inst) → PhiResolution(type, cond_ssa, true_expr, false_expr)
                  OR Expr (for non-diamond cases)
```

Add a `PhiResolution` dataclass to `lowering.py`:
```python
@dataclasses.dataclass
class PhiResolution:
    cond_ssa: str          # SSA name of the branch condition
    cond_negated: bool     # whether the condition should be negated
    true_expr: str         # expression for the true branch
    false_expr: str        # expression for the false branch
```

Modify `_lower_phi()`:
- Case 1 (diamond): Instead of returning `Expr(f"jnp.where({cond}, {true_e}, {false_e})")`, return `PhiResolution(cond_ssa=cond_ssa, cond_negated=False, true_expr=true_e, false_expr=false_e)`
- Case 1b, 1.5, 2: Continue returning `Expr` (these are not batchable or already optimized)

The caller of `_lower_phi()` (which populates `cse_hoists`) must handle both return types:
- `Expr` → store as before: `(ssa_name, expr_text)`
- `PhiResolution` → store in a separate list: `phi_resolutions: list[tuple[str, PhiResolution]]`

#### 2B. `emitter.py` — Batched Emission

Currently `_emit_hoists()` (or `_hoist_lines()` around line 598) iterates `cse_hoists` and emits one line per entry. Add a grouping + batching phase.

**Algorithm**:

1. **Collect**: After lowering completes, separate `phi_resolutions` from regular `cse_hoists`.

2. **Group by condition**:
   ```python
   groups: dict[tuple[str, bool], list[tuple[str, str, str]]] = {}
   # key = (cond_ssa, is_negated)
   # value = [(result_ssa, true_expr, false_expr), ...]
   ```

3. **Emit batched groups**:
   - Groups with 1 PHI → emit traditional `v_N = jnp.where(cond, true, false)`
   - Groups with 2+ PHIs → emit tree_map:
     ```python
     (v_1, v_2, ...) = jax.tree_util.tree_map(
         lambda _t, _f: jnp.where(cond, _t, _f),
         (true_1, true_2, ...),
         (false_1, false_2, ...),
     )
     ```

4. **Ordering**: PHI batches must be emitted at the correct point in the hoist sequence — they depend on their operands being already defined. The simplest approach: emit all non-PHI hoists first (in current order), then emit PHI batches. This works because PHI operands are always defined before the PHI (SSA dominance property).

   If PHIs depend on other PHIs (chained diamonds), topological sort the PHI groups.

**Key files to modify in bosdi**:
- `src/bosdi/va/lowering.py`: `_lower_phi()` return type, `PhiResolution` dataclass
- `src/bosdi/va/emitter.py`: `_hoist_lines()` or `_emit_hoists()` — add grouping + tree_map emission

**Key files to reference from vajax**:
- `openvaf_jax/codegen/function_builder.py` lines 133-258: `_emit_batched_phis()` algorithm
- `openvaf_jax/mir/ssa.py` lines 45-74: `PHIResolution` dataclass

### Testing

1. Run existing bosdi test suite: `pixi run python -m pytest tests/ -x` (194 tests)
2. Emit PSP103 source before and after, diff the `jnp.where` count:
   ```bash
   grep -c "jnp.where" before.py after.py
   ```
3. Run the ring oscillator benchmark at N=3 with `va` and `va_static` variants, compare µs/step

---

## Task 3: Benchmark Before/After

**Goal**: Measure the XLA op reduction and µs/step improvement from PHI batching.

**Metrics to collect**:

| Metric | How to measure |
|---|---|
| `jnp.where` count in emitted source | `grep -c "jnp.where" emitted.py` |
| Emitted source size (chars) | `wc -c emitted.py` |
| XLA HLO op count | `jax.make_jaxpr(eval_fn)(...)` then count ops |
| µs/step at N=3 | `pixi run python benchmarks/ring/bench_circulax.py --variant va_static --n-stages 3` |
| µs/step at N=9 | Same with `--n-stages 9` |

**Baseline (current, pre-batching)**:
- va_static N=3: ~13,100 µs/step
- va N=3: ~20,500 µs/step
- all-static source size: 489,331 chars (post Case 1.5 fix)

**Commands**:
```bash
# All commands from repo root
PYTHONPATH="/home/cdaunt/code/bosdi/src:$PYTHONPATH"

# Emit source and count where calls
pixi run python -c "
from circulax.va import compile_va_unopt_with_split, lower
from circulax.va.emitter import emit_source
from circulax.va.va_defaults import parse_va_defaults_expanded
va = 'tests/data/va/psp103v4/psp103.va'
dump = compile_va_unopt_with_split(va)
defaults = parse_va_defaults_expanded(va)
static = {n: float(s.default) for n, s in defaults.items()
          if isinstance(s.default, (int, float)) and not isinstance(s.default, bool)}
static['TYPE'] = 1
dev = lower(dump.modules[0], va_defaults=defaults, collapse_nodes=True,
            static_params=static, class_name='PSP103N')
src = emit_source([dev])
print(f'chars: {len(src)}')
print(f'jnp.where count: {src.count(\"jnp.where\")}')
print(f'tree_map count: {src.count(\"tree_map\")}')
"

# Run ring benchmark
pixi run python benchmarks/ring/bench_circulax.py --variant va_static --n-stages 3
```

---

## Implementation Order

1. **Task 1** (verify Jacobian) — 15 min, read-only, can run independently
2. **Task 2A** (lowering.py refactor) — 1-2 hrs, in bosdi repo
3. **Task 2B** (emitter.py batching) — 1-2 hrs, in bosdi repo, depends on 2A
4. **Task 3** (benchmark) — 30 min, depends on 2A+2B

Tasks 1 and 2A can run in parallel. Task 2B depends on 2A. Task 3 depends on 2A+2B.
