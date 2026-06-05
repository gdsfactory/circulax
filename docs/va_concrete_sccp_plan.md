# VA Performance: Concrete Init Evaluation for Eval-Phase SCCP

**Date**: 2026-05-18
**Repos**: `bosdi` (`/home/cdaunt/code/bosdi`), `circulax` (`/home/cdaunt/code/circulax/circulax`)
**Branch**: `circulax/feat/va-performance-sensitivity`
**Reference**: vajax at `/home/cdaunt/code/vajax/` achieves this via concrete cache seeding

---

## Problem Statement

circulax-VA is ~29x slower per-device than vajax for PSP103 ring oscillators. The `_PSP103N_combined` function is ~4,897 lines with 1,357 `jnp.where` calls. vajax's equivalent eval function has ~743 `jnp.where` after SCCP (67% reduction from 2,247 baseline).

The gap is NOT about jnp.where vectorization (stack/where/unstack increases XLA ops) or tree_map batching (already implemented, <2% impact). The gap is about **SCCP depth**: how many constants are available when the eval function's dead branches are pruned.

## Root Cause

bosdi's SCCP (`bosdi/src/bosdi/va/sccp.py`, `_eval_opcode` at line 140) returns BOTTOM for:
- `fdiv` (excluded due to Python vs JAX 0/0 semantics divergence)
- ALL math intrinsics: `exp`, `log`, `sqrt`, `pow`, `floor`, `ceil`, `abs`, `min`, `max`, etc.
- Any function call or unrecognized opcode (line 225: `return _BOTTOM`)

It CAN fold: `iadd/isub/imul/idiv`, `fadd/fsub/fmul/frem`, comparisons (`flt/fle/fgt/fge/feq/fne`), casts, boolean negation.

PSP103's init function computes temperature-dependent coefficients through chains of `fdiv`, `exp`, `log`, `sqrt`. Result: most init cache values stay BOTTOM in the lattice. The init→eval bridge at `lowering.py:1215-1217` only propagates CONSTANT lattice values:

```python
lat = init_sccp.lattice_value(init_val)
if lat.is_constant:
    eval_sccp_init[eval_arg] = lat.value
```

So eval SCCP runs with only a handful of known constants (integer switches, simple float arithmetic). Most branches stay unresolvable.

### How vajax solves this

vajax (`vajax/analysis/openvaf_models.py:935-960`) runs the init function concretely and seeds actual float values:

```python
# Seed shared cache values (hidden_state from init)
cache_row = np.asarray(cache[0])  # actual computed values from running init
for cache_col, mapping in enumerate(translator.cache_mapping):
    if cache_col in shared_set:
        eval_param_idx = mapping["eval_param"]
        value_id = translator.param_idx_to_val.get(eval_param_idx)
        if value_id is not None:
            sccp_known_values[value_id] = float(cache_row[cache_col])
```

This seeds hundreds of concrete float values into eval SCCP, enabling it to prove 89% of blocks dead and eliminate 67% of jnp.where calls.

### Evidence

PSP103N emitted by bosdi (from `/tmp/tmp236ok4r6/psp103_va_bench.py`):
- `_PSP103N_setup`: 1,845 lines, 1,241 `jnp.where`, 22 `tree_map`
- `_PSP103N_combined`: 4,897 lines, 1,357 `jnp.where`, 34 `tree_map`
- `init[]` lookups in combined: 324
- Total N-type `jnp.where`: 2,598

vajax PSP103 eval after SCCP: ~743 `jnp.where` (from vajax `docs/performance_analysis.md` line 243).

---

## Implementation Plan

### Step 1: Concrete evaluation of init hoists

**File**: `bosdi/src/bosdi/va/lowering.py`
**Location**: After init hoists are lowered and init SCCP has run (~line 1190), before `_bind_cslot_args`.

After `_resolve_ssa` has produced init hoists (each is a `(ssa_name, expr_text)` pair), evaluate them concretely:

1. Build a Python namespace from `effective_static` (all param values as locals)
2. Add `jnp` module functions to the namespace (for `jnp.where`, `jnp.exp`, etc.)
3. Walk the init hoists in order, evaluating each expression text with Python `eval()`
4. Store results in `concrete_cache: dict[str, float]` mapping cslot bridge SSA → value

Pseudocode:
```python
import jax.numpy as jnp as _jnp_mod

concrete_cache: dict[str, float] = {}
if effective_static:
    ns: dict[str, Any] = {"jnp": _jnp_mod}
    # Seed namespace with static param values
    for name, val in effective_static.items():
        ns[name] = val
    # Evaluate init hoists sequentially (they depend on each other in order)
    for ssa, expr in init_hoists:  # the (name, expr_text) pairs from cse
        try:
            result = eval(expr, {"__builtins__": {}}, ns)
            ns[ssa] = result
            # Store as Python float if it's a scalar
            if hasattr(result, 'item'):
                concrete_cache[ssa] = float(result.item())
            elif isinstance(result, (int, float)):
                concrete_cache[ssa] = float(result)
        except Exception:
            pass  # non-evaluable (depends on runtime input like signals/state)
```

### Step 2: Feed concrete values into eval SCCP initial constants

**File**: `bosdi/src/bosdi/va/lowering.py`
**Location**: Lines 1208-1217, the init→eval bridge loop.

Replace:
```python
for init_val, eval_arg in zip(...):
    lat = init_sccp.lattice_value(init_val)
    if lat.is_constant:
        eval_sccp_init[eval_arg] = lat.value
```

With:
```python
for init_val, eval_arg in zip(...):
    # Prefer symbolic constant (already proven by SCCP)
    lat = init_sccp.lattice_value(init_val)
    if lat.is_constant:
        eval_sccp_init[eval_arg] = lat.value
    else:
        # Fall back to concretely evaluated value
        # Look up the init hoist SSA that feeds this cslot
        init_ssa = init_env.get(init_val)
        if init_ssa is not None:
            ssa_text = init_ssa.text if hasattr(init_ssa, 'text') else str(init_ssa)
            # Check if we have a concrete value for this SSA or its text
            if ssa_text in concrete_cache:
                eval_sccp_init[eval_arg] = concrete_cache[ssa_text]
```

The exact lookup path from `init_val` → `init hoist SSA name` → `concrete_cache` key needs to match bosdi's cslot bridge mapping. Trace the code path in `_bind_cslot_args` to understand how init SSAs map to eval args.

### Step 3: Verify and benchmark

No changes to the emitter needed. The eval SCCP now has hundreds more constants seeded, so `rewrite_function(cm.eval_fn, eval_sccp)` will prune more dead blocks, and the lowering walk will produce fewer hoists.

---

## Key Files

| File | Line(s) | Role |
|------|---------|------|
| `bosdi/src/bosdi/va/lowering.py` | 1103-1123 | Init SCCP run + rewrite |
| `bosdi/src/bosdi/va/lowering.py` | 1195-1243 | `_bind_cslot_args` + eval SCCP seeding + eval SCCP run |
| `bosdi/src/bosdi/va/lowering.py` | 1208-1217 | Init→eval constant bridge (THE CHANGE POINT) |
| `bosdi/src/bosdi/va/sccp.py` | 140-225 | `_eval_opcode` — shows what SCCP can/cannot fold |
| `bosdi/src/bosdi/va/sccp.py` | 233-250 | `SccpResult` — lattice query API |
| `bosdi/src/bosdi/va/emitter.py` | 888-1004 | `_prep_combined_body` — eval hoist transform pipeline |
| `vajax/analysis/openvaf_models.py` | 935-960 | Reference: how vajax seeds concrete cache values |

## Safety Considerations

1. **`eval()` on generated strings**: The expressions come from bosdi's own lowering (not user input). They contain JAX ops like `jnp.where(...)`, `jnp.exp(...)`, arithmetic, and SSA variable references. The restricted namespace (`__builtins__: {}`) prevents arbitrary execution.

2. **Numerical divergence**: Python `eval()` uses float64 arithmetic, same as JAX's default dtype. Results should match JAX to machine epsilon. If a value involves `nan` or `inf` (e.g., `1/0`), the `except` clause handles it — the value stays BOTTOM and the branch remains live.

3. **Phi results**: `_inject_sccp_constants` already excludes phi results (line 637-642) to avoid baking in values from dead branches. The concrete eval approach inherits this exclusion because it only evaluates the init hoists, not phi nodes.

4. **Voltage-tainted values**: Init hoists that depend on `signals.X` or `s.X` won't have those variables in the namespace, so `eval()` will raise `NameError` → caught by `except` → stays BOTTOM. This is correct: voltage-dependent values are not constant.

## Alternative: Extend `_eval_opcode` with Math Intrinsics

Instead of concrete eval, teach SCCP to fold `fdiv`, `exp`, `log`, `sqrt`, etc. Add to `sccp.py:_eval_opcode`:

```python
import math

if op == "fdiv":
    if b == 0:
        return _BOTTOM  # or handle inf/nan
    return LatticeValue(LatticeState.CONSTANT, a / b, "float")

# Math intrinsics (MIR opcodes from OpenVAF)
_MATH_UNARY = {"exp": math.exp, "log": math.log, "sqrt": math.sqrt,
               "floor": math.floor, "ceil": math.ceil, "fabs": abs}
_MATH_BINARY = {"pow": math.pow, "fmin": min, "fmax": max, "atan2": math.atan2}

if len(operands) == 1 and op in _MATH_UNARY:
    try:
        return LatticeValue(LatticeState.CONSTANT, _MATH_UNARY[op](a), "float")
    except (ValueError, OverflowError):
        return _BOTTOM

if len(operands) == 2 and op in _MATH_BINARY:
    try:
        return LatticeValue(LatticeState.CONSTANT, _MATH_BINARY[op](a, b), "float")
    except (ValueError, OverflowError):
        return _BOTTOM
```

**Pros**: Pure symbolic, no eval(). Catches constants during the SCCP lattice walk itself.
**Cons**: Must handle overflow/domain errors carefully. Need to map OpenVAF MIR opcode names to Python math functions (check `lowering.py` for the opcode→jnp mapping). ~15-20 opcodes to add.

**Recommendation**: Do both. Extend `_eval_opcode` for common cases (this helps init SCCP prove more constants natively), AND use concrete eval as belt-and-braces for anything the lattice misses.

## Expected Impact

| Metric | Current | Expected After |
|--------|---------|----------------|
| `jnp.where` in `_PSP103N_combined` | 1,357 | 450–700 |
| Combined function lines | 4,897 | 2,000–3,000 |
| Dead MIR blocks (eval) | ~30% (est.) | ~70-89% (matching vajax) |
| va_static N=3 µs/step | 9,560 | ~4,000–6,000 |
| JIT compile time | ~197s | ~100–140s |

## Verification

```bash
# 1. bosdi tests (194 tests)
cd /home/cdaunt/code/bosdi && pixi run python -m pytest tests/ -x

# 2. circulax VA tests
cd /home/cdaunt/code/circulax/circulax && pixi run python -m pytest tests/ -x

# 3. Count constants bridged (add temporary debug print at line ~1217)
# Before: count how many lat.is_constant == True
# After: count how many concrete_cache entries + lat.is_constant entries

# 4. Emit PSP103 and compare (NOTE: parse all string defaults as float)
PYTHONPATH=/home/cdaunt/code/bosdi/src pixi run python -c "
import math
from bosdi.va import compile_va_unopt_with_split, lower
from bosdi.va.emitter import emit_source
from bosdi.va.va_defaults import parse_va_defaults_expanded
va = '/home/cdaunt/code/bosdi/src/devices/ihp_sg13g2/psp103/psp103.va'
dump = compile_va_unopt_with_split(va)
defaults = parse_va_defaults_expanded(va)
static = {}
for n, s in defaults.items():
    try:
        v = float(s.default)
        if math.isfinite(v):
            static[n] = v
    except (ValueError, TypeError):
        pass
static['TYPE'] = 1
dev = lower(dump.modules[0], va_defaults=defaults, collapse_nodes=True,
            static_params=static, class_name='PSP103N')
src = emit_source([dev])
print(f'chars: {len(src)}')
print(f'jnp.where count: {src.count(\"jnp.where\")}')
print(f'tree_map count: {src.count(\"tree_map\")}')
print(f'lines: {src.count(chr(10))}')
"

# 5. Benchmark
pixi run python benchmarks/ring/bench_circulax.py --variant va_static --n-stages 3

# 6. Numerical accuracy: DC operating point must match pre-change values
pixi run python -m pytest tests/ -x -k "psp103"
```

---

## Implementation Results (2026-05-19)

Implemented as planned. bosdi commit: `cab2d2e`.

### What was built

**`sccp.py`**: Added `_SCCP_MATH1` and `_SCCP_MATH2` tables; extended `_eval_opcode` to fold
`fdiv` (b≠0), all unary math intrinsics (`exp`, `ln`, `log`, `sqrt`, `floor`, `ceil`, all trig and
inverse-trig, `fabs`), and binary math (`pow`, `hypot`, `atan2`). Returns BOTTOM for non-finite
results.

**`lowering.py`**: After `_init_hoist_end`, evaluates init hoists in emission order using Python
`eval()` with a namespace of `{jnp, **{param_name: value}}`. Finite scalar results feed into
`eval_sccp_init` via the cslot bridge as fallback when the lattice returned BOTTOM.

### Key implementation finding

The plan's concrete-eval namespace seeded values by **MIR SSA name** (`v15281 = 1.0`). This is
wrong — hoist expressions are Python source text that uses the **parameter name** (`SWIGATE`,
`TOXE`, …), not the SSA name. The namespace must be seeded with `_kind.name`, not `_ssa`:

```python
# Wrong (plan pseudocode):
_eval_ns[_ssa] = effective_static[_kind.name]

# Correct:
_eval_ns[_kind.name] = effective_static[_kind.name]
```

### Verification script fix

The plan's emit script used `isinstance(s.default, (int, float))` to build `static`, but
`parse_va_defaults_expanded` returns **all defaults as strings** (`'0'`, `'1.0e-9'`, …). This
produced `static = {TYPE: 1}` (1 param), which is not enough to eliminate meaningful branches.
The correct approach parses every string default with `float()`, giving 808 static params for
PSP103.

### Actual impact (808 static params, TYPE=1)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| `jnp.where` (total) | 2,732 | **651** | −76% |
| Lines (total) | 4,872 | **1,313** | −73% |
| `tree_map` batches | 41 | **13** | −68% |

The SCCP math-intrinsic extension drives the majority of the gain: eval SCCP seeds went from 12
to 1,104 (1,096 from lattice folding, 8 from concrete eval). Temperature-dependent hoists (the
first `_init_hoist_end` hoists reference `_temperature`) are correctly left BOTTOM since
`_temperature` is not in the static namespace.

### Tests

- bosdi: 150 passed, 10 skipped
- circulax PSP103/VA: 3 passed
