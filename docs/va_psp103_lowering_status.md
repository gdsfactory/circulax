# PSP103 VA-path lowering — current status

**Date**: 2026-05-04 (resolved)
**Branch**: `worktree-va-pyo3-binding` (circulax) + `development` (bosdi)
**Status**: ✅ **VA lowering matches OSDI within 0.7 % at every measured
PSP103 bias point**, including deep sub-threshold (Vgs=0). Tier-0
(juncap200), Tier-1 (mosvar), Tier-3 (PSP103) all green via the unopt-MIR
ingestion path. The historical "32× sub-threshold gap" was a phantom
caused by comparing against a stale OSDI reference value generated under
a different fixture; re-running OSDI on the same fixture shows the values
agree.

## Resolution log — 2026-05-04

The dominant lowering bug was the `--dump-json` path's heuristic phi-merge
dropping nested-conditional branches. Resolution stack:

- `bosdi:f6aed6e` + `4628eb4` — unopt-MIR ingestion preserves
  nested-conditional phis; dominator-based diamond detection handles
  branches with internal control flow (Cooper-Harvey-Kennedy idom).
- `bosdi:5f39086` — sqrt floor `1e-30` → `1e-300` (juncap200 tunneling
  pre-factor at moderate forward bias).
- `bosdi:615edf2` — `~` not Python `not` for `bnot` opcode.
- `bosdi:02b6171` — `_temperature` default 300.0 → 300.15 K (SPICE/OSDI
  convention).

**PSP103 single-NMOS DC sweep at Vds=0.6 V (W=10 µm, L=1 µm) — current state**:

| Vgs   | OSDI         | VA path      | rel err  |
|-------|--------------|--------------|----------|
| 0.00  | -1.83e-08    | -1.84e-08    | +0.66 %  |
| 0.10  | -4.54e-07    | -4.56e-07    | +0.49 %  |
| 0.30  | -4.61e-05    | -4.62e-05    | +0.06 %  |
| 0.50  | -2.46e-04    | -2.46e-04    | -0.02 %  |
| 0.70  | -5.92e-04    | -5.92e-04    | -0.04 %  |
| 0.90  | -1.03e-03    | -1.03e-03    | -0.05 %  |
| 1.10  | -1.46e-03    | -1.46e-03    | -0.05 %  |

The MIR interpreter (`bosdi.va.interpret`) confirmed the lowering itself
is correct: the interpreter on the unopt-MIR `eval_fn` produces the same
residual as the lowered Python (-1.84e-08 at Vgs=0). Under the same
fixture, OSDI also produces -1.83e-08; the previously-cited 5.7e-10 was
not reproducible.

**Status (resolved):** PSP103 VA path is production-ready at the same
accuracy level as juncap200 and mosvar. JSON path remains broken (2.8e+05
A drain current at all biases) — it was the wrong abstraction; do not
revive without first auditing its phi-merge heuristic against
unopt-MIR's structural phis.

## Problem statement

`bosdi.va.lower()` converts OpenVAF MIR (via `--dump-json`) into Python `@component`
classes that JAX can trace and differentiate. For simple devices (diode), the result
matches the analytical reference within 0.1 % at 35 µs/step. For PSP103 (and BSIM3v3,
BSIM4), the lowered model produces currents that are wildly wrong:

| Test point             | OSDI (correct) | VA (current state) |
|------------------------|----------------|--------------------|
| VD=0, VG=0             | ~0 A           | -2.6 A             |
| VD=1.2, VG=0.9         | -0.44 mA       | -568 kA            |
| VD=0.0001, VG=0.9      | ~0 A           | -1.1 × 10¹⁰⁶ (overflow) |

The PSP103 ring oscillator does not oscillate via the VA path because of these
errors. The diode benchmark continues to pass, demonstrating that the lowering
*infrastructure* (compile_va → SCCP → emitter → @component) is sound — the issue
is specific to surface-potential MOSFET models.

## What's been fixed in this session

Three lowering bugs committed to `bosdi/main`:

### 1. `f4ae251` — `fix(va): tighten ParamGivenRef seeding for non-sentinel models`
Old code aggressively seeded `$param_given(X) = False` for every float param not
listed in `static_params`. PSP103 (no sentinel-defaulted params) has user-supplied
instance settings (W, L, AD, AS, PD, PS); these were being ignored as "not given"
and the model used VA-source defaults instead. New: only seed False for sentinels;
non-sentinel runtime params return True at the `ParamGivenRef` lowering site.

### 2. `744ca6f` — `feat(va): handle 3-edge phis from nested if/else (expll macro)`
PSP103's `expll(x, xlow, expxlow, xhigh, expxhigh)` macro produces a 3-way merge
in the IR (small-x, normal-exp, large-x branches). Previously fell through to
Case 2 fallback in `_lower_phi`, dropping the conditional structure and emitting
a single branch unconditionally. New `_find_nested_diamond` detects the pattern:
```
outer_dec: br(outer_cond, T, F=inner_dec)
inner_dec: br(inner_cond, T_inner, F_inner)
```
and emits `jnp.where(outer_cond, edge0, jnp.where(inner_cond, edge1, edge2))`.

### 3. `23a2f2c` — `fix(va): handle direct-edge diamonds where merge IS a br target`
When a 2-edge phi has `t_true == merge_block` (or `t_false == merge_block`), the
phi-edge predecessor IS the decision block itself. The old `edges_by_from.get(t_true)`
lookup failed because `dec_a` is not in `{t_true, t_false}`. New two-pass resolution
matches indirect edges first, then fills the remaining target by elimination when
`from == dec_a`. Fixes the `max(v, 0)` clamp pattern used in PSP103 init for
threshold-related cache slots (cache[195], [196], [197]).

**All 109 VA-suite tests pass with these fixes.** Diode VA path remains correct.

## Why PSP103 is still wrong

The three fixes take PSP103 from *constant 12 A regardless of voltage* (severe
structural break) to *voltage-dependent but numerically wrong*. The remaining
issue is that intermediate computed values diverge from what OSDI's downstream
codegen produces for the same MIR.

Specific cascade at VD=0.0001, VG=0.9:

```
v901216 = -2.5e-10            should be ~ -0.5 V (off by ~10⁹)
v907945 = v901216 + 1e-30      ≈ -2.5e-10
v907947 = i_v3305 · v907942 / v907945  ≈ -4 × 10¹⁰
v907948 = -v907947                      ≈ +4 × 10¹⁰
expll(v907948) → large-x branch:
  1e+100 × (1 + 4 × 10¹⁰)  ≈  4 × 10¹¹⁰    overflow
```

The root: `v901216` (a threshold-related intermediate) is 9 orders of magnitude
too small at small VDS in our lowered code, but is a normal -0.5 V quantity in
the OSDI binary's evaluation of the same MIR.

This is **not** a phi-detection bug — the phis are correctly resolved. It's that
SCCP runs constant propagation on the MIR exposed by `--dump-mir`/`--dump-json`,
but OpenVAF's actual OSDI codegen path runs **further** optimization passes
(induction-variable analysis, branch-condition propagation, GVN, range analysis)
that we don't replicate. Without matching those passes, intermediate SSA values
take different code paths that happen to produce different numerical answers.

## What's required to fully fix it

Three options, in increasing order of effort:

### Option A — accept and document (lowest effort)
- Mark PSP103 / BSIM3v3 / BSIM4 as not-yet-supported on the VA path.
- Use `OsdiModelDescriptor` for these models (production-quality, GPL-free
  via the `--dump-json` subprocess).
- Use VA path only for simple devices (diode, capacitor, resistor) and custom
  user-written `.va` files where differentiability is the primary need.
- **Effort**: minutes (mostly documentation).

### Option B — side-by-side numerical diff (medium effort)
- Instrument both OSDI and our lowered Python at the SSA level.
- For a single bias point, dump every intermediate value from both paths.
- `diff` to find the first SSA where they diverge.
- Walk back through the IR to find the missing optimization / transform.
- Apply targeted lowering rewrites until divergence is gone.
- **Effort**: 3–5 days. Requires building or extending an OSDI-side tracer.
- **Risk**: each cascade-layer fix may expose another. The diode worked because
  it has shallow chains; PSP103 has 5+ layers of interdependent optimizations.

### Option C — match OpenVAF's full optimization pipeline (highest effort)
- Port OpenVAF's MIR optimization passes that run between `--dump-json` and
  OSDI codegen into `bosdi.va.lowering`.
- Specifically: `inductive_var_analysis`, `branch_propagation`, `gvn`,
  `simplify_cfg` — these are in OpenVAF's `mir_optimize` crate.
- **Effort**: 1–2 weeks. Requires deep familiarity with OpenVAF internals.
- **Benefit**: makes the VA path work for ALL OSDI-supported models, not just
  the ones we patch case-by-case.

## Recommended path

For circulax's near-term goal (differentiable circuit simulation):
1. **Option A** for production work — use OSDI for established compact models.
2. Keep the diode VA benchmark green as the regression test for the lowering.
3. If/when a user really needs `jax.grad` through PSP103/BSIM4, schedule
   **Option B** as a focused 1-week task.

## Key file locations

- bosdi lowering: `/home/cdaunt/code/bosdi/src/bosdi/va/lowering.py`
- bosdi SCCP: `/home/cdaunt/code/bosdi/src/bosdi/va/sccp.py`
- bosdi IR client: `/home/cdaunt/code/bosdi/src/bosdi/va/ir_client.py`
- PSP103 VA source: `tests/data/va/psp103v4/psp103.va` (+ includes/)
- Diode VA source: `tests/data/va/diode.va`
- Diode benchmark: `benchmarks/diode/bench_circulax.py`
- PSP103 ring benchmark: `benchmarks/ring/bench_circulax.py`
- VA test suite: `tests/test_va_*.py` (109 tests, all passing)

## Sanity-check commands

```bash
# Diode VA (should pass cleanly, < 1 second):
pixi run python benchmarks/diode/bench_circulax.py --variant va

# PSP103 OSDI (should oscillate at ~900 MHz):
pixi run python benchmarks/ring/bench_circulax.py --variant osdi --n-stages 3

# PSP103 VA (currently runs but does NOT oscillate):
pixi run python benchmarks/ring/bench_circulax.py --variant va --n-stages 3

# VA test suite (109 tests):
pixi run python -m pytest tests/test_va_*.py --timeout=60
```

## Reproduction of the v901216 bug for further debugging

```python
import sys, jax, jax.numpy as jnp, dataclasses, importlib.util, tempfile
jax.config.update('jax_enable_x64', True)
sys.path.insert(0, 'tests')
from pathlib import Path
from fixtures.psp103_models import PSP103N_DEFAULTS, geom_settings
from circulax.va import compile_va, lower
from circulax.va.emitter import emit_source
from circulax.va.va_defaults import parse_va_defaults_expanded

settings = {**dict(PSP103N_DEFAULTS), **geom_settings(w=10e-6, length=1e-6)}
VA = Path('tests/data/va/psp103v4/psp103.va')
dump = compile_va(str(VA))
defs = parse_va_defaults_expanded(VA)
int_static = {k: int(v.default) for k, v in defs.items() if v.type_ == 'int'}
dev = lower(dump.modules[0], va_defaults=defs, collapse_nodes=True,
            static_params={**int_static, 'TYPE': 1}, class_name='PSP103N')
tmp = Path(tempfile.mkdtemp()) / 'psp103.py'
tmp.write_text(emit_source([dev]))
spec = importlib.util.spec_from_file_location('psp103', tmp)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

fn = {f.name for f in dataclasses.fields(mod.PSP103N) if f._field_type.name == '_FIELD'}
p = {k: float(defs[k].default) for k in fn if k in defs and defs[k].type_ == 'float'}
p.update({k: float(v) for k, v in settings.items() if k in fn})
p['_simparam_gmin'] = 0.0
n = mod.PSP103N(**p)

y = jnp.array([0.0001, 0.9, 0.0, 0.0, 0.0, 0.0])  # VD VG VS VB v_NOI i_NOII
f, q = n.solver_call(0.0, y, n)
print(f'I_D = {float(f[0]):.4e}')   # expected ~0; observed -1.1e+106
```

To debug: patch the emitted file at `tmp` to add `jax.debug.print(...)` before
the SSA you want to inspect, then re-import. The variables `v901216`, `v907945`,
`v907947`, `v907948`, `v911147`, `v911150`, `v911351` are the cascade chain.
