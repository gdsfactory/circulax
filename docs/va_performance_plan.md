# Plan: VA Performance Investigation & Hybrid OSDI+JAX Optimization

## Context

The circulax VA (Verilog-A lowered to JAX) path runs at 10,000–20,000 µs/step — **42–164× slower** than the OSDI native path (~100–500 µs/step). At this speed, brute-force OSDI parameter sweeps finish before a single VA backward pass, defeating the purpose of differentiable simulation.

The primary use cases are **parameter fitting** and **inverse design** — both scalar-loss-w.r.t.-parameters problems. This means we don't need a fast VA forward pass; we need OSDI-speed forward simulation with a mechanism to compute parameter gradients.

### Root cause analysis

The VA path already uses analytical Jacobians via `_fast_combined` (not `jax.jacfwd` through physics). The bottleneck is:

1. **XLA graph size**: PSP103's combined function (physics + Jacobian) compiles to thousands of XLA ops. Even one call is ~100× slower than OSDI's tight native loop doing the same arithmetic.
2. **Unoptimized MIR**: Using `--dump-unopt-mir-with-split` produces 3.5× more code than optimized MIR (42k vs 12k lines for PSP103).
3. **vmap-over-tangents overhead**: `_primal_and_jac_real` vmaps n unit tangent vectors through the custom JVP, producing n copies of `_fast_combined` in the trace for XLA to CSE — wasteful when the combined function already returns the full Jacobian matrix.
4. **Irreducible XLA vs native gap**: Even a perfectly optimized VA path will be ~10–50× slower than OSDI due to XLA interpreter overhead vs compiled C.

### Strategy

Given the irreducible XLA-vs-native gap and the gradient use cases, the highest-impact approach is:

- **Phase 1**: Profile with a simple diode to quantify each bottleneck component
- **Phase 2**: Direct assembly bypass — eliminate vmap-over-tangents overhead (~2–5× improvement)
- **Phase 3**: Hybrid OSDI forward + parameter gradients via implicit differentiation (the architectural solution — gets forward simulation to OSDI speed while enabling `jax.grad` w.r.t. parameters)
- **Phase 4**: Optimized MIR — audit previous attempts, determine if further optimization is possible

---

## Phase 1: Diode Profiling Harness

**Goal**: Quantify where time is spent in the VA path using a simple diode (sub-second JIT, fast iteration).

**Files to modify**:
- `benchmarks/diode/bench_circulax.py` — add `osdi` variant using `tests/data/va/diode.osdi`

**New files**:
- `benchmarks/diode/bench_profile.py` — detailed breakdown script

**Measurements**:
1. Isolated `_fast_combined` kernel time (one device, one call): `jax.jit(combined_fn).lower(args).compile()` then time 1000 calls
2. Isolated OSDI `osdi_eval` call time for the same device
3. `_primal_and_jac_real` time (includes vmap-over-tangents overhead)
4. Full `assemble_system_real` time (includes scatter/gather)
5. HLO operation count via `jax.jit(fn).lower(args).as_text()` — compare VA vs analytical
6. End-to-end µs/step for diode: analytical vs VA vs OSDI

**Expected output**: A table showing where the 164× gap breaks down into: (a) assembly overhead, (b) XLA graph execution, (c) vmap-over-tangents duplication, (d) irreducible XLA-vs-native gap.

**Estimated effort**: 0.5 days

---

## Phase 2: Direct Assembly Bypass for VA Components

**Goal**: When a VA component has a `combined_fn`, call it directly in assembly instead of going through the `_primal_and_jac_real` → `jax.jvp` → `custom_jvp` → `_fast_combined` → `j_f @ y_dot` roundabout.

**Current path** (`assembly.py:286-293`):
```
vmap(_primal_and_jac_real)  →  vmap(vmap(jax.jvp(g, v, eye[i])))
  → custom_jvp fires n times  →  _fast_combined called n times (XLA CSE may or may not merge)
  → j_f @ y_dot computed n times  →  transpose to get full Jacobian
```

**Proposed path**:
```
vmap(_fast_combined(v, params, t))  →  one call per device  →  returns (f, q, j_f, j_q) directly
j_eff = j_f + (alpha/dt) * j_q   →  done
```

**Files to modify**:
- `circulax/solvers/assembly.py` — add direct-combined path in `assemble_system_real` and `assemble_gc_real`
  - Check `hasattr(group, '_has_combined_fn')` or similar flag
  - Call `jax.vmap(group.combined_func)(v_locs, params)` directly
  - Extract `(f_l, q_l, j_f, j_q)` and compute `j_eff = j_f + (alpha/dt) * j_q`
  - Also optimize `assemble_residual_only_real` — call physics directly (already does this, no change needed)

- `circulax/compiler.py` — store a reference to the combined function on `ComponentGroup` when available
  - Detect `cls._fast_physics` having custom JVP and a `_fast_combined` closure
  - Or add an explicit attribute on the component class during `_install_custom_jvp`

- `bosdi/src/bosdi/circulax/va_component.py` — expose `_fast_combined` as a callable attribute on the class
  - After `_install_custom_jvp`, set `cls._combined_fn = staticmethod(_fast_combined)` (line 308 area)

**Key constraint**: This optimization only works for the Newton forward pass (where params are static / not being differentiated). When `jax.grad` is active w.r.t. parameters, the JVP-based path must still be used so parameter tangents flow correctly. The assembly code should check whether this is a gradient context or use a `jax.custom_vjp` wrapper.

**Expected speedup**: 2–5× if XLA wasn't already CSE-merging the n `_fast_combined` calls. Even if CSE was working, eliminates Python-level tracing overhead from the vmap-over-tangents.

**Verification**: Compare `j_eff` values from old path vs new path on the diode benchmark — must match to machine epsilon.

**Estimated effort**: 1–2 days

---

## Phase 3: Hybrid OSDI Forward + JAX Parameter Gradients

**Goal**: Run transient simulation at OSDI speed (~100–500 µs/step) and compute parameter gradients via implicit differentiation, avoiding the VA transient loop entirely.

### Architecture

```
                    FORWARD (OSDI speed)
compile_netlist → ComponentGroups (OSDI)
    → solve_dc()           → y_dc*          (OSDI Newton, ~100 µs/step)
    → setup_transient()    → y(t) trajectory (OSDI time-stepping)
    → loss = L(y(t_final))

                    BACKWARD (implicit diff)
∂loss/∂p = -(∂L/∂y) · J⁻¹ · (∂F/∂p)
             ↑ JAX AD   ↑ OSDI   ↑ finite-diff through OSDI
                                    or VA single-point eval
```

### Sub-approach A: DC Parameter Fitting (simpler, implement first)

For DC operating point fitting: `F(y*, p) = 0` → implicit function theorem gives `dy*/dp = -J(y*)⁻¹ · ∂F/∂p(y*)`.

**New file**: `circulax/solvers/sensitivity.py`
- `dc_parameter_gradient(groups, solver, y_star, loss_fn, param_indices, *, method="fd")`:
  1. Evaluate `∂L/∂y` via `jax.grad(loss_fn)(y_star)` (cheap, small vector)
  2. Solve `J^T · λ = ∂L/∂y` using the already-factored KLU from DC solve (one adjoint solve)
  3. Compute `∂F/∂p` via finite differences through OSDI: perturb each param, eval `osdi_eval`, difference. Cost: `n_params × n_devices` OSDI evals (~µs each).
  4. Return `∂loss/∂p = -λ^T · ∂F/∂p`

**Integration**: Wrap the DC solve + loss in `jax.custom_vjp`:
- `fwd`: run OSDI DC solve, return y* and residuals for backward
- `bwd`: compute parameter gradient via the implicit diff formula above

### Sub-approach B: Transient Parameter Fitting (adjoint method)

For transient inverse design: compute gradients through the time-stepping loop.

**New file**: `circulax/solvers/adjoint.py`
- `transient_parameter_gradient(groups, solver, y_trajectory, ts, loss_fn, param_indices, *, method="fd")`:
  1. Forward: run OSDI transient, save `y(t_k)` at checkpoints
  2. Evaluate `∂L/∂y(t_final)` via `jax.grad(loss_fn)`
  3. Backward in time: solve adjoint ODE `λ'(t) = -J(y(t))^T · λ(t)` using OSDI Jacobians at each checkpoint
  4. Accumulate `∂loss/∂p = -Σ_k λ(t_k)^T · ∂F/∂p(y(t_k), p)` where `∂F/∂p` is via finite differences through OSDI
  5. Return total parameter gradient

**Alternative for ∂F/∂p**: Instead of finite differences, evaluate the VA-lowered function at each checkpoint point. This gives exact analytical gradients but requires one VA evaluation per checkpoint — at 10 ms per eval, 100 checkpoints = 1 second total, which is acceptable.

**Integration with Diffrax**: Register a `jax.custom_vjp` on the `setup_transient` / `run_fn` that:
- Forward: runs the standard OSDI transient (already fast)
- Backward: runs the adjoint solver using OSDI for `J^T` and FD/VA for `∂F/∂p`

### Cost estimate for the hybrid approach

| Operation | Cost |
|-----------|------|
| Forward OSDI transient (20k steps) | ~2–10 s |
| Adjoint ODE backward (20k steps) | ~2–10 s (same as forward) |
| `∂F/∂p` via FD, 10 params, 100 checkpoints | ~100 ms |
| `∂F/∂p` via VA, 100 checkpoints | ~1 s |
| **Total gradient** | **~5–21 s** |
| Current VA forward+backward | ~400+ s |
| **Speedup** | **~20–80×** |

**Estimated effort**: 1–2 weeks (DC: 2–3 days, transient adjoint: 1 week)

---

## Phase 4: Optimized MIR — Audit & Improvements

**Goal**: Audit the current MIR optimization state and determine whether further optimization is possible without breaking differentiability.

**Background — previous attempts**: There have been multiple prior efforts to improve the MIR optimization pipeline. The current approach (`--dump-unopt-mir-with-split`) was arrived at deliberately: earlier attempts with fully-optimized MIR (`--dump-json` with SCCP+GVN+ADCE+phi-merge) encountered issues — the JSON path was found to be broken, and there were concerns about phi-collapse breaking nested conditional patterns in complex models like PSP103. The current unopt-with-split path was deemed to have folded sufficiently to preserve differentiability while maintaining correctness. This phase is a **fresh audit** of that decision with the benefit of hindsight and the diode model as a simpler test case.

**Audit scope** (produce a written report):
1. **Current state of `--dump-json`**: Is the JSON path in openvaf-r still broken? What specifically fails? Is it a parsing issue, a phi-collapse correctness issue, or something else?
2. **What each optimization pass does to differentiability**: SCCP folds constants (safe for AD), GVN eliminates redundant subexpressions (safe), ADCE removes dead code (safe), phi-merge collapses phi chains (may change control-flow structure — needs verification). Which passes, if any, actually threaten gradient correctness?
3. **Quantitative comparison on the diode**: Test `compile_va` (optimized) vs `compile_va_unopt_with_split` — compare HLO op count, JIT time, and µs/step. If optimized works for the diode, try PSP103.
4. **Is there a middle ground?** Can openvaf-r apply SCCP+GVN+ADCE without phi-merge? Does such a mode exist or would it need to be added?
5. **Recommendation**: Based on the audit, recommend whether to stay with unopt-with-split, move to a partially-optimized mode, or adopt fully-optimized MIR.

**Key files to examine**:
- `/home/cdaunt/code/openvaf-r/OpenVAF/openvaf/openvaf-driver/src/main.rs` — dump-json code path
- `bosdi/src/bosdi/va/ir_client.py` — openvaf-r invocation and JSON parsing
- `bosdi/src/bosdi/va/binding.py` — `compile_va` vs `compile_va_unopt_with_split` implementations

**Expected impact**: If optimized MIR is viable: 2–4× reduction in XLA graph size → proportional speedup in both standalone VA and the hybrid's `∂F/∂p` computation. If not, the audit documents why and identifies the specific blockers for future work.

**Estimated effort**: 1–2 days for audit + report, additional time if openvaf-r changes are needed

---

## Execution Order

```
Phase 1 (profiling)  ──→  Phase 2 (assembly bypass)  ──→  Phase 3A (DC implicit diff)
                                                      ──→  Phase 4 (optimized MIR audit)
                                                                    ↓
                                                           Phase 3B (transient adjoint)
```

Phase 1 data informs Phase 2 priorities. Phase 3A (DC) is a stepping stone to Phase 3B (transient). Phase 4 is independent and benefits all paths.

## Verification

- **Phase 1**: Timing table for diode: analytical vs VA vs OSDI, component breakdown
- **Phase 2**: `j_eff` values match old path to machine epsilon; µs/step improves measurably on diode benchmark
- **Phase 3A**: DC parameter gradient matches finite-difference of loss (perturb param → full DC solve → loss → difference) to ~1e-6 relative tolerance
- **Phase 3B**: Transient gradient matches brute-force FD sweep to ~1e-5 relative tolerance
- **Phase 4**: Written audit report; diode DC point and transient waveform match OSDI reference to ~1e-10 if optimized MIR is tested
