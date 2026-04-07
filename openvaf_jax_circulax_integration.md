# openvaf_jax → circulax Integration Notes

Analysis of integrating `openvaf_jax` (Verilog-A to JAX compiler) with circulax's
`CircuitComponent` / `eqx.Module` architecture. Based on code review of
`openvaf_jax/` in the vajax project.

---

## 1. What openvaf_jax provides

`openvaf_jax` compiles `.va` (Verilog-A) files to JAX functions via a Rust/LLVM
pipeline (OpenVAF). It has zero dependencies on the rest of vajax and is
extractable as a standalone package (~2–3 hours of import changes).

The two generated functions per model:

```python
# Computed once per device instance at circuit assembly time
init_fn(all_params: Array[N_init]) -> (cache: Array[N_cache], collapse: Array[N_collapse])

# Called at every NR step
eval_fn(shared_params, device_params, shared_cache, device_cache,
        simparams, limit_state_in, limit_funcs)
    -> (res_resist, res_react, jac_resist, jac_react,
        lim_rhs_resist, lim_rhs_react, ss_resist, ss_react, limit_state_out)
```

`eval_fn` already has the `shared / device` parameter split built in —
designed for exactly the vmap-over-instances pattern.

---

## 2. vmap-over-instances architecture

For N instances of a device type (e.g. all NMOS in a circuit):

| Array | Varies per instance | vmap axis |
|---|---|---|
| `shared_params` | No — model card constants | `None` (broadcast) |
| `device_params` | Yes — W, L, multiplier | `0` |
| `shared_cache` | No — depends only on model card | `None` (broadcast) |
| `device_cache` | Yes — depends on device params | `0` |
| `simparams` | No — circuit-global (t, gmin) | `None` (broadcast) |
| voltages (inside eval) | Yes — per instance | `0` |

The vmapped call pattern:

```python
batched_eval = vmap(eval_fn, in_axes=(None, 0, None, 0, None, None, None))
f, q, *_ = batched_eval(
    shared_params,          # (n_shared,)
    batched_device_params,  # (N, n_device)
    shared_cache,           # (n_sc,)
    batched_device_cache,   # (N, n_dc)
    simparams,              # (n_sp,)
    zeros_limit_state,
    None,
)
# f: (N, n_nodes)   q: (N, n_nodes)
```

### eqx.Module field layout

Per-instance fields (vmapped leafs):

```python
class NMOS(CircuitComponent):
    ports: ClassVar = ("d", "g", "s", "b")

    device_params: jax.Array   # shape (n_device,)  e.g. [W, L, nf]
    device_cache:  jax.Array   # shape (n_dc,)  precomputed at assembly
```

Model-card data (`shared_params`, `shared_cache`) is circuit-level state, not
instance-level. It lives outside the module, closed over in `_fast_physics`.

### `cache_split` computation

`translate_eval` accepts `cache_split=(shared_cache_indices, device_cache_indices)`.
Determines which cache entries depend only on model-card params (shared) vs
instance params (per-device). Can be computed automatically by tracing `init_fn`
sensitivity via `jax.jacfwd` at model-registration time (~50–80 lines).

---

## 3. The analytical Jacobian

`eval_fn` returns `jac_resist` and `jac_react` — sparse COO arrays with sparsity
defined by `eval_meta["jacobian_indices"]` (list of `(row_idx, col_idx)` tuples).

**Critical: this is `d(residuals)/d(voltages)` only.** It does not cover
`d(residuals)/d(params)`.

- Useful for: NR solver Jacobian assembly (stamping MNA matrix)
- Not useful for: optimization gradient w.r.t. device parameters

The NR Jacobian and the optimization gradient are fundamentally different
operations and should be handled via separate code paths (see §5).

---

## 4. Cache and backpropagation — the core tension

For circuit optimization via `jax.grad(loss)(circuit_module)`:

```
d(loss)/d(params) = d(loss)/d(V*) · [d(f)/d(V)]⁻¹ · d(f)/d(params)
```

The `d(f)/d(params)` factor requires tracing through `eval_fn` **and** `init_fn`
(because `cache` depends on `params`). If `device_cache` is stored as a static
module field, JAX's backward pass misses the `params → cache → f` pathway and
produces **silently wrong gradients**.

| Approach | NR speed | Gradient correctness |
|---|---|---|
| Static cache (baked field) | Fast | Wrong — misses cache pathway |
| Fused `init_fn + eval_fn` per step | Slow | Correct |
| `custom_vjp` rematerialization | Fast | Correct |

The `custom_vjp` approach is the right answer.

---

## 5. Recommended design

### 5a. `custom_vjp` on `_fast_physics`

```python
@jax.custom_vjp
def _physics(vars_vec, device_params, device_cache, t):
    # FORWARD: pre-computed cache — fast NR, no rematerialization cost
    result = eval_fn(shared_params, device_params,
                     shared_cache, device_cache, simparams(t), ...)
    return result[0], result[1]  # f, q

def _physics_fwd(vars_vec, device_params, device_cache, t):
    f, q = _physics(vars_vec, device_params, device_cache, t)
    return (f, q), (vars_vec, device_params, t)   # cache NOT saved

def _physics_bwd(saved, g):
    vars_vec, device_params, t = saved
    # BACKWARD: recompute cache from params for correct d(f)/d(params)
    def fused(vv, dp):
        c = init_fn(dp)
        r = eval_fn(shared_params, dp, shared_cache, c, simparams(t), ...)
        return r[0], r[1]
    _, vjp_fn = jax.vjp(fused, vars_vec, device_params)
    g_vars, g_params = vjp_fn(g)
    return g_vars, g_params, jnp.zeros_like(device_cache), jnp.zeros(())

_physics.defvjp(_physics_fwd, _physics_bwd)
```

**Forward pass** (every NR iteration): uses pre-computed `device_cache` — fast.

**Backward pass** (once per `jax.grad` call): recomputes `init_fn(params)` inside
the VJP — correct `d(f)/d(params)` gradients. This is a one-time JIT cost per
optimization step, not per NR iteration.

### 5b. Separate NR Jacobian path

`custom_vjp` and `custom_jvp` cannot coexist on the same function in JAX. This
means the NR Jacobian **cannot** use the `custom_jvp` analytical shortcut.

The clean resolution: **do not use `jacfwd(_fast_physics)` for the NR Jacobian**.
Instead, expose the analytical Jacobian via a separate ClassVar or return it from
`_fast_physics` as optional outputs.

Two options:

**Option A** — extend `_fast_physics` return signature:
```python
# Returns (f, q) for optimization path; (f, q, J_resist, J_react) for NR path
_fast_physics(vars_vec, params, t, return_jacobian=False) -> tuple
```

**Option B** — add `_fast_jacobian` ClassVar:
```python
_fast_jacobian: ClassVar[Any] = None  # (vars_vec, params, t) -> (f, q, J_f, J_q)
```
NR solver checks: use `_fast_jacobian` if available, else fall back to `jacfwd(_fast_physics)`.

Option B is architecturally cleaner — it separates the NR path (analytical,
sparse COO) from the optimization path (`_fast_physics` with `custom_vjp`).

---

## 6. Internal nodes

`eval_meta["internal_nodes"]` lists nodes that appear in the residual vector but
have no corresponding terminal. Examples: BSIM4 source/drain resistance nodes,
PSP103's `NOI` noise correlation node.

Most are collapsible at compile time via `module.collapsible_pairs` — zero-resistance
nodes that OpenVAF can eliminate. For the rest:

- **PSP103 `NOI` node**: has 1/mig conductance to ground (mig=1e-40 → G=1e40).
  If V(NOI) ≠ 0, residual = 1e40·V(NOI) — numerical explosion.
  Resolution: zero-initialize NOI and mask its residual row in `f_vec` before
  returning from `_fast_physics`.

- **General approach**: internal nodes that cannot be collapsed should be exposed
  as additional entries in the residual vector, handled by the MNA stamping layer.
  They do not map to circulax `states` (which are integration variables), but to
  algebraic unknowns in the DAE.

---

## 7. Port ordering and simparams

**Port ordering**: `eval_meta["terminals"]` gives the node ordering used by
`f_vec`/`q_vec`. The `ports` ClassVar must match this ordering. For standard
SPICE devices it follows canonical ordering (d, g, s, b for MOSFET).

**simparams**: `eval_meta["simparam_indices"]` gives the dynamic layout per model.
- `simparams[0]` = analysis_type (0=DC, 1=AC, 2=transient, 3=noise) — always present
- `simparams[1]` = gmin — common convergence aid
- Additional entries (tnom, abstol, $abstime) vary by model
- `t` from circulax maps to `$abstime` in simparams

---

## 8. Extracting openvaf_jax as a standalone package

Changes required:
1. Change relative imports (`from ..mir` → `from openvaf_jax.mir`)
2. Add `pyproject.toml` with deps: `jax`, `openvaf_py` (Rust binary)
3. Update test paths (currently reference `../../vendor/VACASK`)

The `openvaf_py` Rust extension must be distributed alongside. It wraps OpenVAF's
LLVM-based Verilog-A compiler — building it requires Rust + LLVM (the main
installation friction for end users).

---

## 9. Work estimate

| Task | Effort |
|---|---|
| Extract `openvaf_jax` to standalone package | 3–5 hours |
| `cache_split` computation helper (jacfwd sensitivity) | 1 day |
| `CircuitComponent` factory from `.va` file | 2–3 days |
| `custom_vjp` rematerialization wrapper | 1 day |
| `_fast_jacobian` ClassVar + NR solver integration | 1–2 days |
| Internal node handling (collapse + NOI masking) | 1–2 days |
| Tests against known VA models (resistor, diode, BSIM4) | 1–2 days |
| **Total** | **~8–12 days** |

The new code is primarily glue (~500–800 lines). The compiler infrastructure
(`mir/`, `codegen/`, SSA analysis) is unchanged and opaque to the integration.

---

## 10. Key metadata from translate_eval

Everything needed to generate a `CircuitComponent` subclass:

```python
eval_fn, eval_meta = translator.translate_eval(model_card_params, temperature=T)

eval_meta["terminals"]          # port names in residual order
eval_meta["internal_nodes"]     # dict of non-terminal nodes
eval_meta["node_names"]         # all node names (terminals + internal)
eval_meta["jacobian_indices"]   # list of (row_idx, col_idx) for jac_resist/jac_react
eval_meta["shared_inputs"]      # validated shared param array (ready to use)
eval_meta["shared_indices"]     # which param positions are model-card-level
eval_meta["voltage_indices"]    # which param positions are voltages
eval_meta["simparam_indices"]   # dict of simparam name -> array index
eval_meta["simparam_count"]     # length of simparams array
```
