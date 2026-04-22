# Request: vmap-friendly `osdi_eval` in bosdi

**Audience:** bosdi maintainer.
**Status:** open feature request / integration note from the circulax side.

## Summary

`osdi_eval` currently uses `vmap_method="sequential"` in its
`jffi.ffi_call` registration
(`src/osdi_jax.py:61`):

```python
return jffi.ffi_call("OsdiEvalCpu", out_shapes, vmap_method="sequential")(
    m_id, v, p, s
)
```

Under `jax.vmap` — the canonical JAX pattern for circulax's
parameter-sweep workloads — this means JAX loops over the batch
dimension and calls the FFI once per replica. So a batched run over
4 rings × 5 Newton iters/step × 4000 transient steps takes
**80 000 FFI crossings**, most of which is pure overhead (the 9
PSP103 devices evaluated per call is a small amount of actual work).

All the machinery needed to collapse this into one FFI call per
Newton iter is already on the bosdi side:

- `osdi_shim.cpp:88` reads `num_devices = v_dims[0]`, so the C++
  handler doesn't care whether the leading dimension is "N devices
  from one replica" or "N devices × B replicas from a vmap flatten".
- `batched_osdi_eval_ffi` in Rust already uses the "Rayon zipper" as
  the comment on `osdi_shim.cpp:49` states, so the per-device work
  is already parallelised inside a single batched call.

The missing piece is the same Python-side change klujax-rs just
landed (v0.1.7, April 2026): a vmap rule (or switching
`vmap_method`) that flattens the replica axis into the leading
device axis so the existing batched C++ handler sees it.

Below: the measured cost, the two-tier ask, and a pointer to the
klujax-rs commit for reference.

## Measured cost

From the earlier vmap harness (since folded into `benchmarks/ring/`), 9-stage PSP103 ring
oscillator, `TrapRefactoring` + klu_rs_split, 200 ns sim at dt = 50 ps
(= 4000 timesteps, ~5 Newton iters/step):

| Batch | Wall (s) | µs / ring / step |
|-------|----------|------------------|
| 4 | 13.7 | 858 |
| 8 | 27.2 | 851 |

Linear in batch. We verified via isolated testing that klujax-rs's
own Rayon parallelism is firing correctly — the wall isn't growing
because of KLU. KLU on a 34 × 34 sparse system is ~a few µs of the
850 µs per step. **The bulk of the 850 µs/step is the sequential FFI
loop into `osdi_eval`**, one call per replica. With a vmap batching
rule this becomes **one call per Newton iter** regardless of batch,
and Rayon's `par_iter` inside `batched_osdi_eval_ffi` parallelises
all 4 × 9 = 36 devices in one go.

Expected impact: per-ring cost at batch = 8 drops from ~850 µs/step
to approximately `serial_step / (batch * min(batch × N_dev, n_cpu))
+ ffi_overhead`. For our 24-core box with batch = 8 × 9 = 72
devices, that's roughly 3× faster on this small ring, and unbounded
better on larger circuits where per-device work dominates.

## What's required

### Tier 1 — switch `vmap_method` or register a batching rule

The simplest working form is to change line 61 of `src/osdi_jax.py`
from `vmap_method="sequential"` to a concrete rule. Two options that
would do the right thing:

**Option A** — tell JAX to expand the batched axis as an extra
leading dim and let the existing C++ handler deal with it:

```python
return jffi.ffi_call(
    "OsdiEvalCpu", out_shapes,
    vmap_method="expand_dims",
)(m_id, v, p, s)
```

This requires `batched_osdi_eval_ffi` to either handle rank-3 `v`
(shape `(B, N_dev, N_pins)`) or for the Python side to reshape on
entry and un-reshape on exit. The C++ handler today expects rank-2,
so this needs a thin reshape wrapper. ~5 lines.

**Option B** — register an explicit batching rule on
`osdi_eval`'s primitive (the `@jax.custom_jvp` decorator wraps a
primitive; we can attach a `batching.primitive_batchers` rule to
it). The rule flattens `(B, N_dev, …)` into `(B * N_dev, …)`,
calls the existing rank-2 path, then reshapes outputs back:

```python
from jax.interpreters import batching

def _osdi_eval_batch(batched_args, batch_dims):
    model_id, v, p, s = batched_args
    bd_m, bd_v, bd_p, bd_s = batch_dims
    if bd_m is not None:
        raise ValueError("Cannot batch over model_id")

    size = next(x.shape[d] for x, d in zip(batched_args, batch_dims)
                if d is not None)

    # Normalise: broadcast non-batched args, move batch to axis 0.
    def _norm(x, d):
        if d is None:
            return jnp.broadcast_to(x[None], (size,) + x.shape)
        return jnp.moveaxis(x, d, 0)

    v = _norm(v, bd_v)   # (B, N_dev, N_pins)
    p = _norm(p, bd_p)   # (B, N_dev, N_params)
    s = _norm(s, bd_s)   # (B, N_dev, N_states)

    B, N_dev = v.shape[0], v.shape[1]
    v_flat = v.reshape(B * N_dev, -1)
    p_flat = p.reshape(B * N_dev, -1)
    s_flat = s.reshape(B * N_dev, -1)

    cur, cond, chg, cap, new_s = _osdi_eval_raw(model_id, v_flat, p_flat, s_flat)

    # Reshape outputs back to (B, N_dev, …).
    cur = cur.reshape(B, N_dev, -1)
    chg = chg.reshape(B, N_dev, -1)
    new_s = new_s.reshape(B, N_dev, -1)
    cond = cond.reshape(B, N_dev, -1)
    cap = cap.reshape(B, N_dev, -1)

    return (cur, cond, chg, cap, new_s), (0, 0, 0, 0, 0)

# Attach to the primitive that backs osdi_eval.
batching.primitive_batchers[osdi_eval_p] = _osdi_eval_batch
```

This is the same shape as klujax-rs's `_refactor_and_solve_batch`
(klujax_rs.py:1144). No C++ changes needed — the existing handler
reads `num_devices = v_dims[0]`, which happily becomes `B * N_dev`
after the flatten.

Either option should drop the 80 000 FFI calls/run down to
~20 000 (one per Newton iter, independent of batch size), and let
Rayon inside `batched_osdi_eval_ffi` parallelise `B × N_dev`
devices in one go.

### Tier 2 — residual-only FFI entrypoint (follow-up)

Inside circulax's Newton inner loop, `assemble_residual_only_*`
(used on every iter after the first when the Jacobian is cached)
needs only `cur` and `chg`. Today it still receives `cond` and
`cap` from `osdi_eval`, then throws them away. Computing the
Jacobian entries (`cond`, `cap`) is typically the majority of the
OSDI evaluator's work for BSIM4/PSP103-sized models.

A second FFI target like `batched_osdi_residual_eval_ffi` that
skips the ∂/∂V entries and returns just `(cur, chg, new_state)`
would let circulax cut per-step OSDI work roughly in half for
strongly-nonlinear transients where we do 4–5 Newton iters per
step (only the first iter needs the full stamp; subsequent iters
can reuse a frozen Jacobian). Not urgent; Tier 1 is the main ask.

### Tier 3 — Rayon over the device loop in non-batched path

Since the Rust side already has "Rayon zipper" in
`batched_osdi_eval_ffi`, this may already be done — flagging it
just in case. For thousand-device circuits (BSIM4 stacks,
photonic netlists with many resonators) this is the dominant
speedup. Our ring is too small for this one to show up.

## Reference: klujax-rs's analogous change

klujax-rs 0.1.7 shipped the same shape of fix for its
`refactor_and_solve_f64_p` primitive:

- Python-side: `_refactor_and_solve_batch` batching rule
  (klujax_rs.py:1144).
- Registration: `batching.primitive_batchers[refactor_and_solve_f64_p]
  = _refactor_and_solve_batch` (klujax_rs.py:1219).
- No Rust or C++ changes needed — the existing batched backend
  already iterated over `n_lhs` and already had rayon.

The net effect on circulax was immediate and clean: vmap'd
transients that previously raised `NotImplementedError` now
complete end-to-end. Our integration pulled in the change with zero
circulax-side modifications because we already called the fused
primitive (`refactor_and_solve_jacobian`) that the rule is
registered on.

We'd expect the same pattern here: circulax already calls
`osdi_eval` as an `ffi_call`, so a batching rule on that primitive
lands for free on our side.

## Happy to help

- Test a Tier-1 draft against our ring-oscillator benchmark
  (`benchmarks/ring/`) and report per-step cost.
- Share the circulax test setup that exercises vmap'd
  `refactor_and_solve` + vmap'd `osdi_eval` end-to-end, so you can
  reproduce.
- Draft a PR against bosdi if useful — the Option B batching rule
  is maybe 30 lines of Python, no C++ changes.

Thanks — shipping the rule would close the remaining gap between
circulax's vmap'd workflows and VACASK single-circuit wall time on
larger circuits, and open the door to parameter sweeps as a
first-class circulax workload.
