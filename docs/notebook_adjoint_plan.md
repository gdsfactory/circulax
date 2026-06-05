# Plan: Update PSP103 Ring Param Fitting Notebook to Use Adjoint Method

## Context

The existing notebook `examples/inverse_design/05_psp103_ring_param_fitting.ipynb` uses full JAX autodiff (`jax.value_and_grad`) through the VA-lowered PSP103 model. This approach:
- Requires JIT-compiling the entire backward pass through PSP103 eval + Newton solve + trapezoidal integrator
- Has caused OOM issues on machines with limited memory (the XLA backward graph is very large)
- Takes ~5 min for first JIT compilation, ~15,000 µs/step per forward evaluation
- Only works with VA-lowered models (cannot fit OSDI-only models like BSIM4)

The new `transient_parameter_sensitivity()` in `circulax/solvers/adjoint.py` provides an alternative: run the forward simulation with OSDI at ~200 µs/step, then compute parameter gradients via discrete adjoint + finite differences. This eliminates the large backward graph entirely.

## Key Architectural Change

| Aspect | Current (VA autodiff) | Proposed (OSDI + adjoint FD) |
|--------|----------------------|------------------------------|
| Forward model | VA-lowered PSP103 (JAX) | OSDI-compiled PSP103 (native C) |
| Forward speed | ~15,000 µs/step | ~200 µs/step |
| Gradient method | `jax.value_and_grad` through full graph | Discrete adjoint + FD through OSDI |
| Memory | Full backward graph (OOM risk) | One linear solve per timestep (no backward graph) |
| JIT compile | ~5 min first run | ~1s (OSDI is pre-compiled) |
| Gradient accuracy | Exact (machine epsilon) | FD-limited (~1e-6 relative) |
| Parameter flexibility | Any JAX-traced param | Any OSDI model parameter |

## Implementation Plan

### Cell 1-2: Imports — minimal changes

Replace VA-specific imports with OSDI imports:

```python
# REMOVE:
# from circulax.va import compile_va_unopt_with_split, lower
# from circulax.va.emitter import emit_source
# from circulax.va.va_defaults import parse_va_defaults_expanded

# ADD:
from circulax.solvers import transient_parameter_sensitivity
from circulax.solvers import dc_parameter_sensitivity  # optional, for DC-only fitting
```

Remove the JAX persistent compilation cache setup (no longer needed — OSDI has no JIT).

### Cell 3-4: Model Setup — replace VA lowering with OSDI loading

Replace the entire VA lowering pipeline (compile_va → lower → emit_source) with OSDI model loading:

```python
from fixtures.psp103_models import make_psp103_descriptors, PSP103N_DEFAULTS, PSP103P_DEFAULTS, geom_settings

psp103n_desc, psp103p_desc = make_psp103_descriptors()
```

Key files:
- `tests/fixtures/psp103_models.py` — contains `make_psp103_descriptors()` which returns `(OsdiModelDescriptor, OsdiModelDescriptor)` for NMOS and PMOS
- `circulax/components/osdi/compiled/psp103v4_psp103.osdi` — pre-compiled OSDI binary
- The descriptors carry `param_names`, `_name_to_idx`, and the model ID needed by `transient_parameter_sensitivity`

The same 10 `PARAM_NAMES` (VFBO, NSUBO, TOXO, UO, BETN, THESAT, MUEO, THEMU, AX, RSW1) can be used. These are standard OSDI parameter names.

### Cell 5-6: Circuit Build — use OSDI ring builder

Replace the manual VA netlist construction with the existing OSDI ring builder:

```python
from scripts.ring_one_case import build_netlist
groups, sys_size, port_map = build_netlist(c_load=0.0, n_stages=9)
```

Or replicate the netlist construction using `circulax.osdi_component()` instead of `cls_n`/`cls_p`:

```python
from circulax import compile_netlist
from circulax.components.electronic import Resistor, SmoothPulse, VoltageSource

# psp103n_desc, psp103p_desc are OsdiModelDescriptors
# Instance settings come from PSP103N_DEFAULTS + geom_settings
```

DC solve remains the same (gmin stepping from VDD/2).

### Cell 7-8: Reference Data — faster simulation

The reference transient now runs at OSDI speed. No warmup JIT needed:

```python
run_fn = setup_transient(groups, solver, transient_solver=TrapFactorizedTransientSolver)
sol_ref = run_fn(t0=0.0, t1=T_END, dt0=DT, y0=y0, saveat=saveat, ...)
```

Expected: ~4-8 seconds for 20,000-step transient at N=9 (vs minutes for VA).

### Cell 9-10: Optimization Loop — adjoint replaces `jax.value_and_grad`

This is the core change. Replace:

```python
# OLD: Full autodiff
loss_val, grads = jax.value_and_grad(loss_fn)(log_params)
```

With:

```python
# NEW: OSDI forward + adjoint backward
# 1. Update parameters on the OSDI group
groups_current = update_osdi_group_params(groups, "psp103_0", param_names, current_values)

# 2. Run forward transient (OSDI speed)
sol = run_fn(t0=0.0, t1=T_END, dt0=DT, y0=y0, saveat=saveat, ...)

# 3. Compute loss
loss_val = loss_fn(sol.ys, sol.ts)

# 4. Compute parameter gradients via discrete adjoint + FD
grad_dict = transient_parameter_sensitivity(
    component_groups=groups_current,
    solver=solver,
    y_trajectory=sol.ys,
    ts=sol.ts,
    loss_fn=lambda ys, ts: jnp.mean((ys[loss_start:, out_idx] - v_ref_window) ** 2),
    osdi_group_key="psp103_0",  # key for the NMOS OSDI group
    param_names=list(PARAM_NAMES),
    model_descriptor=psp103n_desc,
    eps=1e-6,
)

# 5. Assemble gradient vector for optax
grads = jnp.array([grad_dict[name].sum() for name in PARAM_NAMES])
```

**Parameter update mechanism**: The OSDI group stores parameters in a `(N_devices, N_params)` array. To update a parameter across all devices:

```python
def update_osdi_group_params(groups, group_key, param_names, values, descriptor):
    """Update OSDI group parameters and rebuild the batch handle."""
    group = groups[group_key]
    new_params = group.params.copy()
    for name, val in zip(param_names, values):
        col = descriptor._name_to_idx[name.lower()]
        new_params = new_params.at[:, col].set(float(val))
    # Rebuild handle for Tier-3 eval
    new_group = group.replace(params=new_params)
    new_group = new_group.rebuild_handle()  # if available
    return {**groups, group_key: new_group}
```

**Log-space parameterization**: Keep the same log-space approach. The adjoint returns gradients in linear space; chain rule for log-space: `∂loss/∂(log|p|) = ∂loss/∂p · |p|`.

### Key Implementation Details

**1. Loss function signature**: `transient_parameter_sensitivity` accepts either:
- `loss_fn(y_final)` — single argument, final state only
- `loss_fn(y_trajectory, ts)` — two arguments, full trajectory

For waveform MSE fitting, use the two-argument form.

**2. Gradient aggregation**: The adjoint returns per-device gradients `{name: array(n_devices,)}`. Since all NMOS instances share the same parameters, sum across devices: `grad_total = grad_dict[name].sum()`.

**3. OSDI parameter re-baking**: After each Adam step, the OSDI group's internal handle must be rebuilt to reflect the new parameter values. Check if `OsdiComponentGroup` has a `rebuild_handle()` or equivalent method. If not, re-run `compile_netlist` with updated settings (more expensive but correct).

Alternatively, if the batch handle doesn't need rebuilding (parameters are read from `group.params` directly), just update the params array in-place.

**4. DC re-solve per iteration**: Unlike the VA notebook which re-uses the same `y0` (since the DC solve is inside the loss function), the adjoint approach requires:
- Either re-solving DC after each parameter update (correct but adds ~5ms per step)
- Or keeping y0 fixed and accepting a small bias (faster, acceptable for small perturbations)

Recommendation: re-solve DC if parameters change significantly (>5% from previous), otherwise reuse.

**5. Checkpoint density**: `transient_parameter_sensitivity` uses every saved checkpoint. With `N_SAVE=3000` and 10 params and 9 devices, the FD loop does `3000 × 10 × 9 = 270,000` OSDI calls. At ~1 µs each, this is ~0.27 seconds. Acceptable.

For a shorter adjoint backward sweep, reduce `N_SAVE` (e.g., 300 checkpoints). The adjoint accuracy degrades gracefully with fewer checkpoints since it uses linear interpolation between saved states.

### Cell 11-13: Analysis — keep same visualizations

The loss convergence, parameter trajectory, and waveform overlay plots remain unchanged. Only the data feeding them changes.

### New Cell: Performance Comparison

Add a summary cell comparing old vs new approach:

```python
print(f"Forward transient: {tran_time:.1f}s ({tran_time/N_STEPS*1e6:.0f} µs/step)")
print(f"Adjoint gradient:  {adjoint_time:.1f}s")
print(f"Total per Adam step: {per_step:.1f}s")
print(f"No JIT compilation required")
```

## Files to Modify

| File | Change |
|------|--------|
| `examples/inverse_design/05_psp103_ring_param_fitting.ipynb` | Major rewrite: VA→OSDI, autodiff→adjoint |

## Files to Reference (read-only)

| File | Role |
|------|------|
| `circulax/solvers/adjoint.py` | `transient_parameter_sensitivity()` API |
| `circulax/solvers/sensitivity.py` | `dc_parameter_sensitivity()` API, `_resolve_param_cols` |
| `tests/fixtures/psp103_models.py` | `make_psp103_descriptors()`, PSP103 defaults |
| `scripts/ring_one_case.py` | `build_netlist()` — OSDI ring builder reference |
| `tests/test_adjoint.py` | Working examples of `transient_parameter_sensitivity` usage |
| `tests/test_sensitivity.py` | Working examples of `dc_parameter_sensitivity` usage |

## Verification

1. **Forward simulation matches**: OSDI ring at N=9 should oscillate at ~289 MHz (same as VA)
2. **Gradient sanity**: Initial gradients should be non-zero for all 10 parameters
3. **Convergence**: Adam should reduce loss monotonically, recovering parameters within ~5% in 60 steps (may need tuning LR since FD gradients have different scale/noise than exact gradients)
4. **Memory**: Should not OOM — no backward graph, just one linear solve per checkpoint
5. **Speed**: Each Adam step should take ~10-20 seconds (forward transient ~8s + adjoint ~2-10s) vs minutes for the VA approach

## Potential Issues

1. **OSDI parameter update**: Need to verify how `OsdiComponentGroup` handles parameter changes between optimization steps. May need to re-compile the netlist or rebuild the batch handle.
2. **Learning rate tuning**: FD gradients may have different magnitude than exact gradients, requiring LR adjustment.
3. **Gradient noise**: FD with `eps=1e-6` gives ~1e-6 relative accuracy. For parameters spanning many orders of magnitude (NSUBO~3e23), the absolute FD step should be scaled appropriately. The `eps` parameter in `transient_parameter_sensitivity` already uses relative stepping: `h = eps * max(|p|, 1)`.
4. **Adjoint timestep assumption**: The adjoint assumes Backward Euler. The forward solver uses `TrapFactorizedTransientSolver` (trapezoidal). This mismatch introduces O(dt) bias. For dt=50ps this should be negligible, but worth noting. Alternative: use `FactorizedTransientSolver` (pure BE) for the forward pass to get exact adjoint.
