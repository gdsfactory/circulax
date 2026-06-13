# Fitting PSP103 Process Parameters to Ring Oscillator Measurements

Traditional compact-model parameter extraction is a manual, iterative process: an engineer
sweeps one parameter at a time against measured I-V or C-V curves, re-simulates, and repeats
until the model "looks right." For a production model like PSP103 (783 parameters), this can
take weeks.

This notebook demonstrates the OSDI workflow for gradient-based parameter fitting. The PSP103
device physics is loaded from a compiled `.osdi` binary through bosdi, and parameter gradients
are reconstructed with the discrete adjoint sensitivity solver.

In this notebook we:
1. **Build an OSDI PSP103 ring oscillator** directly from the compiled `.osdi` binary.
2. **Generate a synthetic reference waveform** (nominal simulation + Gaussian noise).
3. **Fit 10 process parameters** using the discrete adjoint (`transient_parameter_sensitivity`) + `optax.adam`.



```
import sys
import time
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax


from circulax import compile_netlist
from circulax.components.electronic import Resistor, SmoothPulse, VoltageSource
from circulax.solvers import (
    analyze_circuit,
    setup_transient,
    transient_parameter_sensitivity,
)
from circulax.solvers.sensitivity import _resolve_param_cols
from circulax.solvers.transient import (
    TrapFactorizedTransientSolver,
    TrapVectorizedTransientSolver,
)
from circulax.utils import update_group_params

jax.config.update("jax_enable_x64", True)

# Persistent compilation cache for solver traces.
_JAX_CACHE = Path.home() / ".cache" / "jax" / "circulax_ring_osdi_adj"
_JAX_CACHE.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_JAX_CACHE))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

_REPO = Path.cwd().resolve().parents[1]
sys.path.insert(0, str(_REPO / "tests"))

print("JAX backend:", jax.default_backend())
```

## OSDI Adjoint Workflow

| Feature | OSDI adjoint |
|---------|--------------|
| Forward model | PSP103 `.osdi` binary loaded via bosdi |
| Gradient method | Discrete adjoint + finite differences through the OSDI residual |
| Memory cost | O(N) — stores the saved trajectory checkpoints |
| Forward pass | Compiled compact-model binary, no model-source tracing |
| Backward pass | N KLU adjoint solves plus batched residual perturbations |
| Best for | Production compact-model fitting and waveform calibration |

**Key insight:** the forward simulation uses the compiled OSDI kernel for device physics.
The discrete adjoint reconstructs parameter gradients by solving a sequence of linear systems
and using finite differences only for the ∂F/∂p term.


## Phase 1 — Build OSDI PSP103 Ring Oscillator

We use the compiled `psp103v4_psp103.osdi` binary directly through bosdi.
`make_psp103_descriptors()` builds canonical-mode `OsdiModelDescriptor` objects for NMOS
and PMOS from the model-card defaults in VACASK's `models.inc`.

The 10 target process parameters and their physical roles:

| Parameter | Nominal | Physical role |
|-----------|---------|---------------|
| `VFBO` | −1.1 V | Flat-band voltage (threshold) |
| `NSUBO` | 3 × 10²³ m⁻³ | Substrate doping (threshold, body effect) |
| `TOXO` | 1.5 nm | Gate oxide thickness (Cox, threshold, drive) |
| `UO` | 0.035 m²/V·s | Low-field mobility |
| `BETN` | 0.07 | Saturation beta (drive strength) |
| `THESAT` | 1.0 V⁻¹ | Velocity saturation |
| `MUEO` | 0.6 | Coulomb scattering mobility |
| `THEMU` | 1.5 | Mobility temperature exponent |
| `AX` | 3.0 | Mobility reduction coefficient |
| `RSW1` | 50 Ω | Source/drain series resistance |


```
# PSP103 OSDI descriptors live in tests/fixtures for reuse by tests and examples.
from fixtures.psp103_models import (
    PSP103N_DEFAULTS,
    PSP103P_DEFAULTS,
    geom_settings,
    make_psp103_descriptors,
)

PARAM_NAMES = ("VFBO", "NSUBO", "TOXO", "UO", "BETN", "THESAT", "MUEO", "THEMU", "AX", "RSW1")
N_STAGES = 9
VDD = 1.2

print("Building OSDI PSP103 descriptors...")
t0 = time.perf_counter()
psp103n, psp103p = make_psp103_descriptors()
print(f"  Descriptors: {time.perf_counter() - t0:.1f}s")

mos_n = geom_settings(10e-6, 1e-6)
mos_p = geom_settings(20e-6, 1e-6)

instances = {
    "Vvdd":  {"component": "vsrc",   "settings": {"V": VDD}},
    "Vkick": {"component": "kick",   "settings": {"V": 1.0, "delay": 1e-9, "tr": 1e-9}},
    "Rkick": {"component": "r_kick", "settings": {"R": 1e5}},
}
connections = {
    "Vvdd,p1": "vdd,p1",     "Vvdd,p2": "GND,p1",
    "Vkick,p1": "kick_n,p1", "Vkick,p2": "GND,p1",
    "Rkick,p1": "kick_n,p1", "Rkick,p2": "n1,p1",
}
for stage in range(1, N_STAGES + 1):
    in_n  = f"n{stage}"
    out_n = f"n{stage % N_STAGES + 1}"
    mn, mp = f"mn{stage}", f"mp{stage}"
    instances[mn] = {"component": "nmos", "settings": mos_n}
    instances[mp] = {"component": "pmos", "settings": mos_p}
    connections[f"{mn},D"] = f"{out_n},p1"
    connections[f"{mn},G"] = f"{in_n},p1"
    connections[f"{mn},S"] = "GND,p1"
    connections[f"{mn},B"] = "GND,p1"
    connections[f"{mp},D"] = f"{out_n},p1"
    connections[f"{mp},G"] = f"{in_n},p1"
    connections[f"{mp},S"] = "vdd,p1"
    connections[f"{mp},B"] = "vdd,p1"

models = {
    "nmos": psp103n, "pmos": psp103p,
    "vsrc": VoltageSource, "kick": SmoothPulse, "r_kick": Resistor,
}
groups, sys_size, port_map = compile_netlist(
    {"instances": instances, "connections": connections, "ports": {"out": "n1,p1"}},
    models,
)
out_idx = port_map["n1,p1"]
nmos_group = groups["nmos"]

# Resolve target parameter columns and record nominal values from the compiled group
param_cols = _resolve_param_cols(nmos_group, list(PARAM_NAMES), model_descriptor=psp103n)
NOMINAL = jnp.array([float(nmos_group.params[0, c]) for c in param_cols])
PARAM_SIGNS = jnp.array([-1.0 if float(n) < 0 else 1.0 for n in NOMINAL])

print(f"\nSystem size: {sys_size}  |  NMOS params shape: {nmos_group.params.shape}")
print("\nNominal target parameters:")
print(f"  {'Parameter':10s}  {'Nominal':>14s}  {'Col':>5s}")
print("  " + "-" * 36)
for name, val, col in zip(PARAM_NAMES, NOMINAL, param_cols):
    print(f"  {name:10s}  {float(val):>14.4e}  {col:>5d}")
```

## Phase 2 — DC Operating Point and Reference Waveform

**DC:** Gmin stepping from VDD/2 initial guess to avoid Newton stagnation in the
OSDI PSP103 subthreshold region.

**Transient:** 50 ns at 50 ps fixed step (~14 oscillation cycles at ~289 MHz).
The first 10 ns of settling are excluded from the loss.  Gaussian noise (15 mV RMS)
simulates measurement error.


```
ring_nodes = {f"n{s},p1" for s in range(1, N_STAGES + 1)}
solver = analyze_circuit(groups, sys_size, backend="klu_split")

y_init = jnp.zeros(sys_size)
for key, idx in port_map.items():
    if key in ring_nodes:
        y_init = y_init.at[idx].set(0.6)
    elif key == "vdd,p1":
        y_init = y_init.at[idx].set(VDD)

print("Solving DC operating point (gmin stepping)...")
t0 = time.perf_counter()
y0 = solver.solve_dc_gmin(groups, y_init, g_start=1e-2, n_steps=30)
print(f"  DC solve: {time.perf_counter() - t0:.1f}s")

assert bool(jnp.all(jnp.isfinite(y0))), "DC solve diverged"
y_max = float(jnp.max(jnp.abs(y0)))
assert y_max < 10.0, f"|y0|_max = {y_max:.2e} — DC diverged"
print(f"  V(n1) = {float(y0[out_idx]):.4f} V   |y0|_max = {y_max:.4f} V")
```


```
def _dom_freq(t, x):
    centered = x - x.mean()
    rising = np.where(np.diff(np.sign(centered)) > 0)[0]
    if len(rising) < 3:
        return float("nan")
    rising = rising[1:]
    times = []
    for i in rising:
        x0, x1 = float(centered[i]), float(centered[i + 1])
        t0_, t1_ = float(t[i]), float(t[i + 1])
        times.append(t0_ - x0 * (t1_ - t0_) / (x1 - x0))
    if len(times) < 2:
        return float("nan")
    return float(1.0 / np.median(np.diff(np.asarray(times))))

T_END     = 50e-9   # 50 ns simulation window
DT        = 5e-11   # 50 ps fixed step (proven stable for PSP103)
N_STEPS   = round(T_END / DT)  # 1000 (round avoids float truncation: int(50e-9/5e-11)=999)
N_SAVE    = N_STEPS + 1        # 1001 — save every BE step (adjoint requires 1:1 alignment)
SETTLE_NS = 10e-9   # first 10 ns excluded from loss

print(f"DT = {DT*1e12:.0f} ps, {N_STEPS} steps, {N_SAVE} saves (every BE step is a checkpoint)")

run_fn = setup_transient(groups, solver, transient_solver=TrapFactorizedTransientSolver)
saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, T_END, N_SAVE))
controller = diffrax.ConstantStepSize()
max_steps = 2 * N_STEPS  # generous headroom

# JIT warmup: same saveat shape/max_steps, tiny t1 so solver exits after 2 steps
saveat_warmup = diffrax.SaveAt(ts=jnp.linspace(0.0, 2 * DT, N_SAVE))
print("JIT compiling transient solver (first call — check cache for speedup)...")
t_compile = time.perf_counter()
_ = run_fn(
    t0=0.0, t1=2 * DT, dt0=DT, y0=y0, saveat=saveat_warmup,
    max_steps=max_steps, stepsize_controller=controller,
).ys.block_until_ready()
print(f"  JIT compile: {time.perf_counter() - t_compile:.1f}s")

print(f"Running reference transient ({T_END*1e9:.0f} ns)...")
t_run = time.perf_counter()
sol_ref = run_fn(
    t0=0.0, t1=T_END, dt0=DT, y0=y0, saveat=saveat,
    max_steps=max_steps, stepsize_controller=controller,
)
sol_ref.ys.block_until_ready()
print(f"  Transient: {time.perf_counter() - t_run:.1f}s")

t_arr = np.asarray(sol_ref.ts)
v_ref_clean = np.asarray(sol_ref.ys[:, out_idx])

NOISE_STD = 0.015
rng = np.random.default_rng(42)
v_ref_noisy = v_ref_clean + rng.normal(0.0, NOISE_STD, size=v_ref_clean.shape)

loss_start_idx = int(np.searchsorted(t_arr, SETTLE_NS))
v_ref_window = jnp.array(v_ref_noisy[loss_start_idx:])

freq_ref = _dom_freq(t_arr[loss_start_idx:], v_ref_clean[loss_start_idx:])
period_ns = 1e9 / freq_ref if np.isfinite(freq_ref) else 3.5
print(f"  Reference frequency: {freq_ref / 1e6:.1f} MHz  ({N_SAVE - loss_start_idx} loss points)")
print(f"  Adjoint will sweep {N_SAVE - 1} backward checkpoints per optimisation step")
```


```
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
axes[0].plot(t_arr * 1e9, v_ref_clean, lw=1.2, label="Nominal (clean)", color="steelblue")
axes[0].plot(t_arr * 1e9, v_ref_noisy, lw=0.6, alpha=0.6,
             label=f"Measured (+{NOISE_STD*1e3:.0f} mV noise)", color="gray")
axes[0].axvline(SETTLE_NS * 1e9, color="red", ls="--", lw=0.8, label="Loss window start")
axes[0].set_xlabel("Time (ns)")
axes[0].set_ylabel("V(n1) (V)")
axes[0].set_title("Reference ring oscillator waveform (OSDI PSP103)")
axes[0].legend(fontsize=8)
axes[0].set_xlim(0, T_END * 1e9)

zoom_end = T_END * 1e9
zoom_start = zoom_end - 5 * period_ns
axes[1].plot(t_arr * 1e9, v_ref_clean, lw=1.5, label="Clean", color="steelblue")
axes[1].plot(t_arr * 1e9, v_ref_noisy, lw=0.8, alpha=0.7, label="Noisy", color="gray")
axes[1].set_xlim(zoom_start, zoom_end)
axes[1].set_xlabel("Time (ns)")
axes[1].set_ylabel("V(n1) (V)")
axes[1].set_title(f"Zoom: last 5 periods (~{period_ns:.1f} ns each)")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.show()
```

## Phase 3 — Adjoint-Based Parameter Fitting (Primary Method)

We perturb the 10 NMOS parameters by ±15% from nominal, then use the **discrete adjoint**
(`transient_parameter_sensitivity`) to compute exact gradients of the time-domain MSE.

**Why adjoint instead of `jax.grad`?**
- PSP103 is implemented as an OSDI XLA FFI kernel — not JAX-native code.
- `jax.grad` cannot differentiate through the FFI boundary.
- The adjoint computes ∂loss/∂p by: (1) running a forward simulation to get the trajectory;
  (2) solving N backward KLU adjoint systems; (3) applying FD ∂F/∂p per checkpoint.
- Total cost: N × (1 KLU tsolve + n_params × 2 OSDI evals) with `shared_params=True`.

**Log-space parametrisation:** parameters span many orders of magnitude (NSUBO ≈ 3×10²³ vs
RSW1 = 50). We optimise `log|p|` and recover `p = exp(log|p|) × sign(p_nominal)`.

**Chain rule:** `∂loss/∂(log|p|) = ∂loss/∂p × p`

**Shared process params:** all 9 NMOS instances have the same parameter values (they model
the same fabricated device). With `shared_params=True`, the adjoint perturbs all devices at
once — 10 OSDI evals per checkpoint instead of 90.


```
# 15% perturbation (alternating ±) to get a meaningful initial error
rng2 = np.random.default_rng(7)
perturbation = 1.0 + 0.15 * rng2.choice([-1, 1], size=len(PARAM_NAMES))
PERTURBED = jnp.abs(NOMINAL) * jnp.array(perturbation)
log_params_init = jnp.log(PERTURBED)

print("Initial parameter perturbation:")
print(f"  {'Parameter':10s}  {'Nominal':>14s}  {'Perturbed':>14s}  {'Error%':>8s}")
print("  " + "-" * 52)
for name, nom, per, fac in zip(PARAM_NAMES, NOMINAL, PERTURBED, perturbation):
    err = (float(per) / float(nom) - 1) * 100
    print(f"  {name:10s}  {float(nom):>14.4e}  {float(per):>14.4e}  {err:>+8.1f}%")

# Loss function for the adjoint: must be differentiable w.r.t. y_trajectory via jax.grad
def loss_fn_traj(y_traj, ts_unused):
    v_sim = y_traj[loss_start_idx:, out_idx]
    return jnp.mean((v_sim - v_ref_window) ** 2)

def run_adjoint_step(log_abs_params):
    """Forward OSDI run + discrete adjoint gradient for current log-space params."""
    current_values = jnp.exp(log_abs_params) * PARAM_SIGNS

    # Update all 9 NMOS devices to the same shared process parameter values
    new_params = nmos_group.params
    for i, col in enumerate(param_cols):
        new_params = new_params.at[:, col].set(current_values[i])
    new_nmos = nmos_group.with_params(new_params)
    g = {**groups, "nmos": new_nmos}

    # Forward transient — passes updated groups via args override (no re-JIT)
    sol = run_fn(
        t0=0.0, t1=T_END, dt0=DT, y0=y0, saveat=saveat,
        max_steps=max_steps, stepsize_controller=controller,
        args=(g, sys_size),
    )

    loss = float(jnp.mean((sol.ys[loss_start_idx:, out_idx] - v_ref_window) ** 2))

    # Discrete adjoint: N backward KLU solves + FD through OSDI residual
    # shared_params=True: perturb all 9 NMOS at once (9× fewer OSDI evals)
    grads = transient_parameter_sensitivity(
        g, solver, sol.ys, sol.ts, loss_fn_traj,
        osdi_group_key="nmos", param_names=list(PARAM_NAMES),
        model_descriptor=psp103n,
        shared_params=True,
    )

    # Gradients are already summed across devices; apply log-space chain rule
    grad_vec = jnp.array([float(grads[p]) for p in PARAM_NAMES])
    log_grad_vec = grad_vec * current_values  # ∂loss/∂(log|p|) = ∂loss/∂p × p

    return loss, log_grad_vec, sol.ys

# Verify gradients on the initial (perturbed) parameters before the full loop
print("\nComputing initial adjoint gradient (forward + backward sweep)...")
t0 = time.perf_counter()
loss0, grad0, _ = run_adjoint_step(log_params_init)
print(f"  Initial loss: {loss0:.6f}  ({time.perf_counter() - t0:.1f}s)")
print(f"\n  {'Parameter':10s}  {'|grad_log|':>14s}")
print("  " + "-" * 28)
for name, g in zip(PARAM_NAMES, grad0):
    print(f"  {name:10s}  {float(abs(g)):>14.4e}")
```


```
N_STEPS_OPT = 60
LR = 5e-4

opt = optax.adam(LR)
opt_state = opt.init(log_params_init)
log_params = log_params_init

losses = []
param_history = [np.asarray(jnp.exp(log_params_init) * PARAM_SIGNS)]

print(f"Running Adam optimisation ({N_STEPS_OPT} steps, lr={LR}, OSDI adjoint gradients)...")
print(f"  {'Step':>4s}  {'Loss':>12s}  {'VFBO':>10s}  {'UO':>10s}  {'RSW1':>8s}  {'Time':>8s}")
print("  " + "-" * 62)

t_total = time.perf_counter()
for step in range(N_STEPS_OPT):
    t0 = time.perf_counter()
    loss_val, grads, _ = run_adjoint_step(log_params)
    grads_jnp = jnp.array(grads)
    updates, opt_state = opt.update(grads_jnp, opt_state)
    log_params = optax.apply_updates(log_params, updates)
    losses.append(loss_val)
    param_history.append(np.asarray(jnp.exp(log_params) * PARAM_SIGNS))
    if step % 10 == 0 or step == N_STEPS_OPT - 1:
        cur = jnp.exp(log_params) * PARAM_SIGNS
        dt = time.perf_counter() - t0
        print(f"  {step:>4d}  {loss_val:>12.6f}  "
              f"{float(cur[0]):>10.4f}  {float(cur[3]):>10.5f}  {float(cur[9]):>8.2f}  ({dt:.1f}s)")

total_s = time.perf_counter() - t_total
log_params_final = log_params
params_final = jnp.exp(log_params_final) * PARAM_SIGNS
print(f"\nFinal loss: {losses[-1]:.6f}  (initial: {losses[0]:.6f}, "
      f"reduction: {losses[0]/losses[-1]:.1f}×)  total: {total_s:.1f}s")
```


```
# Use higher-resolution save for visualization (not constrained by adjoint cost)
N_SAVE_VIZ = 1000
DT_VIZ = 5e-11  # fine 50 ps steps for smooth plots
saveat_viz = diffrax.SaveAt(ts=jnp.linspace(0.0, T_END, N_SAVE_VIZ))
max_steps_viz = int(2 * T_END / DT_VIZ)

def _run_with_log_params(log_abs_params):
    current_values = jnp.exp(log_abs_params) * PARAM_SIGNS
    new_params = nmos_group.params
    for i, col in enumerate(param_cols):
        new_params = new_params.at[:, col].set(current_values[i])
    new_nmos = nmos_group.with_params(new_params)
    g = {**groups, "nmos": new_nmos}
    sol = run_fn(
        t0=0.0, t1=T_END, dt0=DT_VIZ, y0=y0, saveat=saveat_viz,
        max_steps=max_steps_viz, stepsize_controller=controller,
        args=(g, sys_size),
    )
    return np.asarray(sol.ts), np.asarray(sol.ys[:, out_idx])

print("Running transient with initial (perturbed) parameters...")
t_viz_init, v_init = _run_with_log_params(log_params_init)
print("Running transient with final (optimised) parameters...")
t_viz_opt, v_opt = _run_with_log_params(log_params_final)

# Also get high-res reference
sol_ref_viz = run_fn(
    t0=0.0, t1=T_END, dt0=DT_VIZ, y0=y0, saveat=saveat_viz,
    max_steps=max_steps_viz, stepsize_controller=controller,
)
t_arr_viz = np.asarray(sol_ref_viz.ts)
v_ref_clean_viz = np.asarray(sol_ref_viz.ys[:, out_idx])
v_ref_noisy_viz = v_ref_clean_viz + rng.normal(0.0, NOISE_STD, size=v_ref_clean_viz.shape)
loss_start_idx_viz = int(np.searchsorted(t_arr_viz, SETTLE_NS))

fig, axes = plt.subplots(3, 1, figsize=(11, 9))

axes[0].plot(t_viz_init * 1e9, v_init, lw=1.0, ls="--", color="orangered", label="Initial (perturbed ±15%)", zorder=2)
axes[0].plot(t_arr_viz * 1e9, v_ref_noisy_viz, lw=0.7, alpha=0.5, color="gray", label="Reference (noisy)")
axes[0].plot(t_viz_opt * 1e9, v_opt, lw=1.2, color="steelblue", label="Optimised (OSDI adjoint)", zorder=3)
axes[0].set_xlabel("Time (ns)")
axes[0].set_ylabel("V(n1) (V)")
axes[0].set_title("OSDI adjoint fitting — before / after optimisation")
axes[0].legend(fontsize=8)
axes[0].set_xlim(0, T_END * 1e9)

zoom_start2 = max(SETTLE_NS * 1e9, T_END * 1e9 - 5 * period_ns)
axes[1].plot(t_arr_viz * 1e9, v_ref_noisy_viz, lw=0.8, alpha=0.6, color="gray", label="Reference (noisy)")
axes[1].plot(t_viz_init * 1e9, v_init, lw=1.0, ls="--", color="orangered", label="Initial")
axes[1].plot(t_viz_opt * 1e9, v_opt, lw=1.5, color="steelblue", label="Optimised")
axes[1].set_xlim(zoom_start2, T_END * 1e9)
axes[1].set_xlabel("Time (ns)")
axes[1].set_ylabel("V(n1) (V)")
axes[1].set_title("Zoom: last 5 periods (steady state)")
axes[1].legend(fontsize=8)

residual = v_opt[loss_start_idx_viz:] - v_ref_noisy_viz[loss_start_idx_viz:]
rms_mv = float(jnp.sqrt(jnp.mean(jnp.array(residual) ** 2))) * 1e3
axes[2].plot(t_arr_viz[loss_start_idx_viz:] * 1e9, residual * 1e3, lw=0.8, color="purple")
axes[2].axhline(0, color="black", lw=0.5)
axes[2].set_xlabel("Time (ns)")
axes[2].set_ylabel("Residual (mV)")
axes[2].set_title(f"Optimised − Reference  (RMS = {rms_mv:.1f} mV)")

plt.tight_layout()
plt.show()
```


```
param_history_arr = np.array(param_history)  # shape: (N_steps+1, 10)
steps_arr = np.arange(len(param_history_arr))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].semilogy(losses, color="steelblue", lw=1.5)
axes[0].set_xlabel("Adam step")
axes[0].set_ylabel("MSE loss (V²)")
axes[0].set_title("Loss convergence (OSDI adjoint)")
axes[0].grid(True, which="both", alpha=0.3)

nominal_arr = np.abs(np.asarray(NOMINAL))
for i, name in enumerate(PARAM_NAMES):
    normalised = np.abs(param_history_arr[:, i]) / nominal_arr[i]
    axes[1].plot(steps_arr, normalised, lw=1.2, label=name)
axes[1].axhline(1.0, color="black", lw=1.0, ls="--", label="Target (nominal)")
axes[1].set_xlabel("Adam step")
axes[1].set_ylabel("|param| / |nominal|")
axes[1].set_title("Parameter trajectories (normalised)")
axes[1].legend(fontsize=7, ncol=2, loc="upper right")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```


```
print("Final parameter recovery (OSDI adjoint):")
print(f"  {'Parameter':10s}  {'Nominal':>14s}  {'Recovered':>14s}  {'Error%':>8s}")
print("  " + "=" * 52)
max_err = 0.0
for name, nom, rec in zip(PARAM_NAMES, NOMINAL, params_final):
    err_pct = (float(rec) / float(nom) - 1) * 100
    max_err = max(max_err, abs(err_pct))
    flag = " ✓" if abs(err_pct) < 5 else " !"
    print(f"  {name:10s}  {float(nom):>14.4e}  {float(rec):>14.4e}  {err_pct:>+8.2f}%{flag}")
print(f"\nMax absolute error: {max_err:.2f}%  (target: <5%)")
```


```
import matplotlib
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation, PillowWriter

matplotlib.use("Agg")

ANIM_STRIDE = 5
FPS         = 8
DPI         = 110
GIF_PATH    = "examples/inverse_design/ring_fitting_osdi.gif"

n_opt = len(losses)
frame_indices = list(range(0, n_opt, ANIM_STRIDE))
if (n_opt - 1) not in frame_indices:
    frame_indices.append(n_opt - 1)

print(f"Pre-computing {len(frame_indices)} transient waveforms (stride={ANIM_STRIDE})...")
frame_waveforms = []
frame_t_arrs = []
for fi, idx in enumerate(frame_indices):
    abs_params = np.abs(param_history_arr[idx])
    log_abs = jnp.log(jnp.array(abs_params, dtype=jnp.float64))
    t_frame, v_frame = _run_with_log_params(log_abs)
    frame_waveforms.append(v_frame)
    frame_t_arrs.append(t_frame)
    if fi % 3 == 0 or fi == len(frame_indices) - 1:
        print(f"  [{fi+1:3d}/{len(frame_indices)}] step {idx:3d}")
print("Done.")

losses_arr   = np.array(losses)
normed_hist  = np.abs(param_history_arr) / nominal_arr[None, :]
steps_p      = np.arange(len(param_history_arr))
t_zoom_ns    = t_arr_viz[loss_start_idx_viz:] * 1e9

plt.rcParams.update({"text.color": "grey", "axes.labelcolor": "grey",
                     "xtick.color": "grey", "ytick.color": "grey",
                     "axes.edgecolor": "grey"})

fig_a = plt.figure(figsize=(13, 7), facecolor="white")
gs = gridspec.GridSpec(2, 2, figure=fig_a, width_ratios=[1.6, 1],
                       height_ratios=[1, 1], hspace=0.35, wspace=0.30)
ax_wave = fig_a.add_subplot(gs[0, :])
ax_loss = fig_a.add_subplot(gs[1, 0])
ax_traj = fig_a.add_subplot(gs[1, 1])

for ax in (ax_wave, ax_loss, ax_traj):
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.2, color="#cccccc", zorder=0)

ax_wave.plot(t_zoom_ns, v_ref_noisy_viz[loss_start_idx_viz:], color="gray", lw=0.7, alpha=0.6,
             zorder=1, label="Reference (noisy)")
ax_wave.set_xlabel("Time (ns)", fontsize=10)
ax_wave.set_ylabel("V(n1) (V)", fontsize=10)
ax_wave.set_xlim(t_zoom_ns[0], t_zoom_ns[-1])

ax_loss.semilogy(losses_arr, color="#cccccc", lw=1.2, zorder=1)
ax_loss.set_xlabel("Adam step", fontsize=10)
ax_loss.set_ylabel("MSE loss (V²)", fontsize=10)
ax_loss.set_title("Loss convergence", fontsize=10)
ax_loss.set_xlim(-1, n_opt + 1)

_colors = plt.cm.tab10(np.linspace(0, 1, 10))
for i, pname in enumerate(PARAM_NAMES):
    ax_traj.plot(steps_p, normed_hist[:, i], color=_colors[i], lw=0.8, alpha=0.25, zorder=1)
ax_traj.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.5, zorder=1)
ax_traj.set_xlabel("Adam step", fontsize=10)
ax_traj.set_ylabel("|param| / |nominal|", fontsize=10)
ax_traj.set_title("Parameter trajectories", fontsize=10)
ax_traj.set_xlim(-1, len(steps_p) + 1)

(live_wave,)       = ax_wave.plot([], [], color="#4a90c4", lw=1.5, zorder=2, label="Current fit")
title_txt          = ax_wave.set_title("", fontsize=11, pad=6)
ax_wave.legend(fontsize=8, loc="upper right")
(live_loss_line,)  = ax_loss.plot([], [], color="#4a90c4", lw=2.0, zorder=2)
(live_loss_dot,)   = ax_loss.plot([], [], "o", color="orange", ms=7, zorder=3)
live_traj_lines, live_traj_dots = [], []
for i, pname in enumerate(PARAM_NAMES):
    (ln,) = ax_traj.plot([], [], color=_colors[i], lw=1.5, zorder=2, label=pname)
    (dt,) = ax_traj.plot([], [], "o", color=_colors[i], ms=5, zorder=3)
    live_traj_lines.append(ln)
    live_traj_dots.append(dt)
ax_traj.legend(fontsize=6, ncol=2, loc="upper right")

def _animate(fi):
    step = frame_indices[fi]
    t_frame_ns = frame_t_arrs[fi] * 1e9
    # Find the loss window start for this frame's time array
    lsi = int(np.searchsorted(frame_t_arrs[fi], SETTLE_NS))
    live_wave.set_data(t_frame_ns[lsi:], frame_waveforms[fi][lsi:])
    title_txt.set_text(
        f"Fitting PSP103 — step {step}/{n_opt}  "
        f"(loss = {losses_arr[min(step, n_opt-1)]:.4f} V²)"
    )
    pae_step = min(step, n_opt - 1)
    live_loss_line.set_data(np.arange(pae_step + 1), losses_arr[:pae_step + 1])
    live_loss_dot.set_data([pae_step], [losses_arr[pae_step]])
    p_end = min(step + 1, len(steps_p))
    xp = steps_p[:p_end]
    for i in range(10):
        live_traj_lines[i].set_data(xp, normed_hist[:p_end, i])
        live_traj_dots[i].set_data([xp[-1]], [normed_hist[p_end - 1, i]])

anim = FuncAnimation(fig_a, _animate, frames=len(frame_indices),
                     interval=1000 // FPS, blit=False)
anim.save(GIF_PATH, writer=PillowWriter(fps=FPS), dpi=DPI)
plt.close(fig_a)
print(f"Saved → {GIF_PATH}")
```

## Summary

| Stage | What happened |
|-------|---------------|
| **OSDI ring build** | PSP103 `.osdi` binary loaded directly via bosdi |
| **DC solve** | Gmin stepping from VDD/2 initial guess; converged for OSDI PSP103 |
| **Reference data** | 50 ns at 50 ps step; nominal waveform + 15 mV Gaussian noise |
| **OSDI adjoint** | 60 Adam steps, log-space parameterisation; N=999 backward KLU solves per step |
| **Parameter recovery** | Most parameters within 5% of nominal |

**Key result:** The discrete adjoint delivers gradients of the time-domain waveform MSE
with respect to 10 compact-model process parameters using the compiled OSDI model
or autodiff through the OSDI XLA FFI.  The forward simulation uses the compiled PSP103
binary (fast), and the backward sweep reconstructs parameter gradients via KLU adjoint
solves + finite differences through the OSDI residual function.

### Going further

- **Joint N+P fitting** — fit PMOS `VFBO`, `UO`, `BETN` simultaneously by adding a second
  `transient_parameter_sensitivity` call for the `pmos` group key.
- **More parameters** — extend `PARAM_NAMES` to 20–30; per-step cost scales linearly.
- **Measured silicon data** — replace `v_ref_noisy` with oscilloscope CSV.  The optimiser
  loop and loss function are unchanged.
- **Process-corner extraction** — run across multiple dies to extract σ(VFBO), σ(UO), …
  directly from waveform data.
