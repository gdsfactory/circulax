# Transient Simulation

Transient analysis integrates the circuit DAE

$$F(y) + \frac{d}{dt}Q(y) = 0$$

forward in time from an initial condition, typically the DC operating point.  Each time step solves a nonlinear system of equations using Newton-Raphson iteration.

---

## Time Discretisation and Accuracy

Circulax provides four implicit time-stepping schemes.  The default is selected automatically based on the linear solver backend.

| Solver | Order | Stability | Notes |
|--------|-------|-----------|-------|
| `Trap` | 2nd | A-stable, **zero damping** | **Default.** Symmetric trapezoidal rule. The SPICE historical default. |
| `BDF2` | 2nd | A-stable, slight L²-damping | Variable-step Gear/BDF2. BE on step 1, BDF2 from step 2. |
| `SDIRK3` | 3rd | L-stable, **strong damping** | Alexander SDIRK (3 stages). 3 Newton solves per step. |
| `BackwardEuler` | 1st | L-stable, strong damping | First-order; rarely needed directly. |

All four methods are stable for arbitrarily stiff circuits, but they differ in *how they treat oscillation*:

- **Trap** has zero numerical damping at any frequency. Limit-cycle frequencies are preserved exactly. This is what circuit designers expect — it's the SPICE historical default. Trade-off: trap can develop spurious "trapezoidal ringing" at the Nyquist frequency on circuits with very sharp digital edges combined with stiffness; the symptom is a sawtooth at twice the step rate riding on top of the real signal.
- **BDF2** introduces slight L²-stable damping that kills trap-style ringing without measurably hurting accuracy. Use this when you see trap ringing.
- **SDIRK3** is *L-stable* — it aggressively damps high-frequency modes. This is great for stiff DC-settling transients (power-up sequences, biasing into the operating point) and sharp event capture in non-oscillatory circuits. **It is the wrong choice for any self-oscillating circuit** (ring oscillator, LC tank, Colpitts, relaxation oscillator, clock): the same L-stable damping that suppresses unphysical fast modes also pulls real limit-cycle frequencies. On the PSP103 ring benchmark SDIRK3 reports a ~1.7× slower frequency than VACASK trap, while circulax's trap and BDF2 both match VACASK to 0.5 % (see `docs/bosdi_psp103_ring_oscillator_issue.md`). SDIRK3 also costs ~3× more wall time per step.
- **BackwardEuler** is the simplest fallback (1st order, strong damping); useful for smoke testing.

Higher-order solvers reduce **global truncation error**:

- Trap achieves $\mathcal{O}(h^2)$ accuracy and is symmetric (time-reversible).
- BDF2 achieves $\mathcal{O}(h^2)$ — halving the step size reduces the error by 4×.
- SDIRK3 achieves $\mathcal{O}(h^3)$ — halving the step size reduces the error by 8×.

### Picking an integrator

| Circuit type | Recommended solver |
|--------------|--------------------|
| Analog / RF / photonic oscillators, LC tanks, ring oscillators, harmonic balance | **`Trap` (default)** |
| Digital edges or switching circuits where trap shows ringing | `BDF2` |
| Stiff DC-settling transients, ESD pulses, amplifier startup, non-oscillatory event capture | `SDIRK3` |
| Smoke test or first-step warmup | `BackwardEuler` |

To select a non-default integrator explicitly:

```python
from circulax.solvers.transient import (
    BDF2RefactoringTransientSolver,
    SDIRK3RefactoringTransientSolver,
)

transient_sim = setup_transient(
    groups=groups,
    linear_strategy=linear_strat,
    transient_solver=BDF2RefactoringTransientSolver,  # or SDIRK3, etc.
)
```

Each integrator has three variants (`*Vectorized`, `*Factorized`, `*Refactoring`) trading off Newton convergence robustness against per-step cost. Use `*Refactoring` for strongly nonlinear circuits (e.g. PSP103, BSIM4 stacks) where the Jacobian must be re-factorised at every Newton iteration; use `*Factorized` (frozen Jacobian) for mildly nonlinear circuits where one factorisation per step is enough.

---

## Step Size Control

### Constant step size (default)

By default the step size is fixed at `dt0` throughout the simulation:

```python
sol = transient_sim(
    t0=0.0, t1=1e-6, dt0=1e-9,
    y0=y_op, saveat=saveat, max_steps=10000,
)
```

This is simple and JIT-compiles efficiently, but requires you to choose a `dt0` that is small enough for accuracy across the whole simulation.

### Adaptive step size with PIDController

For circuits with widely varying timescales — fast switching events followed by slow settling — an adaptive step size controller dramatically reduces the number of steps while maintaining accuracy.

Pass a `diffrax.PIDController` as the `stepsize_controller` argument:

```python
import diffrax

pid = diffrax.PIDController(
    rtol=1e-3,          # relative tolerance per step
    atol=1e-6,          # absolute tolerance per step
    pcoeff=0.2,         # proportional gain
    icoeff=0.5,         # integral gain
    dcoeff=0.0,         # derivative gain (usually 0)
    error_order=2,      # match the solver order (2 for BDF2, 3 for SDIRK3)
    dtmin=1e-14,        # hard floor on step size
    dtmax=1e-9,         # hard ceiling on step size
    force_dtmin=True,   # accept steps at dtmin rather than aborting
)

sol = transient_sim(
    t0=0.0, t1=1e-6, dt0=1e-11,
    y0=y_op,
    saveat=saveat, max_steps=50000,
    stepsize_controller=pid,
)
```

The controller estimates the local truncation error at each step and scales the next step size using a PID law:

$$h_{n+1} = h_n \cdot \min\!\left(h_{\max},\, \max\!\left(h_{\min},\, \text{safety} \cdot \varepsilon_n^{-k_I} \cdot \varepsilon_{n-1}^{k_P} \cdot \varepsilon_{n-2}^{k_D}\right)\right)$$

where $\varepsilon_n = \|e_n\| / \text{tol}$ is the normalised error at step $n$.

Steps where the error exceeds tolerance are **rejected** and retried at a smaller `h`.  The `sol.stats` dict reports how many steps were accepted and rejected:

```python
print(sol.stats)
# {'num_steps': 1847, 'num_accepted_steps': 1731, 'num_rejected_steps': 116, ...}
```

**Tuning tips:**

- Set `error_order` to match your solver (2 for Trap/BDF2, 3 for SDIRK3).
- `rtol=1e-3, atol=1e-6` is a good starting point for most circuits.
- Tighten tolerances if you see waveform artifacts; loosen them to reduce step count.
- `dtmax` should be no larger than the fastest feature you care about (e.g., 1/10 of the shortest rise time).
- **Always set a sensible `dtmin` (and `force_dtmin=True`) on stiff/oscillating circuits.** Without it, the PID error estimator will burn compute hunting for nonexistent microstructure at sharp transitions — we observed a 12× step-count blow-up on a PSP103 ring oscillator when `dtmin` was left at 1 fs. A good rule of thumb for a ring-oscillator-like circuit at ~GHz is `dtmin ≈ 0.1 × dtmax` (allows 10× subdivision at transitions). See `benchmarks/ring/` for the ring-oscillator benchmark harness.
- `dt0` (initial step) should be small — diffrax's first-step predictor extrapolates from `dt0` and a too-aggressive value blows up Newton on strongly-nonlinear circuits. Use ≤ 1 ps for MOSFET-level circuits; diffrax's PID will ramp up quickly from there.
- `force_dtmin=True` prevents the solver aborting at very stiff moments; check `num_rejected_steps` to see if this was needed.

**For pure oscillator workloads** (ring oscillators, LC tanks, PLLs in steady state) prefer `ConstantStepSize` — the error estimator's per-step overhead isn't paying back when every stage transitions at the same cadence. We measured a 17% wall penalty for PID-at-matched-cadence vs fixed-dt on a PSP103 ring.

---

## Workflow

```python
import diffrax
import jax.numpy as jnp
from circulax import compile_circuit, setup_transient

# 1. Compile and solve DC operating point
circuit = compile_circuit(net_dict, models_map)
y_op = circuit()

# 2. Set up transient solver
transient_sim = setup_transient(groups=circuit.groups, linear_strategy=circuit.solver)

# 3. Run with adaptive stepping
pid = diffrax.PIDController(rtol=1e-3, atol=1e-6, pcoeff=0.2, icoeff=0.5,
                             dcoeff=0.0, error_order=2, dtmin=1e-14, force_dtmin=True)
saveat = diffrax.SaveAt(ts=jnp.linspace(0, 1e-6, 1000))

sol = transient_sim(
    t0=0.0, t1=1e-6, dt0=1e-11,
    y0=y_op, saveat=saveat, max_steps=50000,
    stepsize_controller=pid,
)

# 4. Extract results
ts = sol.ts
v_out = circuit.get_port_field(sol.ys, "R1,p2")
```
