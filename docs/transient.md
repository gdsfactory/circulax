# Transient Simulation

Transient analysis integrates the circuit DAE

$$F(y) + \frac{d}{dt}Q(y) = 0$$

forward in time from an initial condition, typically the DC operating point.  Each time step solves a nonlinear system of equations using Newton-Raphson iteration.

---

## Time Discretisation and Accuracy

Circulax provides three implicit time-stepping schemes.  The default is selected automatically based on the linear solver backend.

| Solver | Order | Method | Notes |
|--------|-------|--------|-------|
| `BDF2` | 2nd | Variable-step Gear/BDF2 | **Default.** Backward Euler on step 1, BDF2 from step 2. |
| `SDIRK3` | 3rd | Alexander SDIRK (3 stages) | A-stable; 3 Newton solves per step. Higher accuracy at larger step sizes. |
| `BackwardEuler` | 1st | Backward Euler | First-order fallback; rarely needed directly. |

All three methods are **A-stable**: they remain stable for arbitrarily stiff circuits without restricting the step size for stability reasons.

Higher-order solvers reduce **global truncation error**:

- BDF2 achieves $\mathcal{O}(h^2)$ accuracy — halving the step size reduces the error by 4×.
- SDIRK3 achieves $\mathcal{O}(h^3)$ — halving the step size reduces the error by 8×.

For most circuits BDF2 is the right choice.  Use SDIRK3 when high accuracy at large step sizes is needed (e.g., long simulations with tightly-toleranced outputs).

To select SDIRK3 explicitly:

```python
from circulax.solvers.transient import SDIRK3VectorizedTransientSolver

transient_sim = setup_transient(
    groups=groups,
    linear_strategy=linear_strat,
    transient_solver=SDIRK3VectorizedTransientSolver,
)
```

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

- Set `error_order` to match your solver (2 for BDF2, 3 for SDIRK3).
- `rtol=1e-3, atol=1e-6` is a good starting point for most circuits.
- Tighten tolerances if you see waveform artifacts; loosen them to reduce step count.
- `dtmax` should be no larger than the fastest feature you care about (e.g., 1/10 of the shortest rise time).
- `force_dtmin=True` prevents the solver aborting at very stiff moments; check `num_rejected_steps` to see if this was needed.

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
