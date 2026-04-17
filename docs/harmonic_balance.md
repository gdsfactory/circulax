## Harmonic Balance Analysis

Harmonic Balance (HB) solves directly for the Fourier coefficients of the periodic steady state, rather than integrating forward in time until transients decay.

For single-tone circuits, HB typically converges in ~10-20 Newton iterations instead of thousands of time steps.

---

### Why use Harmonic Balance?

| | Transient | Harmonic Balance |
|---|---|---|
| **Goal** | Full time evolution from initial conditions | Periodic steady state only |
| **Result** | Waveform over time | Fourier spectrum + one period of waveform |
| **Cost for steady state** | Must wait for transients to decay | Converges in ~10–20 Newton steps |
| **Nonlinear harmonics** | Captured automatically | Captured up to $N$ harmonics |
| **Differentiable?** | Yes (Diffrax) | Yes (JAX end-to-end) |

Use HB for steady-state amplitude/phase, THD measurement, frequency sweeps, or gradient-based optimisation through the periodic solution.

---

### Mathematical Foundation

#### The circuit equations

Circulax formulates every circuit as a Differential Algebraic Equation (DAE):

$$F(y) + \frac{d}{dt}Q(y) = 0$$

where $y(t)$ is the state vector (all node voltages and internal states), $F$ collects the resistive (instantaneous) contributions, and $Q$ collects the reactive (charge/flux) terms.

#### Representing the solution as time samples

Assume $y(t)$ is periodic with period $T = 1/f_0$.  With $N$ harmonics retained, a band-limited periodic signal is determined exactly by $K = 2N + 1$ equally-spaced samples:

$$t_k = \frac{k \, T}{K}, \quad k = 0, 1, \dots, K-1$$

The unknown is the matrix $Y \in \mathbb{R}^{K \times n}$ of these $K$ samples of the $n$-dimensional state vector.  Crucially, because $K$ is odd ($K = 2N+1$), the real FFT of $K$ real samples yields exactly $N+1$ independent complex coefficients — no aliasing.

#### The HB residual

Differentiating the Fourier series of $Q(y(t))$ is equivalent to multiplying the $k$-th Fourier coefficient by $jk\omega_0$ (where $\omega_0 = 2\pi f_0$).  The HB equation at harmonic $k$ is therefore:

$$R_k = \underbrace{\mathcal{F}\{F(y(t))\}_k}_{F_k} + jk\omega_0 \underbrace{\mathcal{F}\{Q(y(t))\}_k}_{Q_k} = 0, \quad k = 0, 1, \dots, N$$

where $\mathcal{F}\{\cdot\}_k$ denotes the $k$-th coefficient of the (unnormalised) DFT.

Inverting: applying $\mathcal{F}^{-1}$ to $R_k$ gives a **real** time-domain residual of shape $(K, n)$.  Newton iteration on this residual finds the periodic steady state.

#### Newton iteration

Given a current iterate $Y^{(\ell)}$:

1. Evaluate $F(Y^{(\ell)})$ and $Q(Y^{(\ell)})$ at all $K$ time points.
2. Apply the DFT: $F_k$, $Q_k$.
3. Form $R_k = F_k + jk\omega_0 Q_k$; invert to get the residual $r = \text{IFFT}(R_k)$.
4. Compute the Jacobian $J = \partial r / \partial Y$.
5. Solve $J \, \delta = -r$ and update $Y^{(\ell+1)} = Y^{(\ell)} + \alpha \, \delta$ (with damping $\alpha \leq 1$).

---

### Jacobian via AD

Traditional HB implementations require manually deriving the block-circulant HB Jacobian. With JAX this is unnecessary:

```python
def residual_fn(y_flat):
    y_time = y_flat.reshape(K, sys_size)
    # (1) Evaluate circuit physics at all K time points in one vmapped call
    f_time, q_time = jax.vmap(
        lambda y_t, t: assemble_residual_only_real(y_t, groups, t, 1.0)
    )(y_time, t_points)
    # (2) FFT, apply jkw scaling, IFFT — all differentiable JAX primitives
    F_k = jnp.fft.rfft(f_time, axis=0)
    Q_k = jnp.fft.rfft(q_time, axis=0)
    R_k = F_k + (1j * omega) * k[:, None] * Q_k
    return jnp.fft.irfft(R_k, n=K, axis=0).flatten()

J = jax.jacobian(residual_fn)(y_flat)   # exact Jacobian, no manual derivation
```

| Feature | Role in HB |
|---|---|
| `jax.vmap` | Evaluates physics at all $K$ time points in one vectorised call |
| `jnp.fft.rfft/irfft` | Differentiable DFT/IDFT |
| `jax.jacobian` | Exact $KN \times KN$ Jacobian via forward-mode AD |
| `jax.jit` | Compiles residual + Jacobian once; Newton steps reuse the XLA program |
| `jax.grad` | The full `hb.solve()` is differentiable for inverse design |

---

### Usage

The workflow mirrors the DC/transient pattern:

```python
import jax.numpy as jnp
from circulax import compile_circuit, setup_harmonic_balance
from circulax.components.electronic import Capacitor, Inductor, Resistor, VoltageSourceAC

# 1. Define and compile the netlist
net = { ... }   # SAX-format netlist dict
models = {"R": Resistor, "L": Inductor, "C": Capacitor, "Vs": VoltageSourceAC}
circuit = compile_circuit(net, models, backend="dense")

# 2. Find the DC operating point (used as the initial guess for HB)
y_dc = circuit()

# 3. Set up and run the Harmonic Balance solver
f_drive = 1e6   # Hz — must match the frequency in VoltageSourceAC settings
run_hb = setup_harmonic_balance(circuit.groups, circuit.sys_size, freq=f_drive, num_harmonics=5)
y_time, y_freq = run_hb(y_dc)

# The solver is also compatible with jax.jit for repeated calls:
# y_time, y_freq = jax.jit(run_hb)(y_dc)
```

`setup_harmonic_balance` parameters:

| Parameter | Default | Description |
|---|---|---|
| `groups` | — | Compiled component groups (from `circuit.groups`) |
| `num_vars` | — | System size (from `circuit.sys_size`) |
| `freq` | — | Fundamental frequency in Hz |
| `num_harmonics` | `5` | Number of harmonics $N$; solver uses $K = 2N+1$ time points |

`run_hb` call parameters:

| Parameter | Default | Description |
|---|---|---|
| `y_dc` | — | DC operating point; used as the zero-AC initial guess |
| `max_iter` | `50` | Maximum Newton iterations |
| `tol` | `1e-6` | Convergence tolerance (infinity norm of residual) |

---

### Interpreting the Output

`run_hb(y_dc)` returns `(y_time, y_freq)`:

**`y_time`** — shape $(K, n)$, dtype `float64`

The state vector sampled at $K$ equally-spaced time points over one period.  Time points are:

```python
t_points = jnp.linspace(0, 1.0 / freq, K, endpoint=False)
```

Plot any node's waveform as `plt.plot(t_points, y_time[:, node_idx])`.

**`y_freq`** — shape $(N+1, n)$, dtype `complex128`

Normalised Fourier coefficients.  Index 0 is DC, index 1 is the fundamental, index $k$ is the $k$-th harmonic.

Because `rfft` folds negative frequencies into positive ones, the two-sided (physical) amplitude of harmonic $k \geq 1$ is:

$$\hat{V}_k = 2 \left| Y_k \right|$$

where $Y_k$ = `y_freq[k, node]`.  The DC value is simply `y_freq[0, node].real` (no factor of 2).

```python
node = net_map["R1,p2"]
harmonics = jnp.arange(hb.num_harmonics + 1)
amplitudes = jnp.abs(y_freq[:, node]) * jnp.where(harmonics == 0, 1.0, 2.0)
```

---

### Frequency Sweep

`jax.vmap` compiles the entire frequency sweep into a single XLA call.

The source frequency is baked into the compiled parameters, so it must be updated at each sweep point via `update_group_params`:

```python
import jax
import jax.numpy as jnp
from circulax import compile_circuit, setup_harmonic_balance
from circulax.utils import update_group_params

circuit = compile_circuit(net, models, backend="dense")
y_dc = circuit()

node_idx = circuit.port_map["RL,p1"]   # output node index

def hb_solve_freq(sweep_freq):
    # Update the source's freq param so it drives at sweep_freq.
    # The group key matches the component key used in models_map.
    updated_groups = update_group_params(circuit.groups, "Vs", "freq", sweep_freq)
    run_hb = setup_harmonic_balance(
        updated_groups, circuit.sys_size, freq=sweep_freq, num_harmonics=10
    )
    _, y_freq = run_hb(y_dc)
    return 2.0 * jnp.abs(y_freq[1, node_idx])   # fundamental amplitude

# Compile once, sweep 100 frequencies in one call:
sweep_freqs = jnp.logspace(2, 5, 100)   # 100 Hz – 100 kHz
amps = jax.jit(jax.vmap(hb_solve_freq))(sweep_freqs)
```

---

### Scalability

The Newton Jacobian is $(K \cdot n) \times (K \cdot n)$, stored dense.  For 10 harmonics ($K=21$) and 200 nodes: $4200 \times 4200 \approx 80\,\text{MB}$.

For larger circuits, the block-circulant structure could be exploited to reduce to $n$ independent $K \times K$ systems via `jnp.fft.ifft` — not yet implemented.
