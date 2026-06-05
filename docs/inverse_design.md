# Inverse Design

Circulax is fully differentiable: `jax.grad` computes exact gradients from any scalar loss back through the solver into component parameters.

## Back-propagation vs Traditional Methods

| | Finite Differences | Adjoint / Sensitivity | Back-propagation (AD) |
|---|---|---|---|
| Gradient cost | O(N) simulations per step | O(1) but requires manual derivation | O(1), automatic |
| Accuracy | Approximate (step-size dependent) | Exact (if correctly derived) | Exact (machine precision) |
| Scales to | ~10 parameters | Large, if hand-derived | Arbitrary |
| Non-linear models | Works but expensive | Requires linearisation | Works on arbitrary non-linearities |

Finite differences costs one simulation per parameter. Back-propagation costs one forward + one backward pass regardless of parameter count.

## How it works in circulax

```python
import jax
import optax

# compile once — the circuit topology is fixed
circuit = compile_circuit(netlist, models)

def loss_fn(params):
    # Update instance parameters without recompiling the topology.
    y = circuit.dc(params={"C1.C": params[0], "L1.L": params[1]})
    return jnp.mean((circuit.port(y, "out") - target) ** 2)

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

grads = jax.grad(loss_fn)(params)          # exact gradients, one pass
updates, opt_state = optimizer.update(grads, opt_state)
params = optax.apply_updates(params, updates)
```

Use the same pattern for frequency-domain and periodic analyses:

```python
S = circuit.ac(params={"L1.L": L, "C1.C": C}, ports=["in", "out"], freqs=freqs)
y_time, y_freq = circuit.hb(params={"Vdp.mu": mu}, freq=f0, harmonics=7, y0=y_dc)
```

Advanced users can still update compiled `groups` directly with
`update_params_dict(...)` when building custom transforms or solver-control
loops, but `circuit.dc/ac/hb(params={...})` is the normal differentiable
workflow.

## Examples

- [LC Filter Synthesis](examples/01_lc_filter_synthesis.md) — gradient-descent tuning of L/C values to match a Butterworth transfer function
- [Oscillator HB Tuning](examples/02_oscillator_hb_tuning.md) — optimising oscillator harmonics via harmonic balance
- [Photonic CMZ Demux](examples/03_photonic_cmz_demux.md) — inverse design of a cascaded Mach-Zehnder wavelength demux
- [HEMT PA Optimisation](examples/04_hemt_pa_optimization.md) — power amplifier matching network optimisation
