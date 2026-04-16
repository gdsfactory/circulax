# Inverse Design

Circulax is fully differentiable — gradients flow from a scalar loss function back through the solver and into every component parameter. This enables **gradient-based inverse design**: define a target, write a loss, and let `jax.grad` + an optimizer find the circuit parameters automatically.

## Back-propagation vs Traditional Methods

| | Finite Differences | Adjoint / Sensitivity | Back-propagation (AD) |
|---|---|---|---|
| Gradient cost | O(N) simulations per step | O(1) but requires manual derivation | O(1), fully automatic |
| Accuracy | Approximate (step-size dependent) | Exact (if correctly derived) | Exact (machine precision) |
| Implementation | Trivial but slow | Complex, error-prone | Zero effort — `jax.grad` handles it |
| Scales to | ~10 parameters | Large, if hand-derived | Arbitrary — 10 or 10,000 parameters |
| Non-linear models | Works but expensive | Requires linearisation | Works on arbitrary non-linearities |

With finite differences you re-run the simulation once per parameter. With back-propagation the cost is essentially **one forward + one backward pass**, regardless of parameter count. This makes it practical to optimise circuits with hundreds of design variables — something infeasible with parameter sweeps.

## How it works in circulax

```python
import jax
import optax
from circulax.utils import update_params_dict

# compile once — the circuit topology is fixed
circuit = compile_circuit(netlist, models)
groups  = circuit.groups

def loss_fn(params):
    # update component values inside the compiled groups (no recompilation)
    g = update_params_dict(groups, "capacitor", "C1", "C", params[0])
    g = update_params_dict(g,      "inductor",  "L1", "L", params[1])
    sol = simulate(g)
    return jnp.mean((sol - target) ** 2)

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

grads = jax.grad(loss_fn)(params)          # exact gradients, one pass
updates, opt_state = optimizer.update(grads, opt_state)
params = optax.apply_updates(params, updates)
```

## Examples

- [LC Filter Synthesis](examples/01_lc_filter_synthesis.md) — gradient-descent tuning of L/C values to match a Butterworth transfer function
- [Oscillator HB Tuning](examples/02_oscillator_hb_tuning.md) — optimising oscillator harmonics via harmonic balance
- [Photonic CMZ Demux](examples/03_photonic_cmz_demux.md) — inverse design of a cascaded Mach-Zehnder wavelength demux
- [HEMT PA Optimisation](examples/04_hemt_pa_optimization.md) — power amplifier matching network optimisation
