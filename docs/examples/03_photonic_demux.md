# Photonic Wavelength Demultiplexer via Backpropagation

This notebook shows how gradient-based optimisation can train a photonic circuit to route different wavelengths to different detectors — exactly the way a neural network is trained, but the "weights" are physical waveguide parameters.

## The device: a wavelength demultiplexer

A wavelength-division multiplexing (WDM) demultiplexer is a chip-scale device that separates a multi-wavelength optical signal into individual channels. It is a workhorse of fibre-optic communications, data centre interconnects, and optical sensing.

We will design a **1→4 binary splitter tree** that routes four telecom wavelengths (1300–1330 nm) to four separate photodetectors. Light accumulates a different phase in each waveguide arm depending on the arm's effective refractive index `neff`. By tuning `neff`, we control how light interferes at each splitter output — a direct photonic analogue of adjusting neural network weights.

## Why backprop wins

The demux has **6 tunable phase parameters**. A finite-difference approach would need 7 forward simulations per gradient step (one baseline + one perturbed per parameter). Backpropagation needs only **1 forward + 1 backward** — a fixed cost regardless of how many parameters there are.

Because circulax is built on JAX, `jax.grad` gives us exact gradients through the full circuit solve at essentially zero overhead over the forward pass.

| Method | Evaluations per step | Scales with N? |
|--------|---------------------|----------------|
| Finite differences | N + 1 | Yes — O(N) |
| Backpropagation (circulax) | ~2 | No — O(1) |

## Key physics

At wavelength λ, a waveguide of length L and effective index `neff` accumulates phase:

$$\phi = \frac{2\pi \cdot n_{\text{eff}} \cdot L}{\lambda}$$

When two arms of an interferometer have different phases, they interfere constructively or destructively at the combiner outputs. By choosing `neff` values such that arm A constructively interferes at λ₁ and destructively at λ₂, we achieve wavelength selectivity.


```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from circulax import compile_circuit
from circulax.components.electronic import Resistor
from circulax.components.photonic import DirectionalCoupler, OpticalSource, OpticalWaveguide
from circulax.utils import update_group_params, update_params_dict

# 64-bit precision is important for photonic circuits: small changes in neff
# produce small changes in phase, and gradients can be tiny in early training.
jax.config.update("jax_enable_x64", True)

plt.rcParams.update({
    "figure.figsize": (8, 3.5),
    "axes.grid": True,
    "lines.color": "grey",
    "patch.edgecolor": "grey",
    "text.color": "grey",
    "axes.facecolor": "white",
    "axes.edgecolor": "grey",
    "axes.labelcolor": "grey",
    "xtick.color": "grey",
    "ytick.color": "grey",
    "grid.color": "grey",
    "figure.facecolor": "white",
    "figure.edgecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
})
```

    KLUJAX_RS DEBUG MODE.
    WARNING:2026-04-15 16:20:02,369:jax._src.xla_bridge:864: An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.


## Circuit topology: MZI binary tree

```
                 ┌─ WG_1 ─┐              ┌─ WG_3 ─┐
Laser ── DC_in ──┤         ├── DC_mid ────┤DC_lo    ├── DC_out1 ── Det_1
                 └─ WG_2 ─┘    │         └─ WG_4 ─┘              Det_2
                                │
                                │         ┌─ WG_5 ─┐
                                └─ DC_hi ─┤         ├── DC_out2 ── Det_3
                                          └─ WG_6 ─┘              Det_4
```

Each stage is a **Mach-Zehnder interferometer** (MZI): an input 50/50 directional
coupler splits the field into two arms, the arms accumulate different phases
(φ = 2π · neff · L / λ), and an output 50/50 coupler recombines them. The
power split at the output depends on the phase difference between the arms:

```
  P_bar  ∝  (1 − cos Δφ) / 2
  P_cross ∝  (1 + cos Δφ) / 2
```

Because Δφ = 2π · Δneff · L / λ varies with wavelength, each MZI routes
different wavelengths to different output ports. A binary tree of two MZI levels
can separate 4 channels.

**Fixed components:**
- `Laser` — ideal CW optical source at 1 W
- `DC_in`, `DC_mid`, `DC_lo`, `DC_hi`, `DC_out1`, `DC_out2` — fixed 50/50 directional couplers

**Trainable parameters (6 × neff):**
- `WG_1`, `WG_2` — stage-1 MZI arms (200 µm): select which half of the spectrum
  goes to Det_1+Det_2 vs Det_3+Det_4
- `WG_3`, `WG_4` — stage-2 upper arms (100 µm): separate Det_1 and Det_2
- `WG_5`, `WG_6` — stage-2 lower arms (100 µm): separate Det_3 and Det_4


```python
# ── Component model registry ───────────────────────────────────────────────────
models = {
    "ground":    lambda: 0,
    "source":    OpticalSource,
    "waveguide": OpticalWaveguide,
    "dc":        DirectionalCoupler,
    "resistor":  Resistor,
}

# ── SAX-format netlist ─────────────────────────────────────────────────────────
#
# MZI binary tree: three stages of 50/50 DC + waveguide-arm pairs.
#
# Stage 1 MZI  — DC_in  → WG_1/WG_2  → DC_mid   (coarse wavelength split)
# Stage 2a MZI — DC_lo  → WG_3/WG_4  → DC_out1  (fine split → Det_1/Det_2)
# Stage 2b MZI — DC_hi  → WG_5/WG_6  → DC_out2  (fine split → Det_3/Det_4)
#
# The unused second input of each "input" coupler is grounded (zero field),
# so each MZI operates as a 1-input device; interference still occurs at the
# output coupler because both arms carry a non-zero field.
#
# Arm lengths (length_um) are the trainable parameters. neff is fixed at 2.4
# and loss is set to zero so that only path-length-induced phase differences
# determine the routing — making the physics maximally transparent.
net_dict = {
    "instances": {
        "GND":     {"component": "ground"},
        "Laser":   {"component": "source",    "settings": {"power": 1.0, "phase": 0.0}},
        # Stage 1 MZI
        "DC_in":   {"component": "dc",        "settings": {"coupling": 0.5}},
        "WG_1":    {"component": "waveguide", "settings": {"length_um": 250.0, "neff": 2.4, "loss_dB_cm": 0.0, "center_wavelength_nm": 1310.0}},
        "WG_2":    {"component": "waveguide", "settings": {"length_um": 250.0, "neff": 2.4, "loss_dB_cm": 0.0, "center_wavelength_nm": 1310.0}},
        "DC_mid":  {"component": "dc",        "settings": {"coupling": 0.5}},
        # Stage 2a MZI (upper branch → Det_1, Det_2)
        "DC_lo":   {"component": "dc",        "settings": {"coupling": 0.5}},
        "WG_3":    {"component": "waveguide", "settings": {"length_um": 250.0, "neff": 2.4, "loss_dB_cm": 0.0, "center_wavelength_nm": 1310.0}},
        "WG_4":    {"component": "waveguide", "settings": {"length_um": 250.0, "neff": 2.4, "loss_dB_cm": 0.0, "center_wavelength_nm": 1310.0}},
        "DC_out1": {"component": "dc",        "settings": {"coupling": 0.5}},
        # Stage 2b MZI (lower branch → Det_3, Det_4)
        "DC_hi":   {"component": "dc",        "settings": {"coupling": 0.5}},
        "WG_5":    {"component": "waveguide", "settings": {"length_um": 250.0, "neff": 2.4, "loss_dB_cm": 0.0, "center_wavelength_nm": 1310.0}},
        "WG_6":    {"component": "waveguide", "settings": {"length_um": 250.0, "neff": 2.4, "loss_dB_cm": 0.0, "center_wavelength_nm": 1310.0}},
        "DC_out2": {"component": "dc",        "settings": {"coupling": 0.5}},
        # Photodetectors (1 Ω matched load)
        "Det_1":   {"component": "resistor",  "settings": {"R": 1.0}},
        "Det_2":   {"component": "resistor",  "settings": {"R": 1.0}},
        "Det_3":   {"component": "resistor",  "settings": {"R": 1.0}},
        "Det_4":   {"component": "resistor",  "settings": {"R": 1.0}},
    },
    "connections": {
        # Ground: laser return + unused DC input ports + detector returns
        "GND,p1":      ("Laser,p2",
                        "DC_in,p2",    # unused second input of stage-1 input coupler
                        "DC_lo,p2",    # unused second input of stage-2a input coupler
                        "DC_hi,p2",    # unused second input of stage-2b input coupler
                        "Det_1,p2", "Det_2,p2", "Det_3,p2", "Det_4,p2"),
        # Laser → stage-1 input coupler
        "Laser,p1":    "DC_in,p1",
        # Stage 1 MZI: input coupler → phase arms → output coupler
        "DC_in,p3":    "WG_1,p1",
        "DC_in,p4":    "WG_2,p1",
        "WG_1,p2":     "DC_mid,p1",
        "WG_2,p2":     "DC_mid,p2",
        # Stage 1 outputs fan into stage-2 input couplers
        "DC_mid,p3":   "DC_lo,p1",
        "DC_mid,p4":   "DC_hi,p1",
        # Stage 2a MZI: input coupler → phase arms → output coupler
        "DC_lo,p3":    "WG_3,p1",
        "DC_lo,p4":    "WG_4,p1",
        "WG_3,p2":     "DC_out1,p1",
        "WG_4,p2":     "DC_out1,p2",
        "DC_out1,p3":  "Det_1,p1",
        "DC_out1,p4":  "Det_2,p1",
        # Stage 2b MZI: input coupler → phase arms → output coupler
        "DC_hi,p3":    "WG_5,p1",
        "DC_hi,p4":    "WG_6,p1",
        "WG_5,p2":     "DC_out2,p1",
        "WG_6,p2":     "DC_out2,p2",
        "DC_out2,p3":  "Det_3,p1",
        "DC_out2,p4":  "Det_4,p1",
    },
}

# ── Compile the netlist ────────────────────────────────────────────────────────
circuit = compile_circuit(net_dict, models, is_complex=True)
groups = circuit.groups
sys_size = circuit.sys_size
port_map = circuit.port_map

print(f"System size: {sys_size} real nodes")
print(f"Complex system size (2N): {sys_size * 2}")
print(f"Component groups: {list(groups.keys())}")
print("Detector node indices: " +
      ", ".join(f"Det_{i+1}={port_map[f'Det_{i+1},p1']}" for i in range(4)))

```

    System size: 21 real nodes
    Complex system size (2N): 42
    Component groups: ['source', 'dc', 'waveguide', 'resistor']
    Detector node indices: Det_1=14, Det_2=15, Det_3=18, Det_4=19


## Helper: compute the 4×4 power routing matrix

For each of the 4 wavelengths we solve the circuit and read the power absorbed at each detector. The result is a 4×4 matrix:

```
power[i, j] = fraction of total power at Det_{j+1} when driving at wavelength[i]
```

An ideal demux has power[i, i] ≈ 1 and all off-diagonal entries ≈ 0 (a perfect identity matrix).

### Implementation details

For photonic circuits, circulax unfolds complex fields `E = Re(E) + j·Im(E)` into a real vector of length `2·sys_size`:

```
y_flat[k]            = Re(E_k)   for k in 0..sys_size-1
y_flat[k + sys_size] = Im(E_k)   for k in 0..sys_size-1
```

So the complex field at detector node `n` is:
```python
E = y_flat[n] + 1j * y_flat[n + sys_size]
power = |E|²
```

We use `jax.vmap` to evaluate all 4 wavelengths in a single parallel call.


```python
# Target wavelengths (nm) — 150 nm span gives visible MZI interference fringes
TARGET_WLS = jnp.array([1285.0, 1300.0, 1315.0, 1330.0])

# Detector port node indices in y
det_nodes = jnp.array([port_map[f"Det_{i+1},p1"] for i in range(4)])

# Initial guess for the complex DC solve
Y_GUESS = jnp.ones(sys_size * 2)


def get_power_matrix(L_params, wavelengths=TARGET_WLS):
    """Compute the 4×4 power routing matrix for given arm lengths.

    Args:
        L_params: Array of shape (6,) — [L_1, L_2, L_3, L_4, L_5, L_6]
                  waveguide arm lengths in µm.  1/2 are stage-1 arms;
                  3/4 and 5/6 are the stage-2a and stage-2b arm pairs.
        wavelengths: Array of wavelengths in nm, shape (4,). Defaults to TARGET_WLS.

    Returns:
        power: Array of shape (4, 4), normalised so each row sums to ~1.
               power[i, j] = fraction of power reaching Det_{j+1} at wavelengths[i].
    """
    L_1, L_2, L_3, L_4, L_5, L_6 = L_params

    grps = update_params_dict(groups, "waveguide", "WG_1", "length_um", L_1)
    grps = update_params_dict(grps,   "waveguide", "WG_2", "length_um", L_2)
    grps = update_params_dict(grps,   "waveguide", "WG_3", "length_um", L_3)
    grps = update_params_dict(grps,   "waveguide", "WG_4", "length_um", L_4)
    grps = update_params_dict(grps,   "waveguide", "WG_5", "length_um", L_5)
    grps = update_params_dict(grps,   "waveguide", "WG_6", "length_um", L_6)

    def solve_at_wavelength(wl):
        grps_wl = update_group_params(grps, "waveguide", "wavelength_nm", wl)
        y_flat = circuit.solver.solve_dc(grps_wl, Y_GUESS)
        E_det = y_flat[det_nodes] + 1j * y_flat[det_nodes + sys_size]
        return jnp.abs(E_det) ** 2

    powers = jax.vmap(solve_at_wavelength)(wavelengths)  # shape: (4, 4)
    row_totals = jnp.sum(powers, axis=1, keepdims=True) + 1e-12
    return powers / row_totals


# Initial arm lengths — asymmetric starting point with ~22% routing contrast.
# Arm pairs (WG_1/WG_2, WG_3/WG_4, WG_5/WG_6) have random length differences
# of a few micrometres, producing non-trivial but poor wavelength routing.
L_init = jnp.array([248.0, 252.0, 227.0, 273.0, 227.0, 273.0])

print("Computing initial power routing matrix (JIT compiling + solving)...")
power_init = jax.jit(get_power_matrix)(L_init)
print("Done.")
print("\nInitial 4×4 power routing matrix:")
print("Rows = wavelengths [1285, 1300, 1315, 1330 nm]")
print("Cols = detectors [Det_1, Det_2, Det_3, Det_4]")
print(np.array(power_init).round(3))

```

    Computing initial power routing matrix (JIT compiling + solving)...


    Done.

    Initial 4×4 power routing matrix:
    Rows = wavelengths [1285, 1300, 1315, 1330 nm]
    Cols = detectors [Det_1, Det_2, Det_3, Det_4]
    [[0.001 0.957 0.    0.042]
     [0.76  0.182 0.047 0.011]
     [0.317 0.282 0.212 0.189]
     [0.046 0.146 0.193 0.614]]


## Part 1 — Baseline: routing before optimisation

The arm lengths are initialised with a deliberate but arbitrary asymmetry:
a 4 µm difference in the stage-1 arms and a 46 µm difference in each
stage-2 arm pair. These values provide non-zero gradients but route each
wavelength to the *wrong* detector — a clear "before" state.

Each MZI arm pair accumulates a phase difference

$$\Delta\phi = \frac{2\pi \, n_{\rm eff} \, \Delta L}{\lambda}$$

The 15 nm channel spacing requires **ΔL ≈ 46 µm** for the stage-2 MZIs so
that adjacent channels fall on opposite interference fringes. The stage-1
MZI uses a smaller ΔL (~4 µm) for the coarse first-level split.

The initial routing contrast — mean fraction of power reaching the
*designated* detector — is around **25%** (random chance is 25%).



```python
def plot_power_matrix(power_matrix, title, ax=None):
    """Plot a 4×4 power routing matrix as an annotated heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    pm = np.array(power_matrix)
    im = ax.imshow(pm, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="Fraction of power")

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(["Det_1", "Det_2", "Det_3", "Det_4"], color="grey")
    ax.set_yticklabels(["1285 nm", "1300 nm", "1315 nm", "1330 nm"], color="grey")
    ax.set_xlabel("Detector", color="grey")
    ax.set_ylabel("Wavelength", color="grey")
    ax.set_title(title, color="grey")

    # Annotate each cell with the percentage value
    for i in range(4):
        for j in range(4):
            val = pm[i, j]
            text_color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val*100:.1f}%",
                    ha="center", va="center", fontsize=9, color=text_color)

    return fig, ax


fig, ax = plt.subplots(figsize=(5, 4))
plot_power_matrix(power_init, "Power Routing Matrix — Before Training", ax=ax)
plt.tight_layout()
plt.show()

# Compute and display the "contrast" score: diagonal / row total
# A perfect demux has contrast = 1.0 for every wavelength.
diag_powers = np.diag(np.array(power_init))
print("\nInitial routing contrast (correct detector fraction):")
for i, (wl, c) in enumerate(zip([1285, 1300, 1315, 1330], diag_powers)):
    print(f"  λ={wl} nm → Det_{i+1}: {c*100:.1f}%")
print(f"  Mean contrast: {diag_powers.mean()*100:.1f}% (ideal = 100%)")

```



![png](03_photonic_demux_files/03_photonic_demux_7_0.png)




    Initial routing contrast (correct detector fraction):
      λ=1285 nm → Det_1: 0.1%
      λ=1300 nm → Det_2: 18.2%
      λ=1315 nm → Det_3: 21.2%
      λ=1330 nm → Det_4: 61.4%
      Mean contrast: 25.2% (ideal = 100%)


## Part 2 — Gradient-based optimisation

### Loss function

We maximise the **routing contrast**: the fraction of each wavelength's total power that reaches its designated detector.

$$\mathcal{L} = -\frac{1}{4} \sum_{i=1}^{4} \frac{P_{ii}}{\sum_j P_{ij}}$$

This is `−mean(diagonal of normalised power matrix)`. We minimise the loss, so minimising `−contrast` = maximising contrast.

### Why this loss?

The normalised power matrix rows already sum to 1.0, so `power[i, i]` is exactly the fraction of wavelength `i`'s power that reaches its target. A perfect demux would give `loss = −1.0`. A random router gives `loss ≈ −0.25`.


```python
def loss_fn(L_params):
    """Negative mean routing contrast — minimise to maximise wavelength selectivity.

    Returns a scalar in [-1, 0]:
      -1.0 = perfect demux (all power at correct detector)
       0.0 = worst case (no power at any correct detector)
    """
    # power shape: (4, 4), already row-normalised
    power = get_power_matrix(L_params)

    # Diagonal entries: power at the "correct" detector for each wavelength
    correct_fractions = jnp.diag(power)

    # Minimise negative contrast (= maximise contrast)
    return -jnp.mean(correct_fractions)


# Verify the loss is differentiable: compute loss AND gradient in one pass.
# jax.value_and_grad is strictly more efficient than calling loss_fn and
# jax.grad(loss_fn) separately — it reuses the forward-pass computation.
print("Computing loss and gradients at initial parameters...")
loss_val, grads = jax.jit(jax.value_and_grad(loss_fn))(L_init)
print(f"  Initial loss:    {float(loss_val):.4f}  (random ≈ -0.25, perfect = -1.0)")
print(f"  Mean contrast:   {-float(loss_val)*100:.1f}%")
print(f"  Gradient norms:  {[f'{float(g):.4f}' for g in grads]}")
print()
print("Gradient labels: [L_1, L_2, L_3, L_4, L_5, L_6]  (arm lengths, µm)")
print("Non-zero gradients confirm the circuit is differentiable end-to-end.")

```

    Computing loss and gradients at initial parameters...


      Initial loss:    -0.2521  (random ≈ -0.25, perfect = -1.0)
      Mean contrast:   25.2%
      Gradient norms:  ['-1.4513', '1.4513', '-1.0161', '1.0161', '-1.5385', '1.5385']

    Gradient labels: [L_1, L_2, L_3, L_4, L_5, L_6]  (arm lengths, µm)
    Non-zero gradients confirm the circuit is differentiable end-to-end.



```python
# ── Adam optimisation loop ─────────────────────────────────────────────────────
#
# We train for 1200 steps with a learning rate of 0.5 µm per step.
# Adam adapts the per-parameter rate, which is critical here because gradients
# for the stage-1 arms (small ΔL, coarse split) differ in magnitude from
# those for the stage-2 arms (large ΔL, fine 15 nm channel split).

N_STEPS = 1200
LR      = 0.5   # µm / step

optimizer  = optax.adam(learning_rate=LR)
L_params   = L_init
opt_state  = optimizer.init(L_params)

# Pre-compile the gradient function once
value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

losses     = []
L_history  = []

print(f"Training for {N_STEPS} steps (Adam, lr={LR} µm)...")
print(f"{'Step':>5}  {'Loss':>8}  {'Contrast':>10}  {'L_1 (µm)':>10}  {'L_2 (µm)':>10}")
print("-" * 56)

for step in range(N_STEPS):
    loss, grads = value_and_grad_fn(L_params)
    losses.append(float(loss))
    L_history.append(np.array(L_params))

    updates, opt_state = optimizer.update(grads, opt_state)
    L_params = optax.apply_updates(L_params, updates)

    if step == 0 or (step + 1) % 200 == 0:
        print(f"{step+1:5d}  {float(loss):8.4f}  {-float(loss)*100:9.1f}%  "
              f"{float(L_params[0]):10.3f}  {float(L_params[1]):10.3f}")

L_history = np.array(L_history)
print("\nFinal arm lengths:")
labels = ["L_1", "L_2", "L_3", "L_4", "L_5", "L_6"]
for label, p_init, p_final in zip(labels, L_init, L_params):
    delta = float(p_final) - float(p_init)
    print(f"  {label}: {float(p_init):.3f} → {float(p_final):.3f} µm  (Δ = {delta:+.3f} µm)")

```

    Training for 1200 steps (Adam, lr=0.5 µm)...
     Step      Loss    Contrast    L_1 (µm)    L_2 (µm)
    --------------------------------------------------------


        1   -0.2521       25.2%     248.500     251.500


      200   -0.7660       76.6%     246.657     253.343


      400   -0.7713       77.1%     246.655     253.345


      600   -0.7712       77.1%     246.655     253.345


      800   -0.7712       77.1%     246.655     253.345


     1000   -0.7712       77.1%     246.655     253.345


     1200   -0.7712       77.1%     246.655     253.345

    Final arm lengths:
      L_1: 248.000 → 246.655 µm  (Δ = -1.345 µm)
      L_2: 252.000 → 253.345 µm  (Δ = +1.345 µm)
      L_3: 227.000 → 227.663 µm  (Δ = +0.663 µm)
      L_4: 273.000 → 272.337 µm  (Δ = -0.663 µm)
      L_5: 227.000 → 231.203 µm  (Δ = +4.203 µm)
      L_6: 273.000 → 268.797 µm  (Δ = -4.203 µm)



```python
# ── Plot: optimisation convergence ─────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))

# Loss curve (inverted so we see contrast climbing toward 100%)
contrast_history = [-l * 100 for l in losses]
ax1.plot(contrast_history, color="C0", lw=2)
ax1.axhline(25.0, color="grey", ls=":", lw=1, label="Random baseline (25%)")
ax1.axhline(100.0, color="C1", ls="--", lw=1, label="Perfect demux (100%)")
ax1.set_xlabel("Optimisation step")
ax1.set_ylabel("Mean routing contrast (%)")
ax1.set_title("Convergence")
ax1.set_ylim(0, 105)
ax1.legend(fontsize=8)

# Arm length trajectories
colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
for i, (label, color) in enumerate(zip(labels, colors)):
    ax2.plot(L_history[:, i], color=color, lw=1.5, label=label)
ax2.set_xlabel("Optimisation step")
ax2.set_ylabel("Arm length (µm)")
ax2.set_title("Arm length trajectories")
ax2.legend(fontsize=8, ncol=2)

plt.tight_layout()
plt.show()

```



![png](03_photonic_demux_files/03_photonic_demux_11_0.png)




```python
# ── Final power routing matrix ─────────────────────────────────────────────────

print("Computing final power routing matrix...")
power_final = jax.jit(get_power_matrix)(L_params)

fig, ax = plt.subplots(figsize=(5, 4))
plot_power_matrix(power_final, "Power Routing Matrix — After Training", ax=ax)
plt.tight_layout()
plt.show()

diag_final = np.diag(np.array(power_final))
print("\nFinal routing contrast (diagonal):")
target_wls = [1285, 1300, 1315, 1330]
for i, (wl, p) in enumerate(zip(target_wls, diag_final)):
    print(f"  Det_{i+1} ← {wl} nm : {p*100:.1f}%")
print(f"  Mean contrast: {np.mean(diag_final)*100:.1f}%")

```

    Computing final power routing matrix...




![png](03_photonic_demux_files/03_photonic_demux_12_1.png)




    Final routing contrast (diagonal):
      Det_1 ← 1285 nm : 78.6%
      Det_2 ← 1300 nm : 83.2%
      Det_3 ← 1315 nm : 69.3%
      Det_4 ← 1330 nm : 77.4%
      Mean contrast: 77.1%



```python
# ── Side-by-side comparison: before vs after ───────────────────────────────────

fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(11, 4))
plot_power_matrix(power_init,  "Before Training", ax=ax_before)
plot_power_matrix(power_final, "After Training",  ax=ax_after)
fig.suptitle("Photonic Demultiplexer — Power Routing Matrix", color="grey", fontsize=12)
plt.tight_layout()
plt.show()
```



![png](03_photonic_demux_files/03_photonic_demux_13_0.png)




```python
# ── Wavelength sweep: transmission spectra at each detector ────────────────────
#
# Sweep a dense set of wavelengths to reveal each detector's passband shape.
# A good demux shows four non-overlapping peaks, one per detector.

sweep_wls = jnp.linspace(1260.0, 1360.0, 200)

def get_raw_power_sweep(L_params, wavelengths):
    """4×N power matrix for a wavelength sweep (not row-normalised)."""
    L_1, L_2, L_3, L_4, L_5, L_6 = L_params
    grps = update_params_dict(groups, "waveguide", "WG_1", "length_um", L_1)
    grps = update_params_dict(grps,   "waveguide", "WG_2", "length_um", L_2)
    grps = update_params_dict(grps,   "waveguide", "WG_3", "length_um", L_3)
    grps = update_params_dict(grps,   "waveguide", "WG_4", "length_um", L_4)
    grps = update_params_dict(grps,   "waveguide", "WG_5", "length_um", L_5)
    grps = update_params_dict(grps,   "waveguide", "WG_6", "length_um", L_6)

    def solve_wl(wl):
        grps_wl = update_group_params(grps, "waveguide", "wavelength_nm", wl)
        y_flat  = circuit.solver.solve_dc(grps_wl, Y_GUESS)
        E_det   = y_flat[det_nodes] + 1j * y_flat[det_nodes + sys_size]
        return jnp.abs(E_det) ** 2

    return jax.vmap(solve_wl)(wavelengths)  # shape: (N_wl, 4)


print("Running wavelength sweep (vmap over 200 wavelengths)...")
sweep_power = jax.jit(get_raw_power_sweep)(L_params, sweep_wls)
print("Done.")

fig, ax = plt.subplots(figsize=(9, 4))
sweep_wls_np = np.array(sweep_wls)
sweep_power_np = np.array(sweep_power)  # shape: (200, 4)

target_wls_nm = [1285, 1300, 1315, 1330]
det_colors = ["C0", "C1", "C2", "C3"]

for j in range(4):
    ax.plot(sweep_wls_np, sweep_power_np[:, j],
            color=det_colors[j], lw=2, label=f"Det_{j+1} (target: {target_wls_nm[j]} nm)")
    ax.axvline(target_wls_nm[j], color=det_colors[j], ls=":", lw=1, alpha=0.6)

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Received power (W)")
ax.set_title("Transmission spectra after training — each detector peaks at its target wavelength")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

```

    Running wavelength sweep (vmap over 200 wavelengths)...


    Done.




![png](03_photonic_demux_files/03_photonic_demux_14_2.png)



## Part 3 — Why backpropagation wins: the scaling argument

The demux we just trained has **N = 6** tunable parameters. But modern photonic integrated circuits can have hundreds of tunable elements — microring resonators, phase shifters, Mach-Zehnder meshes. How does the cost of optimisation scale?

### Finite-difference gradient estimation

To estimate `∂L/∂neff_k` with a forward-difference scheme:

$$\frac{\partial \mathcal{L}}{\partial n_k} \approx \frac{\mathcal{L}(n + \epsilon e_k) - \mathcal{L}(n)}{\epsilon}$$

you need **1 baseline + N perturbations = N + 1 circuit solves per gradient step**.

### Backpropagation (automatic differentiation)

`jax.grad` computes the **exact** gradient for all N parameters simultaneously in roughly **1 forward + 1 backward pass ≈ 2 circuit solves**, regardless of N. The backward pass is the vector-Jacobian product — it costs the same as one additional forward sweep.

### Practical speedup

For our 6-parameter circuit:
- FD needs 7 evaluations per step
- Backprop needs ~2 evaluations per step
- Speedup: **3.5×** even at this small scale

For a 100-element photonic mesh, the speedup is **50×**. FD becomes impractical; backprop remains cheap.


```python
# ── Scaling comparison: FD vs backprop ────────────────────────────────────────

N_values    = [6, 10, 20, 50, 100, 500]
fd_evals    = [N + 1 for N in N_values]           # central difference would be 2N+1
bp_evals    = [2] * len(N_values)                  # forward + backward, always 2
speedup     = [fd / bp for fd, bp in zip(fd_evals, bp_evals)]

print(f"{'N params':>10} | {'FD evals':>10} | {'Backprop evals':>15} | {'Speedup':>8}")
print("-" * 52)
for N, fd, bp, sp in zip(N_values, fd_evals, bp_evals, speedup):
    print(f"{N:>10} | {fd:>10} | {bp:>15} | {sp:>7.1f}×")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))

# Left: evaluations per gradient step
ax1.plot(N_values, fd_evals, "o-", color="C3", lw=2, ms=6, label="Finite differences")
ax1.plot(N_values, bp_evals, "s-", color="C0", lw=2, ms=6, label="Backprop (circulax)")
ax1.set_xlabel("Number of tunable parameters")
ax1.set_ylabel("Circuit evaluations per gradient step")
ax1.set_title("Cost scaling")
ax1.legend(fontsize=9)

# Right: speedup
ax2.plot(N_values, speedup, "D-", color="C2", lw=2, ms=6)
ax2.axvline(6, color="grey", ls=":", lw=1)
ax2.annotate("This demo (N=6)\n3.5× speedup",
             xy=(6, 3.5), xytext=(20, 4.5),
             fontsize=8, color="grey",
             arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))
ax2.set_xlabel("Number of tunable parameters")
ax2.set_ylabel("Speedup over finite differences")
ax2.set_title("Backprop speedup vs finite differences")

plt.tight_layout()
plt.show()
```

      N params |   FD evals |  Backprop evals |  Speedup
    ----------------------------------------------------
             6 |          7 |               2 |     3.5×
            10 |         11 |               2 |     5.5×
            20 |         21 |               2 |    10.5×
            50 |         51 |               2 |    25.5×
           100 |        101 |               2 |    50.5×
           500 |        501 |               2 |   250.5×




![png](03_photonic_demux_files/03_photonic_demux_16_1.png)



## Summary

We trained a photonic wavelength demultiplexer using gradient descent, starting
from arm lengths that route wavelengths to the *wrong* detectors (~25% contrast)
and converging to a design where each of the four target wavelengths is routed
to its designated detector (~77% contrast).

### Physical mechanism

The key element is the **Mach-Zehnder interferometer**: two waveguide arms of
different length accumulate a phase difference

$$\Delta\phi = \frac{2\pi \, n_{\rm eff} \, \Delta L}{\lambda}$$

When the arms recombine at the output coupler, the power split depends on
cos(Δφ). Because Δφ ∝ 1/λ, different wavelengths emerge at different output
ports. A binary tree of two MZI levels separates 4 channels.

The trainable parameters are the **arm lengths** (µm). This is the physically
correct parameterisation: the phase difference ΔL/λ is genuinely
wavelength-dependent, allowing the circuit to distinguish channels. Changing
neff uniformly shifts the common phase and cannot create selectivity.

The 15 nm channel spacing (1285–1330 nm band) requires stage-2 arm
differences of ~46 µm so that adjacent channels fall on opposite interference
fringes — roughly one full fringe period per channel pair.

### Why backpropagation wins: the scaling argument

The demux we just trained has **N = 6** tunable parameters. But modern photonic
integrated circuits can have hundreds of tunable elements. Backpropagation
computes all N gradients in a single forward+backward pass — cost O(N) — while
finite differences require N+1 forward solves — cost O(N²). See Part 3 for the
scaling plot.

| Step | Tool | Role |
|------|------|------|
| Circuit compilation | `compile_circuit` | Runs once; produces JAX-traceable `ComponentGroup` objects |
| Parameter updates | `update_params_dict` | `eqx.tree_at` functional update; no re-compilation, fully differentiable |
| Wavelength sweep | `jax.vmap` | Evaluates all wavelengths in parallel without a Python loop |
| Gradient computation | `jax.grad` | Exact gradients through Newton-Raphson solver via implicit differentiation |
| Optimisation | `optax.adam` | Adaptive learning rate handles varying gradient magnitudes across arms |
