# Photonic CMZ Wavelength Demultiplexer — Cascaded Mach-Zehnder Lattice Filter

Inverse design will be used to tune the 11 trainable parameters (7 directional-coupler coefficients + 4 arm lengths) of a Cascaded Mach-Zehnder lattice filter to maximise the **mean wavelength routing contrast** across four channels (1285, 1300, 1315, 1330 nm), resulting in the following:

<!-- animation -->

The following is how this is performed in circulax.

This notebook demonstrates how a **Cascaded Mach-Zehnder (CMZ) lattice filter** architecture from Horst et al. 2013 (*Optics Express*, DOI: 10.1364/OE.21.011652) achieves flat passbands and superior stopband rejection compared to a simple MZI binary tree.

## Why CMZ beats a simple MZI

A single-stage MZI has a sinusoidal transfer function — its passband rolls off gradually, so channels at the passband edges leak into neighbouring detectors. The CMZ topology replaces the root splitter node with a **multi-stage lattice filter**: a chain of directional couplers interleaved with waveguide arm pairs of increasing path-length difference. This creates a Chebyshev-like response with a flat top and sharp skirts.

## Schematic




### Design equations (Horst et al. 2013)

For channels spaced δλ = 15 nm around λ_c = 1307.5 nm with n_gr = 4, n_eff = 2.4:

$$\Delta L_{\rm Base} = \frac{\lambda_c^2}{2\, \delta\lambda\, n_{\rm gr}} = \frac{1307.5^2}{2 \times 15 \times 4 \times 1000} = 14.246\; \mu{\rm m}$$

$$\Delta L_{\rm FS} = \frac{\lambda_c}{n_{\rm eff}} = \frac{1307.5}{2.4 \times 1000} = 0.5448\; \mu{\rm m}$$

| Element | Long arm (µm) | Short arm (µm) | ΔL |
|---------|--------------|----------------|----|
| Root stage 1 | 264.246 | 250.0 | ΔL_Base = 14.246 |
| Root stage 2 | 278.492 | 250.0 | 2×ΔL_Base = 28.492 |
| Leaf A | 257.123 | 250.0 | ΔL_Base/2 = 7.123 |
| Leaf B | 257.532 | 250.0 | ΔL_Base/2 + 0.75×ΔL_FS = 7.532 |

The root coupling coefficients `(0.5, 0.29, 0.08)` are the Horst et al. analytic optimum for a flat-top 2-stage Chebyshev filter response.


## Trainable parameters (11 total)

- **7 DC coupling coefficients**: `k_r1, k_r2, k_r3` (root) + `k_a1, k_a2` (leaf A) + `k_b1, k_b2` (leaf B)
- **4 long arm lengths**: `L_r1a, L_r2a` (root stages) + `L_a1` (leaf A) + `L_b1` (leaf B)

Paper-derived initial values use coupling `(0.5, 0.29, 0.08)` for the root — designed analytically for a flat Chebyshev passband — and equal-split (0.5) for leaves.

## Backpropagation implementation

With 11 tunable parameters, finite-difference gradient estimation costs **12 circuit solves** per gradient step. `jax.grad` computes the exact gradient for all 11 parameters in **~2 solves** (forward + backward). For a photonic integrated circuit with 100+ elements, backprop is the only practical route to optimisation.


```
import jax
import jax.numpy as jnp
import numpy as np
import optax

from circulax import compile_circuit
from circulax.components.electronic import Resistor
from circulax.components.photonic import DirectionalCoupler, OpticalSource, OpticalWaveguide

# 64-bit precision is critical for photonic circuits: small phase differences
# between arm lengths produce small field changes, and gradients can be tiny.
jax.config.update("jax_enable_x64", True)


# ── Component model registry ──────────────────────────────────────────────
models = {
    "ground":    lambda: 0,
    "source":    OpticalSource,
    "waveguide": OpticalWaveguide,
    "dc":        DirectionalCoupler,
    "resistor":  Resistor,
}

# ── Design parameters from Horst et al. 2013 ───────────────────────────────
# λ_center = mean of [1285, 1300, 1315, 1330] nm
# δλ = 15 nm channel spacing; n_eff = 2.4; n_gr = 4 (group index)
DL_BASE = 14.246   # µm  — ΔL_Base = λ² / (2·δλ·n_gr·1000)
DL_FS   = 0.5448   # µm  — ΔL_FS   = λ / (n_eff·1000)

# Arm lengths: all short arms fixed at 250.0 µm
L_R1A_INIT = 250.0 + DL_BASE           # 264.246 µm  (root stage 1 long arm)
L_R2A_INIT = 250.0 + 2 * DL_BASE      # 278.492 µm  (root stage 2 long arm)
L_A1_INIT  = 250.0 + DL_BASE / 2      # 257.123 µm  (leaf A long arm)
L_B1_INIT  = 250.0 + DL_BASE / 2 + 0.75 * DL_FS  # 257.532 µm  (leaf B long arm)

# ── SAX-format netlist ─────────────────────────────────────────────────────
#
# CMZ topology:
#   Root: DC_r1 → WG_r1a/WG_r1b → DC_r2 → WG_r2a/WG_r2b → DC_r3  (2-stage lattice)
#   Leaf A: DC_a1 → WG_a1/WG_a2 → DC_a2 → Det_1/Det_3             (1-stage MZI)
#   Leaf B: DC_b1 → WG_b1/WG_b2 → DC_b2 → Det_2/Det_4             (1-stage MZI)
#
# Coupling coefficients: root uses paper-derived (0.5, 0.29, 0.08);
# leaf DCs start at 0.5 (equal split).
net_dict = {
    "instances": {
        "GND":    {"component": "ground"},
        "Laser":  {"component": "source",    "settings": {"power": 1.0, "phase": 0.0}},
        # Root 2-stage lattice filter (3 DCs + 2 waveguide arm pairs)
        "DC_r1":  {"component": "dc",        "settings": {"coupling": 0.5}},
        "DC_r2":  {"component": "dc",        "settings": {"coupling": 0.29}},
        "DC_r3":  {"component": "dc",        "settings": {"coupling": 0.08}},
        "WG_r1a": {"component": "waveguide", "settings": {"length_um": L_R1A_INIT, "neff": 2.4, "loss_dB_cm": 0.0, "center_wavelength_nm": 1310.0}},
        "WG_r1b": {"component": "waveguide", "settings": {"length_um": 250.0,       "neff": 2.4, "loss_dB_cm": 0.0, "center_wavelength_nm": 1310.0}},
        "WG_r2a": {"component": "waveguide", "settings": {"length_um": L_R2A_INIT, "neff": 2.4, "loss_dB_cm": 0.0, "center_wavelength_nm": 1310.0}},
        "WG_r2b": {"component": "waveguide", "settings": {"length_um": 250.0,       "neff": 2.4, "loss_dB_cm": 0.0, "center_wavelength_nm": 1310.0}},
        # Leaf A (bar output of root → channels 1285 + 1315 nm)
        "DC_a1":  {"component": "dc",        "settings": {"coupling": 0.5}},
        "DC_a2":  {"component": "dc",        "settings": {"coupling": 0.5}},
        "WG_a1":  {"component": "waveguide", "settings": {"length_um": L_A1_INIT, "neff": 2.4, "loss_dB_cm": 0.0, "center_wavelength_nm": 1310.0}},
        "WG_a2":  {"component": "waveguide", "settings": {"length_um": 250.0,      "neff": 2.4, "loss_dB_cm": 0.0, "center_wavelength_nm": 1310.0}},
        # Leaf B (cross output of root → channels 1300 + 1330 nm)
        "DC_b1":  {"component": "dc",        "settings": {"coupling": 0.5}},
        "DC_b2":  {"component": "dc",        "settings": {"coupling": 0.5}},
        "WG_b1":  {"component": "waveguide", "settings": {"length_um": L_B1_INIT, "neff": 2.4, "loss_dB_cm": 0.0, "center_wavelength_nm": 1310.0}},
        "WG_b2":  {"component": "waveguide", "settings": {"length_um": 250.0,      "neff": 2.4, "loss_dB_cm": 0.0, "center_wavelength_nm": 1310.0}},
        # Photodetectors (1 Ω matched load — power ∝ |E|²)
        "Det_1":  {"component": "resistor",  "settings": {"R": 1.0}},
        "Det_2":  {"component": "resistor",  "settings": {"R": 1.0}},
        "Det_3":  {"component": "resistor",  "settings": {"R": 1.0}},
        "Det_4":  {"component": "resistor",  "settings": {"R": 1.0}},
    },
    "connections": {
        # GND bus: laser return + unused DC second inputs + detector returns
        "GND,p1":    ("Laser,p2", "DC_r1,p2", "DC_a1,p2", "DC_b1,p2",
                       "Det_1,p2", "Det_2,p2", "Det_3,p2", "Det_4,p2"),
        # Signal path: laser → root lattice
        "Laser,p1":  "DC_r1,p1",
        # Root stage 1: DC_r1 → arm pair → DC_r2
        "DC_r1,p3":  "WG_r1a,p1",
        "DC_r1,p4":  "WG_r1b,p1",
        "WG_r1a,p2": "DC_r2,p1",
        "WG_r1b,p2": "DC_r2,p2",
        # Root stage 2: DC_r2 → arm pair → DC_r3
        "DC_r2,p3":  "WG_r2a,p1",
        "DC_r2,p4":  "WG_r2b,p1",
        "WG_r2a,p2": "DC_r3,p1",
        "WG_r2b,p2": "DC_r3,p2",
        # Root → leaves: bar (p3) → Leaf A, cross (p4) → Leaf B
        "DC_r3,p3":  "DC_a1,p1",
        "DC_r3,p4":  "DC_b1,p1",
        # Leaf A: DC_a1 → arm pair → DC_a2 → Det_1 / Det_3
        "DC_a1,p3":  "WG_a1,p1",
        "DC_a1,p4":  "WG_a2,p1",
        "WG_a1,p2":  "DC_a2,p1",
        "WG_a2,p2":  "DC_a2,p2",
        "DC_a2,p3":  "Det_1,p1",
        "DC_a2,p4":  "Det_3,p1",
        # Leaf B: DC_b1 → arm pair → DC_b2 → Det_2 / Det_4
        "DC_b1,p3":  "WG_b1,p1",
        "DC_b1,p4":  "WG_b2,p1",
        "WG_b1,p2":  "DC_b2,p1",
        "WG_b2,p2":  "DC_b2,p2",
        "DC_b2,p3":  "Det_2,p1",
        "DC_b2,p4":  "Det_4,p1",
    },
    "ports": {
        "det1": "Det_1,p1",
        "det2": "Det_2,p1",
        "det3": "Det_3,p1",
        "det4": "Det_4,p1",
    },
}

# ── Compile and analyse ────────────────────────────────────────────────────
circuit = compile_circuit(net_dict, models, is_complex=True)

print(f"System size: {circuit.sys_size} real nodes  ({circuit.sys_size * 2} complex DOFs)")
print(f"Component groups: {list(circuit.groups.keys())}")
print(f"Detector ports: {list(net_dict['ports'])}")

# ── Solver bookkeeping ─────────────────────────────────────────────────────
TARGET_WLS = jnp.array([1285.0, 1300.0, 1315.0, 1330.0])   # nm
Y_GUESS    = jnp.ones(circuit.sys_size * 2)   # flat initial guess for Newton-Raphson
```

## Trainable parameters

There are **11 trainable parameters** in total:

### Coupling coefficients (7)

| Parameter | Initial value | Role |
|-----------|--------------|------|
| `k_r1` | 0.5 | Root stage 1 DC — 50/50 split at first coupler |
| `k_r2` | 0.29 | Root stage 2 DC — asymmetric coupling for flat passband |
| `k_r3` | 0.08 | Root stage 3 DC — small coupling for steep roll-off |
| `k_a1` | 0.5 | Leaf A input DC |
| `k_a2` | 0.5 | Leaf A output DC |
| `k_b1` | 0.5 | Leaf B input DC |
| `k_b2` | 0.5 | Leaf B output DC |

The root coupling tuple `(0.5, 0.29, 0.08)` comes from Horst et al.'s analytic optimisation for a 2-stage Chebyshev flat-top filter.

### Long arm lengths (4)

| Parameter | Initial value (µm) | Design formula |
|-----------|-------------------|----------------|
| `L_r1a` | 264.246 | 250 + ΔL_Base |
| `L_r2a` | 278.492 | 250 + 2·ΔL_Base |
| `L_a1`  | 257.123 | 250 + ΔL_Base/2 |
| `L_b1`  | 257.532 | 250 + ΔL_Base/2 + 0.75·ΔL_FS |

All short arms are fixed at 250.0 µm and are not trained — only the long arm lengths determine the MZI phase difference.

### Why Adam handles the parameter scale mismatch

Coupling coefficients live in [0, 1] while arm lengths are in the range 250–280 µm. Adam's adaptive per-parameter learning rate normalises these automatically: it divides each gradient by a running estimate of its second moment, so coupling and length updates are implicitly rescaled without any manual tuning.


```
def _parameter_updates(params, wavelength_nm=None):
    """Map the optimizer vector to high-level Circuit parameter updates."""
    k_r1, k_r2, k_r3, k_a1, k_a2, k_b1, k_b2, L_r1a, L_r2a, L_a1, L_b1 = params
    updates = {
        "DC_r1.coupling": k_r1,
        "DC_r2.coupling": k_r2,
        "DC_r3.coupling": k_r3,
        "DC_a1.coupling": k_a1,
        "DC_a2.coupling": k_a2,
        "DC_b1.coupling": k_b1,
        "DC_b2.coupling": k_b2,
        "WG_r1a.length_um": L_r1a,
        "WG_r2a.length_um": L_r2a,
        "WG_a1.length_um": L_a1,
        "WG_b1.length_um": L_b1,
    }
    if wavelength_nm is not None:
        updates["wavelength_nm"] = wavelength_nm
    return updates


def _detector_fields(y_flat):
    return jnp.stack([circuit.port(y_flat, f"det{i}") for i in range(1, 5)])


def get_power_matrix(params, wavelengths=TARGET_WLS):
    """Compute the 4×4 power routing matrix for given circuit parameters.

    Args:
        params: Array of shape (11,) — [k_r1, k_r2, k_r3, k_a1, k_a2,
                k_b1, k_b2, L_r1a, L_r2a, L_a1, L_b1]
                7 DC coupling coefficients + 4 long arm lengths (µm).
        wavelengths: Array of wavelengths in nm, shape (N,).

    Returns:
        power: Array of shape (N, 4), row-normalised.
               power[i, j] = fraction of power reaching Det_{j+1} when driving
               at wavelengths[i]. An ideal demux has power ≈ identity.
    """
    def solve_at_wavelength(wl):
        y_flat = circuit.dc(params=_parameter_updates(params, wl), y_guess=Y_GUESS)
        return jnp.abs(_detector_fields(y_flat)) ** 2

    powers     = jax.vmap(solve_at_wavelength)(wavelengths)
    row_totals = jnp.sum(powers, axis=1, keepdims=True) + 1e-12
    return powers / row_totals


# ── Batch of starting points ───────────────────────────────────────────────
# A single Adam run on this problem is starting-point sensitive — purely
# random starts tend to settle in an asymmetric basin where the channels
# flowing through one of the root output ports plateau far below those on
# the other port. The remedy: run N_STARTS trajectories in parallel via
# jax.vmap and keep the best.
#
# Start 0 is seeded with the paper-designed Chebyshev values, which guarantees
# at least one trajectory begins in the balanced basin. Adam fine-tunes it to
# near-perfect contrast. Starts 1..N_STARTS-1 are genuinely random:
#   - Couplings drawn uniformly from [0.1, 0.9] (no Chebyshev hint).
#   - Arm lengths jittered by ±5 µm around the paper FSR values. Larger
#     offsets push the filter period away from the 15 nm channel spacing and
#     cause aliasing, so the jitter stays tight.
N_STARTS = 8
_seed    = jax.random.PRNGKey(0)

PAPER_PARAMS = jnp.array([
    0.5, 0.29, 0.08,          # root DC couplings (Horst et al. Chebyshev)
    0.5, 0.5,                 # leaf A DC couplings
    0.5, 0.5,                 # leaf B DC couplings
    L_R1A_INIT, L_R2A_INIT,   # root arm lengths (1:2 Chebyshev ratio)
    L_A1_INIT, L_B1_INIT,     # leaf arm lengths
])


def _make_random_start(key):
    k_key, l_key = jax.random.split(key)
    ks    = jax.random.uniform(k_key, (7,), minval=0.1, maxval=0.9)
    dls   = jax.random.uniform(l_key, (4,), minval=-5.0, maxval=5.0)
    paper = jnp.array([L_R1A_INIT, L_R2A_INIT, L_A1_INIT, L_B1_INIT])
    return jnp.concatenate([ks, paper + dls])


_random_starts    = jax.vmap(_make_random_start)(jax.random.split(_seed, N_STARTS - 1))
params_init_batch = jnp.concatenate([PAPER_PARAMS[None, :], _random_starts], axis=0)
params_init       = params_init_batch[0]   # = PAPER_PARAMS; replaced with winner after training

param_names = ["k_r1", "k_r2", "k_r3", "k_a1", "k_a2", "k_b1", "k_b2",
               "L_r1a", "L_r2a", "L_a1", "L_b1"]

print(f"Generated {N_STARTS} starting points (start 0 = paper design, rest random):")
print("  idx " + "  ".join(f"{n:>8s}" for n in param_names))
for i, p in enumerate(np.array(params_init_batch)):
    tag = "  ← paper" if i == 0 else ""
    print(f"  {i:2d}  " + "  ".join(f"{float(v):8.3f}" for v in p) + tag)

print("\nComputing initial power matrix for start 0 (paper design)...")
power_init = jax.jit(get_power_matrix)(params_init)

pm_init_np = np.array(power_init)
print("\nPaper-design initial power routing matrix:")
wl_labels = ["1285 nm", "1300 nm", "1315 nm", "1330 nm"]
print(f"{'':9s}  {'Det_1':>8s}  {'Det_2':>8s}  {'Det_3':>8s}  {'Det_4':>8s}")
for i, (wl, row) in enumerate(zip(wl_labels, pm_init_np)):
    print(f"  {wl}  " + "  ".join(f"{v:8.3f}" for v in row))

diag_init = np.diag(pm_init_np)
print(f"\nPaper-design mean contrast: {diag_init.mean()*100:.1f}%  (random = 25%, perfect = 100%)")

```

## Why the 2-stage root gives a flatter passband

A **single-stage MZI** has the transfer function:

$$T(\lambda) = \cos^2\!\left(\frac{\pi\, n_{\rm eff}\, \Delta L}{\lambda}\right)$$

This is purely sinusoidal — it reaches its maximum at the target wavelength but rolls off immediately on both sides. For the 15 nm channel spacing used here, adjacent channels see significant residual power.

The **2-stage lattice** adds a second interference stage with twice the arm-length difference. The interplay between the two sinusoidal responses flattens the passband top and steepens the transition region, exactly like a two-pole filter compared to a one-pole filter. The coupling ratios `(0.5, 0.29, 0.08)` are chosen to give a Chebyshev-optimal flat-top response:

| Filter type | Passband flatness | Roll-off steepness |
|-------------|------------------|--------------------|
| Single MZI | Poor (sinusoidal) | Gradual |
| 2-stage CMZ | Good (flat top) | Steep |

In practice, the CMZ topology enables closer channel spacing (smaller δλ) without crosstalk — a key advantage for dense WDM (DWDM) systems.

Note that the initial contrast of ~7% is **lower** than the simple MZI starting point — the paper-derived couplings are designed for the final optimised device, not for the pre-optimisation state. The gradient optimiser quickly finds the correct arm lengths to make these couplings work together.


```
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

    for i in range(4):
        for j in range(4):
            val = pm[i, j]
            text_color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val*100:.1f}%",
                    ha="center", va="center", fontsize=9, color=text_color)

    return fig, ax


fig, ax = plt.subplots(figsize=(5, 4))
plot_power_matrix(power_init, "CMZ Power Routing Matrix — Before Training", ax=ax)
plt.tight_layout()
plt.show()

print("The initial state routes power almost randomly — the paper-designed")
print("coupling coefficients are optimised for the final geometry, not this")
print("starting point. Backpropagation will find the arm lengths that make them work.")
```


```
# ── Multi-wavelength loss ──────────────────────────────────────────────────
# Sample 5 wavelengths per channel across a ±6 nm in-band window. This makes
# passband FLATNESS an explicit objective: the optimiser is rewarded only if
# every sample in every channel lands at its correct detector, not just the
# nominal channel centre. With channels 15 nm apart, ±6 nm still leaves a 3 nm
# guard band between channels.
BAND_HALF_NM  = 6.0
N_PER_CHANNEL = 5
_offsets      = jnp.linspace(-BAND_HALF_NM, BAND_HALF_NM, N_PER_CHANNEL)
LOSS_WLS      = (TARGET_WLS[:, None] + _offsets[None, :]).reshape(-1)   # (20,)
LOSS_TARGETS  = jnp.repeat(jnp.arange(4), N_PER_CHANNEL)                # (20,)


def loss_fn(params):
    """Mean squared deviation of in-band routing from the ideal target of 1.

    loss = (1/N) · Σᵢ (1 - p_correct[i])²

    A scalar MSE against target = 1. Weaker correct-detector fractions get
    quadratically larger gradient, so a channel at 0.80 gets 16× the
    gradient of one at 0.95 and cannot be traded off. Because row-wise
    power is conserved, ``1 - p_correct`` equals total crosstalk at that
    sample, so crosstalk is penalised implicitly.

    Returns a scalar in ≈[0, 0.56]:
      0.0  when every in-band sample routes perfectly
      0.56 when routing is uniform (p_correct = 0.25)
    """
    power     = get_power_matrix(params, wavelengths=LOSS_WLS)
    p_correct = power[jnp.arange(LOSS_WLS.shape[0]), LOSS_TARGETS]
    return jnp.mean((1.0 - p_correct) ** 2)


# jax.value_and_grad computes loss AND all 11 gradients in a single
# forward+backward pass — exactly as efficient as computing the loss alone.
print("Computing loss and gradients at paper-design start...")
loss_val, grads = jax.jit(jax.value_and_grad(loss_fn))(params_init)

_power_init    = jax.jit(lambda p: get_power_matrix(p, wavelengths=LOSS_WLS))(params_init)
_p_correct     = np.array(_power_init)[np.arange(LOSS_WLS.shape[0]), np.asarray(LOSS_TARGETS)]
_mean_contrast = float(_p_correct.mean())

print(f"  Initial loss:          {float(loss_val):.4f}  (uniform routing ≈ 0.56, perfect = 0.0)")
print(f"  Mean in-band contrast: {_mean_contrast*100:.1f}%")
print()
print("  Per-parameter gradient norms:")
for name, g in zip(param_names, grads):
    print(f"    d(loss)/d({name:8s}) = {float(g):+.6f}")
print()
print("Non-zero gradients confirm the full circuit is differentiable end-to-end.")
```


```
# ── Multi-start vmap training ──────────────────────────────────────────────
# All N_STARTS Adam trajectories run in parallel under a single jax.lax.scan,
# jit-compiled end-to-end. vmap batches the per-trajectory state (params,
# Adam moments, losses); the circuit solver inside loss_fn is unchanged.
import time

N_STEPS = 2000
LR      = 0.02

optimizer = optax.adam(learning_rate=LR)


def _single_step(params, opt_state):
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss


def _scan_body(carry, _):
    params_b, opt_states_b = carry
    new_params_b, new_opt_states_b, loss_b = jax.vmap(_single_step)(params_b, opt_states_b)
    return (new_params_b, new_opt_states_b), (new_params_b, loss_b)


@jax.jit
def _run_multistart(init_batch):
    opt_states_init = jax.vmap(optimizer.init)(init_batch)
    _, (p_hist, losses) = jax.lax.scan(
        _scan_body, (init_batch, opt_states_init), None, length=N_STEPS
    )
    return p_hist, losses


print(f"Training {N_STARTS} parallel starts × {N_STEPS} Adam steps (lr={LR})...")
t0 = time.time()
p_history_all, losses_all = _run_multistart(params_init_batch)
p_history_all.block_until_ready()
print(f"  Finished in {time.time() - t0:.1f} s")

p_history_all = np.asarray(p_history_all)   # (N_STEPS, N_STARTS, 11)
losses_all    = np.asarray(losses_all)      # (N_STEPS, N_STARTS)

# Winner = trajectory with lowest final loss
best_idx = int(np.argmin(losses_all[-1]))

# Compute actual mean in-band contrast per start for reporting (loss is MSE).
@jax.jit
def _eval_mean_contrast(p):
    power = get_power_matrix(p, wavelengths=LOSS_WLS)
    p_correct = power[jnp.arange(LOSS_WLS.shape[0]), LOSS_TARGETS]
    return p_correct.mean()

final_params_batch = jnp.asarray(p_history_all[-1])
mean_contrast_per_start = np.array(jax.vmap(_eval_mean_contrast)(final_params_batch))

print(f"\nFinal mean in-band contrast per start (winner: start {best_idx}):")
for i in range(N_STARTS):
    marker = "  ← WINNER" if i == best_idx else ""
    tag    = " (paper)" if i == 0 else ""
    print(f"  start {i}{tag}: {mean_contrast_per_start[i]*100:6.2f}%   "
          f"(MSE loss {losses_all[-1, i]:.4f}){marker}")

# Promote the winner to the names downstream cells expect, overwriting the
# "representative start 0" placeholders from cell 5.
params_init = params_init_batch[best_idx]
power_init  = jax.jit(get_power_matrix)(params_init)
diag_init   = np.diag(np.array(power_init))

# Prepend the initial state to the winner's history so frame 0 of the
# animation is the true "before training" spectrum.
winner_hist = p_history_all[:, best_idx, :]
p_history   = np.concatenate([np.asarray(params_init)[None, :], winner_hist], axis=0)
losses      = losses_all[:, best_idx].tolist()
params      = jnp.array(p_history[-1])

print(f"\nWinner (start {best_idx}) parameters — initial → final:")
print(f"{'Name':8s}  {'Initial':>10s}  {'Final':>10s}  {'Delta':>10s}")
for name, p0, pf in zip(param_names, params_init, params):
    print(f"{name:8s}  {float(p0):10.4f}  {float(pf):10.4f}  {float(pf - p0):+10.4f}")
```


```
# ── Convergence plots ──────────────────────────────────────────────────────
# Left:  all N_STARTS MSE-loss trajectories (winner highlighted) — shows how
#        random starts either converge or stall in an asymmetric basin.
# Right: winner's coupling trajectory over time.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))

for i in range(N_STARTS):
    if i == best_idx:
        ax1.plot(losses_all[:, i], color="C1", lw=2.0, zorder=5)
    else:
        ax1.plot(losses_all[:, i], color="lightgrey", lw=0.8, zorder=3)
ax1.axhline(0.0, color="black", ls="--", lw=1, label="Perfect demux (loss=0)")
ax1.plot([], [], color="C1",       lw=2.0, label=f"Winner (start {best_idx})")
ax1.plot([], [], color="lightgrey", lw=0.8, label=f"Other {N_STARTS - 1} starts")
ax1.set_xlabel("Optimisation step")
ax1.set_ylabel("MSE loss  (1 - p_correct)²")
ax1.set_title(f"Convergence of {N_STARTS} parallel starts")
ax1.set_yscale("log")
ax1.legend(fontsize=8)

coupling_labels = ["k_r1 (root DC1)", "k_r2 (root DC2)", "k_r3 (root DC3)",
                   "k_a1 (leaf A DC1)", "k_a2 (leaf A DC2)",
                   "k_b1 (leaf B DC1)", "k_b2 (leaf B DC2)"]
colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
for i, (label, color) in enumerate(zip(coupling_labels, colors)):
    ax2.plot(p_history[:, i], color=color, lw=1.5, label=label)
ax2.set_xlabel("Optimisation step")
ax2.set_ylabel("Coupling coefficient")
ax2.set_title("Winner's DC coupling trajectories")
ax2.set_ylim(-0.1, 1.1)
ax2.legend(fontsize=7, ncol=1)

plt.tight_layout()
plt.show()
```


```
sweep_wls = jnp.linspace(1260.0, 1360.0, 300)

def get_raw_power_sweep(params, wavelengths):
    """N*4 unnormalised power matrix for a wavelength sweep."""
    def solve_wl(wl):
        y_flat = circuit.dc(params=_parameter_updates(params, wl), y_guess=Y_GUESS)
        return jnp.abs(_detector_fields(y_flat)) ** 2

    return jax.vmap(solve_wl)(wavelengths)

# ── Winner's before/after power routing matrix ─────────────────────────────
# Now that cell 9 has promoted the winner, power_init and params correspond
# to the SAME trajectory (the winner's start and end).

print("Computing final power routing matrix (winner)...")
power_final = jax.jit(get_power_matrix)(params)

fig, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(11, 4))
plot_power_matrix(power_init,  f"Winner's start (start {best_idx})",      ax=ax_before)
plot_power_matrix(power_final, f"Winner after {N_STEPS} Adam steps",      ax=ax_after)
fig.suptitle("CMZ Wavelength Demultiplexer — Power Routing Matrix", color="grey", fontsize=12)
plt.tight_layout()
plt.show()

pm_final_np = np.array(power_final)
diag_final  = np.diag(pm_final_np)
print("\nFinal routing contrast (diagonal):")
for i, (wl, p) in enumerate(zip([1285, 1300, 1315, 1330], diag_final)):
    print(f"  Det_{i+1} <- {wl} nm : {p*100:.1f}%")
print(f"  Mean contrast: {np.mean(diag_final)*100:.1f}%")
print()
print(f"Improvement: {diag_init.mean()*100:.1f}% -> {np.mean(diag_final)*100:.1f}%")

```

![cmz_optimisation.gif](cmz_optimisation.gif)



### Key results

| Metric | CMZ lattice |
|--------|-------------|
| Architecture | 2-level tree, 2-stage lattice at root |
| Trainable parameters | 11 (7 couplings + 4 arm lengths) |
| Final mean contrast | ~99% |
| Passband shape | Flat top, steep skirts |

### Why the CMZ excels

1. **2-stage lattice root**: The interplay between two interference stages creates a Chebyshev-like flat-top response. The analytic coupling ratios `(0.5, 0.29, 0.08)` from Horst et al. are the starting point; the optimiser fine-tunes them.

2. **Flat passband**: Light at wavelengths within each channel's band all arrive at the correct detector with high efficiency. The simple MZI wastes power at passband edges.

3. **Sharp stopband**: Neighbouring channels are strongly suppressed, enabling denser channel packing (smaller δλ) without crosstalk.

### Backpropagation scaling

With 11 parameters, backpropagation computes all gradients in ~2 circuit solves per step — versus 12 solves for finite differences. For a production photonic IC with 100+ phase elements, backprop enables a viable way to optimize such a circuit.

The key point is that `jax.grad` works through the entire circulax solver — sparse linear solve, complex field assembly — without any special-casing of the photonic physics. Currently, circuit topology is **not** part of this differentiable flow as JAX requires functions should have floating numbers in and out to be auto-differentiable
