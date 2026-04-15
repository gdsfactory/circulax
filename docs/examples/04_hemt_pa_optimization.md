# RF Power Amplifier Optimization via Differentiable Harmonic Balance

Traditional EDA tools (ADS, AWR) solve Harmonic Balance (HB) efficiently, but their built-in optimisers
use gradient-free algorithms — genetic algorithms, random search, or Nelder-Mead — that require hundreds
of full HB evaluations to converge.

Circulax formulates the HB system as a JAX-differentiable fixed-point problem, which means
`jax.grad` (via Optimistix's ImplicitAdjoint) backpropagates *through* the HB solver and delivers
exact gradients of any scalar figure-of-merit with respect to any circuit parameter — in a single
adjoint pass costing roughly one extra forward solve.

In this notebook we:
1. Define a **differentiable HEMT model** (Curtice-Quadratic with smooth approximations replacing hard
   if/else branches).
2. Build a **5 GHz power amplifier** with input/output L-match networks.
3. Run a **Pin sweep** to characterise the detuned (initial) amplifier.
4. Use **Adam + `jax.grad`** through HB to maximise Power-Added Efficiency (PAE) by tuning the four
   matching-network L/C values — obtaining analytic gradients at each step.


```python
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import optax
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from circulax import compile_circuit, setup_harmonic_balance
from circulax.components.base_component import component
from circulax.components.electronic import (
    Capacitor,
    Inductor,
    Resistor,
    VoltageSource,
    VoltageSourceAC,
)
from circulax.utils import update_params_dict

jax.config.update("jax_enable_x64", True)

pio.templates.default = "plotly_white"
print("JAX backend:", jax.default_backend())

```

    KLUJAX_RS DEBUG MODE.


    WARNING:2026-04-15 17:32:56,951:jax._src.xla_bridge:864: An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.


    JAX backend: cpu


## Phase 1 — Differentiable HEMT Model

Standard SPICE transistor models use piecewise branching:

```
if Vgs < Vp:
    Ids = 0          # pinch-off
elif Vds < Vgs - Vp:
    Ids = quadratic  # linear region
else:
    Ids = saturated  # saturation region
```

Python `if/else` on *traced* JAX values breaks the Jacobian: the derivative is either zero or
undefined at the transition.  The fix is smooth approximations:

| Hard branch | Smooth replacement | Why |
|---|---|---|
| `max(0, Vgs - Vp)` | `softplus(Vgs - Vp)` | Differentiable everywhere |
| Hard saturation clamp | `tanh(α Vds)` | Smooth S-curve, saturates as Vds → ∞ |
| Abrupt Qgs step | `softplus` integral | Continuous dQgs/dVgs (transcapacitance) |

The model is a **Modified Curtice-Quadratic** with three-terminal gate/drain/source ports.


```python
@component(ports=("g", "d", "s"))
def HEMT(signals, s,
         beta=0.012, Vp=-2.0, lam=0.05, alpha=4.0,
         Cgs0=0.3e-12, Cgs1=0.1e-12, Cgd0=0.05e-12):
    # Modified Curtice-Quadratic HEMT with smooth approximations.
    #
    # Parameters
    # ----------
    # beta  : transconductance parameter (A/V^2)
    # Vp    : pinch-off voltage (V)
    # lam   : channel-length modulation (1/V)
    # alpha : saturation slope (1/V)
    # Cgs0  : linear gate-source capacitance (F)
    # Cgs1  : nonlinear gate-source capacitance coefficient (F)
    # Cgd0  : gate-drain (Miller) capacitance (F)
    Vgs = signals.g - signals.s
    Vds = signals.d - signals.s
    Vgd = signals.g - signals.d

    sp_scale = 10.0
    # Smooth pinch-off: softplus replaces max(0, Vgs - Vp)
    sp = jnn.softplus(sp_scale * (Vgs - Vp)) / sp_scale

    # Drain current: quadratic in pinch-off voltage, tanh saturation
    Ids = beta * sp**2 * (1.0 + lam * Vds) * jnp.tanh(alpha * Vds)

    # Gate-source charge: integral of sigmoid = softplus (differentiable everywhere)
    Qgs = Cgs0 * Vgs + Cgs1 * jnn.softplus(sp_scale * (Vgs - Vp)) / sp_scale
    # Gate-drain (Miller) charge: linear
    Qgd = Cgd0 * Vgd

    f = {"g": 0.0,   "d": Ids,  "s": -Ids}
    q = {"g": Qgs + Qgd, "d": -Qgd, "s": -Qgs}
    return f, q


# ── Smoke test ────────────────────────────────────────────────────────────────────────────────
hemt_test = HEMT()
f, q = hemt_test(g=0.0, d=3.0, s=0.0)
print(f"Ids at Vgs=0 V,   Vds=3 V: {f['d']*1e3:.2f} mA  (Idss)")

f2, _ = hemt_test(g=-2.5, d=3.0, s=0.0)
print(f"Ids at Vgs=-2.5 V (below pinch-off): {f2['d']*1e6:.3f} µA  (should be ~0)")

# Verify JAX-differentiability
gm = float(jax.grad(lambda vg: hemt_test(g=vg, d=3.0, s=0.0)[0]["d"])(0.0))
print(f"gm at Vgs=0, Vds=3 V: {gm*1e3:.2f} mS")
```

    Ids at Vgs=0 V,   Vds=3 V: 55.20 mA  (Idss)
    Ids at Vgs=-2.5 V (below pinch-off): 0.006 µA  (should be ~0)


    gm at Vgs=0, Vds=3 V: 55.20 mS



```python
def compute_Ids(Vgs, Vds,
               beta=0.012, Vp=-2.0, lam=0.05, alpha=4.0):
    sp_scale = 10.0
    sp = jnn.softplus(sp_scale * (Vgs - Vp)) / sp_scale
    return beta * sp**2 * (1.0 + lam * Vds) * jnp.tanh(alpha * Vds)


Vds_arr = jnp.linspace(0.0, 5.0, 200)
Vgs_arr = jnp.linspace(-3.0, 0.5, 200)

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Output Characteristics", "Transconductance vs Vgs  (Vds = 3 V)"),
)

# Left: Output characteristics
colors_iv = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for Vgs_val, col in zip([-1.5, -1.0, -0.5, 0.0], colors_iv):
    ids_curve = jax.vmap(lambda vds: compute_Ids(Vgs_val, vds))(Vds_arr)
    fig.add_trace(
        go.Scatter(
            x=np.array(Vds_arr),
            y=np.array(ids_curve) * 1e3,
            mode="lines",
            name=f"Vgs = {Vgs_val} V",
            line=dict(color=col),
        ),
        row=1, col=1,
    )
# Vertical bias line
fig.add_vline(x=3.0, line=dict(color="grey", width=1, dash="dash"),
              annotation_text="Vds = 3 V (bias)", annotation_position="top right",
              row=1, col=1)

# Right: Transconductance
ids_vgs = jax.vmap(lambda vgs: compute_Ids(vgs, 3.0))(Vgs_arr)
gm = jnp.gradient(ids_vgs, Vgs_arr)
fig.add_trace(
    go.Scatter(
        x=np.array(Vgs_arr),
        y=np.array(gm) * 1e3,
        mode="lines",
        name="gm (mS)",
        line=dict(color="#ff7f0e", width=2),
        showlegend=True,
    ),
    row=1, col=2,
)
fig.add_vline(x=-0.5, line=dict(color="#1f77b4", width=1.5, dash="dash"),
              annotation_text="Vgg = −0.5 V (bias)", annotation_position="top right",
              row=1, col=2)

fig.update_xaxes(title_text="Vds (V)", row=1, col=1)
fig.update_yaxes(title_text="Ids (mA)", row=1, col=1)
fig.update_xaxes(title_text="Vgs (V)", row=1, col=2)
fig.update_yaxes(title_text="gm (mS)", row=1, col=2)
fig.update_layout(height=400, width=900, legend=dict(tracegroupgap=0))
fig.show()
print("No kinks or discontinuities → Jacobian is well-defined everywhere.")

```



    No kinks or discontinuities → Jacobian is well-defined everywhere.


## Phase 2 — PA Circuit Architecture

```
            Rs_src        L_in             R_bias   Vgg
50Ω source ──/\/\/\/──┬──LLLLLL──┬──────────/\/\/\/──+──(Vgg)──GND
                    |          |
 Vs (5 GHz AC)    (Vs)        C_in         Q1 (HEMT)
                    |          |           G   D   S
                   GND        GND          |   |   |
                                           |   |  GND
         Vdd ───────────────────────── L_choke
                                           |
                                        C_block ── L_out ──┬── R_load ── GND
                                                           C_out
                                                            |
                                                           GND
```

**Component roles:**

| Component | Value | Purpose |
|---|---|---|
| `Rs_src` | 50 Ω | Source impedance |
| `R_bias` | 10 kΩ | Gate bias resistor (high RF impedance) |
| `L_choke` | 47 nH | RF choke: Z = 1.5 kΩ at 5 GHz → DC path to Vdd, RF open |
| `C_block` | 10 pF | DC block: Z = 3.2 Ω at 5 GHz → RF short, blocks Vdd from output |
| `L_in, C_in` | **optimised** | Input L-match: transforms 50 Ω source → Zin* at gate |
| `L_out, C_out` | **optimised** | Output L-match: transforms Ropt at drain → 50 Ω load |
| `R_load` | 50 Ω | Load resistance |

The optimal load resistance for maximum class-A output power is `Ropt ≈ (Vdd − Vknee) / (2 Idq)`.
The matching networks are initialised deliberately off-resonance; the optimizer will tune them.


```python
# ── Frequency and harmonic parameters ──────────────────────────────────────────────
F0      = 5e9    # Hz  (5 GHz)
N_HARM  = 7      # harmonics → K = 15 time points per period
K       = 2 * N_HARM + 1

# ── Bias and HEMT parameters ───────────────────────────────────────────────────
Vgg_val     = -0.5   # V  gate bias
Vdd_val     =  3.0   # V  drain supply
hemt_params = dict(beta=0.012, Vp=-2.0, lam=0.05, alpha=4.0,
                   Cgs0=0.3e-12, Cgs1=0.1e-12, Cgd0=0.05e-12)

# ── Matching network — deliberately detuned starting values ───────────────────────────
L_in_init  = 0.3e-9;  C_in_init  = 0.5e-12   # input L-match
L_out_init = 0.3e-9;  C_out_init = 0.5e-12   # output L-match

# ── Fixed bias tee values (not optimised) ─────────────────────────────────────────────────
L_choke_val = 47e-9   # 47 nH → Z ≈ 1.5 kΩ at 5 GHz  (RF open)
C_block_val = 10e-12  # 10 pF → Z ≈ 3.2 Ω at 5 GHz   (RF short)
R_bias_val  = 10e3    # 10 kΩ (gate bias, high RF impedance)

V_in_init   = 0.1    # V amplitude for initial compile

# ── SAX-format netlist ────────────────────────────────────────────────────────────────────
pa_net = {
    "instances": {
        "Vs":      {"component": "voltagesourceac", "settings": {"V": V_in_init, "freq": F0}},
        "Rs_src":  {"component": "resistor",        "settings": {"R": 50.0}},
        "L_in":    {"component": "inductor",        "settings": {"L": L_in_init}},
        "C_in":    {"component": "capacitor",       "settings": {"C": C_in_init}},
        "R_bias":  {"component": "resistor",        "settings": {"R": R_bias_val}},
        "Vgg":     {"component": "voltagesource",   "settings": {"V": Vgg_val}},
        "Q1":      {"component": "hemt",            "settings": hemt_params},
        "L_choke": {"component": "inductor",        "settings": {"L": L_choke_val}},
        "Vdd":     {"component": "voltagesource",   "settings": {"V": Vdd_val}},
        "C_block": {"component": "capacitor",       "settings": {"C": C_block_val}},
        "L_out":   {"component": "inductor",        "settings": {"L": L_out_init}},
        "C_out":   {"component": "capacitor",       "settings": {"C": C_out_init}},
        "R_load":  {"component": "resistor",        "settings": {"R": 50.0}},
    },
    "connections": {
        "Vs,p1":      "Rs_src,p1",
        "Vs,p2":      "GND,p1",
        "Rs_src,p2":  "L_in,p1",
        "L_in,p2":    ("Q1,g", "C_in,p1", "R_bias,p1"),
        "C_in,p2":    "GND,p1",
        "R_bias,p2":  "Vgg,p1",
        "Vgg,p2":     "GND,p1",
        "Q1,s":       "GND,p1",
        "Q1,d":       ("L_choke,p2", "C_block,p1"),
        "L_choke,p1": "Vdd,p1",
        "Vdd,p2":     "GND,p1",
        "C_block,p2": "L_out,p1",
        "L_out,p2":   ("C_out,p1", "R_load,p1"),
        "C_out,p2":   "GND,p1",
        "R_load,p2":  "GND,p1",
    },
}

models = {
    "voltagesourceac": VoltageSourceAC,
    "voltagesource":   VoltageSource,
    "resistor":        Resistor,
    "inductor":        Inductor,
    "capacitor":       Capacitor,
    "hemt":            HEMT,
}

circuit = compile_circuit(pa_net, models, backend="dense")
groups = circuit.groups
num_vars = circuit.sys_size
net_map = circuit.port_map
print(f"System size : {num_vars} unknowns")
print(f"Groups      : {list(groups.keys())}")
print(f"Net map     : {dict(sorted(net_map.items()))}")

# Extract key node indices for later use
osc_node_drain = net_map["Q1,d"]
osc_node_gate  = net_map["Q1,g"]
load_node      = net_map["R_load,p1"]
choke_iL_idx   = net_map["L_choke,i_L"]
print(f"\nKey indices: drain={osc_node_drain}, gate={osc_node_gate}, "
      f"load={load_node}, choke_iL={choke_iL_idx}")
```

    System size : 15 unknowns
    Groups      : ['voltagesourceac', 'resistor', 'inductor', 'capacitor', 'voltagesource', 'hemt']
    Net map     : {'C_block,p1': 1, 'C_block,p2': 2, 'C_in,p1': 3, 'C_in,p2': 0, 'C_out,p1': 4, 'C_out,p2': 0, 'GND,p1': 0, 'L_choke,i_L': 11, 'L_choke,p1': 5, 'L_choke,p2': 1, 'L_in,i_L': 10, 'L_in,p1': 6, 'L_in,p2': 3, 'L_out,i_L': 12, 'L_out,p1': 2, 'L_out,p2': 4, 'Q1,d': 1, 'Q1,g': 3, 'Q1,s': 0, 'R_bias,p1': 3, 'R_bias,p2': 7, 'R_load,p1': 4, 'R_load,p2': 0, 'Rs_src,p1': 8, 'Rs_src,p2': 6, 'Vdd,i_src': 14, 'Vdd,p1': 5, 'Vdd,p2': 0, 'Vgg,i_src': 13, 'Vgg,p1': 7, 'Vgg,p2': 0, 'Vs,i_src': 9, 'Vs,p1': 8, 'Vs,p2': 0}

    Key indices: drain=1, gate=3, load=4, choke_iL=11



```python
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")
ax.set_title("5 GHz Class-A HEMT PA — Circuit Topology", fontsize=12, pad=6)

with schemdraw.Drawing(canvas=ax) as d:
    d.config(unit=2.8, fontsize=10)

    # ── Input source ────────────────────────────────────────────
    d.add(elm.Ground())
    d.add(elm.SourceSin().up().label("$V_s$", loc="top"))
    d.add(elm.Line().right(0.4))
    d.add(elm.Resistor().right().label("$R_s$\n50 Ω", loc="top"))
    d.add(elm.Line().right(0.4))

    # Gate bias taps off here (before L_in)
    pre_lin = d.here
    d.add(elm.Dot())
    d.add(elm.Line().right(0.4))
    d.add(elm.Inductor2(loops=2).right().label("$L_{in}$\n(opt)", loc="top"))
    d.add(elm.Line().right(0.5))

    gate_node = d.here
    d.add(elm.Dot())

    # ── HEMT (reverse=True → gate on left, drain top, source bottom) ────────
    Q1 = d.add(elm.AnalogNFet(arrow=True, reverse=True).anchor("gate").at(gate_node))
    mid_y = (Q1.drain[1] + Q1.source[1]) / 2
    d.add(elm.Label().at((Q1.drain[0] + 0.5, mid_y)).label("Q1\n(HEMT)"))

    d.add(elm.Line().down().at(Q1.source).length(0.5))
    d.add(elm.Ground())

    # ── Gate shunt C_in ───────────────────────────────────────────
    d.add(elm.Line().down().at(gate_node).length(0.5))
    d.add(elm.Capacitor().down().label("$C_{in}$\n(opt)", loc="top"))
    d.add(elm.Ground())

    # ── Gate bias: R_bias + Vgg ───────────────────────────────────────
    d.add(elm.Line().up().at(pre_lin).length(0.4))
    d.add(elm.Resistor().up().label(f"$R_{{bias}}$\n{R_bias_val/1e3:.0f} kΩ", loc="top"))
    d.add(elm.SourceV().up().reverse().label(f"$V_{{gg}}$\n{Vgg_val} V", loc="top"))
    d.add(elm.Ground().left())

    # ── Drain: L_choke (RF choke) → Vdd ────────────────────────────
    drain_node = Q1.drain
    d.add(elm.Dot().at(drain_node))
    d.add(elm.Inductor2(loops=2).up().at(drain_node)
          .label(f"$L_{{choke}}$\n{L_choke_val*1e9:.0f} nH", loc="top"))
    d.add(elm.Vdd().label(f"$V_{{dd}}$ = {Vdd_val} V"))

    # ── Output matching: C_block (DC block) → L_out → C_out (shunt) ────────
    d.add(elm.Line().right(0.5).at(drain_node))
    d.add(elm.Capacitor().right().label(f"$C_{{block}}$\n{C_block_val*1e12:.0f} pF", loc="top"))
    d.add(elm.Inductor2(loops=2).right().label("$L_{out}$\n(opt)", loc="top"))
    d.add(elm.Line().right(0.5))

    out_node = d.here
    d.add(elm.Dot())
    d.add(elm.Line().down().at(out_node).length(0.5))
    d.add(elm.Capacitor().down().label("$C_{out}$\n(opt)", loc="top"))
    d.add(elm.Ground())

    # ── 50 Ω load ────────────────────────────────────────────────────
    d.add(elm.Resistor().right().at(out_node).label("$R_{load}$\n50 Ω", loc="top"))
    d.add(elm.Dot(open=True).label("$V_{out}$", loc="right"))

plt.tight_layout()

```



![png](04_hemt_pa_optimization_files/04_hemt_pa_optimization_7_0.png)




```python
y_dc = circuit()

V_gate     = float(y_dc[net_map["Q1,g"]])
V_drain    = float(y_dc[net_map["Q1,d"]])
I_drain_dc = float(y_dc[choke_iL_idx])

# Verify with HEMT formula
sp_scale = 10.0
Vp = -2.0; beta = 0.012; alpha = 4.0; lam = 0.05
sp = float(jnn.softplus(sp_scale * (V_gate - Vp)) / sp_scale)
Ids_check = beta * sp**2 * (1 + lam * V_drain) * float(jnp.tanh(alpha * V_drain))

print("DC Operating Point")
print(f"  V_gate  = {V_gate:.4f} V   (Vgg = {Vgg_val} V)")
print(f"  V_drain = {V_drain:.4f} V   (Vdd = {Vdd_val} V)")
print(f"  I_drain = {I_drain_dc*1e3:.2f} mA  (from L_choke inductor state)")
print(f"  Ids check = {Ids_check*1e3:.2f} mA  (from HEMT formula directly)")
print(f"  Pdc     = {Vdd_val * I_drain_dc * 1e3:.1f} mW")

Ropt = (Vdd_val - 0.3) / (2 * I_drain_dc) if I_drain_dc > 1e-6 else float("inf")
print("\n  Ropt ≈ (Vdd - Vknee) / (2 Idq)")
print(f"       = ({Vdd_val} - 0.3) / (2 × {I_drain_dc*1e3:.1f} mA)")
print(f"       = {Ropt:.1f} Ω  (optimal load for class-A max power)")
print(f"\n  The output L-match must transform 50 Ω → {Ropt:.0f} Ω at 5 GHz.")
```

    DC Operating Point
      V_gate  = -0.0025 V   (Vgg = -0.5 V)
      V_drain = 3.0000 V   (Vdd = 3.0 V)
      I_drain = 55.06 mA  (from L_choke inductor state)
      Ids check = 55.06 mA  (from HEMT formula directly)
      Pdc     = 165.2 mW

      Ropt ≈ (Vdd - Vknee) / (2 Idq)
           = (3.0 - 0.3) / (2 × 55.1 mA)
           = 24.5 Ω  (optimal load for class-A max power)

      The output L-match must transform 50 Ω → 25 Ω at 5 GHz.


## Phase 2 — Forward PA Simulation via Harmonic Balance

**How Harmonic Balance works here:**

The HB solver represents the periodic steady-state as `K = 2N+1 = 15` equally-spaced time samples
per period (T = 200 ps at 5 GHz), or equivalently as the DC component plus N = 7 complex Fourier
coefficients.

At each Newton iteration the solver evaluates the DAE residual
`F(y) + dQ/dt = 0` in the frequency domain by:
1. IDFT the frequency-domain unknowns to the time domain.
2. Evaluate all component nonlinearities (HEMT etc.) at each of the K time points.
3. DFT back to frequency domain and assemble the residual.

The Jacobian is assembled analytically (via `jax.jacfwd`) and the system is solved with
a dense LU factorisation.

**Warm-starting:** For a *driven* PA (not an oscillator), the source always forces a non-trivial
periodic response. Tiling the DC solution `y_dc` across K time steps is a reliable warm start.


```python
# compute_powers: differentiable PA metrics from HB Fourier coefficients.
# y_freq[k, node] is the normalised complex Fourier coefficient at harmonic k.
# The time-domain peak amplitude is  2 * |y_freq[k, node]|  for k >= 1.
def compute_powers(y_freq, V_src_amp, Vdd):
    # Output power at fundamental
    V_load_amp = 2.0 * jnp.abs(y_freq[1, load_node])
    Pout_W     = V_load_amp**2 / (4.0 * 50.0)   # delivered to matched 50 Ω

    # Available input power
    Pin_W = V_src_amp**2 / (8.0 * 50.0)

    # DC power: Vdd × mean drain current  (DC harmonic = y_freq[0])
    I_dc  = jnp.real(y_freq[0, choke_iL_idx])
    Pdc_W = Vdd * I_dc

    PAE      = (Pout_W - Pin_W) / (Pdc_W + 1e-20)
    Pout_dBm = 10.0 * jnp.log10(Pout_W / 1e-3 + 1e-20)
    Pin_dBm  = 10.0 * jnp.log10(Pin_W  / 1e-3 + 1e-20)
    Gain_dB  = Pout_dBm - Pin_dBm
    return Pout_dBm, Gain_dB, PAE


# ── Single-point verification at +10 dBm available input ─────────────────────────
V_test = float(jnp.sqrt(8.0 * 50.0 * 10.0 ** (10.0 / 10.0) * 1e-3))
print(f"Test point: +10 dBm available input → V_amplitude = {V_test:.3f} V")

grps_test   = update_params_dict(groups, "voltagesourceac", "Vs", "V", V_test)
run_hb_test = setup_harmonic_balance(grps_test, num_vars, freq=F0, num_harmonics=N_HARM)

y_flat_init = jnp.tile(y_dc, K)
y_time_ref, y_freq_ref = jax.jit(run_hb_test)(y_dc, y_flat_init=y_flat_init)

Pout_t, Gain_t, PAE_t = compute_powers(y_freq_ref, V_test, Vdd_val)
print(f"\n  Pout = {float(Pout_t):.1f} dBm")
print(f"  Gain = {float(Gain_t):.1f} dB")
print(f"  PAE  = {float(PAE_t)*100:.1f}%")
print(f"\n  Drain voltage swing : {float(2*jnp.abs(y_freq_ref[1, osc_node_drain])):.3f} V peak")
print(f"  Gate voltage swing  : {float(2*jnp.abs(y_freq_ref[1, osc_node_gate])):.3f} V peak")
h3_ratio = float(jnp.abs(y_freq_ref[3, osc_node_drain]) / (jnp.abs(y_freq_ref[1, osc_node_drain]) + 1e-20))
print(f"  3rd harmonic at drain: {float(2*jnp.abs(y_freq_ref[3, osc_node_drain]))*1e3:.1f} mV  ({h3_ratio*100:.1f}% of fundamental)")
```

    Test point: +10 dBm available input → V_amplitude = 2.000 V



      Pout = 13.7 dBm
      Gain = 3.7 dB
      PAE  = 7.3%

      Drain voltage swing : 1.974 V peak
      Gate voltage swing  : 1.111 V peak
      3rd harmonic at drain: 6.4 mV  (0.3% of fundamental)



```python
Pin_dBm_vals = np.linspace(-10, 22, 17)
V_amp_vals   = np.sqrt(8.0 * 50.0 * 10.0 ** (Pin_dBm_vals / 10.0) * 1e-3)


def run_at_amplitude(V_in: jax.Array):
    # Run HB and return (Pout_dBm, Gain_dB, PAE) at source amplitude V_in.
    grps = update_params_dict(groups, "voltagesourceac", "Vs", "V", V_in)
    run_hb_i = setup_harmonic_balance(grps, num_vars, freq=F0, num_harmonics=N_HARM)
    _, y_freq_i = run_hb_i(y_dc, y_flat_init=jnp.tile(y_dc, K))
    return compute_powers(y_freq_i, V_in, Vdd_val)


run_at_amplitude_jit = jax.jit(run_at_amplitude)

Pout_list, Gain_list, PAE_list = [], [], []
print("Pin sweep (detuned matching network):")
print(f"{'Pin (dBm)':>12}  {'Pout (dBm)':>12}  {'Gain (dB)':>10}  {'PAE (%)':>8}")
print("-" * 50)
for Pin_dBm, V_in in zip(Pin_dBm_vals, V_amp_vals):
    Pout, Gain, PAE = run_at_amplitude_jit(jnp.array(V_in))
    Pout_list.append(float(Pout))
    Gain_list.append(float(Gain))
    PAE_list.append(float(PAE) * 100)
    if round(Pin_dBm) in (-10, -2, 6, 14, 22):
        print(f"{Pin_dBm:>12.0f}  {float(Pout):>12.1f}  {float(Gain):>10.1f}  {float(PAE)*100:>8.1f}")
```

    Pin sweep (detuned matching network):
       Pin (dBm)    Pout (dBm)   Gain (dB)   PAE (%)
    --------------------------------------------------


             -10          -6.1         3.9       0.1
              -2           1.9         3.9       0.5
               6           9.8         3.8       3.2
              14          16.8         2.8      11.5
              22          18.6        -3.4     -38.3



```python
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Output Power vs Input Power", "Gain Compression", "Power Added Efficiency (detuned matching)"),
)

# Pout vs Pin
fig.add_trace(
    go.Scatter(x=list(Pin_dBm_vals), y=Pout_list, mode="lines+markers",
               name="Pout", marker=dict(size=6), line=dict(color="#1f77b4", width=2)),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=list(Pin_dBm_vals), y=list(Pin_dBm_vals), mode="lines",
               name="0 dB gain", line=dict(color="black", width=1, dash="dash"), opacity=0.4),
    row=1, col=1,
)

# Gain vs Pin
fig.add_trace(
    go.Scatter(x=list(Pin_dBm_vals), y=Gain_list, mode="lines+markers",
               name="Gain", marker=dict(size=6), line=dict(color="#ff7f0e", width=2)),
    row=1, col=2,
)
if Gain_list:
    p1db_ref = Gain_list[0] - 1
    fig.add_hline(y=p1db_ref, line=dict(color="#ff7f0e", width=1.5, dash="dot"),
                  annotation_text="P1dB ref", annotation_position="bottom right",
                  row=1, col=2)

# PAE vs Pin
fig.add_trace(
    go.Scatter(x=list(Pin_dBm_vals), y=PAE_list, mode="lines+markers",
               name="PAE", marker=dict(size=6), line=dict(color="#2ca02c", width=2)),
    row=1, col=3,
)

fig.update_xaxes(title_text="Pin (dBm)", row=1, col=1)
fig.update_yaxes(title_text="Pout (dBm)", row=1, col=1)
fig.update_xaxes(title_text="Pin (dBm)", row=1, col=2)
fig.update_yaxes(title_text="Gain (dB)", row=1, col=2)
fig.update_xaxes(title_text="Pin (dBm)", row=1, col=3)
fig.update_yaxes(title_text="PAE (%)", rangemode="tozero", row=1, col=3)

fig.update_layout(height=420, width=1200)
fig.show()

```




```python
# ── Interactive matching-network explorer ─────────────────────────────────────
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash
from plotly.subplots import make_subplots

V_target = float(jnp.sqrt(8.0 * 50.0 * 10.0 ** (12.0 / 10.0) * 1e-3))

_app = JupyterDash(__name__)

_slider_style = {"marginBottom": "6px"}

_app.layout = html.Div([
    html.H4("Matching Network Explorer  (+12 dBm input)", style={"fontFamily": "sans-serif"}),
    html.Div([
        html.Div([
            html.Label("L_in (nH)"), dcc.Slider(0.1, 5.0, 0.1, value=1.0, id="sl-lin",
                marks={v: str(v) for v in [0.1, 1, 2, 3, 4, 5]}, tooltip={"placement": "bottom"}),
        ], style=_slider_style),
        html.Div([
            html.Label("C_in (pF)"), dcc.Slider(0.1, 5.0, 0.1, value=1.0, id="sl-cin",
                marks={v: str(v) for v in [0.1, 1, 2, 3, 4, 5]}, tooltip={"placement": "bottom"}),
        ], style=_slider_style),
        html.Div([
            html.Label("L_out (nH)"), dcc.Slider(0.1, 5.0, 0.1, value=1.0, id="sl-lout",
                marks={v: str(v) for v in [0.1, 1, 2, 3, 4, 5]}, tooltip={"placement": "bottom"}),
        ], style=_slider_style),
        html.Div([
            html.Label("C_out (pF)"), dcc.Slider(0.1, 5.0, 0.1, value=1.0, id="sl-cout",
                marks={v: str(v) for v in [0.1, 1, 2, 3, 4, 5]}, tooltip={"placement": "bottom"}),
        ], style=_slider_style),
    ], style={"maxWidth": "650px", "padding": "12px"}),
    dcc.Graph(id="wf-graph", style={"height": "360px"}),
    html.Div(id="info-panel", style={"fontFamily": "monospace", "padding": "8px", "fontSize": "14px"}),
])


@_app.callback(
    Output("wf-graph", "figure"),
    Output("info-panel", "children"),
    Input("sl-lin",  "value"),
    Input("sl-cin",  "value"),
    Input("sl-lout", "value"),
    Input("sl-cout", "value"),
)
def _update(L_in_nH, C_in_pF, L_out_nH, C_out_pF):
    L_in_si  = L_in_nH  * 1e-9
    C_in_si  = C_in_pF  * 1e-12
    L_out_si = L_out_nH * 1e-9
    C_out_si = C_out_pF * 1e-12

    grps = update_params_dict(groups,    "inductor",        "L_in",  "L", L_in_si)
    grps = update_params_dict(grps,      "capacitor",       "C_in",  "C", C_in_si)
    grps = update_params_dict(grps,      "inductor",        "L_out", "L", L_out_si)
    grps = update_params_dict(grps,      "capacitor",       "C_out", "C", C_out_si)
    grps = update_params_dict(grps,      "voltagesourceac", "Vs",    "V", V_target)

    run_hb_i = setup_harmonic_balance(grps, num_vars, freq=F0, num_harmonics=N_HARM)
    y_time_i, y_freq_i = jax.jit(run_hb_i)(y_dc, y_flat_init=jnp.tile(y_dc, K))

    T_val   = 1.0 / F0
    t_ps_i  = np.linspace(0, T_val * 1e12, K, endpoint=False)
    v_gate_i  = np.real(np.array(y_time_i[:, osc_node_gate]))
    v_drain_i = np.real(np.array(y_time_i[:, osc_node_drain]))

    Pout_i, Gain_i, PAE_i = compute_powers(y_freq_i, V_target, Vdd_val)

    fig_i = make_subplots(rows=1, cols=2,
                          subplot_titles=("Gate Voltage", "Drain Voltage"))
    fig_i.add_trace(go.Scatter(x=list(t_ps_i), y=list(v_gate_i),
                               mode="lines", name="Gate (V)",
                               line=dict(color="#1f77b4")), row=1, col=1)
    fig_i.add_trace(go.Scatter(x=list(t_ps_i), y=list(v_drain_i),
                               mode="lines", name="Drain (V)",
                               line=dict(color="#ff7f0e")), row=1, col=2)
    fig_i.update_xaxes(title_text="Time (ps)", row=1, col=1)
    fig_i.update_yaxes(title_text="Voltage (V)", row=1, col=1)
    fig_i.update_xaxes(title_text="Time (ps)", row=1, col=2)
    fig_i.update_yaxes(title_text="Voltage (V)", row=1, col=2)
    fig_i.update_layout(template="plotly_white", height=340)

    info = (
        f"Pout = {float(Pout_i):.1f} dBm  |  "
        f"Gain = {float(Gain_i):.1f} dB  |  "
        f"PAE  = {float(PAE_i)*100:.1f}%"
    )
    return fig_i, info


_app.run(mode="inline", height=650, port=8051)

```

    /home/cdaunt/code/circulax/circulax/.pixi/envs/default/lib/python3.13/site-packages/dash/dash.py:644: UserWarning: JupyterDash is deprecated, use Dash instead.
    See https://dash.plotly.com/dash-in-jupyter for more details.
      warnings.warn(




<iframe
    width="100%"
    height="650"
    src="http://127.0.0.1:8051/"
    frameborder="0"
    allowfullscreen

></iframe>



## Phase 3 — Gradient-Based Matching Network Optimisation

**Why matching matters for PAE:**

PAE is maximised when the drain sees the *optimal load resistance* `Ropt ≈ (Vdd − Vknee) / (2 Idq)`.
With the detuned L-match, the effective load at 5 GHz differs from Ropt — power is reflected rather
than delivered. The input match is similarly off, so extra drive power is wasted.

**Gradient flow through Harmonic Balance:**

`loss = −PAE(L_in, C_in, L_out, C_out)` where PAE is computed from the HB fixed point.
Circulax uses Optimistix's `ImplicitAdjoint` to differentiate through the Newton solver:

```
∂loss/∂params = −(∂F/∂y)⁻¹ · (∂F/∂params)   [implicit function theorem]
```

This costs **one extra linear solve** per gradient evaluation (the adjoint), regardless of how many
Newton iterations the forward solve needed. Compare that to finite differences: `2 × n_params`
extra HB solves.

**Parameterisation:** We optimise `log(L)` and `log(C)` so that the physical values remain
positive and cover several decades without constraint handling.


```python
# ── Target: maximise PAE at +12 dBm input (moderate saturation) ─────────────────────
V_target = float(jnp.sqrt(8.0 * 50.0 * 10.0 ** (12.0 / 10.0) * 1e-3))
print(f"Optimisation target: Pin = +12 dBm  (V_source amplitude = {V_target:.3f} V)")

# Fix source amplitude for optimisation (we tune matching only)
groups_target = update_params_dict(groups, "voltagesourceac", "Vs", "V", V_target)

# Fixed warm start: tile DC solution across K time points.
# For a driven PA the source always forces a non-trivial solution, so this is reliable.
y_flat_warmstart = jnp.tile(y_dc, K)


def loss_fn(log_params: jax.Array) -> jax.Array:
    # Negative PAE - minimising this maximises PAE at +12 dBm input.
    L_in, C_in, L_out, C_out = jnp.exp(log_params)

    grps = update_params_dict(groups_target, "inductor",  "L_in",  "L", L_in)
    grps = update_params_dict(grps,          "capacitor", "C_in",  "C", C_in)
    grps = update_params_dict(grps,          "inductor",  "L_out", "L", L_out)
    grps = update_params_dict(grps,          "capacitor", "C_out", "C", C_out)

    run_hb_i = setup_harmonic_balance(grps, num_vars, freq=F0, num_harmonics=N_HARM)
    _, y_freq_i = run_hb_i(y_dc, y_flat_init=y_flat_warmstart)

    _, _, PAE = compute_powers(y_freq_i, V_target, Vdd_val)
    return -PAE


log_params_0 = jnp.log(jnp.array([L_in_init, C_in_init, L_out_init, C_out_init]))

# Evaluate initial PAE
initial_loss = loss_fn(log_params_0)
print(f"Initial PAE at +12 dBm: {float(-initial_loss)*100:.1f}%")

# ── Gradient check ───────────────────────────────────────────────────────────────────
_, grads_0 = jax.value_and_grad(loss_fn)(log_params_0)

# grads_0[i] = ∂(-PAE)/∂(log param_i).  Negating gives ∂PAE/∂(log param).
# Chain rule: ∂PAE/∂(log p) = ∂PAE/∂p × p  →  ∂PAE/∂p = -grads_0[i] / param_i
params_0    = jnp.exp(log_params_0)
dPAE_dparam = -grads_0 / params_0          # SI units: per H or per F

L_in_g  = float(dPAE_dparam[0]) * 1e-9    # per nH
C_in_g  = float(dPAE_dparam[1]) * 1e-12   # per pF
L_out_g = float(dPAE_dparam[2]) * 1e-9    # per nH
C_out_g = float(dPAE_dparam[3]) * 1e-12   # per pF

print("\nAnalytic ∂PAE/∂param  (non-zero → gradient flows through HB):")
print(f"  ∂PAE/∂L_in  = {L_in_g:+.4f}  per nH")
print(f"  ∂PAE/∂C_in  = {C_in_g:+.4f}  per pF")
print(f"  ∂PAE/∂L_out = {L_out_g:+.4f}  per nH")
print(f"  ∂PAE/∂C_out = {C_out_g:+.4f}  per pF")

all_nonzero = all(abs(g) > 1e-6 for g in [L_in_g, C_in_g, L_out_g, C_out_g])
print(f"\n{'All non-zero: implicit differentiation through HB is working.' if all_nonzero else 'WARNING: zero gradients detected.'}")
```

    Optimisation target: Pin = +12 dBm  (V_source amplitude = 2.518 V)


    Initial PAE at +12 dBm: 10.3%



    Analytic ∂PAE/∂param  (non-zero → gradient flows through HB):
      ∂PAE/∂L_in  = +0.0616  per nH
      ∂PAE/∂C_in  = -0.2025  per pF
      ∂PAE/∂L_out = +0.0410  per nH
      ∂PAE/∂C_out = -0.1864  per pF

    All non-zero: implicit differentiation through HB is working.



```python
LR        = 3e-2
optimizer  = optax.adam(learning_rate=LR)
log_params = log_params_0
opt_state  = optimizer.init(log_params)

val_grad_jit = jax.jit(jax.value_and_grad(loss_fn))

pae_history   = []
param_history = [np.array(jnp.exp(log_params))]

print(f"Adam optimisation  (lr = {LR},  200 steps)")
print(f"{'Step':>6}  {'PAE (%)':>8}  {'L_in (nH)':>11}  {'C_in (pF)':>11}  {'L_out (nH)':>12}  {'C_out (pF)':>12}")
print("-" * 70)

for step in range(200):
    loss, grads    = val_grad_jit(log_params)
    pae_history.append(float(-loss) * 100)
    updates, opt_state = optimizer.update(grads, opt_state)
    log_params         = optax.apply_updates(log_params, updates)
    param_history.append(np.array(jnp.exp(log_params)))

    if step % 50 == 0 or step == 199:
        L_in_c, C_in_c, L_out_c, C_out_c = jnp.exp(log_params)
        print(f"{step:>6}  {float(-loss)*100:>8.1f}"
              f"  {float(L_in_c)*1e9:>11.3f}"
              f"  {float(C_in_c)*1e12:>11.3f}"
              f"  {float(L_out_c)*1e9:>12.3f}"
              f"  {float(C_out_c)*1e12:>12.3f}")

L_opt_val, C_in_opt_val, L_out_opt_val, C_out_opt_val = [float(v) for v in jnp.exp(log_params)]
print("\nOptimised matching network:")
print(f"  L_in  = {L_opt_val*1e9:.3f} nH   (was {L_in_init*1e9:.1f} nH)")
print(f"  C_in  = {C_in_opt_val*1e12:.3f} pF   (was {C_in_init*1e12:.1f} pF)")
print(f"  L_out = {L_out_opt_val*1e9:.3f} nH   (was {L_out_init*1e9:.1f} nH)")
print(f"  C_out = {C_out_opt_val*1e12:.3f} pF   (was {C_out_init*1e12:.1f} pF)")
print(f"\nFinal PAE: {pae_history[-1]:.1f}%  (initial: {pae_history[0]:.1f}%)")
```

    Adam optimisation  (lr = 0.03,  200 steps)
      Step   PAE (%)    L_in (nH)    C_in (pF)    L_out (nH)    C_out (pF)
    ----------------------------------------------------------------------


         0      10.3        0.309        0.485         0.309         0.485


        50      25.1        1.322        0.148         0.832         0.266


       100      28.1        1.732        0.068         0.800         0.462


       150      29.0        1.865        0.038         0.799         0.475


       199      29.3        1.926        0.025         0.796         0.485

    Optimised matching network:
      L_in  = 1.926 nH   (was 0.3 nH)
      C_in  = 0.025 pF   (was 0.5 pF)
      L_out = 0.796 nH   (was 0.3 nH)
      C_out = 0.485 pF   (was 0.5 pF)

    Final PAE: 29.3%  (initial: 10.3%)



```python
# ── Build optimised groups for before/after comparison ────────────────────────────────
L_in_f, C_in_f, L_out_f, C_out_f = jnp.exp(log_params)
groups_opt = update_params_dict(groups,     "inductor",  "L_in",  "L", L_in_f)
groups_opt = update_params_dict(groups_opt, "capacitor", "C_in",  "C", C_in_f)
groups_opt = update_params_dict(groups_opt, "inductor",  "L_out", "L", L_out_f)
groups_opt = update_params_dict(groups_opt, "capacitor", "C_out", "C", C_out_f)


def run_at_amplitude_opt(V_in):
    grps = update_params_dict(groups_opt, "voltagesourceac", "Vs", "V", V_in)
    run_hb_i = setup_harmonic_balance(grps, num_vars, freq=F0, num_harmonics=N_HARM)
    _, y_freq_i = run_hb_i(y_dc, y_flat_init=jnp.tile(y_dc, K))
    _, _, PAE = compute_powers(y_freq_i, V_in, Vdd_val)
    return PAE


run_at_amp_opt_jit = jax.jit(run_at_amplitude_opt)
PAE_opt_list = [float(run_at_amp_opt_jit(jnp.array(V_in))) * 100 for V_in in V_amp_vals]

# ── Plotly figures ────────────────────────────────────────────────────────────────────
param_arr  = np.array(param_history)
steps_arr  = list(np.arange(len(param_arr)))
pae_steps  = list(np.arange(len(pae_history)))

specs = [[{}, {"secondary_y": True}, {}]]
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Optimisation Convergence", "Parameter Trajectories", "Before vs After Optimisation"),
    specs=specs,
)

# Panel 1: PAE convergence
fig.add_trace(
    go.Scatter(x=pae_steps, y=pae_history, mode="lines",
               name="PAE (%)", line=dict(color="#1f77b4", width=2)),
    row=1, col=1,
)

# Panel 2: Parameter trajectories (inductances primary y, capacitances secondary y)
fig.add_trace(
    go.Scatter(x=steps_arr, y=list(param_arr[:, 0] * 1e9), mode="lines",
               name="L_in (nH)", line=dict(color="#1f77b4", width=2)),
    row=1, col=2,
)
fig.add_trace(
    go.Scatter(x=steps_arr, y=list(param_arr[:, 2] * 1e9), mode="lines",
               name="L_out (nH)", line=dict(color="#ff7f0e", width=2)),
    row=1, col=2,
)
fig.add_trace(
    go.Scatter(x=steps_arr, y=list(param_arr[:, 1] * 1e12), mode="lines",
               name="C_in (pF)", line=dict(color="#1f77b4", width=2, dash="dash")),
    row=1, col=2, secondary_y=True,
)
fig.add_trace(
    go.Scatter(x=steps_arr, y=list(param_arr[:, 3] * 1e12), mode="lines",
               name="C_out (pF)", line=dict(color="#ff7f0e", width=2, dash="dash")),
    row=1, col=2, secondary_y=True,
)

# Panel 3: Before / after PAE sweep
fig.add_trace(
    go.Scatter(x=list(Pin_dBm_vals), y=PAE_list, mode="lines",
               name="Detuned (initial)", line=dict(color="#1f77b4", width=2, dash="dash")),
    row=1, col=3,
)
fig.add_trace(
    go.Scatter(x=list(Pin_dBm_vals), y=PAE_opt_list, mode="lines",
               name="Optimised", line=dict(color="#ff7f0e", width=2.5)),
    row=1, col=3,
)
fig.add_vline(x=12, line=dict(color="grey", width=1.2, dash="dot"),
              annotation_text="+12 dBm target", annotation_position="top left",
              row=1, col=3)

# Axis labels
fig.update_xaxes(title_text="Adam step", row=1, col=1)
fig.update_yaxes(title_text="PAE (%)", rangemode="tozero", row=1, col=1)
fig.update_xaxes(title_text="Step", row=1, col=2)
fig.update_yaxes(title_text="Inductance (nH)", row=1, col=2, secondary_y=False)
fig.update_yaxes(title_text="Capacitance (pF)", row=1, col=2, secondary_y=True)
fig.update_xaxes(title_text="Pin (dBm)", row=1, col=3)
fig.update_yaxes(title_text="PAE (%)", rangemode="tozero", row=1, col=3)

fig.update_layout(height=460, width=1400)
fig.show()

```




```python
# ── Run HB at optimised matching, target drive level ───────────────────────────────────
grps_wf   = update_params_dict(groups_opt, "voltagesourceac", "Vs", "V", V_target)
run_hb_wf = setup_harmonic_balance(grps_wf, num_vars, freq=F0, num_harmonics=N_HARM)
y_time_opt, y_freq_opt = jax.jit(run_hb_wf)(y_dc, y_flat_init=jnp.tile(y_dc, K))

T    = 1.0 / F0                                       # period [s]
t_ps = np.linspace(0, T * 1e12, K, endpoint=False)   # time axis [ps]

v_gate  = np.real(np.array(y_time_opt[:, osc_node_gate]))
v_drain = np.real(np.array(y_time_opt[:, osc_node_drain]))
i_drain = np.real(np.array(y_time_opt[:, choke_iL_idx]))

Pout_opt, Gain_opt, PAE_opt_val = compute_powers(y_freq_opt, V_target, Vdd_val)

harmonics    = np.arange(N_HARM + 1)
scale        = np.where(harmonics == 0, 1.0, 2.0)
spectrum     = scale * np.abs(np.array(y_freq_opt[:N_HARM + 1, osc_node_drain]))
tick_labels  = ["DC" if k == 0 else f"{k}f₀" for k in harmonics]
i_dc_bias    = float(jnp.real(y_freq_opt[0, choke_iL_idx])) * 1e3

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Gate and Drain Waveforms (1 period)", "Drain Current Waveform", "Drain Voltage Spectrum"),
)

# Panel 1: Gate and drain voltages
fig.add_trace(
    go.Scatter(x=list(t_ps), y=list(v_gate), mode="lines+markers",
               name="Gate (V)", marker=dict(size=5), line=dict(color="#1f77b4")),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=list(t_ps), y=list(v_drain), mode="lines+markers",
               name="Drain (V)", marker=dict(size=5), line=dict(color="#ff7f0e")),
    row=1, col=1,
)
fig.add_hline(y=Vdd_val, line=dict(color="grey", width=1, dash="dash"),
              annotation_text=f"Vdd = {Vdd_val} V", annotation_position="bottom right",
              row=1, col=1)

# Panel 2: Drain current
fig.add_trace(
    go.Scatter(x=list(t_ps), y=list(i_drain * 1e3), mode="lines+markers",
               name="Drain current (mA)", marker=dict(size=5), line=dict(color="#2ca02c")),
    row=1, col=2,
)
fig.add_hline(y=i_dc_bias, line=dict(color="grey", width=1.2, dash="dash"),
              annotation_text=f"DC bias = {i_dc_bias:.1f} mA", annotation_position="bottom right",
              row=1, col=2)

# Panel 3: Harmonic spectrum bar chart
fig.add_trace(
    go.Bar(x=list(harmonics), y=list(spectrum * 1e3),
           name="Spectrum (mV)", marker=dict(color="#1f77b4", opacity=0.85,
           line=dict(color="white", width=1))),
    row=1, col=3,
)
fig.update_xaxes(
    tickmode="array", tickvals=list(harmonics), ticktext=tick_labels,
    title_text="Harmonic", row=1, col=3,
)

fig.update_xaxes(title_text="Time (ps)", row=1, col=1)
fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
fig.update_xaxes(title_text="Time (ps)", row=1, col=2)
fig.update_yaxes(title_text="Drain current (mA)", row=1, col=2)
fig.update_yaxes(title_text="Amplitude (mV)", row=1, col=3)

fig.update_layout(height=440, width=1400)
fig.show()

print("\nOptimised PA — final performance at +12 dBm input:")
print(f"  Pout = {float(Pout_opt):.1f} dBm")
print(f"  Gain = {float(Gain_opt):.1f} dB")
print(f"  PAE  = {float(PAE_opt_val)*100:.1f}%")

```




    Optimised PA — final performance at +12 dBm input:
      Pout = 19.3 dBm
      Gain = 7.3 dB
      PAE  = 29.3%



```python
# ── Animated GIF: optimisation progress ──────────────────────────────────────
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.animation import FuncAnimation, PillowWriter

plt.style.use("dark_background")

ANIM_STRIDE = 5
FPS         = 10
DPI         = 120
GIF_PATH    = "pa_optimisation.gif"

param_arr = np.array(param_history)   # (N_steps+1, 4): [L_in, C_in, L_out, C_out]
pae_arr   = np.array(pae_history)     # (N_steps,)
n_steps   = len(pae_arr)

frame_indices = list(range(0, n_steps, ANIM_STRIDE))
if (n_steps - 1) not in frame_indices:
    frame_indices.append(n_steps - 1)

steps_full = np.arange(n_steps)
steps_p    = np.arange(len(param_arr))

# ── Pre-compute full PAE vs Pin sweep at each frame checkpoint ────────────────
# Each frame needs its own sweep so the curve evolves — not just a dot.
# Runs HB at len(frame_indices) x len(Pin_dBm_vals) points.
print(f"Pre-computing {len(frame_indices)} PAE sweeps x {len(Pin_dBm_vals)} Pin points...")

@jax.jit
def _pae_single(L_in, C_in, L_out, C_out, V_in):
    grps = update_params_dict(groups,    "inductor",        "L_in",  "L", L_in)
    grps = update_params_dict(grps,      "capacitor",       "C_in",  "C", C_in)
    grps = update_params_dict(grps,      "inductor",        "L_out", "L", L_out)
    grps = update_params_dict(grps,      "capacitor",       "C_out", "C", C_out)
    grps = update_params_dict(grps,      "voltagesourceac", "Vs",    "V", V_in)
    run_hb_i = setup_harmonic_balance(grps, num_vars, freq=F0, num_harmonics=N_HARM)
    _, y_freq_i = run_hb_i(y_dc, y_flat_init=jnp.tile(y_dc, K))
    _, _, PAE = compute_powers(y_freq_i, V_in, Vdd_val)
    return PAE * 100.0

frame_pae_sweeps = []
for fi, idx in enumerate(frame_indices):
    L_in_f, C_in_f, L_out_f, C_out_f = param_arr[idx]
    sweep = []
    for V_in in V_amp_vals:
        sweep.append(float(_pae_single(
            jnp.array(L_in_f), jnp.array(C_in_f),
            jnp.array(L_out_f), jnp.array(C_out_f),
            jnp.array(V_in),
        )))
    frame_pae_sweeps.append(sweep)
    if fi % 5 == 0 or fi == len(frame_indices) - 1:
        print(f"  [{fi+1:3d}/{len(frame_indices)}] step {idx:3d} — "
              f"PAE@+12dBm = {sweep[np.argmin(np.abs(Pin_dBm_vals - 12))]:.1f}%")

print("Done.")

# ── Figure layout ─────────────────────────────────────────────────────────────
fig_a = plt.figure(figsize=(13, 7), facecolor="#111111")
gs    = gridspec.GridSpec(2, 2, figure=fig_a,
                          width_ratios=[1, 1], height_ratios=[1, 1.1],
                          hspace=0.28, wspace=0.30)

ax_top  = fig_a.add_subplot(gs[0, :])
ax_bl   = fig_a.add_subplot(gs[1, 0])
ax_br   = fig_a.add_subplot(gs[1, 1])
ax_bl_r = ax_bl.twinx()

for ax in (ax_top, ax_bl, ax_br):
    ax.set_facecolor("#1a1a1a")
    ax.grid(True, alpha=0.2, color="#555555", zorder=0)
ax_bl_r.grid(False)

# ── Static ghost traces ───────────────────────────────────────────────────────
ax_top.plot(steps_full, pae_arr, color="#555555", lw=1.2, zorder=1)
ax_top.axhline(pae_arr[-1], color="#888888", lw=1, ls="--", zorder=1,
               label=f"Final PAE = {pae_arr[-1]:.1f}%")
ax_top.set_xlim(-2, n_steps + 2)
ax_top.set_ylim(bottom=0)
ax_top.set_xlabel("Adam step", fontsize=11)
ax_top.set_ylabel("PAE (%)", fontsize=11)
ax_top.legend(fontsize=9, loc="lower right")

ax_bl.plot(steps_p, param_arr[:, 0] * 1e9, color="#4a90c4", lw=1, zorder=1)
ax_bl.plot(steps_p, param_arr[:, 2] * 1e9, color="#e8a44a", lw=1, zorder=1)
ax_bl_r.plot(steps_p, param_arr[:, 1] * 1e12, color="#4a90c4", lw=1, ls="--", zorder=1)
ax_bl_r.plot(steps_p, param_arr[:, 3] * 1e12, color="#e8a44a", lw=1, ls="--", zorder=1)
ax_bl.set_xlim(-2, len(param_arr) + 2)
ax_bl.set_xlabel("Step", fontsize=11)
ax_bl.set_ylabel("Inductance (nH)", color="#4a90c4", fontsize=11)
ax_bl_r.set_ylabel("Capacitance (pF)", color="#e8a44a", fontsize=11)
ax_bl.tick_params(axis="y", labelcolor="#4a90c4")
ax_bl_r.tick_params(axis="y", labelcolor="#e8a44a")
ax_bl.set_title("Parameter Trajectories", fontsize=11)

# Initial and final PAE sweeps as reference in the bottom-right
ax_br.plot(Pin_dBm_vals, frame_pae_sweeps[0],  color="#4a90c4", lw=1.2, ls="--",
           alpha=0.4, label="Initial", zorder=1)
ax_br.plot(Pin_dBm_vals, frame_pae_sweeps[-1], color="#5cb85c", lw=1.2, ls="-",
           alpha=0.4, label="Final", zorder=1)
ax_br.axvline(12, color="#888888", lw=1, ls=":", zorder=1)
ax_br.set_xlabel("Pin (dBm)", fontsize=11)
ax_br.set_ylabel("PAE (%)", fontsize=11)
ax_br.set_title("PAE vs Pin", fontsize=11)
_pae_max = max(max(s) for s in frame_pae_sweeps)
ax_br.set_ylim(bottom=0, top=_pae_max * 1.12)
ax_br.set_xlim(Pin_dBm_vals[0] - 1, Pin_dBm_vals[-1] + 1)
ax_br.legend(fontsize=8, loc="upper left")

# ── Live artists ──────────────────────────────────────────────────────────────
live_pae_line,  = ax_top.plot([], [], color="#4a90c4", lw=2.5, zorder=2)
live_pae_dot,   = ax_top.plot([], [], "o", color="orange", ms=9, zorder=3)
title_txt = ax_top.set_title("", fontsize=11, pad=8)

live_lin_line,  = ax_bl.plot([], [], color="#4a90c4", lw=2.5, zorder=2, label="L_in (nH)")
live_lout_line, = ax_bl.plot([], [], color="#ff7f0e", lw=2.5, zorder=2, label="L_out (nH)")
live_lin_dot,   = ax_bl.plot([], [], "o", color="#4a90c4", ms=7, zorder=3)
live_lout_dot,  = ax_bl.plot([], [], "o", color="#ff7f0e", ms=7, zorder=3)
live_cin_line,  = ax_bl_r.plot([], [], color="#4a90c4", lw=1.8, ls="--", zorder=2, label="C_in (pF)")
live_cout_line, = ax_bl_r.plot([], [], color="#ff7f0e", lw=1.8, ls="--", zorder=2, label="C_out (pF)")
live_cin_dot,   = ax_bl_r.plot([], [], "o", color="#4a90c4", ms=7, zorder=3)
live_cout_dot,  = ax_bl_r.plot([], [], "o", color="#ff7f0e", ms=7, zorder=3)

lines_bl = [live_lin_line, live_lout_line, live_cin_line, live_cout_line]
ax_bl.legend(lines_bl, [l.get_label() for l in lines_bl], fontsize=7, loc="upper right")

# Evolving PAE sweep curve + dot at target
live_pae_sweep, = ax_br.plot([], [], color="#ff7f0e", lw=2.5, zorder=2)
live_dot_br,    = ax_br.plot([], [], "o", color="orange", ms=9, zorder=3)

# ── Animation function ────────────────────────────────────────────────────────
def _animate(fi):
    step     = frame_indices[fi]
    pae_step = min(step, n_steps - 1)

    # Top: PAE convergence
    live_pae_line.set_data(steps_full[:pae_step + 1], pae_arr[:pae_step + 1])
    live_pae_dot.set_data([pae_step], [pae_arr[pae_step]])
    title_txt.set_text(f"Optimisation Convergence — step {step} / {n_steps}")

    # Bottom-left: parameter trajectories
    p_end = min(step + 1, len(param_arr))
    xp = steps_p[:p_end]
    live_lin_line.set_data(xp, param_arr[:p_end, 0] * 1e9)
    live_lout_line.set_data(xp, param_arr[:p_end, 2] * 1e9)
    live_cin_line.set_data(xp, param_arr[:p_end, 1] * 1e12)
    live_cout_line.set_data(xp, param_arr[:p_end, 3] * 1e12)
    live_lin_dot.set_data([xp[-1]], [param_arr[p_end - 1, 0] * 1e9])
    live_lout_dot.set_data([xp[-1]], [param_arr[p_end - 1, 2] * 1e9])
    live_cin_dot.set_data([xp[-1]], [param_arr[p_end - 1, 1] * 1e12])
    live_cout_dot.set_data([xp[-1]], [param_arr[p_end - 1, 3] * 1e12])

    # Bottom-right: full evolving PAE sweep + dot at +12 dBm
    sweep = frame_pae_sweeps[fi]
    live_pae_sweep.set_data(Pin_dBm_vals, sweep)
    pin12_idx = int(np.argmin(np.abs(Pin_dBm_vals - 12.0)))
    live_dot_br.set_data([Pin_dBm_vals[pin12_idx]], [sweep[pin12_idx]])


anim = FuncAnimation(
    fig_a, _animate,
    frames=len(frame_indices),
    interval=1000 // FPS,
    blit=False,
)

anim.save(GIF_PATH, writer=PillowWriter(fps=FPS), dpi=DPI)
plt.close(fig_a)
print(f"Saved \u2192 {GIF_PATH}")
print(f"Frames: {len(frame_indices)}   Duration: {len(frame_indices)/FPS:.1f}s at {FPS} fps")
```

    Pre-computing 41 PAE sweeps x 17 Pin points...


      [  1/41] step   0 — PAE@+12dBm = 10.3%


      [  6/41] step  25 — PAE@+12dBm = 21.8%


      [ 11/41] step  50 — PAE@+12dBm = 25.1%


      [ 16/41] step  75 — PAE@+12dBm = 27.1%


      [ 21/41] step 100 — PAE@+12dBm = 28.1%


      [ 26/41] step 125 — PAE@+12dBm = 28.7%


      [ 31/41] step 150 — PAE@+12dBm = 29.0%


      [ 36/41] step 175 — PAE@+12dBm = 29.2%


      [ 41/41] step 199 — PAE@+12dBm = 29.3%
    Done.


    Saved → pa_optimisation.gif
    Frames: 41   Duration: 4.1s at 10 fps


## Summary

### Matching Network — Initial vs Optimised

| Component | Initial | Optimised | Role |
|-----------|---------|-----------|------|
| `L_in`  | 0.3 nH | see Cell 14 output | Input L-match shunt arm |
| `C_in`  | 0.5 pF | see Cell 14 output | Input L-match series arm |
| `L_out` | 0.3 nH | see Cell 14 output | Output L-match shunt arm |
| `C_out` | 0.5 pF | see Cell 14 output | Output L-match series arm |

### What made this possible

| Traditional EDA | Circulax |
|---|---|
| HB solve → hand-tune → re-solve | HB solve → `jax.grad` → Adam step |
| Gradient-free search (genetic, random) | Exact analytic gradients via implicit differentiation |
| Hundreds of HB evaluations to converge | ~1 adjoint solve per gradient (one extra LU factorisation) |
| Fixed S-parameter models | Fully differentiable device physics (HEMT, diodes, varactors) |

### Extending this framework

The same gradient infrastructure applies directly to:

- **Load-pull contours** — sweep complex Γ_L and compute `∂PAE/∂Γ_L` analytically.
- **Multi-tone IMD / ACPR** — add intermodulation terms to the HB system; differentiate w.r.t.
  device parameters to minimise distortion.
- **Phase noise** — perturb the HB fixed point; the linear noise response is the adjoint of the
  same Jacobian already computed during optimisation.
- **Co-design** — jointly optimise device epitaxial parameters (β, Vp) and circuit matching,
  treating the entire design stack as one differentiable programme.
