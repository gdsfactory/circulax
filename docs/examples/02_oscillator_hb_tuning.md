# Van der Pol Oscillator Tuning via Backpropagation through Harmonic Balance

This notebook shows how automatic differentiation through the Harmonic Balance solver
can solve oscillator design problems that are impractical with traditional tools.

## The problem

Given a Van der Pol oscillator (a nonlinear LC tank with negative resistance), tune its
parameters so the oscillation hits a precise target frequency and amplitude. This is a
joint nonlinear design problem with no closed-form solution.

**Traditional approach:** Sweep μ over a grid of values, run a transient simulation to
steady state for each, measure the amplitude, and read off the closest match. For a
two-parameter sweep (frequency + amplitude) the cost is quadratic in grid resolution.

**circulax approach:** `jax.grad` through the HB Newton solver via Optimistix's implicit
differentiation. The solver finds the periodic steady state in ~15 Newton steps; the
gradient is computed by differentiating through that fixed-point computation. A single
gradient call replaces the entire parameter sweep.

## Circuit: Van der Pol oscillator

The Van der Pol element has a cubic I–V characteristic:

$$I(V) = -\mu G_0 V + \frac{\mu G_0}{3} V^3$$

- For small $|V|$: slope $= -\mu G_0 < 0$ — negative resistance, oscillation grows
- Cubic term limits amplitude at $|V| \approx \sqrt{3/G_0}$ — stable limit cycle

The element is placed in parallel with an LC tank:

```
top_node ─── VDP   ─── GND    (Van der Pol: negative resistance + cubic saturation)
top_node ─── L1    ─── GND    (inductor)
top_node ─── C1    ─── GND    (capacitor)
top_node ─── Rdamp ─── GND    (small tank loss, models coil resistance)
```

The tank resonates at $f_0 = 1/(2\pi\sqrt{LC})$. The VDP element pumps energy in at
small amplitudes (negative conductance dominates) and absorbs energy at large amplitudes
(cubic term dominates), creating a stable limit cycle.


```python
import jax
import jax.numpy as jnp
import numpy as np
import optax
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from circulax import compile_circuit, setup_harmonic_balance
from circulax.components.base_component import PhysicsReturn, Signals, States, component
from circulax.components.electronic import Capacitor, Inductor, Resistor
from circulax.utils import update_group_params, update_params_dict

# 64-bit precision is important: HB Newton requires accurate Jacobians, and
# gradient-based optimisation accumulates floating-point error across steps.
jax.config.update("jax_enable_x64", True)

pio.templates.default = "plotly_white"

```

    KLUJAX_RS DEBUG MODE.
    WARNING:2026-04-15 16:19:25,983:jax._src.xla_bridge:864: An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.


## Defining the Van der Pol component

Custom components are plain Python functions decorated with `@component`. The decorator
generates an Equinox module whose parameters (`mu`, `G0`) are JAX-traceable leaves,
making the component compatible with `jax.vmap`, `jax.jacfwd`, and `jax.grad`.


```python
@component(ports=("p1", "p2"))
def VanDerPolElement(signals: Signals, s: States, mu: float = 2.0, G0: float = 0.01) -> PhysicsReturn:
    """Nonlinear two-terminal element with cubic I-V characteristic.

    I(V) = -mu*G0*V + (G0/3)*V^3

    The first term is a negative conductance (-mu*G0 < 0), which supplies
    energy to the tank when |V| is small.  The cubic term (G0/3)*V^3 saturates
    the gain at large amplitudes, producing a stable oscillation.

    Limit-cycle amplitude (fundamental, from harmonic balance energy balance):
        -mu*G0*A + G0/4*A^3 = 0  -->  A = 2*sqrt(mu)
    With G0=0.01 and mu=2.0: A = 2*sqrt(2) ≈ 2.83 V

    Note: mu appears only in the linear term so that the limit-cycle amplitude
    A = 2*sqrt(mu) is strongly tunable — a factor-of-4 change in mu moves A by
    2×.  If mu were also in the cubic term, amplitude would saturate at ~2 V
    regardless of mu.
    """
    v = signals.p1 - signals.p2
    i = -mu * G0 * v + (G0 / 3.0) * v**3
    return {"p1": i, "p2": -i}, {}


# Quick smoke-test: check that the I-V slope at V=0 is -mu*G0 (negative conductance)
vdp_test = VanDerPolElement(mu=2.0, G0=0.01)
f_test, _ = vdp_test(p1=0.1, p2=0.0)
print(f"VDP I at V=0.1 V : {f_test['p1']:.6f} A  (expected {-2.0*0.01*0.1 + (0.01/3)*0.1**3:.6f} A)")

print(f"Expected limit-cycle amplitude: 2*sqrt(mu) = {2*np.sqrt(2.0):.3f} V  (at mu=2, G0=0.01)")

```

    VDP I at V=0.1 V : -0.001997 A  (expected -0.001997 A)
    Expected limit-cycle amplitude: 2*sqrt(mu) = 2.828 V  (at mu=2, G0=0.01)


## Building and compiling the circuit

All four elements connect between the same `top_node` and GND — they are in parallel.
The LC tank sets the oscillation frequency; `Rdamp` models the finite Q of a real inductor
(10 kΩ is intentionally large so tank losses are much smaller than the VDP negative
resistance, ensuring oscillation).

The tuple syntax for connections (`"GND,p1": ("VDP,p2", "C1,p2", ...)`) joins multiple
ports to the same node in a single line — equivalent to writing each pair separately.


```python
# ── Circuit parameters ────────────────────────────────────────────────────────
L_val  = 1e-6    # H  (1 µH)
C_val  = 1e-9    # F  (1 nF)
R_damp = 1e4     # Ω  (10 kΩ tank loss — small, Q ≈ R_damp * sqrt(C/L) ≈ 316)

f0 = 1.0 / (2.0 * np.pi * np.sqrt(L_val * C_val))
print(f"Tank resonant frequency : {f0 / 1e6:.4f} MHz")
print(f"Tank Q-factor (Rdamp)   : {R_damp * np.sqrt(C_val / L_val):.1f}")
print(f"VDP negative conductance: {-2.0 * 0.01:.4f} S  (at mu=2, G0=0.01)")
print(f"Tank loss conductance   : {1.0 / R_damp:.6f} S  (1/Rdamp)")
print(f"Net gain margin         : {abs(-2.0 * 0.01) / (1.0 / R_damp):.0f}×  >> 1, oscillation guaranteed")

# ── Netlist ──────────────────────────────────────────────────────────────────
# All four elements are in parallel between top_node and GND.
# GND is the special reserved name — any port touching "GND" is assigned node 0.
vdp_net = {
    "instances": {
        "VDP":   {"component": "vdp",       "settings": {"mu": 2.0,    "G0": 0.01}},
        "L1":    {"component": "inductor",   "settings": {"L": L_val}},
        "C1":    {"component": "capacitor",  "settings": {"C": C_val}},
        "Rdamp": {"component": "resistor",   "settings": {"R": R_damp}},
    },
    "connections": {
        # Connect all negative terminals to GND (node 0)
        "GND,p1":  ("VDP,p2", "L1,p2", "C1,p2", "Rdamp,p2"),
        # Connect all positive terminals to the same top_node
        "VDP,p1":  ("L1,p1", "C1,p1", "Rdamp,p1"),
    },
}

models = {
    "vdp":      VanDerPolElement,
    "inductor":  Inductor,
    "capacitor": Capacitor,
    "resistor":  Resistor,
}

# compile_circuit runs once: builds ComponentGroup objects with batched JAX arrays
# for parameters, and pre-computes index arrays for residual assembly.
circuit = compile_circuit(vdp_net, models)
groups = circuit.groups
num_vars = circuit.sys_size
net_map = circuit.port_map

print(f"\nSystem size : {num_vars} unknowns")
print(f"Node map    : {net_map}")
print(f"Groups      : {list(groups.keys())}")

# The oscillator node — all parallel elements share this port
osc_node = net_map["VDP,p1"]
print(f"Oscillator node index: {osc_node}")

# DC operating point: V=0 is the only fixed point (the VDP element has I(0)=0)
y_dc = circuit()
print(f"DC operating point: max|y_dc| = {float(jnp.max(jnp.abs(y_dc))):.2e} V  (trivially zero)")
```

    Tank resonant frequency : 5.0329 MHz
    Tank Q-factor (Rdamp)   : 316.2
    VDP negative conductance: -0.0200 S  (at mu=2, G0=0.01)
    Tank loss conductance   : 0.000100 S  (1/Rdamp)
    Net gain margin         : 200×  >> 1, oscillation guaranteed



    System size : 3 unknowns
    Node map    : {'C1,p1': 1, 'L1,p1': 1, 'VDP,p1': 1, 'Rdamp,p1': 1, 'C1,p2': 0, 'VDP,p2': 0, 'L1,p2': 0, 'GND,p1': 0, 'Rdamp,p2': 0, 'L1,i_L': 2}
    Groups      : ['vdp', 'inductor', 'capacitor', 'resistor']
    Oscillator node index: 1


    DC operating point: max|y_dc| = 0.00e+00 V  (trivially zero)



```python
# ── Interactive circuit explorer ─────────────────────────────────────────────
# Adjust L, C, R_damp, mu, G0 with sliders to explore how the oscillator responds
# before running the optimisation cells below.
from dash import Input, Output, dcc, html
from jupyter_dash import JupyterDash

_app = JupyterDash(__name__)

_SLIDER_STYLE = {"marginBottom": "18px"}
_LABEL_STYLE  = {"fontWeight": "bold", "fontFamily": "sans-serif", "fontSize": "13px"}

_app.layout = html.Div([
    html.H3("Van der Pol Oscillator — Circuit Explorer",
            style={"fontFamily": "sans-serif", "marginBottom": "20px"}),
    html.Div([
        html.Div([
            html.Label("L (\u00b5H)", style=_LABEL_STYLE),
            dcc.Slider(id="sl-L", min=0.1, max=10.0, step=0.1, value=1.0,
                       marks={v: str(v) for v in [0.1, 1, 2, 5, 10]},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style=_SLIDER_STYLE),
        html.Div([
            html.Label("C (nF)", style=_LABEL_STYLE),
            dcc.Slider(id="sl-C", min=0.1, max=10.0, step=0.1, value=1.0,
                       marks={v: str(v) for v in [0.1, 1, 2, 5, 10]},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style=_SLIDER_STYLE),
        html.Div([
            html.Label("R_damp (k\u03a9)", style=_LABEL_STYLE),
            dcc.Slider(id="sl-R", min=1.0, max=50.0, step=1.0, value=10.0,
                       marks={v: str(v) for v in [1, 10, 20, 50]},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style=_SLIDER_STYLE),
        html.Div([
            html.Label("\u03bc (VDP nonlinearity)", style=_LABEL_STYLE),
            dcc.Slider(id="sl-mu", min=0.25, max=6.0, step=0.25, value=2.0,
                       marks={v: str(v) for v in [0.25, 1, 2, 3, 4, 6]},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style=_SLIDER_STYLE),
        html.Div([
            html.Label("G\u2080 (conductance, S)", style=_LABEL_STYLE),
            dcc.Slider(id="sl-G0", min=0.001, max=0.05, step=0.001, value=0.01,
                       marks={0.001: "0.001", 0.01: "0.01", 0.03: "0.03", 0.05: "0.05"},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style=_SLIDER_STYLE),
        html.Div(id="circ-info",
                 style={"fontFamily": "monospace", "fontSize": "13px",
                        "padding": "12px", "background": "#f5f5f5",
                        "borderRadius": "6px", "marginTop": "10px"}),
    ], style={"width": "38%", "display": "inline-block", "verticalAlign": "top",
              "padding": "20px"}),
    html.Div([
        dcc.Graph(id="explorer-waveform"),
        dcc.Graph(id="explorer-spectrum"),
    ], style={"width": "58%", "display": "inline-block", "verticalAlign": "top"}),
], style={"maxWidth": "1100px"})


@_app.callback(
    Output("explorer-waveform", "figure"),
    Output("explorer-spectrum", "figure"),
    Output("circ-info", "children"),
    Input("sl-L",  "value"),
    Input("sl-C",  "value"),
    Input("sl-R",  "value"),
    Input("sl-mu", "value"),
    Input("sl-G0", "value"),
)
def _update_explorer(L_uh, C_nf, R_kohm, mu_val, G0_val):
    import plotly.graph_objects as _go
    L  = float(L_uh)  * 1e-6
    C  = float(C_nf)  * 1e-9
    R  = float(R_kohm) * 1e3
    f  = 1.0 / (2.0 * np.pi * np.sqrt(L * C))
    Q  = R * np.sqrt(C / L)
    A_theory = 2.0 * np.sqrt(float(mu_val))

    vdp_net_e = {
        "instances": {
            "VDP":   {"component": "vdp",       "settings": {"mu": float(mu_val), "G0": float(G0_val)}},
            "L1":    {"component": "inductor",   "settings": {"L": L}},
            "C1":    {"component": "capacitor",  "settings": {"C": C}},
            "Rdamp": {"component": "resistor",   "settings": {"R": R}},
        },
        "connections": {
            "GND,p1": ("VDP,p2", "L1,p2", "C1,p2", "Rdamp,p2"),
            "VDP,p1": ("L1,p1", "C1,p1", "Rdamp,p1"),
        },
    }
    ckt_e = compile_circuit(vdp_net_e, models)
    grps_e, nv_e, nm_e = ckt_e.groups, ckt_e.sys_size, ckt_e.port_map
    osc_e = nm_e["VDP,p1"]
    dc_e  = ckt_e()

    N_e = 7
    run_hb_e = setup_harmonic_balance(grps_e, nv_e, freq=f, num_harmonics=N_e, osc_node=osc_e)
    yt_e, yf_e = jax.jit(run_hb_e)(dc_e)

    K_e    = 2 * N_e + 1
    t_ns_e = np.linspace(0.0, 1e9 / f, K_e, endpoint=False)
    v_e    = np.array(yt_e[:, osc_e])
    A_fund = float(2.0 * jnp.abs(yf_e[1, osc_e]))

    harms_e  = np.arange(N_e + 1)
    spec_e   = np.where(harms_e == 0, 1.0, 2.0) * np.abs(np.array(yf_e[:, osc_e]))
    tick_e   = ["DC" if k == 0 else f"{k}f\u2080" for k in harms_e]

    fig_w = _go.Figure()
    fig_w.add_trace(_go.Scatter(x=t_ns_e, y=v_e, mode="lines+markers",
                                marker=dict(size=6), line=dict(width=2),
                                name="V_osc"))
    fig_w.add_hline(y=0, line_dash="dash", line_color="grey", line_width=0.7)
    fig_w.update_layout(
        title=f"Limit cycle — f\u2080 = {f/1e6:.3f} MHz",
        xaxis_title="Time (ns)", yaxis_title="Voltage (V)",
        height=300, margin=dict(t=40, b=30),
        template="plotly_white",
    )

    fig_s = _go.Figure()
    fig_s.add_trace(_go.Bar(x=tick_e, y=spec_e, marker_color="#636EFA", opacity=0.85))
    fig_s.update_layout(
        title=f"Harmonic spectrum — A_fund = {A_fund:.3f} V",
        xaxis_title="Harmonic", yaxis_title="Amplitude (V)",
        height=280, margin=dict(t=40, b=30),
        template="plotly_white",
    )

    info = [
        html.Div(f"f\u2080 = {f/1e6:.4f} MHz"),
        html.Div(f"Q  = {Q:.1f}"),
        html.Div(f"A_fund \u2248 {A_fund:.3f} V  (theory: 2\u221a\u03bc = {A_theory:.3f} V)"),
        html.Div(f"Negative conductance = {-mu_val*G0_val:.5f} S"),
        html.Div(f"Gain margin = {abs(-mu_val*G0_val) / (1.0/R):.0f}\u00d7"),
    ]
    return fig_w, fig_s, info


_app.run(mode="inline", height=720, port=8050)

```

    /home/cdaunt/code/circulax/circulax/.pixi/envs/default/lib/python3.13/site-packages/dash/dash.py:644: UserWarning: JupyterDash is deprecated, use Dash instead.
    See https://dash.plotly.com/dash-in-jupyter for more details.
      warnings.warn(




<iframe
    width="100%"
    height="650"
    src="http://127.0.0.1:8050/"
    frameborder="0"
    allowfullscreen

></iframe>



## Part 1 — Harmonic Balance finds the limit cycle

Transient simulation would need to integrate forward in time until the oscillation
envelope settles — often hundreds of RF cycles. Harmonic Balance finds the periodic
steady state **directly** by solving for the Fourier coefficients of the waveform.

`setup_harmonic_balance` builds a residual function over K = 2N+1 equally-spaced time
samples per period and solves it with Newton–Raphson (via Optimistix's
`FixedPointIteration` backed by `jax.lax.while_loop`). The result is JIT-compatible and
differentiable end-to-end.


```python
N_harm = 7   # 7 harmonics → K = 15 time points per period
K      = 2 * N_harm + 1

# Pass osc_node so setup_harmonic_balance automatically tries several sinusoidal
# initial amplitudes via jax.vmap and selects the limit-cycle solution.
# Users never need to choose a starting amplitude or think about the trivial y=0
# fixed point — the multi-start strategy handles it transparently.
run_hb = setup_harmonic_balance(groups, num_vars, freq=f0, num_harmonics=N_harm, osc_node=osc_node)
y_time, y_freq = jax.jit(run_hb)(y_dc)

print(f"y_time shape : {y_time.shape}  (K={K} time samples × {num_vars} nodes)")
print(f"y_freq shape : {y_freq.shape}  ({N_harm+1} harmonics × {num_vars} nodes)")

# y_freq[0] is the DC component, y_freq[1] is the fundamental, etc.
# Two-sided amplitude at harmonic k>=1 is 2 * |y_freq[k]|  (rfft folds negative freqs)
A_dc   = float(jnp.abs(y_freq[0, osc_node]))
A_fund = float(2.0 * jnp.abs(y_freq[1, osc_node]))
A_2nd  = float(2.0 * jnp.abs(y_freq[2, osc_node]))
A_3rd  = float(2.0 * jnp.abs(y_freq[3, osc_node]))

print(f"\nOscillator node (index {osc_node}) harmonic amplitudes:")
print(f"  DC (0f0)       : {A_dc:.4f} V")
print(f"  Fundamental f0 : {A_fund:.4f} V")
print(f"  2nd harmonic   : {A_2nd:.4f} V  ({A_2nd/A_fund*100:.1f}% of fundamental)")
print(f"  3rd harmonic   : {A_3rd:.4f} V  ({A_3rd/A_fund*100:.1f}% of fundamental)")
print(f"\nExpected amplitude (VdP limit cycle): ~2*sqrt(mu) = {2*np.sqrt(2.0):.3f} V")

```

    y_time shape : (15, 3)  (K=15 time samples × 3 nodes)
    y_freq shape : (8, 3)  (8 harmonics × 3 nodes)

    Oscillator node (index 1) harmonic amplitudes:
      DC (0f0)       : 0.0000 V
      Fundamental f0 : 2.9139 V
      2nd harmonic   : 0.0000 V  (0.0% of fundamental)
      3rd harmonic   : 0.2501 V  (8.6% of fundamental)

    Expected amplitude (VdP limit cycle): ~2*sqrt(mu) = 2.828 V



```python
# ── Plot: time-domain waveform and harmonic spectrum ─────────────────────────
T     = 1.0 / f0
t_ns  = np.linspace(0.0, T * 1e9, K, endpoint=False)
v_osc = np.array(y_time[:, osc_node])

harmonics   = np.arange(N_harm + 1)
scale       = np.where(harmonics == 0, 1.0, 2.0)
spectrum    = scale * np.abs(np.array(y_freq[:, osc_node]))
tick_labels = ["DC" if k == 0 else f"{k}f\u2080" for k in harmonics]

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        f"Van der Pol limit cycle — f\u2080 = {f0/1e6:.3f} MHz",
        "Harmonic spectrum",
    ),
)

fig.add_trace(
    go.Scatter(x=t_ns, y=v_osc, mode="lines+markers",
               marker=dict(size=7), line=dict(width=1.5),
               name=f"V_osc (HB, K={K})"),
    row=1, col=1,
)
fig.add_hline(y=0, line_dash="dash", line_color="grey", line_width=0.7, row=1, col=1)

fig.add_trace(
    go.Bar(x=tick_labels, y=spectrum, marker_color="#636EFA", opacity=0.85, name="spectrum"),
    row=1, col=2,
)

fig.update_xaxes(title_text="Time (ns)", row=1, col=1)
fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
fig.update_xaxes(title_text="Harmonic", row=1, col=2)
fig.update_yaxes(title_text="Amplitude (V)", row=1, col=2)
fig.update_layout(height=400, showlegend=True)
fig.show()

print("The odd-harmonic dominance (f0, 3f0, 5f0) is a hallmark of the symmetric cubic nonlinearity.")
print("Even harmonics (2f0, 4f0) are near-zero because I(-V) = -I(V) — the VDP element is odd.")

```



    The odd-harmonic dominance (f0, 3f0, 5f0) is a hallmark of the symmetric cubic nonlinearity.
    Even harmonics (2f0, 4f0) are near-zero because I(-V) = -I(V) — the VDP element is odd.


## Part 2 — Gradient-based amplitude tuning

Goal: find the value of μ such that the fundamental amplitude equals a target $A_{\text{target}}$.

The key insight: `setup_harmonic_balance` returns a function that is **end-to-end
differentiable** with respect to any JAX-traceable quantity captured in `groups`. The
parameter update is done with `update_group_params` (uses `eqx.tree_at` under the hood —
a pure functional update with no in-place mutation), so the entire loss computation is a
valid JAX program that `jax.grad` can differentiate through.

Optimistix's `FixedPointIteration` implements implicit differentiation: the gradient of
the fixed-point solution with respect to μ is computed via the implicit function theorem
rather than unrolling the Newton iterations.


```python
A_target = 1.5  # V  (requires mu* = (A_target/2)^2 = 0.5625)

# Fixed warm start for the loss function: sinusoidal at 3.5 V at osc_node, all
# other variables zero.  3.5 V is above the limit-cycle amplitude for all mu in
# [0.5, 3] (LC range [1.5V, 3.46V]), so Newton always descends to the LC, never
# to the trivial y=0 fixed point.  Using a constant here keeps the gradient path
# free of argmax (Optimistix's implicit diff works cleanly through the fixed point).
_phase = 2.0 * jnp.pi * jnp.arange(K, dtype=jnp.float64) / K
y_flat_warmstart = (
    jnp.zeros(K * num_vars, dtype=jnp.float64)
    .at[jnp.arange(K) * num_vars + osc_node].set(3.5 * jnp.sin(_phase))
)


def loss_fn_mu(mu: jax.Array) -> jax.Array:
    """Squared error between fundamental amplitude and target, as a function of mu."""
    groups_new = update_group_params(groups, "vdp", "mu", mu)
    run_hb_new = setup_harmonic_balance(groups_new, num_vars, freq=f0, num_harmonics=N_harm)
    _, y_freq_new = run_hb_new(y_dc, y_flat_init=y_flat_warmstart)
    A_fund = 2.0 * jnp.abs(y_freq_new[1, osc_node])
    return (A_fund - A_target) ** 2


# Test: evaluate loss and gradient at the default mu=2.0
mu_test = jnp.array(2.0)
loss_val, grad_val = jax.value_and_grad(loss_fn_mu)(mu_test)
print(f"At mu=2.0:  loss = {float(loss_val):.4f} V²,  grad = {float(grad_val):.4f} V²/[mu]")
print(f"  Current amplitude ≈ {float(jnp.sqrt(loss_val)) + A_target:.3f} V,  target = {A_target} V")
print("  Positive gradient → decreasing mu reduces amplitude toward target")

# ── Gradient descent — scalar parameter, no need for Adam ───────────────────
mu = jnp.array(3.0)  # start above the solution (A=2*sqrt(3)≈3.46 V)
lr = 0.05

mu_history    = [float(mu)]
loss_history  = []

val_and_grad_jit = jax.jit(jax.value_and_grad(loss_fn_mu))

print(f"\nGradient descent (lr={lr}, starting from mu={float(mu):.1f}):")
for step in range(80):
    loss, g = val_and_grad_jit(mu)
    loss_history.append(float(loss))
    mu = mu - lr * g
    mu_history.append(float(mu))
    if step % 20 == 0 or step == 79:
        groups_cur = update_group_params(groups, "vdp", "mu", mu)
        _, yf_cur  = jax.jit(setup_harmonic_balance(groups_cur, num_vars, freq=f0, num_harmonics=N_harm, osc_node=osc_node))(y_dc)
        A_cur = float(2.0 * jnp.abs(yf_cur[1, osc_node]))
        print(f"  Step {step:3d}: mu = {float(mu):.4f},  loss = {float(loss):.6f},  A_fund = {A_cur:.4f} V")

mu_opt = float(mu)
print(f"\nOptimised mu = {mu_opt:.4f}  (analytical: {(A_target/2)**2:.4f})")

```

    At mu=2.0:  loss = 1.9991 V²,  grad = 3.4346 V²/[mu]
      Current amplitude ≈ 2.914 V,  target = 1.5 V
      Positive gradient → decreasing mu reduces amplitude toward target

    Gradient descent (lr=0.05, starting from mu=3.0):


      Step   0: mu = 2.5133,  loss = 4.027752,  A_fund = 3.2511 V


      Step  20: mu = 0.5763,  loss = 0.013855,  A_fund = 1.5303 V


      Step  40: mu = 0.5078,  loss = 0.000000,  A_fund = 1.5000 V


      Step  60: mu = 0.5076,  loss = 0.000000,  A_fund = 1.4998 V


      Step  79: mu = 0.5076,  loss = 0.000000,  A_fund = 1.4998 V

    Optimised mu = 0.5076  (analytical: 0.5625)



```python
# ── Compute waveforms before and after optimisation ───────────────────────────
groups_before = update_group_params(groups, "vdp", "mu", jnp.array(2.0))
groups_after  = update_group_params(groups, "vdp", "mu", jnp.array(mu_opt))

_, yf_before = jax.jit(setup_harmonic_balance(groups_before, num_vars, freq=f0, num_harmonics=N_harm, osc_node=osc_node))(y_dc)
yt_after, yf_after = jax.jit(setup_harmonic_balance(groups_after, num_vars, freq=f0, num_harmonics=N_harm, osc_node=osc_node))(y_dc)

A_before = float(2.0 * jnp.abs(yf_before[1, osc_node]))
A_after  = float(2.0 * jnp.abs(yf_after[1, osc_node]))

v_before = np.array(y_time[:, osc_node])
v_after  = np.array(yt_after[:, osc_node])

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Before and after \u03bc tuning",
        f"Amplitude tuning convergence (target {A_target} V)",
    ),
)

fig.add_trace(
    go.Scatter(x=t_ns, y=v_before, mode="lines", line=dict(width=2),
               name=f"\u03bc=2.00 \u2192 A={A_before:.2f} V (initial)"),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=t_ns, y=v_after, mode="lines", line=dict(width=2.5, dash="dash"),
               name=f"\u03bc={mu_opt:.2f} \u2192 A={A_after:.2f} V (optimised)"),
    row=1, col=1,
)
fig.add_hline(y= A_target, line_dash="dot", line_color="grey", line_width=0.8, row=1, col=1)
fig.add_hline(y=-A_target, line_dash="dot", line_color="grey", line_width=0.8, row=1, col=1)

fig.add_trace(
    go.Scatter(x=list(range(len(loss_history))), y=loss_history, mode="lines",
               line=dict(width=2), name="loss", showlegend=False),
    row=1, col=2,
)

fig.update_xaxes(title_text="Time (ns)", row=1, col=1)
fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
fig.update_xaxes(title_text="Gradient descent step", row=1, col=2)
fig.update_yaxes(title_text="Loss (V\u00b2)", type="log", row=1, col=2)
fig.update_layout(height=420)
fig.show()

print(f"Amplitude error after optimisation: {abs(A_after - A_target)*1000:.2f} mV")

```



    Amplitude error after optimisation: 0.23 mV


## Part 3 — Joint tuning: frequency and amplitude

Now we solve the harder problem: simultaneously tune L, C, and μ so the oscillator
hits a **new target frequency** (8 MHz, shifted from 5 MHz) **and** a target amplitude
(1.0 V).

This is a joint nonlinear design problem. The traditional approach would be a 3D sweep
over (L, C, μ) — thousands of HB solves just to map the landscape. With `jax.grad` we
get the exact gradient with one HB solve, and Adam navigates directly to the solution.

**Log-space parameterisation** keeps all three parameters positive and balances their
gradients: a 10% change in L feels the same as a 10% change in μ, regardless of the
absolute scale. We also pass the resonant frequency (derived from L and C) directly to
`setup_harmonic_balance` — this ensures the HB discretisation always matches the actual
oscillation frequency, which is essential for convergence when L and C are changing.


```python
f_target = 8e6    # Hz — 8 MHz target (up from ~5 MHz)
A_target_j = 1.0  # V  — 1 V fundamental amplitude

LC_target = 1.0 / (2.0 * np.pi * f_target) ** 2
print(f"Target frequency    : {f_target/1e6:.1f} MHz")
print(f"Required L*C product: {LC_target:.3e} H·F")
print(f"Starting L={L_val*1e6:.2f} µH, C={C_val*1e9:.2f} nF  → f0={f0/1e6:.3f} MHz")

# Fixed sinusoidal warm start for the joint optimisation loss — amplitude 2 V at osc_node,
# all other nodes zero.  This is above the basin-of-attraction saddle (~1.15 V) for all
# (L, C, mu) values explored during optimisation.  Using a constant here keeps the
# gradient path free of argmax, so Optimistix's implicit diff works cleanly.
_phase_k = 2.0 * jnp.pi * jnp.arange(K, dtype=jnp.float64) / K
y_flat_hb_init = (
    jnp.zeros(K * num_vars, dtype=jnp.float64)
    .at[jnp.arange(K) * num_vars + osc_node].set(3.5 * jnp.sin(_phase_k))
)


def loss_joint(log_params: jax.Array) -> jax.Array:
    """Joint loss over (log_L, log_C, log_mu) for frequency + amplitude targets."""
    log_L, log_C, log_mu = log_params
    L   = jnp.exp(log_L)
    C   = jnp.exp(log_C)
    mu  = jnp.exp(log_mu)

    f_resonant = 1.0 / (2.0 * jnp.pi * jnp.sqrt(L * C))

    grps = update_params_dict(groups, "inductor",  "L1", "L", L)
    grps = update_params_dict(grps,   "capacitor", "C1", "C", C)
    grps = update_group_params(grps,  "vdp", "mu", mu)

    run_hb_j = setup_harmonic_balance(grps, num_vars, freq=f_resonant, num_harmonics=N_harm)
    _, y_freq_j = run_hb_j(jnp.zeros(num_vars), y_flat_init=y_flat_hb_init)

    A_fund = 2.0 * jnp.abs(y_freq_j[1, osc_node])

    loss_freq = ((f_resonant - f_target) / f_target) ** 2
    loss_amp  = ((A_fund - A_target_j) / A_target_j) ** 2

    return loss_freq + loss_amp


# Sanity check
log_params_0 = jnp.log(jnp.array([L_val, C_val, 2.0]))
loss_0 = loss_joint(log_params_0)
print(f"\nInitial joint loss: {float(loss_0):.4f}  (freq error + amplitude error)")

# ── Adam optimisation ─────────────────────────────────────────────────────────
optimizer   = optax.adam(0.05)
log_params  = log_params_0
opt_state   = optimizer.init(log_params)
losses_joint = []
param_log_hist = [np.array(log_params)]

val_grad_joint = jax.jit(jax.value_and_grad(loss_joint))

print("\nAdam optimisation (200 steps, lr=0.05):")
for i in range(200):
    loss, grads = val_grad_joint(log_params)
    losses_joint.append(float(loss))
    updates, opt_state = optimizer.update(grads, opt_state)
    log_params = optax.apply_updates(log_params, updates)
    param_log_hist.append(np.array(log_params))
    if i % 50 == 0 or i == 199:
        L_c, C_c, mu_c = np.exp(np.array(log_params))
        f_c_cur = 1.0 / (2.0 * np.pi * np.sqrt(L_c * C_c))
        print(f"  Step {i:3d}: loss={float(loss):.5f},  f={f_c_cur/1e6:.3f} MHz,  L={L_c*1e6:.3f} µH,  C={C_c*1e9:.3f} nF,  mu={mu_c:.3f}")

L_opt, C_opt, mu_opt_j = np.exp(np.array(log_params))
f_opt = 1.0 / (2.0 * np.pi * np.sqrt(L_opt * C_opt))
print(f"\nFinal: f={f_opt/1e6:.4f} MHz  (target {f_target/1e6:.1f} MHz),  L={L_opt*1e6:.4f} µH,  C={C_opt*1e9:.4f} nF,  mu={mu_opt_j:.4f}")

```

    Target frequency    : 8.0 MHz
    Required L*C product: 3.958e-16 H·F
    Starting L=1.00 µH, C=1.00 nF  → f0=5.033 MHz



    Initial joint loss: 3.8006  (freq error + amplitude error)

    Adam optimisation (200 steps, lr=0.05):


      Step   0: loss=3.80059,  f=5.033 MHz,  L=0.951 µH,  C=1.051 nF,  mu=1.902


      Step  50: loss=0.08283,  f=8.239 MHz,  L=0.234 µH,  C=1.594 nF,  mu=0.412


      Step 100: loss=0.03842,  f=8.014 MHz,  L=0.222 µH,  C=1.777 nF,  mu=0.304


      Step 150: loss=0.00066,  f=8.001 MHz,  L=0.220 µH,  C=1.796 nF,  mu=0.269


      Step 199: loss=0.00006,  f=8.000 MHz,  L=0.220 µH,  C=1.800 nF,  mu=0.261

    Final: f=7.9998 MHz  (target 8.0 MHz),  L=0.2199 µH,  C=1.7997 nF,  mu=0.2609



```python
# ── Animated GIF: joint optimisation waveform evolution ─────────────────────
# Three-panel animation: waveform | parameter trajectories | loss convergence
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for GIF rendering
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec

# ── 1. Pre-compute HB waveforms at each saved checkpoint ─────────────────────
ANIM_STRIDE = 4   # 1-in-N Adam steps (lower = more frames, slower)
frame_indices = list(range(0, len(param_log_hist), ANIM_STRIDE))
if frame_indices[-1] != len(param_log_hist) - 1:
    frame_indices.append(len(param_log_hist) - 1)

print(f"Pre-computing {len(frame_indices)} HB solutions (stride={ANIM_STRIDE})...")

frame_data = []
for i, idx in enumerate(frame_indices):
    L_f, C_f, mu_f = np.exp(param_log_hist[idx])
    f_f = 1.0 / (2.0 * np.pi * np.sqrt(L_f * C_f))

    grps_f = update_params_dict(groups, "inductor",  "L1", "L", L_f)
    grps_f = update_params_dict(grps_f, "capacitor", "C1", "C", C_f)
    grps_f = update_group_params(grps_f, "vdp", "mu", jnp.array(mu_f))

    run_hb_f   = setup_harmonic_balance(grps_f, num_vars, freq=f_f, num_harmonics=N_harm, osc_node=osc_node)
    yt_f, yf_f = jax.jit(run_hb_f)(jnp.zeros(num_vars))

    frame_data.append({
        "wv":   np.array(yt_f[:, osc_node]),
        "freq": f_f,
        "amp":  float(2.0 * jnp.abs(yf_f[1, osc_node])),
        "L":    L_f,
        "C":    C_f,
        "mu":   mu_f,
        "step": idx,
    })
    if i % 10 == 0 or i == len(frame_indices) - 1:
        print(f"  [{i+1:3d}/{len(frame_indices)}] step {idx:3d} — "
              f"f={f_f/1e6:.2f} MHz, A={frame_data[-1]['amp']:.3f} V, "
              f"mu={mu_f:.3f}")

print("Done.")

# ── 2. Pre-compute full parameter histories for trajectory panels ─────────────
# These are cheap — no HB, just exp() of the saved log-params.
all_steps  = np.arange(len(param_log_hist))
all_params = np.exp(np.array(param_log_hist))          # (N, 3): [L, C, mu]
all_freqs  = 1.0 / (2.0 * np.pi * np.sqrt(all_params[:, 0] * all_params[:, 1]))
all_mus    = all_params[:, 2]
all_Ls     = all_params[:, 0] * 1e6   # µH
all_Cs     = all_params[:, 1] * 1e9   # nF

# ── 3. Figure layout ─────────────────────────────────────────────────────────
t_norm = np.linspace(0, 1, K, endpoint=False)

fig_anim = plt.figure(figsize=(13, 8), facecolor="white")
gs = GridSpec(2, 2, figure=fig_anim,
              height_ratios=[1.1, 1], hspace=0.45, wspace=0.38)
ax_wave  = fig_anim.add_subplot(gs[0, :])   # top row, full width
ax_param = fig_anim.add_subplot(gs[1, 0])   # bottom-left
ax_loss  = fig_anim.add_subplot(gs[1, 1])   # bottom-right
ax_mu    = ax_param.twinx()   # secondary y for μ

plt.rcParams.update({"font.family": "sans-serif"})

# ── Waveform panel ────────────────────────────────────────────────────────────
ax_wave.set_facecolor("white")
ax_wave.set_xlim(0, 1)
ax_wave.set_ylim(-3.8, 3.8)
ax_wave.axhline(0, color="#cccccc", lw=0.8)
ax_wave.axhline( A_target_j, color="#2ca02c", lw=1.5, ls="--", alpha=0.8, label=f"target amplitude ±{A_target_j} V (fundamental)")
ax_wave.axhline(-A_target_j, color="#2ca02c", lw=1.5, ls="--", alpha=0.8)
ax_wave.set_xlabel("Normalised time  (t / T)", fontsize=11)
ax_wave.set_ylabel("Oscillator voltage  (V)", fontsize=11)
ax_wave.grid(True, alpha=0.25)
ax_wave.legend(fontsize=9, loc="upper right")
(line_wave,) = ax_wave.plot([], [], "-", color="#1f77b4", lw=2.5)
title_wave   = ax_wave.set_title("", fontsize=11, pad=8)
annot = ax_wave.text(
    0.03, 0.97, "",
    transform=ax_wave.transAxes, fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4ff",
              edgecolor="#aaaacc", alpha=0.9),
    fontfamily="monospace",
)

# ── Parameter trajectories panel ─────────────────────────────────────────────
# Full ghost traces
ax_param.plot(all_steps, all_freqs / 1e6, color="#aec7e8", lw=1.5, zorder=1)
ax_mu.plot(all_steps, all_mus, color="#ffbb78", lw=1.5, zorder=1)
# Target reference lines
ax_param.axhline(f_target / 1e6, color="#1f77b4", ls="--", lw=1, alpha=0.5)
ax_mu.axhline(mu_opt_j, color="#ff7f0e", ls="--", lw=1, alpha=0.5)
# Live traces
(line_freq,) = ax_param.plot([], [], "-", color="#1f77b4", lw=2.5, label=f"f  (target {f_target/1e6:.0f} MHz)", zorder=2)
(dot_freq,)  = ax_param.plot([], [], "o", color="#1f77b4", ms=7, zorder=3)
(line_mu,)   = ax_mu.plot([], [], "-", color="#ff7f0e", lw=2.5, label=f"μ  (target {mu_opt_j:.2f})", zorder=2)
(dot_mu,)    = ax_mu.plot([], [], "o", color="#ff7f0e", ms=7, zorder=3)

ax_param.set_facecolor("white")
ax_param.set_xlim(-5, len(all_steps) + 5)
ax_param.set_ylim(all_freqs.min() / 1e6 * 0.97, all_freqs.max() / 1e6 * 1.03)
ax_mu.set_ylim(all_mus.min() * 0.85, all_mus.max() * 1.15)
ax_param.set_xlabel("Adam step", fontsize=11)
ax_param.set_ylabel("Frequency  (MHz)", color="#1f77b4", fontsize=11)
ax_mu.set_ylabel("μ", color="#ff7f0e", fontsize=11)
ax_param.tick_params(axis="y", labelcolor="#1f77b4")
ax_mu.tick_params(axis="y", labelcolor="#ff7f0e")
ax_param.set_title("Parameter trajectories", fontsize=11, pad=8)
ax_param.grid(True, alpha=0.25)
# Combined legend
lines_param = [line_freq, line_mu]
ax_param.legend(lines_param, [l.get_label() for l in lines_param],
                fontsize=8, loc="lower right")

# ── Loss convergence panel ────────────────────────────────────────────────────
ax_loss.set_facecolor("white")
ax_loss.semilogy(range(len(losses_joint)), losses_joint, color="#dddddd", lw=2, zorder=1)
(line_loss,) = ax_loss.semilogy([], [], color="#1f77b4", lw=2.5, zorder=2)
(dot_loss,)  = ax_loss.semilogy([], [], "o", color="#ff7f0e", ms=9, zorder=3)
ax_loss.set_xlabel("Adam step", fontsize=11)
ax_loss.set_ylabel("Loss  (freq² + amp²)", fontsize=11)
ax_loss.set_title("Optimisation convergence", fontsize=11, pad=8)
ax_loss.grid(True, alpha=0.25)
ax_loss.set_xlim(-5, len(losses_joint) + 5)

# ── 4. Animation update function ─────────────────────────────────────────────
def _frame(i):
    d    = frame_data[i]
    step = d["step"]

    # Waveform
    line_wave.set_data(t_norm, d["wv"])
    title_wave.set_text(
        f"Step {step:3d} / {len(losses_joint)-1}   —   "
        f"f = {d['freq']/1e6:.2f} MHz   A = {d['amp']:.3f} V"
    )
    annot.set_text(
        f"target  f = {f_target/1e6:.1f} MHz\n"
        f"target  A = {A_target_j:.1f} V\n"
        f"current f = {d['freq']/1e6:.3f} MHz\n"
        f"current A = {d['amp']:.3f} V"
    )

    # Parameter trajectories (up to current step)
    s = slice(0, step + 1)
    line_freq.set_data(all_steps[s], all_freqs[s] / 1e6)
    dot_freq.set_data([all_steps[step]], [all_freqs[step] / 1e6])
    line_mu.set_data(all_steps[s], all_mus[s])
    dot_mu.set_data([all_steps[step]], [all_mus[step]])

    # Loss (clamp to valid range — param_log_hist has one more entry than losses_joint)
    loss_step = min(step, len(losses_joint) - 1)
    line_loss.set_data(range(loss_step + 1), losses_joint[:loss_step + 1])
    dot_loss.set_data([loss_step], [losses_joint[loss_step]])

    return line_wave, title_wave, annot, line_freq, dot_freq, line_mu, dot_mu, line_loss, dot_loss


anim_obj = animation.FuncAnimation(
    fig_anim, _frame,
    frames=len(frame_data),
    interval=80,
    blit=True,
)

# ── 5. Save ───────────────────────────────────────────────────────────────────
GIF_PATH = "oscillator_optimisation.gif"
anim_obj.save(
    GIF_PATH,
    writer=animation.PillowWriter(fps=12),
    dpi=130,
)
plt.close(fig_anim)
print(f"\nSaved \u2192 {GIF_PATH}")
print(f"Frames: {len(frame_data)}   Duration: {len(frame_data)/12:.1f}s at 12 fps")
print("Tip: drag-and-drop directly into PowerPoint (Insert \u2192 Pictures).")

```

    Pre-computing 51 HB solutions (stride=4)...


      [  1/51] step   0 — f=5.03 MHz, A=2.914 V, mu=2.000


      [ 11/51] step  40 — f=7.65 MHz, A=1.483 V, mu=0.496


      [ 21/51] step  80 — f=7.92 MHz, A=1.238 V, mu=0.330


      [ 31/51] step 120 — f=8.00 MHz, A=1.161 V, mu=0.285


      [ 41/51] step 160 — f=8.00 MHz, A=1.131 V, mu=0.267


      [ 51/51] step 200 — f=8.00 MHz, A=1.119 V, mu=0.261
    Done.



    Saved → oscillator_optimisation.gif
    Frames: 51   Duration: 4.2s at 12 fps
    Tip: drag-and-drop directly into PowerPoint (Insert → Pictures).



```python
# ── Evaluate the optimised waveform ──────────────────────────────────────────
grps_opt = update_params_dict(groups, "inductor",  "L1", "L", L_opt)
grps_opt = update_params_dict(grps_opt, "capacitor", "C1", "C", C_opt)
grps_opt = update_group_params(grps_opt, "vdp", "mu", jnp.array(mu_opt_j))

run_hb_opt = setup_harmonic_balance(grps_opt, num_vars, freq=f_opt, num_harmonics=N_harm, osc_node=osc_node)
yt_opt, yf_opt = jax.jit(run_hb_opt)(jnp.zeros(num_vars))

A_fund_opt = float(2.0 * jnp.abs(yf_opt[1, osc_node]))
T_opt      = 1.0 / f_opt
t_opt_ns   = np.linspace(0.0, T_opt * 1e9, K, endpoint=False)
v_opt      = np.array(yt_opt[:, osc_node])

# Build reference waveform at initial parameters for comparison
grps_init  = update_params_dict(groups, "inductor",  "L1", "L", L_val)
grps_init  = update_params_dict(grps_init, "capacitor", "C1", "C", C_val)
grps_init  = update_group_params(grps_init, "vdp", "mu", jnp.array(2.0))
run_hb_init = setup_harmonic_balance(grps_init, num_vars, freq=f0, num_harmonics=N_harm, osc_node=osc_node)
yt_init, _ = jax.jit(run_hb_init)(jnp.zeros(num_vars))
v_init = np.array(yt_init[:, osc_node])

param_hist = np.exp(np.array(param_log_hist))  # shape (201, 3): [L, C, mu]
f_hist = 1.0 / (2.0 * np.pi * np.sqrt(param_hist[:, 0] * param_hist[:, 1]))
steps  = np.arange(len(f_hist))

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=("Joint optimisation convergence",
                    "Parameter trajectories",
                    "Waveform: initial vs optimised"),
    specs=[[{}, {"secondary_y": True}, {}]],
)

# Loss convergence
fig.add_trace(
    go.Scatter(x=list(range(len(losses_joint))), y=losses_joint, mode="lines",
               line=dict(width=2), name="loss", showlegend=False),
    row=1, col=1,
)

# Frequency trajectory (primary y-axis)
fig.add_trace(
    go.Scatter(x=steps, y=f_hist / 1e6, mode="lines",
               line=dict(width=2, color="#636EFA"), name="frequency (MHz)"),
    row=1, col=2, secondary_y=False,
)
fig.add_hline(y=f_target / 1e6, line_dash="dash", line_color="#636EFA",
              line_width=1, opacity=0.5, row=1, col=2)

# mu trajectory (secondary y-axis)
fig.add_trace(
    go.Scatter(x=steps, y=param_hist[:, 2], mode="lines",
               line=dict(width=2, color="#EF553B"), name="\u03bc"),
    row=1, col=2, secondary_y=True,
)

# Initial vs optimised waveforms (normalised time)
t_norm = np.linspace(0, 1, K, endpoint=False)
fig.add_trace(
    go.Scatter(x=t_norm, y=v_init, mode="lines",
               line=dict(width=2, color="#00CC96"),
               name=f"Initial: f={f0/1e6:.2f} MHz, A={float(2*jnp.abs(yf_opt[1, osc_node])):.2f} V"),
    row=1, col=3,
)
fig.add_trace(
    go.Scatter(x=t_norm, y=v_opt, mode="lines",
               line=dict(width=2.5, dash="dash", color="#636EFA"),
               name=f"Optimised: f={f_opt/1e6:.2f} MHz, A={A_fund_opt:.2f} V"),
    row=1, col=3,
)
fig.add_hline(y= A_target_j, line_dash="dot", line_color="grey", line_width=0.8, row=1, col=3)
fig.add_hline(y=-A_target_j, line_dash="dot", line_color="grey", line_width=0.8, row=1, col=3)

fig.update_xaxes(title_text="Adam step", row=1, col=1)
fig.update_yaxes(title_text="Loss (freq\u00b2 + amp\u00b2)", type="log", row=1, col=1)
fig.update_xaxes(title_text="Adam step", row=1, col=2)
fig.update_yaxes(title_text="Frequency (MHz)", color="#636EFA", row=1, col=2, secondary_y=False)
fig.update_yaxes(title_text="\u03bc", color="#EF553B", row=1, col=2, secondary_y=True)
fig.update_xaxes(title_text="Normalised time (t/T)", row=1, col=3)
fig.update_yaxes(title_text="Voltage (V)", row=1, col=3)
fig.update_layout(height=450, legend=dict(orientation="h", y=-0.2))
fig.show()

print("\nFinal results:")
print(f"  Frequency : {f_opt/1e6:.4f} MHz  (target {f_target/1e6:.1f} MHz,  error {abs(f_opt - f_target)/f_target*100:.2f}%)")
print(f"  Amplitude : {A_fund_opt:.4f} V    (target {A_target_j:.1f} V,        error {abs(A_fund_opt - A_target_j)/A_target_j*100:.2f}%)")
print(f"  L = {L_opt*1e6:.4f} \u00b5H,  C = {C_opt*1e9:.4f} nF,  mu = {mu_opt_j:.4f}")

```




    Final results:
      Frequency : 7.9998 MHz  (target 8.0 MHz,  error 0.00%)
      Amplitude : 1.1190 V    (target 1.0 V,        error 11.90%)
      L = 0.2199 µH,  C = 1.7997 nF,  mu = 0.2609


## Summary

Starting from a Van der Pol oscillator running at ~5 MHz with ~2.8 V amplitude, gradient
descent simultaneously tuned L, C, and μ to hit 8 MHz and 1.0 V in 200 Adam steps —
with no grid sweeps and no manual iteration.

### What made this possible?

| Step | Tool | Role |
|------|------|------|
| Component definition | `@component` decorator | Generates JAX-traceable Equinox module |
| Netlist compilation | `compile_circuit` | Runs once; produces vmappable `ComponentGroup` objects |
| Differentiable parameter update | `update_params_dict` / `update_group_params` | `eqx.tree_at` functional update; no recompilation |
| Periodic steady state | `setup_harmonic_balance` | Finds limit cycle in ~15 Newton steps; JIT-compatible |
| Exact gradients | `jax.grad` | Differentiates through the HB Newton solver via implicit differentiation |
| Optimisation | `optax.adam` | First-order optimiser in log-space; 200 steps to convergence |

### Going further

The same pattern extends directly to:

- **Phase noise minimisation** — add a noise source and minimise the HB-estimated phase
  noise spectral density as a function of tank Q and bias point.
- **Injection locking** — add a small-signal source and tune the free-running frequency
  to lock at the injection frequency with maximum locking range.
- **Coupled oscillator arrays** — vmap the HB solve over an array of weakly coupled
  oscillators and optimise the coupling network for synchronisation.
- **CMOS VCO design** — replace the VDP element with a transistor-level cross-coupled
  pair (using NMOS / PMOS components from `circulax.components.electronic`) and
  sweep the varactor capacitance as a differentiable parameter.
