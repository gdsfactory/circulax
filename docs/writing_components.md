## Writing Components

Components are pure Python functions with a decorator. They are automatically compatible with `jax.jit`, `jax.vmap`, and `jax.grad`.

---

## Quick Reference

| Decorator | Use case | DC | Transient | HB |
|---|---|:--:|:--:|:--:|
| `@component` | Electrical & photonic — time-invariant physics | ✓ | ✓ | ✓ |
| `@source` | Time-varying sources (AC, pulse, modulated optical) | ✓ | ✓ | ✓ |
| `@fdomain_component` | Electrically frequency-dependent impedance (skin effect, wideband interconnect) | ✓ | ✗ | ✓ |
| `sax_component` | Photonic models already written for the SAX library | ✓ | ✓ | ✓ |
| `osdi_component` | Verilog-A compact models compiled with OpenVAF | ✓ | ✓ | ✗ |

---

### Function Signature

Every component computes the instantaneous balance equations for its ports and states:

```python
def MyComponent(signals, [t], **params):
    # 1. Calculate physics
    # 2. Return (Flows, Storage)
```

!!! note "CamelCase function names"
    Component functions use **CamelCase** (`Resistor`, `VoltageSourceAC`) because the decorator promotes them into `equinox.Module` **classes** — Python convention for classes. In the netlist `"component"` key you reference them by whatever string key you pass to `compile_netlist`'s `models` dict, which is typically the lowercase name (e.g. `"resistor"`). The two names are independent.


### Arguments

 1) ```signals```: A unified view of every port and internal state declared by
 the decorator. Current values use attribute access (for example,
 `signals.p1` or `signals.i_L`), and delayed values use
 `signals.at_delay(tau)`.

 2) ```t``` (Time): Optional. Only present if you use the `@source` decorator.

 3) ```**params```: Keyword arguments defining the physical properties (Resistance, Length, Refractive Index).

### Return Values

The function must return a tuple of two dictionaries: ```(f_dict, q_dict)```.

* ```f_dict``` (The Flow/Balance Vector):

    * For Ports: Represents the "Flow" (Current) entering the node.

    * For internal state variables: Represents the algebraic constraint (should sum to 0).

* ```q_dict``` (The Storage Vector):

    * Represents the time-dependent quantity (Charge, Flux) stored in a variable.

    * The solver computes $\frac{d}{dt}(q\_dict)$.

---

## `@component` — Time-Invariant Physics

For components that do not depend explicitly on time (resistors, transistors, diodes, etc.).

#### Example: A Simple Resistor

```python
import jax.numpy as jnp
from circulax.components.base_component import component, Signals

@component(ports=("p1", "p2"))
def Resistor(signals: Signals, R: float = 1e3):
    """Ohm's Law: I = V / R"""
    i = (signals.p1 - signals.p2) / (R + 1e-12)
    return {"p1": i, "p2": -i}, {}
```

#### Example: A Capacitor (Storage Term)

Reactive components use `q_dict` for the quantity differentiated with respect to time.

```python
@component(ports=("p1", "p2"))
def Capacitor(signals: Signals, C: float = 1e-12):
    """I = C * dV/dt  →  I = dQ/dt"""
    v_drop = signals.p1 - signals.p2
    q_val = C * v_drop
    return {}, {"p1": q_val, "p2": -q_val}
```

#### Example: An Inductor (Internal State Variable)

When a component requires a state variable not directly tied to a port voltage (e.g. the current through an inductor), declare it in `states=`. The solver adds it to the global state vector.

```python
@component(ports=("p1", "p2"), states=("i_L",))
def Inductor(signals: Signals, L: float = 1e-9):
    """V = L * di/dt  →  flux φ = L * i_L"""
    v_drop = signals.p1 - signals.p2
    # f_dict: KCL at ports, algebraic constraint on i_L
    # q_dict: flux = L * i_L  →  solver computes V = dφ/dt
    return (
        {"p1": signals.i_L, "p2": -signals.i_L, "i_L": v_drop},
        {"i_L": -L * signals.i_L},
    )
```

---

## `@source` — Time-Dependent Sources

For components that vary with time (AC sources, pulse generators, modulated optical sources). Injects `t` as the second argument.

Voltage sources need an internal state `i_src` — the voltage is prescribed, so the current is the unknown.

```python
from circulax.components.base_component import source

@source(ports=("p1", "p2"), states=("i_src",))
def VoltageSourceAC(
    signals: Signals,
    t: float,
    V: float = 1.0,
    freq: float = 1e6,
    phase: float = 0.0,
):
    """Sinusoidal voltage source: V_s(t) = V · sin(2πf·t + φ)"""
    v_target = V * jnp.sin(2 * jnp.pi * freq * t + phase)
    constraint = (signals.p1 - signals.p2) - v_target
    return {"p1": signals.i_src, "p2": -signals.i_src, "i_src": constraint}, {}
```

---

## `@Component.setup` — Deferred Initialisation

When a component has expensive derived quantities that depend only on a subset of its parameters (e.g. round-trip coefficients in a ring resonator), use `@Component.setup` to compute them separately. The setup function runs inside the JAX trace, so gradients flow through it.

Declare `init` immediately after `signals`, then register a setup function:

```python
@component(ports=("in_", "thru", "drop"))
def RingMod(signals, init, kappa=0.3, neff=2.4, L=62.8, V_pi=2.0, V=0.0):
    phi = init["phi"] * (1.0 + V / V_pi)
    # ... use init["a"], init["t"], phi in CMT equations

@RingMod.setup
def _(kappa=0.3, neff=2.4, L=62.8):
    a = jnp.exp(-1e-3 * L / 2.0)
    t = jnp.sqrt(1.0 - kappa**2)
    return {"a": a, "t": t, "phi": 2.0 * jnp.pi * neff * L}
```

The setup function only needs to declare the parameters it uses — extra physics params are silently dropped. See the [Ring Modulator example](examples/ring_modulator.md) for a full worked example with gradient verification.

---

## Fixed propagation delay

Use `signals.at_delay(tau)` to read every component-local port and state at
`t - tau`. The delay is a non-negative duration in seconds and may depend on
differentiable component parameters.

```python
@component(ports=("p1", "p2"), states=("a1", "a2"))
def TransmissionLine(signals: Signals, tau: float = 1e-9, z0: float = 50.0):
    past = signals.at_delay(tau)
    b1 = past.a2
    b2 = past.a1
    return {
        "p1": (signals.a1 - b1) / z0,
        "p2": (signals.a2 - b2) / z0,
        "a1": signals.p1 - signals.a1 - b1,
        "a2": signals.p2 - signals.a2 - b2,
    }, {}
```

The same declaration is interpreted as an identity at DC, accepted-history
interpolation in transient analysis, and an exact spectral phase shift in AC
and harmonic balance. A component may currently use one unique delay; reuse
the returned snapshot when reading several values.

---

## Holomorphic Components and AC Analysis

When running S-parameter (AC) analysis on complex-valued circuits, circulax needs to know whether a component's physics function is **holomorphic** (complex-differentiable). This determines whether the fast N×N Wirtinger system or the full 2N×2N real-block system is used.

### What is holomorphic?

A function is holomorphic if it only uses operations that are complex-differentiable: addition, multiplication, division, `jnp.exp`, `jnp.sin`, `jnp.sqrt`, matrix operations, etc.

Operations like `jnp.real()`, `jnp.imag()`, `jnp.conj()`, and `jnp.abs()` (on complex inputs) are **not** holomorphic — they couple the field and its conjugate, so the Jacobian cannot be collapsed into a single N×N complex matrix.

### Default: `holomorphic=False` (safe)

By default, components are marked `holomorphic=False`. This means any circuit containing them will use the full 2N×2N real-block system for AC analysis, which is **always correct** but slower (2N×2N vs N×N).

```python
@component(ports=("p1", "p2", "v_e"), states=("a", "i_out"))
def RingModulator(signals, ...):
    voltage = jnp.real(signals.v_e)  # non-holomorphic operation
    ...
```

### Opting in: `holomorphic=True` (fast)

When you know your component only uses holomorphic operations, set `holomorphic=True` to enable the faster N×N path. If *every* component in a circuit is holomorphic, `circuit.sp()` will automatically use the fast path.

```python
@component(ports=("p1", "p2"), holomorphic=True)
def Resistor(signals, R=1e3):
    i = (signals.p1 - signals.p2) / R
    return {"p1": i, "p2": -i}, {}
```

All built-in passive components (resistors, capacitors, inductors, waveguides, couplers) and linear sources are marked `holomorphic=True`. Non-linear components that use clipping or comparisons on signal values (diodes, MOSFETs) default to `holomorphic=False`.

### Compile-time validation

When compiling a complex-valued circuit, circulax traces each `holomorphic=True` component with complex inputs and inspects the JAX computation graph (jaxpr) for non-holomorphic primitives. If a mismatch is detected, a warning is emitted:

```
UserWarning: Component group 'ring_eo' is marked holomorphic=True but uses
non-holomorphic operations: real. Set holomorphic=False on its
@component/@source decorator.
```

### Auto-detection at circuit level

`circuit.sp()` defaults to `holomorphic="auto"`, which inspects all component groups:

- If every group has `holomorphic=True` → uses the fast N×N Wirtinger system
- If any group has `holomorphic=False` → uses the full 2N×2N real-block system

You can override with `circuit.sp(..., holomorphic=True)` or `circuit.sp(..., holomorphic=False)`.

---

## `@fdomain_component` — Frequency-Domain Components

For components whose admittance depends on the electrical signal frequency and cannot be expressed as an instantaneous time-domain relation.

Examples: skin-effect resistors ($Z(f) = R_0 + a\sqrt{f}$), wideband interconnect models from EM simulation, or alternative reactive formulations ($Y_C(f) = j2\pi f C$).

### Signature contract

The decorated function must:

1. Accept `f` (frequency in Hz) as its **first** positional argument.
2. Accept any number of keyword parameters, **all with defaults**.
3. Return a square Y-matrix of shape `(n_ports, n_ports)` with `dtype=complex128`.

```python
from circulax import fdomain_component
import jax.numpy as jnp

@fdomain_component(ports=("p1", "p2"))
def SkinEffectResistor(f: float, R0: float = 1.0, a: float = 1e-4):
    """Z(f) = R0 + a·√|f| — resistance rises with frequency."""
    Z = R0 + a * jnp.sqrt(jnp.abs(f) + 1e-30)  # +ε avoids √0 at DC
    Y = 1.0 / Z
    return jnp.array([[Y, -Y], [-Y, Y]], dtype=jnp.complex128)
```

### Solver behaviour

| Solver | Behaviour |
|--------|-----------|
| **DC** | Evaluated at `f = 0`. Skin-effect reduces to `R₀`; a capacitor (`Y = j2πfC`) becomes an open circuit. Make sure `Y(0)` is finite — add a small series resistance for components that would otherwise diverge (e.g. pure inductors). |
| **Harmonic Balance** | Evaluated at each harmonic `k·f₀`. The contribution `Y(k·f₀) @ V_k` is added directly to the frequency-domain residual `R_k` before the inverse FFT. |
| **Transient** | **Not supported.** A frequency-dependent admittance requires convolving `h(t) = IFFT{Y(f)}` with the voltage waveform, which is incompatible with the per-time-step Newton loop. Calling `circuit.transient()` or low-level `setup_transient()` with an f-domain component raises `RuntimeError`. |

### Equivalence with time-domain reactive components

For linear components, both formulations produce **identical Harmonic Balance results**:

| Time-domain (`@component`) | F-domain (`@fdomain_component`) | HB contribution at harmonic k |
|---|---|---|
| `q_C = C·V` → solver adds `jkω₀·C·V_k` | `Y_C(f) = j2πfC` → adds `j2πkf₀·C·V_k` | identical |
| flux `φ = L·i_L` → solver adds `jkω₀·L·I_k` | `Y_L(f) = 1/(j2πfL)` → adds `V_k/(j2πkf₀L)` | identical |

The `harmonic_balance` example notebook (Part 3) demonstrates this numerically: replacing the time-domain `Capacitor` and `Inductor` with `@fdomain_component` equivalents gives waveform errors below solver tolerance (~10⁻¹⁰ V).

The f-domain path has one practical advantage: the inductor no longer needs an internal state variable `i_L`, reducing the system size by one variable.

---

## Photonic Components

Optical field amplitudes are treated as complex-valued "voltages" and S-parameter-derived admittances as "conductances". Since photonic S-parameters describe the steady state at a given wavelength — the Y-matrix is constant w.r.t. the electrical solver frequency — photonic components use `@component`, not `@fdomain_component`.

### A: Manual `@component` with `s_to_y()`

Build the S-matrix, convert to Y via `s_to_y`, return `I = Y @ V`.

```python
from circulax.s_transforms import s_to_y

@component(ports=("p1", "p2"))
def OpticalWaveguide(
    signals: Signals,
    length_um: float = 100.0,
    neff: float = 2.4,
    wavelength_nm: float = 1310.0,
    loss_dB_cm: float = 1.0,
):
    """Single-mode waveguide with propagation loss and phase shift."""
    phi = 2.0 * jnp.pi * neff * (length_um / wavelength_nm) * 1000.0
    loss = loss_dB_cm * (length_um / 10000.0)
    T = 10.0 ** (-loss / 20.0) * jnp.exp(-1j * phi)

    S = jnp.array([[0.0, T], [T, 0.0]], dtype=jnp.complex128)
    Y = s_to_y(S)

    v_vec = jnp.array([signals.p1, signals.p2], dtype=jnp.complex128)
    i_vec = Y @ v_vec
    return {"p1": i_vec[0], "p2": i_vec[1]}, {}
```

Use this approach for new photonic components.

### B: `sax_component` — importing SAX models

Wraps existing [SAX](https://flaport.github.io/sax/) models (e.g. from gdsfactory PDK libraries) without rewriting physics. Auto-detects port names and handles S→Y conversion.

```python
from circulax.s_transforms import sax_component

# A pure SAX model function (no circulax dependency)
def sax_coupler(coupling: float = 0.5):
    kappa = coupling ** 0.5
    tau = (1 - coupling) ** 0.5
    return {
        ("in0", "out0"): tau,
        ("in0", "out1"): 1j * kappa,
        ("in1", "out0"): 1j * kappa,
        ("in1", "out1"): tau,
    }

# Ports ('in0', 'in1', 'out0', 'out1') are detected automatically
Coupler = sax_component(sax_coupler)
```

Use `sax_component` only for reusing existing SAX models. For new components, prefer the explicit `@component` pattern.

---

## `osdi_component` — Verilog-A Compact Models

Load industry-standard Verilog-A models (PSP, BSIM, JUNCAP, etc.) compiled to `.osdi` binaries by [OpenVAF](https://openvaf.semimod.de/). Requires the optional `bosdi` package:

```sh
pip install circulax[verilog-a]
```

```python
from circulax import osdi_component

PSP103 = osdi_component(
    osdi_path="psp103.osdi",
    ports=("D", "G", "S", "B"),
    default_params={"LEVEL": 103, "TNOM": 27.0, ...},
)
```

OSDI models bypass the standard `@component` decorator — they are evaluated via FFI and use finite-difference sensitivities rather than AD. See the [OSDI Ring Oscillator](examples/ring_oscillator_osdi.md) and [PSP103 Parameter Fitting](examples/05_psp103_ring_param_fitting.md) examples.

---

## Under the Hood

The `@component` decorator dynamically generates an `equinox.Module` subclass.

When you write:

```python
@component(ports=("a", "b"))
def MyResistor(signals, R=100.0):
```

The decorator:

1. **Introspects** the function signature to find parameters (`R`) and defaults (`100.0`).
2. **Generates** an `eqx.Module` class named `MyResistor`.
3. **Registers** the parameters as module fields, making them JAX-differentiable.
4. **Creates** a `_fast_physics` method that unrolls dict lookups into raw array ops for use inside `jax.jit` / `jax.vmap`.
