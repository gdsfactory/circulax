## Writing Components

Circulax uses a functional, JAX-first approach to component definition. Instead of inheriting from complex base classes, you define components as pure Python functions decorated with specific handlers.

This architecture ensures your components are automatically compatible with JIT compilation (jax.jit), vectorization (jax.vmap), and back-propagation (jax.grad).

---

## Quick Reference

Four decorators are available. Pick based on what your component's physics looks like:

| Decorator | Use case | DC | Transient | HB |
|---|---|:--:|:--:|:--:|
| `@component` | Electrical & photonic — time-invariant physics | ✓ | ✓ | ✓ |
| `@source` | Time-varying sources (AC, pulse, modulated optical) | ✓ | ✓ | ✓ |
| `@fdomain_component` | Electrically frequency-dependent impedance (skin effect, wideband interconnect) | ✓ | ✗ | ✓ |
| `sax_component` | Photonic models already written for the SAX library | ✓ | ✓ | ✓ |

---

### The Core Concept

Every component in circulax is a function that calculates the instantaneous balance equations for a specific node or state. The function signature generally looks like this:

```python
def MyComponent(signals, s, [t], **params):
    # 1. Calculate physics
    # 2. Return (Flows, Storage)
```


### Arguments

 1) ```signals``` (Ports): A NamedTuple containing the potential (Voltage) at every port defined in the decorator. Accessed via dot notation (e.g., signals.p, signals.gate).

 2) ```s``` (States): A NamedTuple containing internal state variables (e.g., current through an inductor, internal node voltages).

 3) ```t``` (Time): Optional. Only present if you use the @source decorator.

 4) ```**params```: Keyword arguments defining the physical properties (Resistance, Length, Refractive Index).

### Return Values

The function must return a tuple of two dictionaries: ```(f_dict, q_dict)```.

* ```f_dict``` (The Flow/Balance Vector):

    * For Ports: Represents the "Flow" (Current) entering the node.

    * For States: Represents the algebraic constraint (should sum to 0).

* ```q_dict``` (The Storage Vector):

    * Represents the time-dependent quantity (Charge, Flux) stored in a variable.

    * The solver computes $\frac{d}{dt}(q\_dict)$.

---

## `@component` — Time-Invariant Physics

Most passive components (Resistors, Transistors, Diodes) do not depend explicitly on time t. For these, use the ```@component``` decorator.

#### Example: A Simple Resistor

```python
import jax.numpy as jnp
from circulax.components.base_component import component, Signals, States

@component(ports=("p1", "p2"))
def Resistor(signals: Signals, s: States, R: float = 1e3):
    """Ohm's Law: I = V / R"""
    i = (signals.p1 - signals.p2) / (R + 1e-12)
    return {"p1": i, "p2": -i}, {}
```

#### Example: A Capacitor (Storage Term)

For reactive components, use `q_dict` to define what is being differentiated with respect to time.

```python
@component(ports=("p1", "p2"))
def Capacitor(signals: Signals, s: States, C: float = 1e-12):
    """I = C * dV/dt  →  I = dQ/dt"""
    v_drop = signals.p1 - signals.p2
    q_val = C * v_drop
    return {}, {"p1": q_val, "p2": -q_val}
```

#### Example: An Inductor (Internal State Variable)

When a component requires a state variable not directly tied to a port voltage (e.g. the current through an inductor), declare it in `states=`. The solver adds it to the global state vector.

```python
@component(ports=("p1", "p2"), states=("i_L",))
def Inductor(signals: Signals, s: States, L: float = 1e-9):
    """V = L * di/dt  →  flux φ = L * i_L"""
    v_drop = signals.p1 - signals.p2
    # f_dict: KCL at ports, algebraic constraint on i_L
    # q_dict: flux = L * i_L  →  solver computes V = dφ/dt
    return (
        {"p1": s.i_L, "p2": -s.i_L, "i_L": v_drop},
        {"i_L": -L * s.i_L},
    )
```

---

## `@source` — Time-Dependent Sources

If your component varies with time (e.g. AC source, pulse generator, modulated optical source), use the `@source` decorator. This injects `t` as the third argument.

Voltage sources require an internal state variable `i_src` to represent the current flowing through the source — the voltage is fixed, so the current is the unknown the solver must find.

```python
from circulax.components.base_component import source

@source(ports=("p1", "p2"), states=("i_src",))
def VoltageSourceAC(
    signals: Signals,
    s: States,
    t: float,
    V: float = 1.0,
    freq: float = 1e6,
    phase: float = 0.0,
):
    """Sinusoidal voltage source: V_s(t) = V · sin(2πf·t + φ)"""
    v_target = V * jnp.sin(2 * jnp.pi * freq * t + phase)
    constraint = (signals.p1 - signals.p2) - v_target
    return {"p1": s.i_src, "p2": -s.i_src, "i_src": constraint}, {}
```

---

## `@fdomain_component` — Frequency-Domain Components

Use this decorator when a component's admittance depends on the **electrical signal frequency** — it cannot be expressed as an instantaneous function of port voltages.

Typical examples:

* **Skin-effect resistor**: $Z(f) = R_0 + a\sqrt{f}$ — resistance rises with frequency as current crowds to conductor surface.
* **Wideband interconnect model**: frequency-dependent loss derived from measurements or EM simulation.
* **Alternative reactive formulation**: Capacitors and inductors can be defined via $Y_C(f) = j2\pi f C$ and $Y_L(f) = 1/(j2\pi f L)$ — mathematically equivalent to the `@component` path for Harmonic Balance.

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
| **Transient** | **Not supported.** A frequency-dependent admittance requires convolving `h(t) = IFFT{Y(f)}` with the voltage waveform, which is incompatible with the per-time-step Newton loop. Calling `setup_transient()` with an f-domain component raises `RuntimeError`. |

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

Photonic circuits are simulated by treating optical field amplitudes as complex-valued "voltages" and scattering-matrix admittances as "conductances". The key insight is that photonic S-parameters describe the optical steady state at a given wavelength — **the Y-matrix is constant with respect to the electrical solver frequency** (the optical wavelength is a parameter, not a solved variable). This is why photonic components use `@component`, not `@fdomain_component`.

### A: Manual `@component` with `s_to_y()`

Build the S-matrix from your component's physics, convert to Y via `s_to_y`, then return `I = Y @ V`. Cast all signals to `complex128`.

```python
from circulax.s_transforms import s_to_y

@component(ports=("p1", "p2"))
def OpticalWaveguide(
    signals: Signals,
    s: States,
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

Use this approach when writing new photonic components — you have full control over the S-matrix construction.

### B: `sax_component` — importing SAX library models

If you have photonic models already written for the [SAX](https://flaport.github.io/sax/) ecosystem (e.g. from gdsfactory PDK libraries), `sax_component` wraps them without rewriting the physics. It auto-detects port names from the SAX dict and handles the S→Y conversion internally.

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

**When to use `sax_component`**: only when reusing models that already exist in SAX format. For new components, prefer the explicit `@component` pattern above — it is easier to understand and debug.

---

## Advanced: Under the Hood

For advanced users familiar with JAX and Equinox, it is helpful to understand what the `@component` decorator actually does.

It does not simply wrap your function. Instead, it dynamically generates a new class that inherits from `equinox.Module`.

**The Transformation Process**

When you write:

```python
@component(ports=("a", "b"))
def MyResistor(signals, s, R=100.0):
```

The decorator performs the following steps:

**Introspection**: It analyses the function signature to identify parameters (`R`) and their default values (`100.0`).

**Class Generation**: It constructs a new `eqx.Module` class named `MyResistor`.

**Field Registration**: The parameters (`R`) become fields of this class. This allows JAX to differentiate with respect to `R` automatically.

**Static Optimization**: It creates a static `_fast_physics` method that unrolls dictionary lookups into raw array operations. This is what the solver calls inside `jax.jit` or `jax.vmap`.
