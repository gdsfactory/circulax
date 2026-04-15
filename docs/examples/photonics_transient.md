# Photonics Transient

In this advanced demonstration, we simulate the non-linear transient response of a photonic circuit. We revisit the standard waveguide model but introduce Two-Photon Absorption (TPA).

In linear optics, loss is constant ($\alpha$). In the non-linear regime, loss becomes intensity-dependent ($\alpha + \beta I$). This creates an Optical Limiting effect: as the optical power increases, the material absorbs more efficiently, effectively "clamping" the output power. This example demonstrates two critical capabilities of the solver:

Dynamic S-Matrices: Unlike the previous linear examples where the system matrix was constant, here the scattering parameters $S(t)$ are a function of the instantaneous state $|E(t)|^2$. The solver re-evaluates the component physics at every femtosecond time-step.

Complex Envelopes: Optical signals oscillate at hundreds of terahertz ($193\text{THz}$). To make simulation feasible, we simulate the complex slowly-varying envelope $A(t)$ rather than the raw field, using a Real-Imaginary Flattening strategy to map the complex state to the solver's real-valued requirements.

What to Expect: A high-power Gaussian pulse will be launched into the waveguide. In the linear regeime, the output is just a scaled, delayed version of the input, however with tpa_coeff > 0, pulse reshaping is observed. The peak of the Gaussian (high intensity) will be flattened or "squashed" due to the non-linear loss, while the tails (low intensity) pass through with standard linear attenuation.


```python
import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from circulax import compile_circuit
from circulax.components.base_component import PhysicsReturn, Signals, States, component
from circulax.components.electronic import Resistor
from circulax.components.photonic import OpticalSourcePulse
from circulax.solvers import setup_transient
```

    KLUJAX_RS DEBUG MODE.
    WARNING:2026-04-15 17:32:29,353:jax._src.xla_bridge:864: An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.



```python
print("--- DEMO: Photonic Transient (Flat Vector Fix) ---")

jax.config.update("jax_enable_x64", True)

models_map = {
    "waveguide": OpticalWaveguide,
    "source": OpticalSourcePulse,
    "resistor": Resistor,
    "ground": lambda: 0,
}

net_dict = {
    "instances": {
        "GND": {"component": "ground"},
        "I1": {"component": "source", "settings": {"power": 100.0, "delay": 0.1e-9}},
        "WG1": {
            "component": "waveguide",
            "settings": {"length_um": 1000.0, "loss_dB_cm": 20.0, "tpa_coeff": 5e-1},
        },
        "R1": {"component": "resistor", "settings": {"R": 1.0}},
    },
    "connections": {"GND,p1": ("I1,p2", "R1,p2"), "I1,p1": "WG1,p1", "WG1,p2": "R1,p1"},
}

print("1. Compiling...")
circuit = compile_circuit(net_dict, models_map, is_complex=True)

print("2. Solving DC Operating Point...")
y_op_flat = circuit()

print(f"   DC Converged. Norm: {jnp.linalg.norm(y_op_flat):.2e}")

transient_sim = setup_transient(groups=circuit.groups, linear_strategy=circuit.solver)

t_max = 1.0e-9
saveat = diffrax.SaveAt(ts=jnp.linspace(0, t_max, 500))

print("3. Running Transient Simulation...")
sol = transient_sim(
    t0=0.0,
    t1=t_max,
    dt0=1e-13,
    y0=y_op_flat,
    args=(circuit.groups, circuit.sys_size),
    saveat=saveat,
    max_steps=100000,
    throw=False,
    stepsize_controller=diffrax.PIDController(rtol=1e-4, atol=1e-6),
)

if sol.result == diffrax.RESULTS.successful:
    print("   ✅ Simulation Successful")

    ts = sol.ts * 1e9

    v_in = circuit.get_port_field(sol.ys, "I1,p1")
    v_out = circuit.get_port_field(sol.ys, "R1,p1")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    axes = axes.ravel()

    axes[0].plot(ts, 20 * jnp.log10(jnp.abs(v_in)), "g--", label="Input Pulse")
    axes[0].plot(ts, 20 * jnp.log10(jnp.abs(v_out)), "r-", label="Output (After WG)")

    axes[0].set_title("Photonic Transient Response")
    axes[0].set_xlabel("Time (ns)")
    axes[0].set_ylabel("Power (dBm)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Ouput Power vs Input Power")
    axes[1].set_xlabel("Input Power (dB)")
    axes[1].set_ylabel("Ouput Power(dB)")
    axes[1].plot(
        20 * jnp.log10(jnp.abs(v_in)),
        (20 * jnp.log10(jnp.abs(v_out))),
        "r-",
        label="Loss",
    )
    axes[1].grid(True, alpha=0.3)
else:
    print(f"❌ Simulation Failed: {sol.result}")
```

    --- DEMO: Photonic Transient (Flat Vector Fix) ---
    1. Compiling...


    2. Solving DC Operating Point...


       DC Converged. Norm: 1.92e+00
    3. Running Transient Simulation...


       ✅ Simulation Successful




![png](photonics_transient_files/photonics_transient_4_4.png)
