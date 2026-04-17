# LRC Circuit

## Introduction

In this example, we simulate a classical Series RLC Circuit operating in the Gigahertz (RF) regime. The circuit consists of a voltage source, a resistor ($10\Omega$), an inductor ($5\text{nH}$), and a capacitor ($10\text{pF}$) connected in series.

The behavior of the circuit is determined by the relationship between the resistance ($R$) and the critical damping resistance, defined as $ R_{c} = 2\sqrt{L/C}$

For this specific configuration:

* **Critical Resistance ($R_c$)**: $2\sqrt{L/C} \approx 44.7\Omega$.
* **Actual Resistance ($R$)**: $10\Omega$.

Since $R < R_c$, the system is underdamped. When the voltage source is activated, energy oscillates—or "sloshes"—between the inductor’s magnetic field and the capacitor’s electric field. This creates a characteristic "ringing" effect (transient oscillation) that gradually decays as the resistor dissipates the energy as heat.



```python
import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from circulax import compile_circuit
from circulax.components.electronic import Capacitor, Inductor, Resistor, VoltageSource
from circulax.solvers import setup_transient
```

    KLUJAX_RS DEBUG MODE.
    WARNING:2026-04-17 17:32:56,782:jax._src.xla_bridge:864: An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.



```python
net_dict = {
    "instances": {
        "GND": {"component": "ground"},
        "V1": {"component": "source_voltage", "settings": {"V": 1.0, "delay": 0.25e-9}},
        "R1": {"component": "resistor", "settings": {"R": 10.0}},
        "C1": {"component": "capacitor", "settings": {"C": 1e-11}},
        "L1": {"component": "inductor", "settings": {"L": 5e-9}},
    },
    "connections": {
        "GND,p1": ("V1,p2", "C1,p2"),
        "V1,p1": "R1,p1",
        "R1,p2": "L1,p1",
        "L1,p2": "C1,p1",
    },
}
```



![svg](lcr_transient_files/lcr_transient_3_0.svg)



## Visualize the nodes


```python
from circulax.netlist import draw_circuit_graph

draw_circuit_graph(netlist=net_dict);
```



![png](lcr_transient_files/lcr_transient_5_0.png)




```python
jax.config.update("jax_enable_x64", True)


models_map = {
    "resistor": Resistor,
    "capacitor": Capacitor,
    "inductor": Inductor,
    "source_voltage": VoltageSource,
    "ground": lambda: 0,
}

print("Compiling...")
circuit = compile_circuit(net_dict, models_map)

print(circuit.port_map)

print(f"Total System Size: {circuit.sys_size}")
for g_name, g in circuit.groups.items():
    print(f"Group: {g_name}")
    print(f"  Count: {g.var_indices.shape[0]}")
    print(f"  Var Indices Shape: {g.var_indices.shape}")
    print(f"  Sample Var Indices:{g.var_indices}")
    print(f"  Jacobian Rows Length: {len(g.jac_rows)}")

print("2. Solving DC Operating Point...")
y_op = circuit()

transient_sim = setup_transient(groups=circuit.groups, linear_strategy=circuit.solver)

t_max = 3e-9
saveat = diffrax.SaveAt(ts=jnp.linspace(0, t_max, 500))
print("3. Running Simulation...")
sol = transient_sim(
    t0=0.0,
    t1=t_max,
    dt0=1e-3 * t_max,
    y0=y_op,
    saveat=saveat,
    max_steps=100000,
    progress_meter=diffrax.TqdmProgressMeter(refresh_steps=100),
)

ts = sol.ts
v_src = circuit.get_port_field(sol.ys, "V1,p1")
v_cap = circuit.get_port_field(sol.ys, "C1,p1")
i_ind = sol.ys[:, 5]

plt.rcParams.update({
    "figure.figsize": (9, 4),
    "axes.grid": True,
    "text.color": "grey",
    "axes.facecolor": "white",
    "axes.edgecolor": "grey",
    "axes.labelcolor": "grey",
    "xtick.color": "grey",
    "ytick.color": "grey",
    "grid.color": "#e0e0e0",
    "figure.facecolor": "white",
    "font.family": "sans-serif",
})

ts_ns = ts * 1e9  # convert to nanoseconds

fig, ax1 = plt.subplots(figsize=(9, 4))
ax2 = ax1.twinx()
ln1 = ax1.plot(ts_ns, v_src, color="#2ca02c", linewidth=2, linestyle="--", label="Source V")
ln2 = ax1.plot(ts_ns, v_cap, color="#1f77b4", linewidth=2.5, label="Capacitor V")
ln3 = ax2.plot(ts_ns, i_ind, color="#d62728", linewidth=2, linestyle=":", label="Inductor I")
ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Voltage (V)", color="#1f77b4")
ax2.set_ylabel("Current (A)", color="#d62728")
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#d62728")
lines = ln1 + ln2 + ln3
ax1.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=9,
           framealpha=0.9, edgecolor="grey")
ax1.set_title("LCR Impulse Response — underdamped ringing at ~1 GHz", color="grey", pad=10)
fig.tight_layout()
plt.show()

```

    Compiling...


    {'L1,p2': 1, 'C1,p1': 1, 'GND,p1': 0, 'V1,p2': 0, 'C1,p2': 0, 'L1,p1': 2, 'R1,p2': 2, 'V1,p1': 3, 'R1,p1': 3, 'V1,i_src': 4, 'L1,i_L': 5}
    Total System Size: 6
    Group: source_voltage
      Count: 1
      Var Indices Shape: (1, 3)
      Sample Var Indices:[[3 0 4]]
      Jacobian Rows Length: 1
    Group: resistor
      Count: 1
      Var Indices Shape: (1, 2)
      Sample Var Indices:[[3 2]]
      Jacobian Rows Length: 1
    Group: capacitor
      Count: 1
      Var Indices Shape: (1, 2)
      Sample Var Indices:[[1 0]]
      Jacobian Rows Length: 1
    Group: inductor
      Count: 1
      Var Indices Shape: (1, 3)
      Sample Var Indices:[[2 1 5]]
      Jacobian Rows Length: 1
    2. Solving DC Operating Point...


    3. Running Simulation...




![png](lcr_transient_files/lcr_transient_6_3.png)




```python

```
