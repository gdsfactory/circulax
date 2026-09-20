# Photonic transient: split paths with propagation delay

This example launches one optical step into a 1×2 splitter and routes the outputs through nonlinear waveguides of different lengths. The longer arm arrives later and experiences both more linear propagation loss and more two-photon absorption (TPA).

Circulax simulates the complex optical envelope rather than the hundreds-of-terahertz carrier. The custom component below combines group delay $\tau=L n_g/c$, linear field transmission $T=10^{-\alpha L/20}\exp(-j\phi)$, and intensity-dependent TPA. Each instance receives its own interpolated history, so different lengths retain different delays.


```python
import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from circulax import compile_circuit
from circulax.components.base_component import PhysicsReturn, Signals, component
from circulax.components.electronic import Resistor
from circulax.components.photonic import OpticalSourcePulse, Splitter

jax.config.update("jax_enable_x64", True)
plt.rcParams.update({"figure.figsize": (8, 3.5), "axes.grid": True, "figure.facecolor": "white"})
```

## Non-linear optical waveguide

The waveguide uses incident-wave states $a_1,a_2$ and outgoing waves $b_1,b_2$ evaluated after the propagation time. This stamp loads both ports at $Z_0$, propagates reflections in either direction, and remains finite for a lossless matched waveguide. TPA makes the transmission depend on the incident power at the corresponding retarded time.


```python
@component(ports=("p1", "p2"), states=("a1", "a2"))
def NonlinearOpticalWaveguide(
    signals: Signals,
    length_um: float = 100.0,
    loss_dB_cm: float = 1.0,
    tpa_coeff: float = 0.0,
    neff: float = 2.4,
    n_group: float = 4.0,
    wavelength_nm: float = 1310.0,
    z0: float = 1.0,
) -> PhysicsReturn:
    """Bidirectional waveguide with fixed delay, linear loss, and TPA."""
    length_cm = length_um / 10_000.0
    alpha_cm = loss_dB_cm * jnp.log(10.0) / 10.0
    effective_length = jnp.where(
        alpha_cm > 1e-12,
        (1.0 - jnp.exp(-alpha_cm * length_cm)) / alpha_cm,
        length_cm,
    )
    linear_amplitude = 10.0 ** (-loss_dB_cm * length_cm / 20.0)
    phase = 2.0 * jnp.pi * neff * length_um * 1000.0 / wavelength_nm
    delay = length_um * n_group / 2.99792458e14
    past = signals.at_delay(delay)

    def propagate(incident):
        tpa_amplitude = 1.0 / jnp.sqrt(1.0 + tpa_coeff * jnp.abs(incident) ** 2 * effective_length)
        return linear_amplitude * tpa_amplitude * jnp.exp(-1j * phase) * incident

    b1 = propagate(past.a2)
    b2 = propagate(past.a1)
    i1 = (signals.a1 - b1) / z0
    i2 = (signals.a2 - b2) / z0
    return {
        "p1": i1,
        "p2": i2,
        "a1": signals.p1 - signals.a1 - b1,
        "a2": signals.p2 - signals.a2 - b2,
    }, {}
```

## Circuit and expected path properties

Both arms have the same propagation loss per centimetre; only their lengths differ. Matched $1\,\Omega$ optical-envelope loads let the 50/50 splitter feed both paths without reflections.


```python
c_um_per_s = 2.99792458e14
source_power, source_delay, source_rise = 1.0, 0.30e-9, 0.025e-9
split_ratio, n_group, loss_dB_cm = 0.5, 4.0, 3.0
tpa_coeff = 4.0  # 1 / (W cm); deliberately strong so pulse compression is visible
short_length_um, long_length_um = 5_000.0, 15_000.0


def path_properties(length_um):
    delay = length_um * n_group / c_um_per_s
    loss_dB = loss_dB_cm * length_um / 10_000.0
    field_transmission = 10.0 ** (-loss_dB / 20.0)
    alpha_cm = loss_dB_cm * np.log(10.0) / 10.0
    effective_length = (1.0 - np.exp(-alpha_cm * length_um / 10_000.0)) / alpha_cm
    return delay, loss_dB, field_transmission, effective_length


tau_short, loss_short_dB, transmission_short, leff_short = path_properties(short_length_um)
tau_long, loss_long_dB, transmission_long, leff_long = path_properties(long_length_um)

print(f"Short arm: {short_length_um / 1000:.1f} mm, {tau_short * 1e12:.1f} ps delay, {loss_short_dB:.1f} dB loss")
print(f"Long arm:  {long_length_um / 1000:.1f} mm, {tau_long * 1e12:.1f} ps delay, {loss_long_dB:.1f} dB loss")
print(f"Differential delay: {(tau_long - tau_short) * 1e12:.1f} ps")
```

    Short arm: 5.0 mm, 66.7 ps delay, 1.5 dB loss
    Long arm:  15.0 mm, 200.1 ps delay, 4.5 dB loss
    Differential delay: 133.4 ps



```python
models = {
    "source": OpticalSourcePulse,
    "splitter": Splitter,
    "waveguide": NonlinearOpticalWaveguide,
    "resistor": Resistor,
    "ground": lambda: 0,
}

netlist = {
    "instances": {
        "GND": {"component": "ground"},
        "SRC": {
            "component": "source",
            "settings": {
                "power": source_power,
                "delay": source_delay,
                "rise": source_rise,
            },
        },
        "SPLIT": {"component": "splitter", "settings": {"split_ratio": split_ratio}},
        "WG_SHORT": {
            "component": "waveguide",
            "settings": {
                "length_um": short_length_um,
                "loss_dB_cm": loss_dB_cm,
                "tpa_coeff": tpa_coeff,
                "n_group": n_group,
            },
        },
        "WG_LONG": {
            "component": "waveguide",
            "settings": {
                "length_um": long_length_um,
                "loss_dB_cm": loss_dB_cm,
                "tpa_coeff": tpa_coeff,
                "n_group": n_group,
            },
        },
        "LOAD_SHORT": {"component": "resistor", "settings": {"R": 1.0}},
        "LOAD_LONG": {"component": "resistor", "settings": {"R": 1.0}},
    },
    "connections": {
        "GND,p1": ("SRC,p2", "LOAD_SHORT,p2", "LOAD_LONG,p2"),
        "SRC,p1": "SPLIT,p1",
        "SPLIT,p2": "WG_SHORT,p1",
        "SPLIT,p3": "WG_LONG,p1",
        "WG_SHORT,p2": "LOAD_SHORT,p1",
        "WG_LONG,p2": "LOAD_LONG,p1",
    },
    "ports": {
        "input": "SRC,p1",
        "out_short": "WG_SHORT,p2",
        "out_long": "WG_LONG,p2",
    },
}

circuit = compile_circuit(netlist, models, is_complex=True, backend="dense")
y_dc = circuit.dc()
print(f"System size: {circuit.sys_size} complex unknowns")
```

    System size: 11 complex unknowns


## Transient simulation

The analytical references combine the split ratio, linear path attenuation, effective nonlinear length, and group delay. TPA is evaluated from the delayed incident power, exactly as in the component. Comparing envelope magnitudes removes carrier phase while retaining arrival time, compression, and loss.


```python
sample_times = jnp.linspace(0.0, 0.8e-9, 801)
solution = circuit.transient(
    t0=0.0,
    t1=float(sample_times[-1]),
    dt0=0.5e-12,
    y0=y_dc,
    saveat=diffrax.SaveAt(ts=sample_times),
    stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-7),
    max_steps=20_000,
    throw=True,
)

field_in = circuit.port(solution.ys, "input")
field_short = circuit.port(solution.ys, "out_short")
field_long = circuit.port(solution.ys, "out_long")
incident_short = jnp.sqrt(split_ratio * source_power) * jax.nn.sigmoid((sample_times - source_delay - tau_short) / source_rise)
incident_long = jnp.sqrt((1.0 - split_ratio) * source_power) * jax.nn.sigmoid(
    (sample_times - source_delay - tau_long) / source_rise
)
linear_short = transmission_short * incident_short
linear_long = transmission_long * incident_long
expected_short = linear_short / jnp.sqrt(1.0 + tpa_coeff * incident_short**2 * leff_short)
expected_long = linear_long / jnp.sqrt(1.0 + tpa_coeff * incident_long**2 * leff_long)

short_error = float(jnp.max(jnp.abs(jnp.abs(field_short) - expected_short)))
long_error = float(jnp.max(jnp.abs(jnp.abs(field_long) - expected_long)))
print(f"Maximum short-arm envelope error: {short_error:.2e}")
print(f"Maximum long-arm envelope error:  {long_error:.2e}")
assert short_error < 5e-4
assert long_error < 5e-4
```

    Maximum short-arm envelope error: 3.40e-06
    Maximum long-arm envelope error:  2.59e-06



```python
time_ns = np.asarray(sample_times) * 1e9
fig, (ax_field, ax_power) = plt.subplots(1, 2, figsize=(12, 4))

ax_field.plot(time_ns, np.abs(field_in), color="0.35", lw=2, label="input")
ax_field.plot(time_ns, np.abs(field_short), color="C0", lw=2, label="5 mm arm")
ax_field.plot(time_ns, np.abs(field_long), color="C1", lw=2, label="15 mm arm")
ax_field.plot(time_ns, expected_short, "k--", lw=1, alpha=0.65, label="analytic shifts")
ax_field.plot(time_ns, expected_long, "k--", lw=1, alpha=0.65)
ax_field.axvline((source_delay + tau_short) * 1e9, color="C0", ls=":")
ax_field.axvline((source_delay + tau_long) * 1e9, color="C1", ls=":")
ax_field.set(xlabel="Time (ns)", ylabel=r"Envelope magnitude $|E|$", title="Different lengths produce different arrival times")
ax_field.legend(fontsize=9)

power_in = np.abs(field_in) ** 2
power_short = np.abs(field_short) ** 2
power_long = np.abs(field_long) ** 2
floor = 1e-12
ax_power.plot(time_ns, 10 * np.log10(np.maximum(power_in, floor)), color="0.35", lw=2, label="input")
ax_power.plot(time_ns, 10 * np.log10(np.maximum(power_short, floor)), color="C0", lw=2, label="5 mm arm + TPA")
ax_power.plot(time_ns, 10 * np.log10(np.maximum(power_long, floor)), color="C1", lw=2, label="15 mm arm + TPA")
ax_power.plot(time_ns, 10 * np.log10(np.maximum(np.asarray(linear_short) ** 2, floor)), "C0--", lw=1.2, label="5 mm linear only")
ax_power.plot(time_ns, 10 * np.log10(np.maximum(np.asarray(linear_long) ** 2, floor)), "C1--", lw=1.2, label="15 mm linear only")
ax_power.set(
    xlabel="Time (ns)", ylabel="Envelope power (dB, 1 W reference)", title="TPA adds intensity-dependent loss", ylim=(-35, 1)
)
ax_power.legend(fontsize=9)
plt.tight_layout()
plt.show()
```



![png](photonics_transient_files/photonics_transient_9_0.png)



## Verified delay and nonlinear loss

Because TPA compresses the envelope, its half-output crossing is not a pure delay measurement. Instead, both arms are measured where their delayed incident envelope reaches the same 10% fraction. The settled powers are compared with the closed-form linear-plus-TPA result.


```python
def arrival_at_incident_fraction(times, envelope, branch_power, transmission, effective_length, fraction=0.1):
    incident = np.sqrt(branch_power) * fraction
    target = transmission * incident / np.sqrt(1.0 + tpa_coeff * incident**2 * effective_length)
    return np.interp(target, envelope, times)


short_branch_power = split_ratio * source_power
long_branch_power = (1.0 - split_ratio) * source_power
arrival_short = arrival_at_incident_fraction(
    np.asarray(sample_times), np.abs(field_short), short_branch_power, transmission_short, leff_short
)
arrival_long = arrival_at_incident_fraction(
    np.asarray(sample_times), np.abs(field_long), long_branch_power, transmission_long, leff_long
)
measured_differential_delay = arrival_long - arrival_short

settled_short_power = float(jnp.abs(field_short[-1]) ** 2)
settled_long_power = float(jnp.abs(field_long[-1]) ** 2)
linear_short_power = short_branch_power * 10.0 ** (-loss_short_dB / 10.0)
linear_long_power = long_branch_power * 10.0 ** (-loss_long_dB / 10.0)
expected_short_power = linear_short_power / (1.0 + tpa_coeff * short_branch_power * leff_short)
expected_long_power = linear_long_power / (1.0 + tpa_coeff * long_branch_power * leff_long)
tpa_short_dB = 10.0 * np.log10(linear_short_power / expected_short_power)
tpa_long_dB = 10.0 * np.log10(linear_long_power / expected_long_power)

print(f"Measured differential delay: {measured_differential_delay * 1e12:.1f} ps")
print(f"Expected differential delay: {(tau_long - tau_short) * 1e12:.1f} ps")
print(
    f"Short-arm settled power: {settled_short_power:.4f} W (expected {expected_short_power:.4f} W; TPA adds {tpa_short_dB:.2f} dB)"
)
print(f"Long-arm settled power:  {settled_long_power:.4f} W (expected {expected_long_power:.4f} W; TPA adds {tpa_long_dB:.2f} dB)")

assert abs(measured_differential_delay - (tau_long - tau_short)) < 1e-13
assert np.isclose(settled_short_power, expected_short_power, rtol=3e-4)
assert np.isclose(settled_long_power, expected_long_power, rtol=3e-4)
```

    Measured differential delay: 133.4 ps
    Expected differential delay: 133.4 ps
    Short-arm settled power: 0.1918 W (expected 0.1918 W; TPA adds 2.66 dB)
    Long-arm settled power:  0.0619 W (expected 0.0619 W; TPA adds 4.58 dB)


The two waveguides are instances of the same bidirectional nonlinear component. Each retains its own length-dependent history query and effective nonlinear length, so the result simultaneously shows the splitter ratio, linear loss, TPA compression, and differential group delay.
