# Fit an S-parameter model

This tutorial fits a passive two-port, checks the result on frequencies excluded from fitting, saves the coefficients, and creates a Circulax circuit. The fitting result is an explicit pole-residue model; circuit construction is a separate step.


```python
from pathlib import Path
import tempfile

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import skrf

from circulax.fitting import (
    ModelFitOptions,
    circuit_from_coefficients,
    fit_model,
    validate_model,
)
```

## Reserve validation frequencies

We use scikit-rf's ring-slot example and reserve every fifth sample. The reserved samples do not influence fitting or order selection.


```python
network = skrf.data.ring_slot
validation_mask = np.arange(len(network.f)) % 5 == 4
training_mask = ~validation_mask

training_frequencies = network.f[training_mask]
training_s = network.s[training_mask]
validation_frequencies = network.f[validation_mask]
validation_s = network.s[validation_mask]

print(f"{len(training_frequencies)} training and {len(validation_frequencies)} validation samples")
```

    161 training and 40 validation samples


## Fit coefficients

The default backend is conventional vector fitting from scikit-rf. Here we prescribe four real starting poles and require the final fit to stay below both error limits.


```python
options = ModelFitOptions(
    vector_fit_order=(4, 0),
    normalized_rmse=1e-4,
    max_absolute_error=1e-3,
)
coefficients = fit_model(training_s, training_frequencies, options=options, z0=50.0)
print(f"{len(coefficients.poles)} stored poles")
```

    4 stored poles


## Validate before simulation

Validation checks the training and reserved errors, sampled passivity, reciprocity, pole stability, converted admittance stability, and requested simulation band. Active models must opt out of passivity explicitly.


```python
report = validate_model(
    coefficients,
    measured_S=training_s,
    freqs=training_frequencies,
    validation_S=validation_s,
    validation_freqs=validation_frequencies,
    simulation_frequency_range=(network.f[0], network.f[-1]),
    normalized_rmse=options.normalized_rmse,
    max_absolute_error=options.max_absolute_error,
)
print(report)
report.raise_for_simulation()
```

    ModelValidationReport(status='pass', training_nrmse=3.393949832678483e-05, training_max_error=7.719185324875572e-05, validation_nrmse=3.377656605015983e-05, validation_max_error=7.342742496725854e-05, maximum_singular_value=0.9995124725308753, maximum_s_pole_real_part=-79662721263.10605, maximum_y_pole_real_part=-1652811937.9770281, findings=())



```python
prediction = coefficients.evaluate(network.f)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
axes[0].plot(network.f / 1e9, 20 * np.log10(np.maximum(abs(network.s[:, 1, 0]), 1e-12)), label="data")
axes[0].plot(network.f / 1e9, 20 * np.log10(np.maximum(abs(prediction[:, 1, 0]), 1e-12)), "--", label="fit")
axes[0].set(xlabel="Frequency (GHz)", ylabel="|S21| (dB)")
axes[1].plot(network.f / 1e9, np.unwrap(np.angle(network.s[:, 1, 0])), label="data")
axes[1].plot(network.f / 1e9, np.unwrap(np.angle(prediction[:, 1, 0])), "--", label="fit")
axes[1].set(xlabel="Frequency (GHz)", ylabel="S21 phase (rad)")
for axis in axes:
    axis.grid(True)
    axis.legend()
fig.tight_layout()
```



![png](vector_fitting_files/vector_fitting_8_0.png)



## Save and create a circuit

The archive contains only coefficients, units, fitting settings, and diagnostics. Loading it does not refit the data. circuit_from_coefficients always returns a Circuit, whether the model contains explicit delays or only a rational core.


```python
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "ring-slot.npz"
    coefficients.save(path)
    circuit = circuit_from_coefficients(path, name="RingSlot")

simulated = circuit.sp(ports=["p1", "p2"], freqs=jnp.asarray(validation_frequencies))
np.testing.assert_allclose(simulated, coefficients.evaluate(validation_frequencies), atol=1e-8)
print(f"Circuit variables: {circuit.sys_size}")
```

    Circuit variables: 11


The fitted frequency range is evidence for interpolation within this band, not arbitrary extrapolation toward DC or infinity. For propagation-delay separation, continue with the [time-delay tutorial](time_delay.md). Advanced AAA, active-network, passivity-enforcement, and timing experiments live under `benchmarks/fitting`.
