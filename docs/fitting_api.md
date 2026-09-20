# S-parameter fitting

Circulax fits sampled S-parameters into a portable pole-residue model. Fitting,
validation, and circuit construction are separate operations:

```python
from circulax.fitting import (
    ModelFitOptions,
    circuit_from_coefficients,
    fit_model,
    validate_model,
)

options = ModelFitOptions(
    normalized_rmse=1e-4,
    max_absolute_error=1e-3,
)
coefficients = fit_model(s_parameters, frequencies_hz, options=options, z0=50.0)

report = validate_model(
    coefficients,
    measured_S=s_parameters,
    freqs=frequencies_hz,
    validation_S=heldout_s,
    validation_freqs=heldout_frequencies_hz,
    simulation_frequency_range=(frequencies_hz[0], frequencies_hz[-1]),
)
report.raise_for_simulation()

coefficients.save("network.npz")
circuit = circuit_from_coefficients("network.npz", name="Network")
```

The [vector-fitting tutorial](examples/vector_fitting.md) applies this workflow
to a passive two-port.

## Inputs and coefficients

`fit_model` expects:

- S data with shape `(frequencies, ports, ports)`;
- finite, strictly increasing frequencies in Hz;
- a common positive real reference impedance;
- reciprocal data by default.

`ModelCoefficients` stores the proper rational S model

$$
S_\mathrm{core}(s)=D+\sum_k\frac{R_k}{s-p_k},
\qquad s=j2\pi f.
$$

Poles use rad/s. Both members of each conjugate pair are stored. Residues have
shape `(ports, ports, poles)`. `evaluate_core(f)` evaluates the rational
core, while `evaluate(f)` also applies explicit port delays.

The versioned NPZ archive contains coefficients, delays, units, fitting options,
and diagnostics. It does not contain pickled Python objects.

## Vector fitting

Conventional scikit-rf vector fitting is the default:

```python
options = ModelFitOptions(
    method="vector_fitting",
    vector_fit_order=(4, 0),  # real starting poles, complex starting pairs
)
```

With no explicit order, scikit-rf selects one automatically. Accuracy limits are
Circulax acceptance checks applied after fitting and optional enforcement; they
are not aliases for scikit-rf's internal stopping criteria.

AAA and the lower-level pole, surface, and enforcement routines remain available
from their implementation submodules for experiments. They are not part of the
stable fitting interface.

## Supplied propagation delays

Known delays are stored as one-way seconds per port:

$$
S_\mathrm{full}(f)=P(f)S_\mathrm{core}(f)P(f),\qquad
P_{ii}(f)=e^{-j2\pi f\tau_i}.
$$

```python
options = ModelFitOptions(
    delay_mode="supplied",
    port_delays=(0.3e-9, 0.7e-9),
)
coefficients = fit_model(s_parameters, frequencies_hz, options=options)
```

`circuit_from_coefficients` realizes these delays with bidirectional
`TransmissionLine` elements. The returned value is always a `Circuit`, for
both delayed and delay-free models.

## Inferred propagation delay

`delay_mode="infer"` is intended for passive, approximately reciprocal,
low-reflection two-ports:

```python
options = ModelFitOptions(
    delay_mode="infer",
    max_delay=2e-9,
)
coefficients = fit_model(s_parameters, frequencies_hz, options=options)
```

`max_delay` is a required upper bound on the total transmission delay. It
prevents phase unwrapping from silently choosing an aliased delay.

Inference performs the following steps:

1. Measure input reciprocity and reject excessive asymmetry.
2. Project acceptable noisy data onto $(S+S^T)/2$.
3. Reserve every fifth frequency for candidate validation.
4. Fit the rational-only baseline and bounded delay candidates on the remaining
   samples.
5. If the baseline qualifies, select a delayed candidate only if it removes at
   least one complete pole group without materially degrading reserved-sample
   accuracy. If the baseline misses the error limits, a delayed candidate may
   qualify on its own reserved-sample accuracy.
6. Refit the selected configuration on all samples.

If inference is not justified but the baseline passes, Circulax returns the
baseline and emits `DelayInferenceWarning`. The `delay_inference` metadata
records the reason, reservation sizes, reciprocity correction, candidates, and
selection. A phase slope is a proposal, not a causality proof.

Expert thresholds are grouped in `DelayInferenceOptions`:

```python
from circulax.fitting import DelayInferenceOptions

options = ModelFitOptions(
    delay_mode="infer",
    max_delay=2e-9,
    delay_inference=DelayInferenceOptions(
        reflection_threshold=0.03,
        phase_residual=0.04,
    ),
)
```

Reflective networks, active or nonreciprocal devices, multiple propagation paths,
and unequal unidentified port delays require supplied delays or a physical
model.

## Validation

`validate_model` checks:

- training and held-out complex-S errors when data are supplied;
- sampled passivity by default;
- reciprocity by default;
- stable S poles and stable converted Y poles;
- finite, supported S-to-Y realization;
- requested simulation-band coverage.

Active models must opt out explicitly:

```python
report = validate_model(
    coefficients,
    measured_S=s_parameters,
    freqs=frequencies_hz,
    expected_passive=False,
)
```

Missing measured or held-out data produce warnings. Validation never refits,
repairs, or enforces a model.

## Passivity enforcement

`ModelFitOptions(enforce_passivity=True)` enables sampled fixed-pole
coefficient correction. It rechecks the requested fitting accuracy after the
correction. Sampled enforcement is not a global passivity certificate; see
[Rational-model passivity](rational_model_enforcement.md).
