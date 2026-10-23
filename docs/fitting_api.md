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

## Transmission lines from RF mode data

For a uniform line, use frequency-dependent mode-solver attenuation, phase
index and characteristic impedance directly:

```python
from circulax.components import mode_transmission_line

line = mode_transmission_line(
    frequencies,                    # Hz
    attenuation_db_per_m=loss,       # scalar, array, or loss(f_hz) callable
    n_eff=rf_phase_index,            # scalar, array, or n_eff(f_hz) callable
    impedance=characteristic_z,      # ohms; may be complex and vary with frequency
)
section = line(length=200e-6, n_sections=2)
```

The conversion uses `gamma = ln(10)/20 * loss + 1j*2*pi*f*n_eff/c`, followed by
`Zprime=gamma*Z` and `Yprime=gamma/Z`. The same passive RLC fitting, fit error
reporting and section-count controls described below apply. Inputs must be
mutually consistent physical mode data; separate arbitrary loss and phase
curves do not automatically admit this passive, causal realization. The
factory rejects invalid inputs and fits exceeding the requested tolerance.

`n_eff` is the real RF **phase index**. Group index is
`n_group = n_eff + f*dn_eff/df`; velocity matching compares the **loaded RF group
index** with the optical group index. If mode-solver data already includes the
PN junction's distributed R/C, do not add those elements again in the circuit.
If the mode data excludes the junctions, add them and verify the resulting
loaded propagation, as in the traveling-wave MZM tutorial.

DC is optional. When included, impedance must be real there. Attenuation must
be nonnegative and the real part of impedance positive. The restricted Foster
family also requires the converted per-metre RLGC values to be nonnegative.
Frequency interpolation and extrapolation of external data are the caller's
responsibility; a callable is evaluated once on the provided fitting grid.

## Transmission lines from frequency-dependent RLGC

`rlgc_transmission_line` fits RLGC per metre once, then generates passive,
causal RLC ladders at different lengths. Series impedance and shunt admittance
are fitted independently, allowing frequency-dependent characteristic impedance.

```python
from circulax.components import rlgc_transmission_line

line = rlgc_transmission_line(
    frequencies,  # Hz, strictly increasing; DC may be included
    R=r_per_m,    # ohm/m
    L=l_per_m,    # H/m
    G=g_per_m,    # S/m
    C=c_per_m,    # F/m
    n_poles=12,
    tolerance=0.01,
)
section = line(length=200e-6, n_sections=2)
# Register the Circuit in a parent circuit's models_map:
models = {"line": section}
```

Each RLGC input can be a real nonnegative scalar or an array matching the
frequency grid. `section` has single-ended `p1`/`p2` ports and midpoint ports
`tap_0`, `tap_1`, etc. for junction loading, all relative to common ground.
It includes no source or termination. `line.netlist(...)` returns the primitive
RLC netlist instead of compiling it.

The fits approximate each per-metre response with
`constant + s*linear + sum(w*s/(s+p))`, with nonnegative coefficients and positive
rates `p`. Positive resistors, inductors and capacitors realize this response;
no S-parameter-to-state-space conversion is involved. The baseline propagation
is represented by the linear L/C terms, rather than a rational fit to a long
phase rotation. Optional `poles=` supplies positive rates in rad/s.

`line.series` and `line.shunt` expose `constant`, `linear`, `weights`, `poles`,
`evaluate(frequencies)`, `normalized_rmse`, and `maximum_relative_error`.
The fit uses relative error with a floor of 1% of the response's maximum
magnitude. It raises `ValueError` if either weighted relative RMS error exceeds
`tolerance`. These errors describe the impedance/admittance fits, not the final
line's S-parameter error.

This Foster branch family supports useful loss and dispersion models, including
finite-band skin-effect approximations, but cannot represent every passive
response. The generated circuit is passive and causal by construction; this
is not a causality certificate for the supplied data. Validate the line response
over the required band and increase `n_sections` until the spatial approximation
converges. Accuracy outside the fitted band is not guaranteed.
