# Fit coefficients, then create a component

For a guided walkthrough, see the [vector-fitting tutorial](../examples/fitting/vector_fitting.ipynb),
which covers ring-slot, resonant four-port, and active-transmitter fits in one notebook.

The recommended S-model workflow has two entry points:

```python
from circulax.fitting import (
    ModelFitOptions,
    fit_model,
    component_from_coefficients,
)

options = ModelFitOptions(
    method="vector_fitting",  # default: scikit-rf
    vector_fit_order=(4, 0),  # real starting poles, complex starting pairs
    normalized_rmse=1e-4,
    max_absolute_error=1e-3,
)
coefficients = fit_model(S, frequencies_hz, options=options, z0=50.0)
coefficients.save("network.npz")

# This can run in another process. It does not fit or enforce anything.
Network = component_from_coefficients("network.npz", name="Network")
from circulax import compile_circuit
circuit = compile_circuit(
    {"instances": {"dut": {"component": "network"}}, "connections": {},
     "ports": {"p1": "dut,p1", "p2": "dut,p2"}},
    models_map={"network": Network},
)
```

`Network` is a component class for ordinary zero-delay cores, or a composed
`Circuit` when delays are present (also for an ideal two-port through).
`compile_circuit` accepts both. Do not assume every result supports `Network()`
or low-level `compile_netlist`. Complex-state mode is inferred automatically.

You can also pass `coefficients` directly to `component_from_coefficients`, with
no file. Inspect `coefficients.poles`, `.residues`, `.D`, `.z0`,
`.frequency_range`, and `.metadata`, or evaluate it using
`coefficients.evaluate(frequencies_hz)`.

## Explicit fitting options

### Delay support: existing solvers versus this API

Circulax's unified `signals.at_delay(...)` contract and `TransmissionLine` already
support DC, AC, harmonic balance, and transient analysis. A rational core with
explicit line elements can therefore retain propagation delays across solvers.
The older `rational_delay_component` factory is a frequency-domain oracle; its
restriction is not a restriction of the unified solver API.

The coefficient API stores one-way seconds per port in `port_delays`:
`S_full = P @ S_core @ P`, where `P[i,i] = exp(-2j*pi*f*port_delays[i])`.
Reflection at port i therefore gets twice its port delay. The legacy
`fit_with_delay` convention uses **twice** these durations; convert explicitly
with `port_delays = tau_per_port / 2`.

```python
options = ModelFitOptions(delay_mode="supplied", port_delays=(0.3e-9, 0.7e-9))
coefficients = fit_model(S, frequencies_hz, options=options)
model = component_from_coefficients(coefficients)
```

Known delays work with arbitrary port counts, optional AAA, and active or
nonreciprocal cores (set `reciprocal=False`). Enforcement acts only on the core;
final errors are checked against the original complex S data. No holdout samples
are used. Numerically constant cores can have zero poles. The ideal lossless
two-port through uses the exact line primitive, without artificial attenuation.
Other singular `I+D` cores raise a supported-scope error. An unloaded ideal through
has an undetermined DC voltage: for standalone `sp`, provide a zero `y_dc` array
of the solver state size, or use a loaded parent circuit.

`delay_mode="none"` remains the reproducible default. Opt-in `"auto"` fits and
validates an undelayed baseline, then tries a bounded set of delay proposals.
It selects a delayed candidate only for fewer rational poles; ties retain the
baseline. If no candidate meets accuracy and realization checks it raises.
Automatic proposals currently require a passive reciprocal two-port with
negligible reflections. Phase slope is a proposal, **not a causality proof**.
Reflections, resonant phase, nulls, active devices, and ambiguous sampling cause
inference to be declined; supplied delays remain the reliable path.

| Automatic option | Default | Meaning |
| --- | --- | --- |
| `auto_max_delay` | `None` | Asserted upper bound on the sum of port delays, seconds; required for inference |
| `auto_min_transmission` | `0.05` | Minimum magnitude in both transmission directions |
| `auto_phase_residual` | `0.05` | Maximum phase-line fit residual, radians |
| `auto_direction_tolerance` | `0.05` | Relative disagreement allowed between directional delay estimates |
| `auto_reflection_threshold` | `1e-3` | Maximum reflection magnitude for equal-split proposals |
| `auto_delay_fractions` | `(1.0, 0.5)` | Fractions of the transmission-delay proposal to test |

The asserted bound times the largest frequency gap must be less than 1/2.
Unwrapping alone cannot detect missing phase turns, and a wrong user bound can
still admit an aliased estimate. Transmission determines only the delay sum;
metadata records the equal split assumption, candidate failures, baseline
outcome, selection, and sampled checks. Supplied initial poles require `none`
or `supplied`, never implicit automatic inference. The safeguards are conservative
heuristics, not a proof that advancing the measured response leaves a causal core.

Metadata reports core poles and states, line algebraic variables, circuit size,
and separate fit/enforcement/conversion times. History stores full solver states;
its cost depends on the transient step budget. Fewer poles do not establish a
simulation speedup. Run `python -m benchmarks.fitting.bench_delay_separation` in
the project environment for sequential, separately warmed measurements.

### Current fitting controls

The default delegates to scikit-rf's `auto_fit()`; `vector_fit_order=(n_real,
n_pairs)` instead calls its prescribed-order `vector_fit()`. Install
`scikit-rf>=1.8,<2` if it is not already available (the documentation environment
includes it). Ordinary component loading does not require scikit-rf. See
[scikit-rf's tutorial](https://scikit-rf.readthedocs.io/en/latest/tutorials/VectorFitting.html)
for fitting theory and its API for advanced solver controls.

`S` must have shape `(frequencies, ports, ports)`. Frequencies are finite,
nonnegative, strictly increasing Hz. Reference impedance is one common positive
real value in ohms. Inputs must already be S data: there is no implicit domain
inference, DC extrapolation, sample conditioning. Delay extraction is opt-in.

| `ModelFitOptions` field | Default | Meaning |
| --- | --- | --- |
| `method` | `"vector_fitting"` | scikit-rf fitting, or experimental `"aaa"` initialization plus NumPy VF |
| `vector_fit_order` | `None` | `(real starting poles, complex starting pairs)`; `None` uses automatic order |
| `aaa_backend` | `"numpy"` | `"numpy"` or `"jax"` for AAA support discovery only |
| `tol` | `1e-8` | AAA approximation tolerance |
| `mmax` | `12` | Maximum AAA support capacity; not the final pole count |
| `iterations` | `6` | NumPy VF iterations for AAA or supplied poles; not scikit-rf's iteration limit |
| `reciprocal` | `True` | Require symmetric input; use reciprocal fitting |
| `screening` | `"compact"` | Compact or `"masked"` NumPy candidate screening |
| `reduction_stage` | `"refined"` | Refine the full discovery before reduction, or use `"initial"` screening |
| `normalized_rmse` | `0.02` | Maximum training normalized complex RMS error |
| `max_absolute_error` | `0.05` | Maximum training absolute S error |
| `enforce_passivity` | `False` | Opt into fixed-pole rational S coefficient correction |
| `passivity_limit` | `0.999999` | Singular-value bound for enforcement |
| `enforcement_iterations` | `300` | Maximum optimizer iterations |
| `enforcement_freqs` | `None` | Explicit enforcement grid in Hz; otherwise use the broad automatic grid |

AAA controls (`aaa_backend`, `tol`, `mmax`, `screening`, `reduction_stage`) apply
only with `method="aaa"`. For that experimental branch, NumPy/SciPy still performs
pole extraction, refinement, and enforcement when `aaa_backend="jax"`.
Scikit-rf uses its own fitting defaults; the public error limits are independent
acceptance checks, not aliases for its internal stopping criteria. Optional
enforcement is the same coefficient correction for either method.

AAA remains available for experiments, but these datasets have not established
a speed or accuracy advantage over conventional vector fitting. Select it
explicitly with `ModelFitOptions(method="aaa", aaa_backend="numpy")`.

For supplied poles instead of automatic discovery:

```python
coefficients = fit_model(
    S,
    frequencies_hz,
    initial_poles=poles_rad_per_second,
    options=ModelFitOptions(iterations=0),
)
```

Supplied poles must be strictly stable, with adjacent complete conjugate pairs.
They bypass either discovery backend and order reduction. With `iterations=0`, only residues and the
constant are fitted; positive iterations relocate the supplied poles. AAA and
screening settings are inapplicable on this branch. Both real poles and full
conjugate pairs count toward the length of the pole array.
Do not combine supplied poles with `vector_fit_order`.

## What is stored?

`ModelCoefficients` always represents the proper rational **S** model

$$
S_{core}(s)=D+\sum_k\frac{R_k}{s-p_k},\qquad s=j2\pi f.
$$

Residues have shape `(ports, ports, poles)`. Both members of every conjugate pair
are stored. There is no proportional term; explicit delays are stored separately.
`evaluate_core(f)` returns the rational core; `evaluate(f)` restores port delays.
Coefficient arrays are NumPy arrays; creating a component is the explicit JAX
realization boundary.

The NPZ archive contains `poles`, `residues`, `D`, `port_delays`, and a JSON header recording
schema version 2, one-way-seconds-per-port convention, domain, reference impedance, measured frequency range, settings,
and diagnostics. Version-1 files load as zero-delay models. No pickle is used. `save` overwrites its specified file;
`ModelCoefficients.load(path)` validates the archive. The stored frequency range
describes the supplied data, not certified extrapolation coverage.

## Enforcement and admission to simulation

Scikit-rf already provides passivity testing and enforcement. Circulax is not
claiming a stricter fitting algorithm: the extra admission checks concern the
converted admittance realization that will enter its circuit equations. A good
S fit can have unstable Y poles, and an excited unstable mode can grow in a
transient simulation. Applying a correction is optional and must be followed by
both accuracy and physical checks.

With `enforce_passivity=True`, fitting performs the sampled constrained rational
correction and rejects optimizer failure. **It checks the requested training
accuracy limits again after correction.** It does not silently relax them.
If enforcement pushes a fit past its limit, `fit_model` raises `ValueError`.
Nonreciprocal fitting is supported, but the present enforcement requires reciprocity.

Sampled enforcement is not a global passivity certificate. The returned metadata
records `global_passivity_certified=False`; the rational-test/refined-grid loop
from the [enforcement benchmark](../benchmarks/fitting/bench_rational_enforcement.py)
remains a separate validation experiment. See the
[engineering explanation](rational_model_enforcement.md) for the distinction.

Component generation revalidates the coefficients, performs S-to-Y realization
conversion, and rejects unstable S poles, unstable Y poles, nonfinite results,
and conversion failures. It does not trust saved diagnostics as proof of safety,
nor does it refit or silently repair coefficients. Static, zero-pole models are
supported. Pole stability alone does not establish passivity: independently
check rational passivity, held-out accuracy, and the intended simulation band.

## Existing APIs

Existing names remain available to avoid breaking notebooks and callers:

- `fit_s_numpy`, `discover_numpy`, and `refine_numpy`: low-level experiments.
- `aaa_scalar_numpy` and `aaa_scalar_jax`: direct discovery/backend comparisons.
- `fit_with_delay`: advanced S/Y fitting and delay extraction.
- `vfdriver` / `FitOptions`: traditional VF controls.
- `rational_component` and `rational_delay_component`: direct state-space factories.
- Surface fitting and validation: multi-corner/parameterized workflows.

Use `ModelFitOptions`, not the older low-level `FitOptions`, with `fit_model`.
Proportional-term Y and parameterized models remain separate workflows.
