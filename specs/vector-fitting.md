# S-parameter fitting

## Goal

Convert sampled S-parameters into a portable rational model, validate it, and
construct a simulation-ready `Circuit`. These are separate operations so that a
fit can be inspected or rejected before JAX compilation.

## Stable interface

The stable `circulax.fitting` namespace contains:

- `fit_model` and `ModelFitOptions`;
- `ModelCoefficients` persistence and evaluation;
- `validate_model` and `ModelValidationReport`;
- `circuit_from_coefficients`;
- `DelayInferenceOptions` and `DelayInferenceWarning`.

The default fitting backend is scikit-rf vector fitting. AAA, pole screening,
surface optimization, and enforcement implementation helpers remain available
from their submodules for experimentation, but are not stable API.

## Model convention

The rational core is

```text
S_core(s) = D + sum_k R_k / (s - p_k),  s = j 2 pi f.
```

Poles use rad/s. Conjugate pairs are stored explicitly. Optional per-port delays
use one-way seconds:

```text
S_full(f) = P(f) S_core(f) P(f)
P_ii(f) = exp(-j 2 pi f tau_i).
```

The coefficient archive is versioned NPZ data without pickled Python objects.

## Delay inference

Inference is intentionally narrow: passive, approximately reciprocal,
low-reflection two-ports with a user-supplied upper delay bound. Acceptable
reciprocity noise is projected onto `(S + S.T) / 2` and recorded.

Candidate selection reserves 20% of the frequencies. The undelayed baseline and
bounded delay candidates are fitted on the other 80%. When both the baseline
and a delayed candidate qualify, delay is selected only if it removes at least
one complete real pole or conjugate-pole pair without violating the
reserved-sample error guard. A qualifying delayed fit may also be selected when
the baseline cannot meet the error limits. The selected configuration is then
refitted on all samples. If inference is unjustified but
the baseline is valid, the baseline is returned with a warning and a structured
report in `metadata["delay_inference"]`.

Reflective, active, nonreciprocal, or multipath networks require supplied delays
or a physical model. See GitHub issue 56 for the tracked scope extension.

## Admission to simulation

`validate_model` checks requested training and held-out errors, sampled
passivity and reciprocity, S-pole stability, converted Y-pole stability, and
simulation-band coverage. Passivity and reciprocity are expected by default;
active or nonreciprocal data must opt out explicitly.

`circuit_from_coefficients` performs no fitting or enforcement. It rejects
unstable S or Y realizations and always returns a flattened `Circuit`, including
for delay-free models.

## Acceptance criteria

- The public workflow fits, validates, saves, loads, and constructs a circuit.
- Construction has one return type and does not compile during fitting.
- Default vector fitting works from a normal core installation.
- Inferred delay is selected using held-out frequencies and complete pole groups.
- Noisy, irregular, approximately reciprocal data are covered by seeded tests.
- Unjustified inference falls back with a warning and structured reason.
- Circulax 0.2.3 component signatures remain supported.

## Verification

Use the repository tasks:

```text
pixi run pytest_run
pixi run nbrun
pixi run nbdocs
pixi run docs-build
```
