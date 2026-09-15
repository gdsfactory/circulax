# Fit coefficients, then create a component

The recommended delay-free S-model workflow has two entry points:

```python
from circulax.fitting import (
    ModelFitOptions,
    fit_model,
    component_from_coefficients,
)

options = ModelFitOptions(
    aaa_backend="numpy",
    normalized_rmse=1e-4,
    max_absolute_error=1e-3,
)
coefficients = fit_model(S, frequencies_hz, options=options, z0=50.0)
coefficients.save("network.npz")

# This can run in another process. It does not fit or enforce anything.
Network = component_from_coefficients("network.npz", name="Network")
component = Network()
```

`Network` is a Circulax component class with ports `p1`, `p2`, etc. Use the
instance in the usual circuit workflow. The existing rational component uses
complex states; AC setup/circuit analysis requires `is_complex=True`.

You can also pass `coefficients` directly to `component_from_coefficients`, with
no file. Inspect `coefficients.poles`, `.residues`, `.D`, `.z0`,
`.frequency_range`, and `.metadata`, or evaluate it using
`coefficients.evaluate(frequencies_hz)`.

## Explicit fitting options

`S` must have shape `(frequencies, ports, ports)`. Frequencies are finite,
nonnegative, strictly increasing Hz. Reference impedance is one common positive
real value in ohms. Inputs must already be S data: there is no implicit domain
inference, DC extrapolation, sample conditioning, or delay extraction.

| `ModelFitOptions` field | Default | Meaning |
| --- | --- | --- |
| `aaa_backend` | `"numpy"` | `"numpy"` or `"jax"` for AAA support discovery only |
| `tol` | `1e-8` | AAA approximation tolerance |
| `mmax` | `12` | Maximum AAA support capacity; not the final pole count |
| `iterations` | `6` | VF relocation iterations per refinement |
| `reciprocal` | `True` | Require symmetric input; use reciprocal fitting |
| `screening` | `"compact"` | Compact or `"masked"` NumPy candidate screening |
| `reduction_stage` | `"refined"` | Refine the full discovery before reduction, or use `"initial"` screening |
| `normalized_rmse` | `0.02` | Maximum training normalized complex RMS error |
| `max_absolute_error` | `0.05` | Maximum training absolute S error |
| `enforce_passivity` | `False` | Opt into fixed-pole rational S coefficient correction |
| `passivity_limit` | `0.999999` | Singular-value bound for enforcement |
| `enforcement_iterations` | `300` | Maximum optimizer iterations |
| `enforcement_freqs` | `None` | Explicit enforcement grid in Hz; otherwise use the broad automatic grid |

NumPy/SciPy still performs pole extraction, refinement, and enforcement when
`aaa_backend="jax"`. This switch does not claim an all-JAX fitting pipeline.
The new defaults use refine-before-reduction; the older `fit_s_numpy` default
is unchanged. Automatic order means no pole count is required from the caller.

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
They bypass AAA and order reduction. With `iterations=0`, only residues and the
constant are fitted; positive iterations relocate the supplied poles. AAA and
screening settings are inapplicable on this branch. Both real poles and full
conjugate pairs count toward the length of the pole array.

## What is stored?

`ModelCoefficients` always represents the proper rational **S** model

$$
S(s)=D+\sum_k\frac{R_k}{s-p_k},\qquad s=j2\pi f.
$$

Residues have shape `(ports, ports, poles)`. Both members of every conjugate pair
are stored. There is no proportional term or hidden reference-plane delay.
Coefficient arrays are NumPy arrays; creating a component is the explicit JAX
realization boundary.

The NPZ archive contains `poles`, `residues`, `D`, and a JSON header recording
schema version, domain, reference impedance, measured frequency range, settings,
and diagnostics. No pickle is used. `save` overwrites its specified file;
`ModelCoefficients.load(path)` validates the archive. The stored frequency range
describes the supplied data, not certified extrapolation coverage.

## Enforcement and admission to simulation

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
This consolidation intentionally does not pretend that the delay-free S workflow
can transparently replace delayed, proportional-term Y, or parameterized models.
