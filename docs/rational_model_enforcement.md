# Rational-model passivity

A passive S-parameter model must satisfy

$$
\sigma_{\max}(S(j\omega))\le 1
$$

at every frequency. Checking each S-parameter magnitude separately is not
sufficient because several ports can be excited together.

Circulax can apply a fixed-pole coefficient correction:

```python
from circulax.fitting import ModelFitOptions, fit_model

coefficients = fit_model(
    s_parameters,
    frequencies_hz,
    options=ModelFitOptions(enforce_passivity=True),
)
```

The correction changes residues and the constant term while leaving poles
fixed. It minimizes the change to the fitted response subject to singular-value
limits on a frequency grid and at the infinite-frequency limit. The requested
fitting error limits are checked again afterwards.

## What the check establishes

A successful correction establishes passivity only on its sampled grid. It does
not prove passivity between samples or outside the modeled band. Use independent
frequency samples and application-specific source and load conditions when
qualifying a model.

Circulax also checks the converted admittance realization. Stable S poles alone
do not guarantee stable Y poles because

$$
Y(s)=\frac{1}{z_0}[I-S(s)][I+S(s)]^{-1}.
$$

Zeros of $\det[I+S(s)]$ become poles of the voltage-driven admittance
realization. This is why `validate_model` checks both S- and Y-domain
stability.

## Active models

Do not enforce passivity on a model whose gain is intentional. Fit it with
`reciprocal=False` when appropriate and validate with
`expected_passive=False`. Stability with a particular source, load, or
feedback network still requires analysis of the assembled circuit.

Algorithm comparisons, optimizer diagnostics, and reproducible timing studies
belong in the repository's `benchmarks/fitting` directory rather than this
user guide.
