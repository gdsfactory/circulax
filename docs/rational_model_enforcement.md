# Rational-model passivity enforcement

This page describes the experimental fixed-pole correction in
[`enforcement_numpy.py`](../circulax/fitting/enforcement_numpy.py). For fitting
and component creation, start with the [notebook](../examples/fitting/vector_fitting.ipynb)
or [fitting API](fitting_api.md).

The benchmark results below use an AAA-initialized seven-pole fit. They are a
separate experiment from the notebook's conventional four-pole scikit-rf fit.
The correction itself works on rational coefficients and does not depend on AAA.

## Model and constraints

For an $n$-port network, the correction accepts a proper rational S model

$$
S(s)=D+\sum_{k=1}^{K}\frac{R_k}{s-p_k},\qquad s=j2\pi f.
$$

Poles are in rad/s, residues are $n\times n$ matrices, and $D$ is the
high-frequency limit. There is no proportional term $sE$.

The input must have strictly stable poles, real or conjugate-paired coefficients,
a real $D$, and reciprocal coefficient matrices. Unstable S poles are rejected.
Reciprocity is a restriction of this implementation, not a general requirement
for passivity.

For power-normalized incident and outgoing waves, $b=Sa$. Absorbed power is
proportional to

$$
a^Ha-b^Hb=a^H(I-S^HS)a.
$$

A passive model therefore needs

$$
\sigma_{\max}(S(j\omega))\le1
$$

at every frequency, along with stable, causal behavior. Here $H$ means conjugate
transpose. Checking each $|S_{ij}|\le1$ is insufficient because several ports
can be excited together. Pole stability and passivity are separate checks; see
[SINTEF's passivity guidance](https://www.sintef.no/en/software/vector-fitting/passivity/).

## Why the converted Y poles matter

Circulax uses admittance in its circuit equations. With a common real positive
reference impedance $z_0$,

$$
Y(s)=\frac{1}{z_0}[I-S(s)][I+S(s)]^{-1}.
$$

Zeros of $\det[I+S(s)]$ can become Y poles, even when the S poles are stable.
For example, with $\tau>0$,

$$
S(s)=\frac{-2}{1+s\tau}
$$

has a stable S pole at $-1/\tau$ but is nonpassive near DC. Its admittance is

$$
Y(s)=\frac{1}{z_0}\frac{s\tau+3}{s\tau-1}
=\frac{1}{z_0}+\frac{4}{z_0\tau}\frac{1}{s-1/\tau}.
$$

The voltage-driven Y realization contains a growing mode $e^{t/\tau}$.
Conversion changes which terminal variables are inputs and outputs; stable S
poles alone don't guarantee a stable realization with voltage as the input.

For $S=D+C(sI-A)^{-1}B$, let $M=(I+D)^{-1}$. The implemented conversion uses

$$
A_Y=A-BMC,\quad B_Y=BM/\sqrt{z_0},\quad
C_Y=-2MC/\sqrt{z_0},\quad D_Y=(I-D)M/z_0.
$$

Changing residues and $D$ can therefore change Y stability while the S poles
stay fixed. On the frequency axis, where $I+S$ is invertible,

$$
\frac{Y+Y^H}{2}
=\frac{1}{z_0}(I+S)^{-H}(I-S^HS)(I+S)^{-1}.
$$

This relates scattering passivity to nonnegative admittance dissipation.
Singular conversions, including ideal lossless cases, need separate treatment.
A finite frequency grid doesn't establish these properties throughout the right
half-plane.

## Fixed-pole correction

### Real coefficient basis

The correction changes residues and $D$, leaving all poles fixed. For a
conjugate pair $p,p^*$ with residue $R=a+jb$,

$$
\frac{R}{s-p}+\frac{R^*}{s-p^*}
=a\left(\frac{1}{s-p}+\frac{1}{s-p^*}\right)
+b\,j\left(\frac{1}{s-p}-\frac{1}{s-p^*}\right).
$$

This gives a linear model in the real unknowns $a,b$. Real-pole and constant
basis functions complete the model. Only the upper triangle of each coefficient
matrix is fitted; mirroring it preserves reciprocity.

### Objective and enforcement grid

Let $S_0$ be the original fit and $\theta$ the new coefficients. The objective is

$$
\min_\theta\sum_i
\left\|S_\theta(j2\pi f_i)-S_0(j2\pi f_i)\right\|_F^2,
$$

subject to

$$
\sigma_{\max}(S_\theta(j2\pi g_\ell))\le\gamma,
\qquad \sigma_{\max}(D_\theta)\le\gamma.
$$

Here $f_i$ are training frequencies, $g_\ell$ are enforcement frequencies, and
$\gamma=0.999999$ by default. The margin keeps the target just inside the passive
boundary. The objective stays close to the original fitted response at the
training frequencies. It doesn't use holdout data, preserve DC exactly, or put
a hard bound on the final measured-data error. The higher-level `fit_model`
checks its requested error limits again after correction.

The default enforcement grid contains DC, the training frequencies, and 500
logarithmic points from $10^{-6}F$ to $10^3F$, where

$$
F=\max(f_{\max},\max_k|p_k|/(2\pi),1\,\mathrm{Hz}).
$$

The separate constraint on $D$ checks the limit at infinity. Intervals between
grid points still need testing.

### Numerical solution

For the complex training basis $B$, the code forms

$$
\begin{bmatrix}\operatorname{Re}B\\\operatorname{Im}B\end{bmatrix}=QR
$$

and uses $R^{-1}$ to rescale the coefficients. This avoids mixing residue
coefficients with very different scales. Off-diagonal entries have weight two
because they occur twice in the full Frobenius norm.

SciPy's SLSQP optimizer solves the quadratic objective with singular-value
constraints. For a simple singular value with left and right vectors $u,v$,

$$
d\sigma=\operatorname{Re}(u^H\,dS\,v)
$$

gives the constraint Jacobian. Frequency evaluations and SVDs are batched NumPy
operations. Repeated singular values are nonsmooth and can cause difficulty for
the optimizer. Although the finite-grid feasible set is convex, the numerical
solver can fail to converge.

A constant, zero-pole model uses a direct projection instead: diagonalize the
real symmetric $D$, clip its eigenvalues to $[-\gamma,\gamma]$, and reconstruct
it.

`converged` requires optimizer success and a maximum sampled singular value,
including $D$, no greater than $\gamma+10^{-9}$. The report always sets
`global_passivity_certified=False`.

## Testing between samples

The [enforcement benchmark](../benchmarks/fitting/bench_rational_enforcement.py)
adds scikit-rf's rational passivity test. Its half-size state-space eigenvalue
problem finds passivity-boundary crossings beyond the selected grid. See the
[implementation and references](https://scikit-rf.readthedocs.io/en/latest/_modules/skrf/vectorFitting.html#VectorFitting.passivity_test).
It is a numerical test, subject to floating-point limitations.

For each violation interval, the benchmark adds 31 frequencies and repeats the
correction against the original model. It allows eight attempts, raising an
error if optimization or final testing fails. This refinement loop belongs to
the benchmark; `enforce_s_passivity_numpy` doesn't run it automatically.

In the ring-slot experiment, the first 661-point grid passed, but the rational
test found a violation near 270–280 GHz. Expanding to 692 points removed that
violation in the rational test.

## Ring-slot benchmark

The experiment uses 160 training samples and 41 untouched holdout samples from
scikit-rf's ring-slot network, spanning 75–110 GHz.

| Metric | Original fit | Corrected model |
| --- | ---: | ---: |
| Poles, counting both members of each pair | 7 | 7 |
| Holdout normalized RMS error | $7.244\times10^{-7}$ | $2.473\times10^{-6}$ |
| Maximum holdout absolute S error | $1.302\times10^{-6}$ | $7.377\times10^{-6}$ |
| Rational passivity test | Fail | Pass |
| Largest real part of a Y pole, rad/s | $+7.738\times10^{12}$ | $-1.745\times10^6$ |
| Circulax circuit validation | Fail | Pass |

Normalized RMS is $\|S_{\mathrm{fit}}-S_{\mathrm{data}}\|_F/
\|S_{\mathrm{data}}\|_F$, summed over frequencies and ports. Correction increases
it from about 0.72 to 2.47 ppm, exceeding the original $8\times10^{-7}$ target.
The two corrections took roughly 0.3–0.52 seconds in local runs, excluding
fitting, rational testing, and circuit validation.

The original unstable Y mode has an e-folding time of about 0.129 ps. The
corrected slowest mode decays with a time constant of about 0.573 microseconds.
These are properties of the fitted realizations; the measurements don't
establish those time scales for the actual device outside the measured band.

The corrected $D_Y$ also has a large eigenvalue, about $4\times10^4$ S. An S
eigenvalue near $-1$ makes $(I+S)^{-1}$ large even on the passive side of the
boundary. Passing passivity therefore doesn't settle whether the extrapolated
admittance is physically reasonable. This experiment doesn't validate transient
waveforms or supply the missing DC-to-75-GHz data.

## Running the correction

To reproduce the benchmark from the repository root:

```bash
pixi run python -m benchmarks.fitting.bench_rational_enforcement
```

For direct use of the lower-level routine:

```python
from circulax.fitting import enforce_s_passivity_numpy

corrected, report = enforce_s_passivity_numpy(model, training_freqs)
if not report["converged"]:
    raise RuntimeError(report["optimizer_message"])
```

Before simulation, check rational passivity, the converted poles, and held-out
error. Test the intended source/load conditions and time-step convergence too.
If correction costs too much accuracy, revisit the fit or obtain wider-band
data. Leave passivity enforcement off for an intentionally active device.

For background on fitting and coefficient correction, see the
[SINTEF overview](https://www.sintef.no/en/software/vector-fitting/) and
[vector-fitting algorithm](https://www.sintef.no/en/software/vector-fitting/algorithm/).
Delay handling is described separately in the [fitting API](fitting_api.md) and
[delay-aware fitting plan](delay_aware_fitting_plan.md).
