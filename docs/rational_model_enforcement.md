# Rational-model passivity enforcement: an engineering explanation

## What problem are we solving?

A rational model can match measured S-parameters extremely well and still be
unsafe as a passive component in a transient circuit simulation. Accuracy over
the measured band, stable poles, and passivity are different requirements.

Our ring-slot experiment illustrates this distinction: the original fitted
**S model had stable poles**, but its conversion to an admittance model produced
a right-half-plane pole. We corrected the rational S-model coefficients, leaving
its seven poles fixed. The converted Y model then had stable poles and passed
the circuit validation checks, at a small cost in measured-band accuracy.

This note describes the experimental implementation in
[`enforcement_numpy.py`](../circulax/fitting/enforcement_numpy.py).
Here, “normal practice” means conventional stable vector fitting followed by
passivity assessment and, when needed, coefficient correction. It does not mean
that ordinary vector fitting automatically guarantees passivity. SINTEF also
separates fitting and passivity enforcement into distinct stages.
([SINTEF overview](https://www.sintef.no/en/software/vector-fitting/))

## 1. The model and its physical requirements

We use the Laplace convention in which a mode evolves as $e^{pt}$ and evaluate
frequency responses at $s=j2\pi f$. For an $n$-port proper rational S model,

$$
S(s)=D+\sum_{k=1}^{K}\frac{R_k}{s-p_k}.
$$

$p_k$ are poles in radians per second; $R_k$ are $n\times n$ residue matrices;
$D$ is the dimensionless high-frequency limit. “Proper” here excludes a
proportional term $sE$: the implemented model has $E=0$.

The requirements serve different purposes:

| Property | Mathematical condition | What it means |
| --- | --- | --- |
| Real time-domain response | Real poles/residues, or conjugate pole/residue pairs; real $D$ | Real excitation produces a real response |
| Reciprocity | $S(s)=S(s)^T$ | Interchanging ports preserves transfer response |
| Strict pole stability | $\operatorname{Re}(p_k)<0$ | Individual causal modes decay |
| Scattering passivity | $I-S(j\omega)^H S(j\omega)\succeq0$ at every frequency, together with stable causal analyticity | The component cannot supply net energy from zero initial energy |
| Fit accuracy | Small error against measured data | The model represents the available measurements |

$H$ denotes conjugate transpose; $T$ denotes transpose without conjugation.
Reciprocity is not a requirement for passivity in general, but this implementation
requires reciprocal models. Strict stability also excludes ideal undamped
lossless modes on the imaginary axis, which require separate treatment.

For power-normalized waves with real positive reference impedances, let incident
and outgoing waves satisfy $b=Sa$. Their power difference is proportional to

$$
a^H a-b^H b=a^H(I-S^H S)a.
$$

Nonnegative absorbed power for every excitation therefore requires

$$
\sigma_{\max}(S(j\omega))\le1.
$$

Checking each element $|S_{ij}|\le1$ is insufficient: coherent excitation of
multiple ports can violate the matrix singular-value condition. Reality at DC
is also insufficient; a real matrix can have singular values greater than one.

## 2. What unstable poles do in the time domain

For a causal realization, the impulse response is

$$
h(t)=D\delta(t)+\sum_k R_k e^{p_k t}u(t).
$$

Writing a pole as $p=\alpha+j\beta$, its amplitude envelope is $e^{\alpha t}$:

- $\alpha<0$: decaying transient, with decay time $1/|\alpha|$.
- $\alpha=0$: persistent mode; not asymptotically stable.
- $\alpha>0$: exponential growth, with growth time $1/\alpha$.

A complex-conjugate pair produces an oscillation at $|\beta|/(2\pi)$ Hz with
that envelope. A positive real pole produces nonoscillatory exponential growth.

Equivalently, in state space,

$$
\dot x=Ax+Bu,\qquad y=Cx+Du,
$$

$$
x(t)=e^{At}x(0)+\int_0^t e^{A(t-\tau)}Bu(\tau)\,d\tau.
$$

An unstable mode can be excited by the input, an initial condition, or numerical
roundoff. Possible symptoms include growing ringing, current or voltage runaway,
time-step collapse, nonlinear-solver failures, or overflow. Nonlinear limits in
the surrounding circuit may instead turn growth into a spurious oscillation.
These are possible consequences, not transient results measured in our benchmark.

Exact unobservable or uncontrollable modes can cancel from an input-output
transfer function. Such cancellations do not justify retaining an internally
unstable numerical realization: imperfect cancellation can expose the mode.

### Why an excellent AC fit can hide the problem

Evaluating a rational expression on $s=j\omega$ does not test whether all its
natural responses decay. An unstable rational expression can be finite and
closely match data at every measured frequency. For a causal unstable system,
that evaluation is not evidence of an attainable stable sinusoidal steady state.

Finite bandwidth adds another blind spot: a narrow out-of-band violation may
barely affect measured error, while still changing the model's transient modes.
An inverse FFT of finite-band samples is a periodic, band-limited reconstruction,
not a substitute for checking the poles of the causal rational realization.

### Why a smaller time step is not a repair

For $\dot x=\alpha x$, exact sampling gives

$$
x_{m+1}=e^{\alpha\Delta t}x_m.
$$

If $\alpha>0$, its amplification exceeds one for every positive time step.
Reducing the step resolves the growth more accurately; it does not remove it.
Strong numerical damping can sometimes conceal instability, but that changes
the computed dynamics rather than repairing the component model.

Not every nonpassive model will blow up in every circuit. A stable active model
can work with some terminations and destabilize others. Conversely, a passive
component does not guarantee convergence of every numerical solver or stability
of an external active circuit. Passivity is an energy property, not a blanket
simulation guarantee. SINTEF specifically identifies pole stability and
passivity as separate requirements for passive time-domain macromodeling.
([SINTEF passivity guidance](https://www.sintef.no/en/software/vector-fitting/passivity/))

## 3. Why stable S poles can become unstable Y poles

For equal real reference impedance $z_0$, conversion is

$$
Y(s)=\frac{1}{z_0}[I-S(s)][I+S(s)]^{-1}.
$$

The inverse introduces potential poles wherever

$$
\det[I+S(s)]=0,
$$

unless cancellation removes them. These need not be poles of $S$.

A one-port counterexample makes this concrete. Let $\tau>0$ and

$$
S(s)=\frac{-2}{1+s\tau}.
$$

Its S pole is stable, $p=-1/\tau$, but it is nonpassive near DC. Conversion gives

$$
Y(s)=\frac{1}{z_0}\frac{s\tau+3}{s\tau-1}
=\frac{1}{z_0}+\frac{4}{z_0\tau}\frac{1}{s-1/\tau}.
$$

The voltage-driven admittance has a growing mode $e^{t/\tau}$. Conversion did not
create a new physical component: it changed which terminal variables are inputs
and outputs, exposing a problem with the fitted terminal relation.

For the S realization $S=D+C(sI-A)^{-1}B$, define $M=(I+D)^{-1}$. The code uses

$$
A_Y=A-BMC,\quad B_Y=BM/\sqrt{z_0},\quad
C_Y=-2MC/\sqrt{z_0},\quad D_Y=(I-D)M/z_0.
$$

Thus changing S residues and $D$ can stabilize $A_Y$ even while $A$ stays fixed.

A globally bounded-real, strictly contractive S model prevents an eigenvalue
$-1$ in the right half-plane and supports a well-defined passive Y conversion.
Boundary singularities and ideal lossless cases need additional care. On the
frequency axis, whenever $I+S$ is invertible,

$$
\frac{Y+Y^H}{2}
=\frac{1}{z_0}(I+S)^{-H}(I-S^HS)(I+S)^{-1}.
$$

This identity explains the connection between scattering passivity and
nonnegative admittance dissipation. A finite-frequency sample check alone does
not establish the required right-half-plane properties.

## 4. The implemented coefficient correction

### Fixed poles and real coefficients

The function accepts a stable, proper, reciprocal, real-rational S model. It
**rejects unstable input S poles**; it is not a pole-stabilization routine.
It adjusts residues and $D$, without discovering, moving, or removing poles.

For a pair $p,p^*$ and residue $R=a+jb$, its contribution can be written

$$
\frac{R}{s-p}+\frac{R^*}{s-p^*}
=a\left(\frac{1}{s-p}+\frac{1}{s-p^*}\right)
+b\,j\left(\frac{1}{s-p}-\frac{1}{s-p^*}\right).
$$

The unknowns $a,b$ are real. Along with real-pole basis functions and a constant
basis function, this builds a linear model in real coefficients. Only the upper
triangle of each coefficient matrix is independent; mirroring it preserves
reciprocity. Conjugacy ensures a real causal impulse response and real DC.

### Optimization objective and constraints

Let $S_0$ be the original fit, $\theta$ the new coefficients, $f_i$ the training
frequencies, and $g_\ell$ the enforcement frequencies. We solve

$$
\min_\theta\ \sum_i
\left\|S_\theta(j2\pi f_i)-S_0(j2\pi f_i)\right\|_F^2
$$

subject to

$$
\sigma_{\max}(S_\theta(j2\pi g_\ell))\le\gamma,
\qquad \sigma_{\max}(D_\theta)\le\gamma,
$$

with default $\gamma=0.999999$. The small margin keeps the target inside the
passive boundary. The objective preserves the **original fitted response**, not
new extrapolated data. Measured holdout samples are not used in optimization.
It does not explicitly preserve the original DC value or impose a hard bound on
the final measured-data error.

The default grid includes DC, training frequencies, and 500 logarithmic points
from $10^{-6}F$ to $10^3F$, where
$F=\max(f_{\max},\max_k|p_k|/(2\pi),1\,\mathrm{Hz})$.
The separate $D$ constraint checks the exact limit at infinity, not just a very
high frequency. The finite grid and this endpoint still leave untested intervals.

### Numerical solution

If $B$ is the complex training basis, the code computes

$$
\begin{bmatrix}\operatorname{Re}B\\\operatorname{Im}B\end{bmatrix}=QR
$$

and changes coordinates using $R^{-1}$. This whitens the least-squares objective,
which otherwise mixes residue coefficients with very different scales. In these
coordinates the objective is a weighted squared coefficient distance. Off-diagonal
entries have weight two because they occur twice in the full Frobenius norm.

SciPy's SLSQP optimizer uses that quadratic objective and singular-value
constraints. For a simple singular value with left/right vectors $u,v$,

$$
d\sigma=\operatorname{Re}(u^H\,dS\,v),
$$

which gives the analytic constraint Jacobian. Frequency evaluations and SVDs
are batched NumPy operations. Repeated singular values are nonsmooth, so the
Jacobian and optimizer deserve care near degeneracies. The finite-grid feasible
set is convex in the real coefficients, but this general-purpose numerical
solver is not a formal certificate and may report failure.

For a zero-pole constant reciprocal model, the implementation instead uses the
exact static projection: diagonalize real symmetric $D$, clip its eigenvalues
to $[-\gamma,\gamma]$, and reconstruct it. No iterative optimizer is needed.

The returned `converged` flag requires optimizer success and a maximum sampled
singular value, including $D$, no larger than $\gamma+10^{-9}$.
`global_passivity_certified` remains `False`. A returned model must not be treated
as qualified simply because the function returned without an exception.

## 5. Rational testing versus sampled testing

The benchmark performs an additional scikit-rf rational passivity test, based on
a half-size state-space eigenvalue problem that identifies passivity-boundary
crossings. This checks the rational function beyond the chosen grid. It is a
numerical test with floating-point assumptions, not a formal proof.
([scikit-rf implementation and references](https://scikit-rf.readthedocs.io/en/latest/_modules/skrf/vectorFitting.html#VectorFitting.passivity_test))

When that test finds a violation interval, the benchmark adds 31 points across
it and repeats the constrained correction against the **original** model. It
allows up to eight attempts and raises an error if optimization or final testing
fails. This oracle-driven refinement is in the benchmark, not automatically
inside `enforce_s_passivity_numpy`.

The ring-slot case needed this distinction: its first 661-point grid passed,
but the rational test found a remaining violation near 270--280 GHz. Refining to
692 points removed the violation in the independent numerical test.

## 6. How this differs from other approaches

| Approach | What changes | What it does not establish by itself |
| --- | --- | --- |
| AAA discovery and vector fitting | Pole locations and fitted residues | Passivity of the resulting rational model |
| Pole reflection into the left half-plane | Unstable poles, followed by a residue refit | Energy conservation or stable conversion to another domain |
| SVD clipping of sampled S data | Singular values at individual frequency samples | One causal, passive rational function interpolating those samples |
| FFT-based sample conditioning | A sampled time/frequency representation | Passivity and stable poles of a subsequent rational fit |
| This fixed-pole correction | Rational residues and $D$ | Global passivity without further testing, or trustworthy extrapolation |

Coefficient perturbation after fitting is established engineering practice,
not a benefit unique to AAA or JAX. The experimental choices here are a
NumPy/SciPy constrained solver, a fixed-pole real basis, a measured-band
perturbation objective, and a separate rational-test-driven refinement loop.
Traditional VF is itself a pole-relocation method and can enforce pole stability
during fitting. ([SINTEF algorithm](https://www.sintef.no/en/software/vector-fitting/algorithm/))

The implementation is independent of SignalIntegrity's source code. Its role
in the earlier investigation was mathematical reference, not code reuse.

## 7. Ring-slot result and its limits

The reproducible experiment uses 160 measured training samples and 41 untouched
holdout samples from scikit-rf's ring-slot network, spanning 75--110 GHz.

| Metric | Original fit | Corrected rational model |
| --- | ---: | ---: |
| Full pole count, counting both members of a pair | 7 | 7 |
| Holdout normalized RMS error | $7.244\times10^{-7}$ | $2.473\times10^{-6}$ |
| Maximum holdout absolute S error | $1.302\times10^{-6}$ | $7.377\times10^{-6}$ |
| Rational passivity test | Fail | Pass |
| Largest real part of a Y pole, rad/s | $+7.738\times10^{12}$ | $-1.745\times10^6$ |
| Circulax circuit validation | Fail | Pass |

Normalized RMS means $\|S_{\mathrm{fit}}-S_{\mathrm{data}}\|_F/
\|S_{\mathrm{data}}\|_F$, aggregating frequencies and ports. The errors above are
approximately 0.72 and 2.47 ppm: still small, but correction worsens the error by
about 3.4 times and exceeds the original $8\times10^{-7}$ target.

The original positive Y pole corresponds to an e-folding time of about
**0.129 ps**, if excited in that realization. The corrected slowest decay mode
has a time constant of about **0.573 microseconds**. Neither number is evidence
that the actual ring-slot device has those dynamics outside the measured band.

The two corrections took approximately **0.3--0.52 seconds** in local runs,
excluding initial fitting, the rational-test oracle, and circuit validation.
This is not a speed improvement over the earlier millisecond-scale fitting.

There is also a significant extrapolation warning: the corrected $D_Y$ has a
large eigenvalue, approximately $4\times10^4$ S. An S eigenvalue near $-1$ makes
$(I+S)^{-1}$ large, even on the passive side of the boundary. Thus passivity can
coexist with extreme admittance and poorly justified out-of-band behavior.
DC-to-110-GHz validation of model properties does not supply missing DC-to-75-GHz
measurement evidence. We have not demonstrated transient waveform accuracy.

## 8. How to use the result

Run the full experiment from the repository root:

```bash
pixi run python -m benchmarks.fitting.bench_rational_enforcement
```

The standalone API is opt-in:

```python
from circulax.fitting import enforce_s_passivity_numpy

corrected, report = enforce_s_passivity_numpy(model, training_freqs)
if not report["converged"]:
    raise RuntimeError(report["optimizer_message"])
# Still perform independent rational passivity testing, converted-model checks,
# and held-out accuracy assessment before circuit use.
```

For a passive circuit component, acceptance should include stable poles in the
actual realization used by the simulator, rational passivity assessment,
acceptable held-out error, and physically credible behavior over the bandwidth
excited by the transient. Test representative terminations and time-step
convergence as well. If correction is too costly in accuracy or extrapolation,
reconsider model order or obtain wider-band data rather than hiding the issue
with numerical damping. Do not enforce passivity on an intentionally active
device without changing the intended modeling problem.
