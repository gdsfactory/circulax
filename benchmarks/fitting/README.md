# AAA backend comparison

Run from the repository root:

```bash
pixi run python -m benchmarks.fitting.bench_aaa_backends
```

The implementations are separate:

- `circulax/fitting/aaa_numpy.py`: original NumPy scalar AAA discovery from
  vfitax, without JAX operations. This remains the default.
- `circulax/fitting/aaa_jax.py`: fixed-capacity arrays, masked SVD, and a
  compiled adaptive loop. Returns padded arrays and an active support count;
  it can be called under `jit` and `vmap`.
- `circulax/fitting/aaa.py`: host adapter and shared pole extraction/realization.
  `aaa_scalar(..., backend="numpy" | "jax")` returns compact NumPy arrays.

`fit_with_delay(..., aaa_backend="numpy" | "jax")` selects discovery only.
Both options still use SciPy pole extraction and the existing JAX residue/VF
stages. Neither label means the full circuit fitting pipeline uses only that
backend. The original vfitax source already had this hybrid architecture.

The benchmark fits the same three reciprocal ring-slot responses, with the
same tolerance, capacity, and holdout. It excludes pole extraction, VF,
stabilization, passivity enforcement, and circuit conversion. JAX timings
wait for device completion and exclude initial input transfer. First-use
timings share a process; they are not isolated cold-start measurements.

On CPU (JAX 0.7.2, NumPy 2.5.0), one run measured:

| Discovery | First use | Warm median |
| --- | ---: | ---: |
| NumPy | 1.04 ms | 0.69 ms |
| JAX sequential | 156.66 ms | 0.74 ms |
| JAX batched | 133.46 ms | 0.95 ms |

All selected 6/5/6 support points and achieved approximately 8.96e-10
normalized error on held-out responses. These are barycentric approximations
of individual responses, not a shared stable/passive circuit model. Support
count is not circuit pole or state count.

NumPy is sufficient for this discovery workload. The JAX variant is retained
for experiments with repeated/batched inputs; no accelerator speedup has been
demonstrated. It recompiles when sample shape or support capacity changes.
Adaptive support choices are discrete, so this does not make topology search
differentiable end to end.

## Complete S-domain reduction workflow

```python
from circulax.fitting import fit_s_numpy

# Train samples only; choose the smallest shortlisted model meeting the
# training NRMSE/max-error limits after six VF relocation iterations.
model, metadata = fit_s_numpy(S_train, frequencies_train, screening="compact")
# Alternative: an extra candidate dimension and zero masks, using stacked SVD.
masked_model, masked_metadata = fit_s_numpy(
    S_train, frequencies_train, screening="masked"
)
```

`reduction_numpy.py` performs AAA discovery once, shared-pole residue fitting,
contribution-ranked screening of complete pole groups, and traditional relaxed
VF refinement entirely with NumPy/SciPy numerical operations. All returned
model arrays are NumPy arrays. The package still imports JAX, but the new
fitting routine does not dispatch JAX computations.

The masks are derived from the discovered order, not a prescribed final pole
count. Both screens solve for real coefficients of conjugate-pair basis
functions. The compact implementation uses one least-squares solve with
multiple right-hand sides per candidate; the masked implementation applies a
singular-value cutoff to padded systems rather than inverting zero singular
values. The latter compacts coefficients before VF refinement.

This is an experimental proper S-model workflow (constant term, no slope),
with uniform weighting and no delay extraction or passivity enforcement.
It preserves diagnostics about reflected AAA poles. It raises if refinement
cannot meet training thresholds. Screening is heuristic: candidates rejected
before relocation might have improved after relocation, so the selected order
is not a proven global minimum. Holdout and physical qualification remain
separate from order selection.

Run the full comparison:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 pixi run python -m benchmarks.fitting.bench_reduction_backends
```

Latest CPU run, medians of five warm repetitions, on the ring-slot training
split (160 samples; 41 frequencies reserved for final validation):

| Workflow | Warm time | Final pole order | Holdout NRMSE |
| --- | ---: | ---: | ---: |
| NumPy AAA + compact screening + VF | 6.34 ms | 4 | 3.648e-5 |
| NumPy AAA + masked screening + VF | 7.30 ms | 4 | 3.648e-5 |
| Same discovery + existing JAX screen + VF | 73.24 ms | 4 | 3.648e-5 |
| Existing two-call JAX notebook workflow | 238.47 ms | 4 | 3.648e-5 |
| scikit-rf VF, prescribed order four | 3.96 ms | 4 | 3.530e-5 |
| scikit-rf auto_fit, default stopping rule | 3.87 ms | 7 | 7.788e-7 |

The NumPy compact and masked fits and matched JAX fit all pass the same final
validation report after conversion to an admittance realization. This checks
held-out accuracy, reciprocity, stable admittance poles, sampled passivity,
asymptotic margins, and band coverage. It does not prove global passivity or
erase the intermediate AAA pole-reflection diagnostic.

For screening alone, compact NumPy took 1.09 ms and masked NumPy 0.77 ms in
this run; their response arrays agree to 1e-10 absolute tolerance. Across
runs, total NumPy workflow times varied from approximately 3 to 7 ms, and
masked screening did not consistently improve the full workflow. Keep the
compact version as the default pending larger workloads.

Timing scope matters: the first two rows end at a fitted S pole-residue model.
The JAX screen additionally computes candidate S-to-Y diagnostics; the notebook
workflow also repeats AAA discovery and performs full-model refinement and
S-to-Y conversion before its selected-model refit. Final validation is excluded
from every fitting timer. scikit-rf automatic order selection uses a different
stopping rule. First-use results share process caches and should not be read
as isolated cold-start comparisons. These measurements are illustrative CPU
results, not a general claim about larger networks or accelerators.

To hand the fitted model to Circulax explicitly:

```python
from circulax.fitting import scattering_state_space_to_admittance
from circulax.fitting.types import vfmodel_to_ss

scattering_ss = vfmodel_to_ss(model, model.D.shape[0])
admittance_ss, condition = scattering_state_space_to_admittance(scattering_ss, z0=50.)
# Apply held-out and physical validation before circuit simulation.
```

## Matching scikit-rf automatic-fit accuracy

```python
model, metadata = fit_s_numpy(
    S_train, frequencies_train,
    reduction_stage="refined",
    normalized_rmse=8e-7,
    max_absolute_error=1e-5,
)
```

This optional strategy refines the full AAA topology first, then enumerates
contribution-ranked budgets and refines them in ascending order. Full-model
VF can turn complex pairs into real poles, exposing odd-order candidates.
It does not reject candidates merely because their fixed-pole screening
error exceeds the final threshold. Training error alone selects the order.

Run `pixi run python -m benchmarks.fitting.bench_accuracy_target` to reproduce
the accuracy comparison and report the separate circuit qualification.
One CPU run selected seven poles automatically, with training NRMSE 7.065e-7
and holdout NRMSE 7.244e-7, versus scikit-rf auto_fit's seven poles and
7.788e-7 holdout NRMSE. Warm times were approximately 12.4 ms and 2.5 ms,
respectively. Targets and algorithms differ; neither run uses holdout for
order selection. These are experiments on this dataset, not a generalization
claim beyond it.

**The seven-pole NumPy model fails circuit qualification.** Its S poles are
stable and its sampled band is passive, but its exact Y realization has a
right-half-plane pole (approximately +7.74e12 rad/s) and a negative D margin.
Matching frequency-fit accuracy alone does not make this a simulation-ready
replacement for the four-pole model. The benchmark prints this failure;
passivity constraints and/or a different realization need further work.

### DC extrapolation experiment

Using scikit-rf 1.12.0, `network[training_mask].extrapolate_to_dc()` gives
the following real DC S matrix for the ring-slot training split:

```text
[[2.31900087, 0.29630126],
 [0.29630126, 3.37576046]]
```

Its largest singular value is 3.453, so this extrapolated endpoint is not
passive. The method estimates the endpoint from the first two training
frequencies and forces zero imaginary part; it does not enforce passivity.
The extrapolation spans the large unmeasured gap from about 75 GHz to DC.

To preserve the original measured samples and holdout, the experiment used
only the extrapolated endpoint, not the method's resampled frequency grid:

```python
training = network[training_mask]
extended = training.extrapolate_to_dc()
frequencies_augmented = np.r_[0., training.f]
S_augmented = np.concatenate([extended.s[:1], training.s])
model, metadata = fit_s_numpy(
    S_augmented, frequencies_augmented, reduction_stage="refined"
)
```

At the default 2% training threshold this selected four poles and gave
0.195% holdout NRMSE, but the Y realization still had an unstable pole at
approximately +1.18e10 rad/s. At the tighter 8e-7 training target, no candidate
met the accuracy limits. The NumPy workflow now accepts a leading DC sample;
negative or non-increasing frequencies remain invalid.

Real DC behavior alone is insufficient: the existing conjugate-pole,
real-coefficient models already have real DC responses. A trusted physical
DC endpoint or measured low-frequency data could help, but this default
extrapolated endpoint is unsuitable as a passive ring-slot constraint.

## Vectorized NumPy sample conditioning

```python
from circulax.fitting import condition_sparameters

# Input grid must start at DC and be uniformly spaced for the FFT projection.
cleaned, report = condition_sparameters(S_grid, freqs_grid, max_iterations=500)
if not report.converged:
    raise RuntimeError("Requested sample constraints did not converge")

# On a measured band without DC, disable the FFT constraint explicitly.
band_cleaned, band_report = condition_sparameters(S_train, freqs_train, causality=False)
```

`conditioning_numpy.py` is an independent mathematical implementation, not a
copy of SignalIntegrity source. It uses stacked SVD to clip singular values,
transpose averaging for reciprocity, and FFTs across all port responses to
remove negative-lag impulse samples. Optional leading batch dimensions are
supported. Input arrays are not modified.

The FFT projection uses an odd-length conjugate extension: DC becomes real,
but the top frequency bin is not artificially forced real as an even-length
Nyquist bin would be. Negative time is interpreted in the centered periodic
FFT window. Consequently this is a finite-grid causality proxy, sensitive to
bandwidth, frequency spacing and extrapolation; it is not a proof of physical
causality for a measured band or a rational extrapolation.

`preserve_dc=True` keeps a real, feasible original DC matrix using an additive
correction distributed over retained impulse taps. It rejects a DC matrix
incompatible with requested passivity or reciprocity. It cannot preserve the
non-passive scikit-rf DC estimate and also enforce passivity. With DC
preservation disabled, the routine may change the DC endpoint substantially.

Every iteration checks all requested constraints on the returned samples.
The report gives convergence, iteration count, maximum singular value,
reciprocity error, negative-time relative norm, DC preservation error and
the size of the data correction. Hitting the iteration limit returns
`converged=False`. A successful report certifies only these sampled checks.
Passivity clipping assumes power-normalized S parameters.

Reproduce the fitting experiment:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 pixi run python -m benchmarks.fitting.bench_conditioning
```

Ring-slot findings (all extrapolation uses training data only):

| Input treatment | Fitted order | Original holdout NRMSE | Circuit gate |
| --- | ---: | ---: | --- |
| Original band, tight fitting target | 7 | 7.24e-7 | Fail: unstable Y, asymptotic passivity |
| Band passivity/reciprocity projection, tight target | 7 | 7.24e-7 | Same failure |
| Conditioned DC endpoint + original band, default target | 4 | 2.57e-3 | Fail: unstable Y, passivity |
| Full conditioned extrapolation, default target | 13 | 3.22e-1 | Fail: original accuracy, unstable Y, passivity |

The original band changes by only roundoff under passivity projection. The
scikit-rf extrapolated grid, however, needs approximately 65% relative
correction. Alternating projections converge in 27 iterations, with final
maximum singular value 1.0000000063 (within the 1e-8 tolerance). A successful
sample-grid projection does not survive unconstrained rational fitting as a
global passivity/stability guarantee.

On one CPU run the stacked passivity projection took 1.09 ms versus 5.67 ms
for an equivalent per-frequency loop on the 590-point two-port grid; the
entire alternating conditioning took about 56 ms. These timings exclude
fitting and are not accelerator benchmarks.

Conclusion: this provides useful explicit data conditioning, but it does not
solve the seven-pole model's extrapolation problem. Enforcement on the fitted
rational model and validation after S-to-Y conversion remain necessary.
# Rational-model enforcement experiment

Run `pixi run python -m benchmarks.fitting.bench_rational_enforcement`.
`enforce_s_passivity_numpy` is an experimental NumPy/SciPy fixed-pole
constrained refit: it minimizes the change to the original fitted S response
at training frequencies, changing real conjugate-basis residues and D while
preserving reciprocity. It constrains singular values at DC, a broad frequency
grid, and infinity. It does not use extrapolated samples or copy SignalIntegrity
implementation code. The function reports finite-grid convergence only, not
global passivity. Failed optimization is returned explicitly in its report.

The benchmark additionally uses scikit-rf's rational passivity test to locate
remaining violation bands and refine the grid, always refitting against the
original model. The first 661-point grid misses a narrow violation near
270--280 GHz; one refinement to 692 points removes it in that numerical test.
No measured holdout samples enter enforcement.

Ring-slot result (same 160 training / 41 holdout split):

| Metric | Before | After |
| --- | ---: | ---: |
| Full pole count | 7 | 7 |
| Holdout normalized RMS | 7.244e-7 | 2.473e-6 |
| Holdout maximum absolute S error | 1.302e-6 | 7.377e-6 |
| scikit-rf rational passivity test | Fail | Pass |
| Maximum real Y pole (rad/s) | +7.738e12 | -1.745e6 |
| Circulax circuit validation | Fail | Pass |

The two constrained corrections took approximately 0.3--0.52 seconds in local
runs, excluding AAA/VF fitting, the independent oracle, and circuit validation.
This is an accuracy/admissibility experiment, not a speed improvement. The
corrected error is about 3.4 times the original and exceeds the original 8e-7
accuracy target. Its large Y feedthrough eigenvalue (~4e4 S) also warns against
interpreting passivity as evidence of physically accurate extrapolation.
The default production fitting path is unchanged; independent rational testing
and application-specific accuracy limits remain necessary.
