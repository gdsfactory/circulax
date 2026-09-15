# Delay-aware fitting validation

The deterministic fixture has a stable passive one-pole core and 0.03 seconds
of one-way delay per port. Both fits use conventional scikit-rf at NRMSE <= 1e-4
and maximum absolute S error <= 1e-3. The delayed branch reduces contribution-ranked
pole subsets and refits residues; the baseline retains scikit-rf automatic order.
Timings below are a single sequential local run, not portable performance guarantees.
Environment: Python 3.13.14, JAX 0.7.2, NumPy 2.5.0, SciPy 1.18.0,
scikit-rf 1.12.0.

| Measurement | Rational only | Explicit delays |
| --- | ---: | ---: |
| Core poles | 9 | 2 |
| Core states | 18 | 4 |
| Circuit unknowns | 23 | 15 |
| Real solver unknowns | 46 | 30 |
| Line algebraic unknowns | 0 | 4 |
| History allocation (bytes, 1000-step budget) | 0 | 248248 |
| Training NRMSE | 2.21407e-07 | 8.63172e-16 |
| Sampled core maximum singular value | 0.4 | 0.4 |
| Fit (s) | 0.00399323 | 0.00414938 |
| Enforcement (s; disabled) | 0 | 0 |
| Conversion (s) | 0.287869 | 0.00905489 |
| Parent compilation (s) | 0.665169 | 0.0275873 |
| Warm coefficient evaluation (s) | 2.23717e-05 | 1.96652e-05 |
| Cold transient including JIT (s) | 1.15523 | 1.71521 |
| Warm transient (s) | 0.00983087 | 0.00688184 |
| Transient steps | 200 | 200 |

The known core has one pole, but numerical fitting/reduction retained two.
The state count includes repeated per-port rational states. History allocation
is `(max_steps + 1) * (real_solver_unknowns + 1) * 8` bytes for float64 state
and timestamps, excluding solver workspaces and adjoint storage. Warm transient
measurements reuse the same jitted callable and synchronize device completion.
No competing benchmark workloads were run concurrently.

Reproduce with `pixi run python -m benchmarks.fitting.bench_delay_separation`.
The output contains the complete machine-readable measurements. Cold conversion
and compilation include initialization overhead and must not be treated as a
fair warm comparison. Automatic inference remains optional and heuristic;
stability and sampled passivity are not a causality certificate.

## Solver checks and numerical limits

Regression fixtures cover loaded nonzero DC, analytic AC transmission/reflection
phase, source-driven HB, round-trip reflected transients, and gated-sine transient propagation for lossless and
attenuating static cores. Unequal delays, nested/repeated models, active supplied
cores, fixed/adaptive stepping and sub-step delays are covered. The complex-state
path uses explicit mode selection or the usual automatic inference.

At a sub-step turn-on the first interpolation interval is allowed 1 mV absolute
error for a 1 V source; later samples use 0.5 mV. Resolved-delay pre-arrival
samples use 0.1 microvolt tolerance. These are numerical fixture tolerances,
not a guarantee of exact onset between accepted time steps. Prehistory is the DC
operating point. Existing shared-delay tests exercise delay gradients and history
interpolation; no differentiable fitting or automatic inference is advertised.

The combined fitting notebook executed 27 code cells in a fresh kernel without
errors; source outputs were stripped after inspection.
