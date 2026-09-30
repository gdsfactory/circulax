# Fixed Delay for Pole-Reduced S-Parameter Models

## Goal

Represent propagation delay exactly after it has been de-embedded from sampled
S-parameters, allowing the remaining response to use far fewer rational poles
without giving DC, transient, AC, and harmonic-balance solvers different physics.

## Scope

This work covers causal, parameter-dependent fixed delay. State-dependent delay,
baseband carrier shifting, asymmetric sidebands, and WDM simulation are explicitly
out of scope.

## Solver contract

For a delayed quantity `z(t) = x(t - tau)`, every analysis implements the same
relation in its natural representation:

| Analysis | Relation |
|---|---|
| DC | `Z - X = 0` |
| Transient | `Z - interp(X, t - tau) = 0` |
| AC | `Z - exp(-j 2 pi f tau) X = 0` |
| HB | `Z[k] - exp(-j 2 pi k f0 tau) X[k] = 0` |

The component declares its fixed delay inline with
`signals.at_delay(tau)`. The solver owns history interpolation or spectral
phase rotation. In particular, a query inside
the current transient step depends on the current Newton trial and must contribute
to the Jacobian.

At transient startup the DC operating point is used as constant prehistory. This
is consistent with the DC identity above and prevents artificial startup edges.

## S-parameter pole reduction

Delay de-embedding factors a measured or simulated network into

```text
S_full(f) = P(f) @ S_reduced(f) @ P(f)
P_ii(f) = exp(-j 2 pi f tau_i / 2)
```

`S_reduced` is fitted by a low-order rational state-space model. Each `P_ii` is
represented by a matched bidirectional `TransmissionLine` between the external
reference plane and port `i` of the reduced model. This construction is equivalent
to the frequency-domain embedding but also has a transient realization.

The transmission line uses incident-wave auxiliary unknowns. For a two-port line:

```text
b1(t) = attenuation * a2(t - tau)
b2(t) = attenuation * a1(t - tau)
V1 = a1 + b1                    I1 = (a1 - b1) / z0
V2 = a2 + b2                    I2 = (a2 - b2) / z0
```

This stamp remains finite for an exactly lossless line and avoids converting an
ideal-through S-matrix to a singular or extremely ill-conditioned Y-matrix.

## Acceptance criteria

- [x] Current-step interpolation has the analytic value and Jacobian for `tau < dt`.
- [x] Accepted-history interpolation has zero current-trial Jacobian.
- [x] Fixed delay is identity in both the DC residual and Jacobian.
- [x] HB rotates every retained harmonic by `exp(-j 2 pi k f0 tau)`.
- [x] Complex periodic delay uses a full FFT and does not assume conjugate symmetry.
- [x] The bidirectional wave-variable line reproduces its analytic two-port S-matrix.
- [x] Exact per-port lines plus a reduced rational core match direct frequency-domain delay embedding.
- [ ] Cross-solver AC/HB/transient agreement is verified end to end for the same delayed rational model.
- [ ] Delay and rational-model parameter gradients match finite differences across analyses.

## Test layout

- `tests/test_delay_contract.py`: independent delay, Jacobian, DC, AC, HB, and complex-spectrum contract tests.
- `tests/test_delay.py`: end-to-end transient interpolation and adaptive-step tests.
- `tests/test_rational.py`: reduced rational core plus exact reference-plane delay equivalence.

## Current limitations

- The stable fitting constructor assembles a rational core plus one
  `TransmissionLine` per delayed external port and returns a flattened `Circuit`.
  Lower-level rational factories are implementation APIs.
- Fixed-delay history currently uses a buffer sized by `max_steps`.
- State-dependent delays and discontinuity propagation are not implemented.
