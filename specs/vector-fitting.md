# Vector Fitting of S-Parameters with Delay De-Embedding

## Goal

Convert frequency-domain S-parameter data into simulation-ready circulax components
via rational approximation (AAA method), with group-delay de-embedding to reduce
pole count for transmission-line-like data.

## Context

S-parameter data (from measurement or EM simulation) describes a linear N-port over
frequency. To simulate this network in transient (time domain), the data must be
converted to a rational transfer function that maps onto circulax's DAE formulation.

For transmission lines, the raw S-parameters contain a large linear phase ramp from
propagation delay. Fitting this directly requires 40-100+ poles. By de-embedding the
group delay first, the smooth remainder fits with ~5 poles — a massive reduction that
shrinks the Jacobian (cost scales as O(poles²)) and improves numerical conditioning.

The fitting engine is **vfitax** (separate repo, branch `scipy-aaa`). Circulax
consumes the fitted `SSModel` to create simulation-ready components.

### Key formulas

| Formula | Description |
|---------|-------------|
| `H(s) = C·diag(1/(s-A))·B + D + s·E` | State-space transfer function |
| `τ = -dφ(S21)/dω` | Group delay extraction (LS phase slope) |
| `S' = P·S·P` where `P = diag(exp(+jωτ/2))` | Reference-plane de-embedding |
| `S_full = P·S·P` where `P = diag(exp(-jπfτ))` | Reference-plane re-embedding |
| `Y = (I-S)(z0·S + z0*·I)⁻¹` | Kurokawa S-to-Y conversion |

### Design decisions

| Decision | Rationale |
|----------|-----------|
| AAA method (not VF iteration) | Automatic pole selection, fast, no initial pole guess needed |
| De-embed before fitting | Removes linear phase → fewer poles, better conditioning of `s_to_y` |
| Passivity on de-embedded Y | `σ_max(PSP) = σ_max(S)` for unitary P, so passivity transfers to full cascade |
| `is_complex=True` required | Real-pair block form (`_ss_to_real_pairs`) not yet implemented; all rational circuits use complex-doubled path |

---

## Acceptance criteria

- [x] `fit_with_delay` returns SSModel + tau with ≥5× pole reduction vs raw fit
- [x] Round-trip: `embed(deembed(S)) == S` to machine precision
- [x] Over-estimated delay triggers pole-flip tripwire
- [x] `rational_component` AC sweep matches `rational_fdomain_component` to 1e-8
- [x] AC sweep Y matches independent pole-residue oracle to 1e-6
- [x] DC response: `i_port = H(0) @ v` from consistent (v, x) state
- [x] Transient: step response settles to correct divider voltage (rtol=0.01)
- [x] `rational_delay_component` S21 group delay matches τ (rtol=0.03)
- [x] `rational_delay_component` with τ=0 matches `rational_fdomain_component`
- [x] Asymmetric delay: S11/S22 group delays match τ1/τ2 independently

---

## Components

| Component | Description | Repo | Depends on |
|-----------|-------------|------|------------|
| S-param pipeline | `extract_group_delay`, `deembed_delay`, `embed_delay`, `fit_with_delay` | vfitax | — |
| `rational_component` | Time-domain DAE component from SSModel | circulax | S-param pipeline |
| `rational_fdomain_component` | Fdomain oracle from SSModel (test reference) | circulax | — |
| `rational_delay_component` | Fdomain composite: delay + rational cascade | circulax | `rational_component` |
| `TransmissionLine` | Exact solver-independent reference-plane delay | circulax | fixed-delay solver contract |

---

## Delegation map

| Component | File scope | Constraints |
|-----------|------------|-------------|
| S-param pipeline | `vfitax/sparam.py`, `vfitax/tests/unit/test_sparam.py` | Duplicate `s_to_y` (5 lines) to avoid cross-dep |
| Rational factories | `circulax/components/rational.py`, `tests/test_rational.py` | Must use `is_complex=True`; `z0` not differentiable |
| Exports | `circulax/__init__.py`, `circulax/components/__init__.py` | — |

---

## Implementation notes

### SSModel → DAE stamp

The state-space model maps directly to circulax's `F(y) + dQ/dt = 0`:

```
Port i:   f[port_i] = (C @ x + D @ v)_i       q[port_i] = (E @ v)_i
State j:  f[x_j]    = -(A_j·x_j + (B @ v)_j)  q[x_j]    = x_j
```

The linearized `G + jωC` block is `[[D+jωE, C], [-B, jωI-A]]`, whose Schur
complement gives exactly `H(jω)`.

### Port termination subtraction

AC sweep Y = s_to_y(S, z0) includes port shunt resistors. To recover the DUT's
Y-parameters: `Y_DUT = Y_circuit - I/z0`. This is needed for oracle comparison
tests but not for normal simulation.

### circulax `s_to_y` default z0

circulax uses `z0=1.0+1e-12j` (photonic convention) while vfitax uses `z0=50.0`.
Explicit `z0=50.0` is required when converting electrical S-parameters.

### Delay over-estimation guard

If the LS phase slope over-estimates τ (e.g. from dispersion), the de-embedded
remainder approximates `exp(+jωδ)` — a non-causal advance. AAA fits this with
RHP poles, and `_collect_poles` flips them to LHP, destroying the fit. The
`fit_with_delay` metadata includes `pole_flips` count as a tripwire.

### Solver-independent delay realization

`rational_delay_component` remains the direct AC/HB oracle. For DC, transient,
AC, and HB agreement, place a matched `TransmissionLine(tau=tau_i/2)` between
each external port and port `i` of `rational_component`. This realizes
`S_full = P @ S_reduced @ P` exactly while retaining the pole-count reduction
from fitting only the de-embedded response. The line uses delayed wave-state
constraints, so an ideal lossless through path never needs a singular S-to-Y
conversion.

### Active and nonreciprocal networks

Use `reciprocal=False`, `enforce_passive=False` for active devices. The default
`delay_mode="auto"` disables reference-plane delay extraction in this case,
because one per-port factor imposes the same transmission delay in both
directions. Request `delay_mode="port"` only when that physical assumption is
appropriate.

Validation must set `expected_reciprocal=False, expected_passive=False` for
such models. This permits intentional gain and directionality, but still
requires bounded fit and holdout error, stable poles, non-negative delay,
finite evaluation, and coverage of the requested simulation band. Inspect
each ordered S-parameter independently so forward gain cannot hide a poor
reverse fit.

The 190 GHz transmitter benchmark uses `fit_domain="s"`: AAA initializes the
topology, the strongest conjugate pole pairs are refined against every ordered
S response, and an exact state-space S-to-Y transformation produces the
simulation model. With `max_poles=20`, the benchmark reaches 1.26% normalized
complex error without requiring scikit-rf at runtime.

### Vmapped pole-count screening

`pole_count_candidates=(...)` retains a fixed AAA pole shape and zeroes
coefficient columns for excluded real poles or complete conjugate pairs.
Nested `vmap` operations refit every ordered S response for every mask, then
reconstruct and score fixed-shape S-to-Y realizations. Shortlisted candidates
must still be compacted, refined, and fully validated.

### Real-pair block form (future)

`_ss_to_real_pairs` would convert conjugate pole pairs to 2×2 real blocks,
halving system size for electrical circuits. Currently all rational circuits
pay 2× via `is_complex=True`. Documented as a future optimization.

---

## Implementation files

| File | Repo | Action |
|------|------|--------|
| `circulax/fitting/sparam.py` | circulax | Delay de-embedding and S-parameter fit pipeline |
| `tests/fitting/test_sparam.py` | circulax | Round-trip, pole reduction, tripwire tests |
| `circulax/components/rational.py` | circulax | New — SSModel → component factories |
| `tests/test_delay_contract.py` | circulax | New — analysis-independent fixed-delay contract tests |
| `tests/test_rational.py` | circulax | Updated — includes exact-line/reduced-core equivalence |
| `circulax/components/__init__.py` | circulax | Updated exports |
| `circulax/__init__.py` | circulax | Updated imports |

---

## Verification

```bash
# fitting and time-delay integration
pytest tests/fitting -v
pytest tests/test_delay_contract.py -v
pytest tests/test_delay.py -v
pytest tests/test_fdomain.py tests/test_rational.py -v
```
