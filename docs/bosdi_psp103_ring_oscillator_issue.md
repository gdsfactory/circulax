# PSP103 ring-oscillator frequency mismatch vs. VACASK

**Audience:** bosdi author.
**Status:** open — circulax's ring-oscillator transient runs ~6.7× slower than
VACASK's on the same `psp103v4_psp103.osdi` binary and the same model card.
This document is the second pass on that investigation, updated after your
feedback; it confirms two of your hypotheses, rules out one of ours, and
narrows the remaining work to a bit-for-bit DC comparison plus a hard look
at how the constraint-row `dQ/dt` participates in our implicit timestep.

---

## 1. Symptom (unchanged from v1)

Same benchmark, same OSDI binary, same PSP103 model card:

| Simulator | Period | Frequency |
|-----------|--------|-----------|
| VACASK `tran1.raw` | 1.07 ns | 936.81 MHz |
| circulax (PSP103 via bosdi) | 6.67 ns | 149.93 MHz |

Ratio: **6.25×** slower in circulax.

Netlist: `/home/cdaunt/code/vacask/VACASK/benchmark/ring/vacask/` — 9-stage
CMOS ring, NMOS W=10 µm, PMOS W=20 µm, L=1 µm, VDD=1.2 V, 10 µA current
pulse on node 1 at t=1 ns. VACASK 0.3.2-103-g6434ed0-dirty. bosdi at
`/home/cdaunt/code/bosdi` with `osdi_loader`/`osdi_jax` on JAX 0.7.2.

## 2. Confirmed: it is not a time-stepping / integrator-tuning issue

Ran the ring transient across six integrator configurations, all with the
same 50 fF lumped load per stage:

| Case | Saved points | Freq [MHz] | Period [ns] | Swing [V] |
|------|--------------|------------|-------------|-----------|
| fixed dt = 10 ps, t=100 ns | 2000 | 149.93 | 6.67 | 1.376 |
| fixed dt = 5 ps,  t=100 ns | 2000 | 149.93 | 6.67 | 1.375 |
| fixed dt = 2 ps,  t=40 ns  | 2000 | 149.93 | 6.67 | 1.375 |
| fixed dt = 1 ps,  t=20 ns  | 2000 | 149.93 | 6.67 | 1.375 |
| PID rtol=1e-4 atol=1e-6    | 2000 | 149.93 | 6.67 | 1.375 |
| PID rtol=1e-6 atol=1e-8    | 2000 | 149.93 | 6.67 | 1.375 |

SDIRK3 + KLU. The answer is invariant to dt down to 1 ps and to the PID
error tolerance. Reproducer: `scripts/ring_dt_sweep.py`.

So the gap is not truncation error, not Newton convergence tolerance, not
the choice of adaptive controller — at least for stepping choices
available inside our current diffrax + SDIRK3 wrapping. This is consistent
with your Q5 observation that 6× is big-for-numerics-alone, but routes the
question at the `dQ/dt`-treatment-of-constraint-rows hypothesis rather
than at step size.

## 3. Your answers mapped to what we see in the Jacobian

### 3.1 Row-by-row dump at Vds=1.0, Vgs=0.7 (your suggested reference)

Script: `scripts/psp103_jacobian_dump.py`. NMOS W=10 µm, L=1 µm, VACASK
`psp103n` model card. Node order (post-collapse, per bosdi): `D, G, S, B,
int0, int1`. We tested three internal-node biases to separate
configuration-dependent entries (sized on V) from configuration-independent
entries (collapse constraints):

```
==== NMOS at V_D=1.0, V_G=0.7, V_S=V_B=0, V_int0=V_int1=0.5 ====
cur : [ +627 µA,  0,  -627 µA,  0,  +0.515,  -0.500 ]
chg : [ O(1e-14),  O(1e-14),  O(1e-14),  O(1e-14),  +85 fF,  0 ]

Row 0 (D)    physics:    G nonzeros: {D: 69 µS, G: 2.11 mS, S: -2.43 mS, B: 0.25 mS, int1: +95.5 mS}
                         C nonzeros: {D: 2.8 fF, G: -43 fF, S: +46 fF, B: -5.3 fF, int0: +85 fF}

Row 1 (G)    physics:    G nonzeros: {} (no DC gate leakage)
                         C nonzeros: {G: +170 fF, …}

Row 2 (S)    physics:    mirror of row 0 with sign flips

Row 3 (B)    physics:    tiny bulk-junction terms only

Row 4 (int0) physics:    G nonzeros: {D: +1.68 mS, G: +17 mS, S: -21 mS, B: +2.4 mS, int0: +31 mS, int1: +1.0}
                         C nonzeros: {int0: +170 fF, … small cross-terms}

Row 5 (int1) constraint: G nonzeros: {int1: -1.0}
                         C nonzeros: {}
                         cur[int1] = -V_int1
```

(Full dump at three V_int biases in the script output.)

### 3.2 What your answers confirm

**Q1 answer — "the ±1 pattern is a constraint row, not a 1 Ω conductance":**
row 5 fits your description exactly. It has a single nonzero `G[5,5] = -1`
and `cur[5] = -V_int1`, which is the compiled form of `V(br_int1, S) <+ 0`
when `rs = 0` in the model card — pinning `V_int1 = V_S` (= 0 here, so
`cur[5]` reduces to `-V_int1`). Zero capacitance, zero off-diagonal
elsewhere, exactly the Lagrange-style identity row you described.

circulax's assembly (`circulax/solvers/assembly.py::_assemble_osdi_group`)
stamps this row into `j_eff = G + (α/dt)·C + reg_diag` verbatim. Because
`C[5,:] = 0` and `reg_diag[5,5] = 0` (our `resistive_mask` is all-True
for PSP103), the stamp stays exactly `[0, 0, 0, 0, 0, -1]`. Newton drives
`V_int1 → 0` cleanly; DC solve passes our whole verification ladder
(single-device DC sweeps, inverter DC transfer, 3-stage chain, ring DC
metastable point). So the constraint row IS being read as a constraint,
not as a 1 S conductance — ruling out that specific hypothesis from v1 of
this doc.

**Q2 answer — "no Schur elimination; internal nodes are honest MNA
unknowns":** confirmed. Our `OsdiComponentGroup.var_indices` exposes the
full 6 post-collapse nodes as global unknowns; `_assemble_osdi_group`
writes the full 6×6 `j_eff` into the sparse matrix. No elimination.

**Q3 answer — "VACASK does not eliminate either":** understood. That
validates our not-eliminating approach and points the investigation away
from "we should've been reducing" and toward "something specific to how
implicit integration handles this MNA structure".

### 3.3 Where the intrinsic capacitance actually lives

This is the row-by-row finding that reshapes our v1 hypothesis.

- `C[D,D] = 2.8 fF`   — drain terminal self-cap is tiny.
- `C[D, int0] = 85 fF` — drain ↔ internal-drain gate-overlap / channel.
- `C[int0, int0] = 170 fF` — **most of the intrinsic capacitance lives on
  the internal-drain node**, not on the terminal.
- `C[G, G] = 170 fF` — gate cap correctly exposed on the gate terminal.

Schur-reducing (informally, since we don't do this in the hot path) gives
an effective terminal capacitance at D of
`C[D,D] + C[D,int0]² / C[int0,int0] ≈ 2.8 + 85²/170 ≈ 45 fF`, which is in
the ball-park of a physically realistic 1 µm-process node capacitance and
matches what VACASK's implicit solver effectively presents to its step
controller at each ring node.

**But** — to stabilise SDIRK3 at 10 ps timesteps in circulax we add 50 fF
of lumped external capacitance per stage, which roughly *doubles* the
effective node capacitance. That accounts for ~2× of the 6.7× gap. A
factor-of-~3 remains unexplained by load capacitance alone.

## 4. Where we currently suspect the remaining ~3× lives

Three concrete candidates, ranked by what the row dump implicates:

1. **Constraint-row `dQ/dt` routing at row 4.** Row 4 (int0) carries the
   big `C[int0, int0] = 170 fF` entry plus a `G[int0, int1] = +1.0`
   off-diagonal coupling to row 5's constraint. Our implicit step computes
   `residual[int0] = cur[int0] + (chg[int0] - chg_prev[int0])/dt` and stamps
   `j_eff[int0, :] = G[int0, :] + (α/dt)·C[int0, :]`. We think this is
   right, but the interaction between the `+1` row-4 entry (pointing at
   V_int1, which row 5 pins to V_S) and the 170 fF row-4 capacitance
   might produce a stale `dQ/dt` term at the constraint boundary that
   VACASK treats differently. This matches your Q5 call-out about `ddt()`
   routed to a constraint row via the wrong integrator coefficient.
2. **Our numerical-stability heuristic forcing extra C_load.** We use a
   Jacobian-diagonal-dominance heuristic to pick `C_load ≥ α·|G_ii|·dt`
   and landed on 50 fF as the safe choice at 10 ps. If the real effective
   `|G_ii|` the solver sees is dominated by `G[D, int1] = 95.5 mS`
   (≈ gds via the internal source), we may be over-stabilising.
3. **SDIRK3 coefficient interaction with row 5.** SDIRK3 has α ≈ 2.29 per
   stage and three internal stage solves; the constraint row's `cur[5] =
   -V_int1` residual gets re-evaluated per stage. We haven't audited
   whether the per-stage `V_int1` trajectory produces the same transient
   behaviour VACASK's trapezoidal integrator does on the same row.

We don't yet know which of (1)/(2)/(3) dominates.

## 5. Your offered helpers — yes please

You offered two; we'd take both and happily pay for them with a concrete
bug report or a circulax-side PR, whichever is more useful.

1. **`schur_reduce_terminal(G_full, C_full, num_pins, alpha, gmin=1e-12)`
   utility**, returning a 4×4 terminal-equivalent `(G_eff, C_eff)` and a
   `warning_if_singular` flag. Even if we don't put it in circulax's hot
   path, it would be a powerful *debugging* tool: if we run the ring with
   the Schur-reduced 4×4 matrix and get 937 MHz, that isolates the gap to
   "internal-node integration in circulax", and if we still get 150 MHz,
   it isolates it to "something in the model eval or the reduced matrix
   itself". We'd be happy to add this to a `bosdi.debug` submodule.
2. **Per-device debug emitter**, printing
   `{(row_idx, col_idx, has_resist, has_react, value, is_likely_constraint)}`
   for one instance. This would let us line-by-line diff our
   `_assemble_osdi_group` output against what bosdi intends the scatter
   to mean, which is the next investigation step for candidate (1) above.

## 6. Concrete next step we're starting now

Per your Q5 paragraph, the deciding experiment is a bit-for-bit
`osdi_eval` vs VACASK DC comparison at a fixed bias. We're extending our
PSP103 verification tests so each ladder stage has a VACASK reference:

- single NMOS at Vds=1.0, Vgs=0.7 — `cur`, `cond`, `chg`, `cap` matrices;
- single PMOS equivalent;
- CMOS inverter DC transfer (7 points);
- inverter step-response delay;
- 3-stage chain DC logic levels.

If `osdi_eval` matches VACASK's internal stamping bit-for-bit, the 6.7×
gap is unambiguously transient-side (our implicit integration of the
constraint row's `dQ/dt`) and your Schur-reduce helper plus the debug
emitter will pin it precisely. If it doesn't match, we've found a
collapse-interpretation bug to triage together.

Status on that: the test scaffolding is next up on this branch
(`osdi_improvements` at `/home/cdaunt/code/circulax/circulax-osdi`) and we
will share dumps with you as soon as they're done.

## 7. Artifacts

- Branch: `osdi_improvements` on `cdaunt/circulax` (worktree at
  `/home/cdaunt/code/circulax/circulax-osdi`).
- Reference fixture: `tests/fixtures/vacask_ring_ref.npz` (24 765 points
  over 1 µs, v(n1) waveform).
- dt-sweep reproducer: `scripts/ring_dt_sweep.py`.
- Jacobian dump reproducer: `scripts/psp103_jacobian_dump.py`.
- Ring test: `tests/test_psp103_ring_oscillator.py::test_ring_oscillator_vs_vacask_reference`.
- Adapter source: `circulax/components/osdi.py` +
  `circulax/solvers/assembly.py::_assemble_osdi_group`.

Thanks — your feedback moved this from "we don't know where to look" to
"we have three testable hypotheses and a decision tree". Let us know when
the `schur_reduce_terminal` and debug-emitter helpers land and we'll run
them against the ring immediately.
