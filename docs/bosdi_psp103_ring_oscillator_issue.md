# PSP103 ring-oscillator frequency mismatch vs. VACASK

**Audience:** bosdi author.
**Status:** open — circulax's ring-oscillator transient runs ~6.7× slower than
VACASK's on the same `psp103v4_psp103.osdi` binary and the same model card.
Third pass. Since the last revision: ran the `C_load=0` crosscheck you
suggested and **ruled out candidate (2)** (stability-heuristic oversizing);
ran your new `osdi_debug.schur_reduce` + `classify_rows` at the Vds=1.0,
Vgs=0.7 reference bias and confirmed the Lagrange-identity classification
of row 5; all six rows classify cleanly. Next experiment is a Schur-reduced
assembly path inside circulax to localise the bug to internal-node
integration vs. something else.

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

## 4. Candidate (2) is out — C_load sweep

You proposed a sharp crosscheck: run the ring with `C_load = 0` and a
larger `gmin`; if it converges, our stability heuristic is the culprit.
We ran it. Single-case runner in `scripts/ring_one_case.py`, results
appended to `reports/ring_sweep.csv`:

| C_load | gmin | Period | Frequency | Swing |
|--------|------|--------|-----------|-------|
| 50 fF | 1e-9 | 6.26 ns | 159.8 MHz | 1.375 V |
| 0 fF  | 1e-3 | 6.01 ns | 166.5 MHz | 1.407 V |
| 0 fF  | 1e-6 | 6.01 ns | 166.5 MHz | 1.407 V |
| 0 fF  | 1e-9 | 6.01 ns | 166.5 MHz | 1.407 V |

Removing the external 50 fF shifts frequency only from 160 → 167 MHz
(~4 %). The ring is essentially just as slow with zero external load as
with our safety 50 fF. So the stability-heuristic oversizing hypothesis
is wrong, or at best a rounding term — it contributes a ~4 % slowdown,
not a 6× slowdown.

Gmin is effectively irrelevant here because our `dc_init` step runs with
a large homotopy gmin and then we switch to the runtime one for the
transient; the transient path itself doesn't lean on gmin once the
initial DC seed is settled.

So the gap lives in candidate (1) and/or (3) — the constraint-row
`dQ/dt` / integrator-coefficient interaction on rows 4 and 5.

## 5. schur_reduce output at Vds=1.0, Vgs=0.7 — ready for you

We ran your `osdi_debug.schur_reduce` + `classify_rows` +
`format_jacobian_table` at the reference bias. Full dump at
`reports/psp103_schur_dump.txt` (reproducer: `scripts/psp103_schur_dump.py`).
Headline:

**`classify_rows` on the 6×6 PSP103 stamp:**

| Row | Node | Kind | Detail |
|-----|------|------|--------|
| 0 | D    | physics        | 5 cond + 5 cap non-zeros |
| 1 | G    | reactive_only  | 5 cap entries, no conductance |
| 2 | S    | physics        | 5 cond + 5 cap non-zeros |
| 3 | B    | reactive_only  | 4 cap entries, no conductance |
| 4 | int0 | physics        | 6 cond + 5 cap non-zeros |
| 5 | int1 | **constraint** | `cond` nonzero at col=5, value=-1.0 — Lagrange identity row |

Your classifier agrees with the hand-reading from §3: row 5 is the
constraint, row 4 is ordinary physics, rows 1 and 3 are pure-capacitive.
Good sanity check.

**`schur_reduce` at α=0 (DC-only), singular=False:**

```
j_eff (4×4)          D          G          S          B
        D    6.902e-05  2.115e-03 -2.434e-03  2.505e-04
        G    0          0          0          0
        S   -6.902e-05 -2.115e-03  2.434e-03 -2.505e-04
        B    0          0          0          0
r_eff  [-0.0471, 0, +0.0471, 0]
```

The G and B rows Schur-reduce to zero at DC, as expected (gate/bulk
carry no DC conductance). The D/S row structure reproduces the
terminal-level gds / gm / gmb pattern one would hand-derive from
PSP103's compact equations. No surprises.

**`schur_reduce` at α=1e11 (dt=10 ps, Backward Euler), singular=False:**

```
j_eff (4×4)          D          G          S          B
        D    3.779e-05 -5.078e-03  5.716e-03 -6.756e-04
        G    3.271e-04  1.709e-02 -1.749e-02  7.754e-05
        S   -3.613e-04 -1.115e-02  1.293e-02 -1.416e-03
        B   -3.582e-06 -8.591e-04 -1.151e-03  2.014e-03
```

Non-trivial. All four rows become non-zero as the reactive network
couples the gate and bulk into the terminal equations via the
internal-node path. The `j_eff[G, G] = 17 mS` entry at this α is
`α · C_G,G` ≈ 1e11 · 170 fF = 17 mS — consistent.

**Finite-differenced `C_eff[D,D]`:** ≈ −1.7 fF. We note your docstring
warning that `(G_eff, C_eff)` doesn't cleanly split except at α=0, so
this negative value is just the nonlinearity in α biting us, not a
physical absurdity.

`scripts/psp103_schur_dump.py` has the full artefact you can diff
against bosdi's own outputs entry-by-entry.

## 6. Next step: the Schur-reduced circulax assembly

With candidate (2) ruled out and bosdi's `schur_reduce` producing clean
terminal stamps, the decisive experiment is now a one-shot change in
`_assemble_osdi_group`: replace the 6×6 `j_eff` stamp with a
Schur-reduced 4×4 per-device stamp, wire its j_eff/r_eff into circulax's
KCL block-assembly instead of the full internal-node system, and re-run
the ring transient.

- If it produces ~937 MHz → our internal-node integration path
  (specifically the handling of the constraint row and/or the row-4
  coupling to it) is the bug, and we can either Schur-reduce in the hot
  path or fix the handling directly.
- If it still produces ~160 MHz → the discrepancy is in something
  shared between both paths (osdi_eval itself, or circulax's Newton
  residual assembly at the terminal-only level, or SDIRK3's treatment
  of the stamp).

Starting that now on the `osdi_improvements` branch. Will send the
result as soon as the reduced assembly is working end-to-end.

## 7. Artifacts

- Branch: `osdi_improvements` on `cdaunt/circulax` (worktree at
  `/home/cdaunt/code/circulax/circulax-osdi`).
- Reference fixture: `tests/fixtures/vacask_ring_ref.npz` (24 765 points
  over 1 µs, v(n1) waveform).
- dt-sweep reproducer: `scripts/ring_dt_sweep.py`.
- Jacobian dump reproducer: `scripts/psp103_jacobian_dump.py`.
- Schur dump reproducer: `scripts/psp103_schur_dump.py` → `reports/psp103_schur_dump.txt`.
- C_load sweep runner: `scripts/ring_one_case.py` → `reports/ring_sweep.csv`.
- Ring test: `tests/test_psp103_ring_oscillator.py::test_ring_oscillator_vs_vacask_reference`.
- Adapter source: `circulax/components/osdi.py` +
  `circulax/solvers/assembly.py::_assemble_osdi_group`.

Thanks for the fast turnaround on `osdi_debug`. The Schur-reduced ring
is the next experiment on our side.
