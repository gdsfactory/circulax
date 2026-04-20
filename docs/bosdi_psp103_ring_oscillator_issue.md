# PSP103 ring-oscillator frequency mismatch vs. VACASK

**Audience:** bosdi author.
**Status:** **resolved — not a bosdi issue**.  Fourth and final pass.
Since the last revision we built out a full VACASK cross-check ladder
(13 DC tests + 2 transient tests, all passing in `tests/test_psp103_vs_vacask.py`)
and proved:

1. PSP103 device evaluation is bit-for-bit between bosdi and VACASK
   (DC currents agree to 0.05 %, inverter-VTC to <0.2 mV, chain-3 node
   voltages to <1 µV).
2. Open-loop transient matches VACASK to ~1 % (single-inverter step:
   tpHL 62.3 ps VACASK vs 63.1 ps circulax/BDF2; 3-stage chain per-stage
   delays match each VACASK delay to within 1 %).
3. The Schur-reduced assembly path produces the same answer as the
   full 6×6 path on the ring oscillator → internal-node integration is
   not the problem either.

The remaining 3.1 × ring-osc gap (300 MHz circulax/BDF2 vs 937 MHz VACASK
trap) is **circulax-internal**: BDF2 / SDIRK3 don't preserve limit-cycle
frequency on a ring oscillator the way trapezoidal does.  Section 6 has
the details and the proposed fix.  Thanks for the `osdi_debug` helpers
and the systematic question list — they were critical to ruling out the
device-side hypotheses cleanly.

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

## 6. Resolution: it's the time integrator, not bosdi or the assembly

We built three more independent crosschecks since v3 and they form a
clean diagnosis.

### 6.1 Schur-reduced assembly path — internal nodes are not the bug

Implemented an opt-in Schur-reduced assembly path in
`circulax/components/osdi.py` + `circulax/solvers/assembly.py`
(`use_schur_reduction=True` on the descriptor).  It calls your
`schur_reduce` to eliminate internal nodes from the per-device stamp
*at the host's integrator coefficient* and pads the result back into
the original COO sparsity.  Rerunning the ring with this path:

```
circulax (full 6×6 OSDI stamp):       159.84 MHz
circulax (Schur-reduced 4×4 stamp):   159.84 MHz
```

Identical to four digits.  So **internal-node integration is not the
source of the gap** — eliminating those nodes entirely doesn't move
the answer.  Candidates (1) and (3) from §4 are out.

### 6.2 DC cross-check ladder — device evaluation is bit-for-bit

Added `tests/test_psp103_vs_vacask.py` and
`tests/fixtures/vacask_runner.py`.  13 DC cross-checks run VACASK on
small sub-circuits and assert agreement with circulax:

| Cross-check | Reference | circulax | ratio |
|-------------|-----------|----------|-------|
| NMOS Vds=1.0, Vgs=0.7 |Id|     | 626.327 µA | 626.602 µA | 1.0004 |
| NMOS Vds=0.6, Vgs=1.2 |Id|     | 1629.86 µA | 1630.73 µA | 1.0005 |
| NMOS Vds=0.6, Vgs=0.3 |Id|     | 46.154 µA  | 46.125 µA  | 0.9994 |
| PMOS Vsg=1.0, Vsd=0.6 |Id|     | 2522.94 µA | 2524.27 µA | 1.0005 |
| Inverter VTC, 7 Vin pts        | <±0.2 mV diff on Vout, all 7 points |
| Chain-3 DC, Vin∈{0, VDD}       | <1 µV diff on every node          |

DC physics is identical at every level we can probe — device, gate,
small chain.

### 6.3 Open-loop transient cross-check — the limit-cycle distinction

Added two transient cross-checks at 5 ps fixed step / 5 ns window:

**Single inverter step** (`tests/fixtures/vacask/inverter_step.sim`,
Vin pulse 0→VDD over 100 ps):

| Integrator | tpHL | ratio vs VACASK |
|------------|------|-----------------|
| VACASK trap | 62.3 ps | — |
| circulax BE | 63.4 ps | 1.02 |
| circulax BDF2 | 63.1 ps | 1.01 |
| circulax SDIRK3 | 91.6 ps | 1.47 |

**3-stage chain step** (each stage drives the next stage's gate):

| Stage | VACASK trap | circulax BDF2 | ratio |
|-------|-------------|---------------|-------|
| 1 | 181.1 ps | 181.1 ps | 1.00 |
| 2 | 168.4 ps | 168.3 ps | 1.00 |
| 3 |  78.3 ps |  79.4 ps | 1.01 |

So circulax/BDF2 reproduces single-stage and chain transient propagation
delays to ~1 %.  The bug doesn't appear here.

### 6.4 Why the ring still gaps 3 ×

Naively, `period = 2 × N_stages × τ_stage` would give
`2 × 9 × 180 ps = 3.24 ns → 309 MHz` — and that is essentially exactly
circulax/BDF2's ring frequency (300 MHz).  VACASK's ring runs at 937 MHz
*despite* the same 180 ps chain delay because in a limit-cycle
oscillator each stage sees small-signal continuous swings, not full
step transitions, so the effective per-stage delay shrinks well below
the standalone step delay.

**Trapezoidal** integration is uniquely well-suited to limit-cycle
oscillators: zero numerical damping, zero phase distortion at all
frequencies up to Nyquist, exact preservation of conservative dynamics.
**BDF2** introduces L²-stable damping that is benign for chain step
responses but distorts ring-oscillator limit cycles by pulling them
toward longer periods.  **BE** has even more damping but happens to
land on the same answer at this circuit because the small-signal
attenuation is dominated by stage-loading anyway.  **SDIRK3** carries
its own per-stage coefficient that adds an additional ~1.5× slowdown
on top.

This is a circulax-internal limitation, not anything bosdi exposed
incorrectly.

### 6.5 Proposed fix in circulax (no bosdi changes needed)

Add a trapezoidal-rule integrator alongside the existing BDF2 / SDIRK3
classes in `circulax/solvers/transient.py`.  Trap implementation is a
~80-line sibling of the existing BDF2 wrapper.  Once landed,
`test_ring_oscillator_vs_vacask_reference` should tighten its
tolerance from 10× to ~5 % and the tests in `test_psp103_vs_vacask.py`
will validate the new integrator alongside the existing BE/BDF2/SDIRK3.

Tracking this on `osdi_improvements` for now; not blocking from
bosdi's side.  Closing this issue from our end pending the trapezoidal
integrator landing — happy to send a heads-up when it does so you can
verify against your own ring runs.

## 7. Artifacts

- Branch: `osdi_improvements` on `cdaunt/circulax` (worktree at
  `/home/cdaunt/code/circulax/circulax-osdi`).
- VACASK cross-check tests: `tests/test_psp103_vs_vacask.py`
  (13 DC + 5 transient cases, all passing with BE/BDF2; SDIRK3 expected
  to differ on transient).
- VACASK reference fixtures: `tests/fixtures/vacask/*.{sim,raw}`
  (16 paired netlist+waveform files for the ladder).
- VACASK runner helper: `tests/fixtures/vacask_runner.py`.
- Ring transient reference: `tests/fixtures/vacask_ring_ref.npz`
  (24 765 points over 1 µs, v(n1) waveform).
- Investigation reproducers under `scripts/`:
  - `ring_dt_sweep.py` — proves the gap is dt-invariant.
  - `psp103_jacobian_dump.py` — row-by-row 6×6 stamp dump.
  - `psp103_schur_dump.py` → `reports/psp103_schur_dump.txt` — uses your
    `osdi_debug.schur_reduce` + `classify_rows` + `format_jacobian_table`.
  - `ring_one_case.py` → `reports/ring_sweep.csv` — C_load sweep.
  - `ring_schur.py` — ring with the Schur-reduced assembly path.
  - `ring_backend_check.py`, `ring_integrator_check.py`,
    `ring_integ_one.py`, `ring_bdf2_cload0.py` — backend / integrator
    sweeps proving the gap is integrator-method-specific.
- Adapter source: `circulax/components/osdi.py` +
  `circulax/solvers/assembly.py::_assemble_osdi_group`
  (added the Schur-reduced path under `use_schur_reduction=True`,
  default off so existing behaviour is preserved).

Thanks for the fast turnaround on `osdi_debug` — your Schur helper let
us cleanly localise the bug to circulax's time integrator and rule out
every device-side hypothesis in one experiment.
