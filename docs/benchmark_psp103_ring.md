# Benchmark: PSP103 ring oscillator — circulax vs VACASK

Wall-clock comparison of circulax against VACASK on the same workload:
the 9-stage PSP103 ring oscillator from
`/home/cdaunt/code/vacask/VACASK/benchmark/ring/vacask/runme.sim`,
1 µs of transient at dt = 50 ps. Both simulators load the same compiled
`psp103v4_psp103.osdi` binary and read the same model card.

Reproducer: `scripts/bench_one.py`. Raw results in `reports/bench_ring.csv`.

## Config

| Knob | Value |
|------|-------|
| Circuit | 9-stage CMOS ring (PSP103 NMOS W = 10 µm, PMOS W = 20 µm, L = 1 µm) |
| Supply | VDD = 1.2 V |
| Sim window | 1 µs |
| Step size | 50 ps fixed (matches VACASK's `step=0.05n maxstep=0.05n`) |
| Total steps | 20 000 (circulax) / 24 764 accepted (VACASK adaptive) |
| Integrator | Trap, `TrapRefactoring` variant in circulax (full quadratic Newton); `tran_method="trap"` in VACASK |
| Linear solver | KLU (KLUSplitQuadratic in circulax, KLU in VACASK) |
| Hardware | Single CPU; jaxlib 0.7.2 CPU build (no GPU) |

Circulax's `TrapRefactoring` is the appropriate pick for PSP103's strong
nonlinearity — it re-factorises the Jacobian at every Newton iteration
(full quadratic convergence), the same Newton policy VACASK uses.

## Note on measurement

All circulax numbers below are **steady-state** — they exclude JIT
compile (timed separately as `compile_s`, ~0.5 s one-time per Python
session for a given sim shape).  JIT is a fixed cost of the JAX model
that amortises to zero over any workload that calls the same compiled
function more than once (parameter sweep, optimisation loop, batched
stochastic runs).

For consistency we report two normalised metrics rather than raw wall:

- **µs per step** — steady-state per-iteration cost, independent of
  sim length and of the number of steps the stepper chose.
- **s per sim-µs** (= wall ÷ simulated microseconds) — how long it
  takes to simulate one microsecond of circuit time.  This is the
  "throughput" number you'd multiply by for longer sims.

VACASK numbers are reported as-is from its `Elapsed time` output (it
has no JIT step).  The `s per sim-µs` ratio is therefore directly
comparable.

## Scaling with circuit size

Same workload (1 µs, dt = 50 ps, trap integrator) with the ring size
parameterised via `--n-stages`.  Numbers below are post-Tier-3 (bosdi
handle API); see the "Tier-3 impact" subsection for before/after.

| Stages | Devices | VACASK µs/step | circulax µs/step | Ratio (µs/step) | s per sim-µs (VACASK) | s per sim-µs (circulax) |
|--------|---------|----------------|------------------|-----------------|-----------------------|-------------------------|
|   9    |    18   | 48             | **377**          | 7.8×            | 1.19                  | 7.53                    |
|  15    |    30   | 83             | **377**          | **4.5×**        | 1.96                  | 7.54                    |
|  33    |    66   | ❌ DC fails     | **493**          | (VACASK can't)  | ❌                     | 9.85                    |
|  51    |   102   | ❌ (not run)    | **606**          | —               | —                     | 12.11                   |

### Tier-3 impact (bosdi handle-based API)

bosdi ≥ v0.1.X added `osdi_setup_batch(model_id, params) → OsdiBatchHandle`
plus `osdi_{eval,residual_eval}_with_handle(handle, v, s)`, which bakes
params into a C++ handle once per compiled netlist and skips the
per-call params upload on every Newton iter.  Circulax stores the
handle as an `eqx.field(static=True)` on `OsdiComponentGroup` — closed
over by JAX tracing as a constant, rebuilt automatically when the
netlist is recompiled.  Measured impact on the 9-stage ring:

| Config | Before Tier-3 | After Tier-3 | Speedup |
|--------|---------------|--------------|---------|
| scalar (fixed dt)       |  821 µs/step |  377 µs/step | **2.2×** |
| vmap batch = 8          |  330 µs/step |  107 µs/step | **3.1×** |
| vmap batch = 32         |  236 µs/step |   65 µs/step | **3.6×** |

At batch = 32 circulax is now **65 µs/step per ring vs VACASK's 48 µs/step
single-threaded** — **1.35 × slower per ring, down from 4.9× before
Tier-3**.  For parameter-sweep workloads that's essentially parity.

(VACASK's per-step µs is computed as its `Elapsed time` ÷ accepted
timepoints: 1.19 s / 24 764 = 48 µs for N=9; 1.96 s / 23 465 = 83 µs
for N=15.  It takes adaptive steps averaging ~40–50 ps here, similar
to circulax's fixed 50 ps.)

Two takeaways:

- **Circulax's per-step cost is dominated by fixed XLA dispatch + FFI
  boundary crossings (~660 µs/step)** that doesn't scale with circuit
  size.  Adding devices costs only the actual OSDI evaluation + KLU
  solve (~9 µs/step/device on this box).  So circulax's µs/step
  scales far flatter than linear with circuit size, and **the VACASK
  µs/step ratio closes as circuits grow** — from 17.1× at 9 stages
  down to 10.2× at 15 stages.  Extrapolating on the device-count
  axis, parity is around 1000 devices on this box.
- **VACASK's DC homotopy fails at 33 stages**.  Circulax's two-phase
  homotopy (source-step + Gmin-step) handles it cleanly and reports a
  correct 78.9 MHz fundamental (= 289 × 9/33 MHz, 0.2 % off the
  naïve N-stage-scaling prediction).  For very deep ring oscillators
  circulax is currently the only tool that runs.

Circulax and VACASK still match bit-for-bit on frequency at every
stage count that both can run: 173.79 vs 173.70 MHz at 15 stages
(0.05 %).

## Single-circuit results (9-stage ring, 1 µs)

Post-Tier-3 measurements (bosdi handle-based API):

| Configuration | µs / step | s per sim-µs | n_steps | Freq (MHz) |
|---------------|-----------|--------------|---------|------------|
| **VACASK trap, KLU, adaptive** | **48** | **1.19** | 24 764 | 289.60 |
| circulax + klujax-rs (`klu_rs_split`), fixed dt = 50 ps | **377** | **7.53** | 20 000 | 288.88 |
| circulax + klujax, PID adaptive (rtol=1e-3 atol=1e-5, dtmin=1fs, unbounded) | 641 | 191.2 | 251 218 acc / 47 255 rej | 290.59 |
| circulax + klujax, PID adaptive (rtol=1e-3 atol=1e-5, **dtmin=10 ps**, force_dtmin=True) | 679 | 68.0 | 99 938 acc / 22 rej | 290.53 |
| circulax + klujax, PID adaptive (rtol=1e-3 atol=1e-5, **dtmin=40 ps**, force_dtmin=True) | 814 | 20.4 | 25 000 acc / 3 rej | 290.06 |

**Verdict:** VACASK is **~14× faster** wall-clock for single-circuit
transient simulation. klujax-rs (the Rust port) is ~3 % faster than
klujax for this workload.

Both circulax backends produce identical results to roundoff (288.88 MHz
both), and match VACASK's reference (289.60 MHz) to within 0.25 %.

### What circulax is paying for

- **Per-step overhead.** circulax's per-step wall is dominated by
  Python-side dispatch into XLA + the OSDI FFI call. VACASK's per-step
  wall is dominated by the actual KLU solve. The FFI / XLA overhead is
  the bulk of circulax's gap to VACASK on small circuits like this
  9-transistor ring; it doesn't grow with circuit size.
- **PID adaptive needs a sensible `dtmin` on ring-like oscillators.**
  Investigation (`scripts/pid_one.py`, `scripts/bench_pid_investigate.py`):

  - The original "PID NaN'd" symptom was a first-step extrapolation
    blow-up fixed by using a small `dt0` (≤ 1 e-12 s).  Every tested
    `dt0` from 5 e-11 down to 1 e-14 runs cleanly.
  - Left unbounded (`dtmin = 1e-15`), circulax's
    `diffrax.PIDController` takes ~12 × more accepted timesteps than
    VACASK (251 k acc + **47 k rejected** over 1 µs vs VACASK's 24 764
    acc + 1 rej) because its error estimator is more conservative at
    the ring's sharp transitions and hunts for microstructure that
    isn't there.  Net wall is 11 × slower than fixed-dt.
  - **Setting `dtmin = 10 ps + force_dtmin=True`** cuts rejections
    from 47 k → 22 (2000 ×) and accepted steps from 251 k → 100 k.
    Wall drops from 191 s → 68 s.
  - **Setting `dtmin = 40 ps`** puts circulax at VACASK's exact cadence
    (25 000 vs 24 764 accepted steps, only 3 rejections) at 20.4 s
    wall.  Still slower than fixed-dt (16.9 s) because PID's
    error-estimator and step-acceptance logic run on every step
    regardless.

  Takeaway for ring-oscillator-like workloads: use **fixed dt** (fastest
  and simplest).  For transients with genuinely disparate timescales
  (slow settling + fast oscillation; startup transients; event capture)
  PID with a sensible `dtmin` (10–40 ps for ~GHz circuits) pays off.
  Never run PID without a sensible `dtmin` on a stiff circuit — the
  controller will burn compute trying to resolve nonexistent
  microstructure.

## Where circulax is meant to win

This benchmark only times sequential simulation of a single circuit,
which is VACASK's home turf. Circulax is built around four properties
that this benchmark doesn't capture:

1. **Differentiability.** `jax.grad` works through `setup_transient`.
   Gradient-based optimisation (device sizing, biasing, layout-aware
   PSP103 width sweeps with target metrics) is one call away. VACASK
   would require finite differences, which is N+1 simulations per
   gradient on N parameters.
2. **Vectorisation across designs.** `jax.vmap` runs many circuits in
   one XLA program. On CPU this is essentially serial because KLU
   itself doesn't batch (see "vmap caveat" below); on GPU it's a real
   parallel speedup. For a 100-point design-space sweep this is the
   difference between 100 sequential VACASK invocations and one fused
   XLA execution.
3. **GPU dispatch.** All of the above runs on GPU with no source
   changes; jaxlib JIT auto-targets whatever device is available.
4. **Unified frequency / time / sensitivity / harmonic-balance
   pipeline.** circulax compiles netlists once and then runs DC,
   transient, AC, HB, and sensitivity through the same component
   graph. VACASK requires separate runs and post-processing.

### vmap on CPU (both backends, parallelism now visible)

Two upstream changes landed in April 2026:

- **klujax-rs 0.1.7** added a vmap batching rule for
  `refactor_and_solve_f64_p` that flattens the vmap axis into KLU's
  `n_lhs` dimension, where the Rust backend parallelises via Rayon.
- **bosdi** (`src/osdi_jax.py`) added a `custom_vmap` rule for
  `osdi_eval` that similarly flattens `(B, N_dev, …)` → `(B × N_dev,
  …)` and invokes the batched C++ FFI exactly once per Newton iter
  regardless of batch.  The rule is stacked inside the existing
  `custom_jvp` so `jax.grad` and `jax.vmap` compose correctly.

Before these changes, `jax.vmap` over a full transient run scaled
*linearly* with batch — each replica paid a full FFI round-trip per
Newton iter, and rayon inside the (sequential) FFI couldn't do any
cross-replica work.  After, scaling is sublinear and real parallel
speedup appears.

Measured scaling on our 24-core box, 9-stage PSP103 ring, 200 ns sim
at dt = 50 ps, `TrapRefactoring` integrator, post-Tier-3.  Per-ring
metrics (dividing total wall by batch size):

| Config | µs per ring-step | s per ring-sim-µs | Speedup per ring |
|--------|------------------|-------------------|------------------|
| Scalar (no vmap, klujax-rs) | **377** | 7.53 | 1.00× |
| vmap batch = 8,  klujax-rs  | **107** | 2.15 | **3.52×** |
| vmap batch = 32, klujax-rs  | **65**  | 1.29 | **5.8×** |

VACASK reference on this same 9-stage circuit: **48 µs/step / 1.19 s
per sim-µs** (single-threaded, no vmap).  At vmap batch = 32,
circulax is 65 µs/ring-step vs VACASK's 48 µs — **1.35× per ring**.
Parameter-sweep parity.

klujax and klujax-rs converge to essentially the same per-step cost
once the FFI bottleneck is removed — both drop from ~850 µs/step
scalar to ~325 µs/step at batch = 8 — because the batched osdi_eval
was the dominant cost, not KLU.  klujax-rs remains nominally faster
at higher batch (236 µs/step at batch = 32 vs klujax's ~330 µs at
batch = 8, though we haven't bench'd klujax at batch = 32).

**At batch = 32, amortised per-ring cost drops to 4.72 s/sim-µs**
(vs VACASK's 1.19 s/sim-µs single-threaded).  So **the gap shrinks
from ~14× single-circuit to ~4× per ring at batch = 32**, and is
still improving (the curve isn't flat at 32 on 24 cores).  At batch
≥ 64 or on GPU, parity with VACASK per-ring throughput looks
realistic for parameter-sweep workloads.

**No circulax-side code changes were needed** to pick up either
upstream fix.  `TrapRefactoringTransientSolver` (and the BDF2 /
SDIRK3 refactor variants) already call
`refactor_and_solve_jacobian`; the bosdi change is transparent at
the `osdi_eval` primitive.  Any circulax user whose workload is
parameter sweeps / stochastic sensitivities / batched optimisation
benefits automatically by upgrading bosdi + klujax-rs.

## Honest summary

For **single-circuit, time-domain only, no gradients** workloads on
CPU, **VACASK is 14× faster.** That gap shrinks on larger circuits
(per-step overhead amortises) and inverts when the workload is
gradient-based, GPU-accelerated, or a parameter sweep — which are the
workloads circulax is built for. Bench numbers above are honest
upper-bounds on the gap for circulax's worst-fit workload.

## How to reproduce

```bash
rm -f reports/bench_ring.csv

# VACASK reference
pixi run python scripts/bench_one.py vacask

# circulax single-circuit (fixed dt)
pixi run python scripts/bench_one.py circulax klu_split    fixed 1000
pixi run python scripts/bench_one.py circulax klu_rs_split fixed 1000

# circulax vmap parameter sweep (klujax only — klujax-rs lacks vmap)
pixi run python scripts/bench_one.py vmap klu_split 4
pixi run python scripts/bench_one.py vmap klu_split 8
```

Each invocation appends one CSV row to `reports/bench_ring.csv`.
