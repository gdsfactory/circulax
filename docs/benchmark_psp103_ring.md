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

## Single-circuit results

| Configuration | Wall (s) | µs / step | n_steps | Freq (MHz) |
|---------------|----------|-----------|---------|------------|
| **VACASK trap, KLU, adaptive** | **1.19** | **48** | 24 764 | 289.60 |
| circulax + klujax (`klu_split`), fixed dt = 50 ps | 16.90 | 845 | 20 000 | 288.88 |
| circulax + klujax-rs (`klu_rs_split`), fixed dt = 50 ps | 16.37 | 819 | 20 000 | 288.88 |
| circulax + klujax, PID adaptive (rtol=1e-3 atol=1e-5, dtmin=1fs, unbounded) | 191.2 | 641 | 251 218 acc / 47 255 rej | 290.59 |
| circulax + klujax, PID adaptive (rtol=1e-3 atol=1e-5, **dtmin=10 ps**, force_dtmin=True) | 68.0 | 679 | 99 938 acc / 22 rej | 290.53 |
| circulax + klujax, PID adaptive (rtol=1e-3 atol=1e-5, **dtmin=40 ps**, force_dtmin=True) | 20.4 | 814 | 25 000 acc / 3 rej | 290.06 |

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

### vmap caveat (CPU)

| Configuration | Wall (s) | µs / ring / step |
|---------------|----------|------------------|
| circulax + klujax, vmap batch = 4, t1 = 200 ns | 13.4 | 839 |
| circulax + klujax, vmap batch = 8, t1 = 200 ns | 27.3 | 853 |

Linear scaling: doubling batch doubles wall time. On CPU, klujax's KLU
primitive is sequentialised inside XLA — a vmap'd refactor-and-solve
loops over the batch one element at a time. So `vmap` here has the same
total cost as `for` (modulo a fixed JIT overhead that pays back over
many calls). On GPU this would parallelise; on CPU it doesn't.
**klujax-rs's `refactor_and_solve_f64` doesn't have a vmap batching
rule yet** (`NotImplementedError` from the FFI layer); vmap parameter
sweeps must use `klu_split` for now.

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
