# Ring oscillator

N-stage CMOS ring oscillator (PSP103, NMOS W = 10 µm / PMOS W = 20 µm,
L = 1 µm, VDD = 1.2 V).  1 µs transient at 50 ps fixed step → 20 000
time points.  Canonical N = 9 mirrors
`/home/cdaunt/code/vacask/VACASK/benchmark/ring/`; other N values
are generated alongside via `vacask_gen.py` / `ngspice_gen.py`.

Four simulators are compared, all running the same PSP103 physics:

- **VACASK** — reference C++ simulator (`vacask --skip-embed
  --skip-postprocess`).  Loads `psp103v4.osdi` (compiled from
  `psp103.va` by openvaf-r 23.5.0).
- **ngspice** — ngspice 45.2 batch mode (`ngspice -b`), KLU direct
  linear solver, `pre_osdi psp103v4.osdi`.  ngspice 45.2 happily loads
  OSDI 0.4 binaries; the harness compiles `psp103v4.osdi` into the
  ngspice working directory on demand.
- **circulax-OSDI** — JAX-based, calls the same `psp103v4.osdi` binary
  via the bosdi FFI shim.  Fast but **not** differentiable through the
  device physics — the .osdi boundary breaks the JAX trace.
- **circulax-VA** — fully differentiable: `psp103.va` is lowered via
  `bosdi.va.compile_va_unopt_with_split` (openvaf-r
  `--dump-unopt-mir-with-split`, ADCE + `simplify_cfg_no_phi_merge`)
  + `bosdi.va.lower` into a JAX-traceable component, then run through
  the same circulax DC + transient path.  JIT-compiled, so the first
  call carries a one-time cost; subsequent calls reuse the compiled
  XLA program (persistent cache at `~/.cache/jax/circulax_ring`).
  Use this variant when you need `jax.grad(loss, params)` through the
  device.

VACASK and circulax both run a fixed 50 ps step (20 000 points over
1 µs); ngspice picks its own internal step.

## Reproduce

    pixi run python benchmarks/ring/run.py           # default sweep
    pixi run python benchmarks/ring/run.py 9 33      # specific N

## Latest results

<!-- RESULTS -->
| N | VACASK (µs/step) | ngspice (µs/step) | circulax-OSDI (µs/step) | circulax-VA (µs/step) | Freq VACASK (MHz) | OSDI Δf | VA Δf |
|---|------------------|-------------------|-------------------------|-----------------------|-------------------|---------|-------|
| 3 | 18.1 | 48.1 | 101.1 | 10666.2 (JIT 223s) | 910.1 | +0.4 % | +0.3 % |
| 9 | 53.4 | 109.5 | 188.2 | 12009.9 (JIT 247s) | 289.6 | -0.2 % | -0.3 % |
| 15 | 101.8 | 1400.7 | 311.3 | 12145.5 (JIT 238s) | 173.7 | +0.1 % | +0.0 % |
| 21 | dc_diverged | 1754.2 | 377.6 | 11681.7 (JIT 203s) | — | — | — |
| 27 | dc_diverged | 2285.9 | 493.9 | 11291.4 (JIT 217s) | — | — | — |
| 31 | dc_diverged | 2500.3 | 557.1 | 16149.0 (JIT 294s) | — | — | — |
| 33 | dc_diverged | 3094.8 | 646.2 | 12832.0 (JIT 216s) | — | — | — |
_2026-05-05_
<!-- /RESULTS -->

## Known failures at large N

VACASK's DC homotopy fails on this ring starting at N = 21 (logged as
"Homotopy failed." in stdout; surfaced as `dc_diverged` in the table).
circulax's two-phase homotopy (source-step + Gmin-step) completes
through N = 51 in prior runs; the sweep here stops at 33 so VACASK
has a row to fail in.

## KLU backend scaling

`klu_split_linear` and `klu_split_refactor` both use klujax's handle-based
API (merged from `klujax_rs` into klujax >= 0.5): the sparse symbolic
factorisation (fill-reduction, CSC pattern analysis) is computed **once** at
`analyze_circuit` time and held in a `KLUHandleManager`.  Each Newton step
only runs the cheap numeric phase (`klujax.factor` or `klujax.refactor`),
which is O(nnz) rather than O(N³).  ngspice repeats the full symbolic +
numeric factorisation every step.

The difference is invisible at small N (overhead-dominated) but the
scaling diverges past N ≈ 9 (sys_size ≈ 50 unknowns).

`klu_rs_*` routes through `klujax_rs` (Rust/Rayon backend at
`/home/cdaunt/code/klujax_rs-static`), which additionally parallelises the
vmap batch dimension (N device instances per group) across a Rayon thread
pool.  The Rayon break-even is n_lhs ≈ 32 for solve; at these ring sizes
(N device instances per type-group) the Rayon benefit is marginal — both
variants scale essentially identically vs ngspice.  The dominant win is the
symbolic factorisation reuse, shared by all four circulax backends.  Rayon
would become the dominant speedup at much larger batches (N ≈ 64+ instances
per group, e.g. a parametric sweep vmapped over device corners).

<!-- BACKEND_RESULTS -->
| N | sys_size | klu_split_linear | klu_split_refactor | klu_rs_linear | klu_rs_refactor | ngspice | best_cx/ngspice |
|---|----------|-----------------|--------------------|---------------|-----------------|---------|-----------------|
| 3 | 20 | 80.4 | 81.4 | 80.4 | 82.8 | 47.4 | 0.6× |
| 9 | 50 | 176.8 | 176.6 | 173.0 | 177.0 | 97.1 | 0.6× |
_2026-05-04_
<!-- /BACKEND_RESULTS -->
