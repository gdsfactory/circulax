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
| 3 | 17.7 | 62.6 | 117.3 | 12944.4 (JIT 243s) | 910.1 | +0.4 % | +0.3 % |
| 9 | 52.5 | 117.9 | 254.1 | 15044.1 (JIT 297s) | 289.6 | -0.2 % | -0.3 % |
| 15 | 99.6 | 1502.8 | 285.8 | 20466.3 (JIT 473s) | 173.7 | +0.1 % | +0.0 % |
| 21 | 121.4 | 2247.1 | 365.8 | 20822.7 (JIT 411s) | 124.1 | +0.2 % | +0.1 % |
| 27 | 179.8 | 3126.3 | 474.3 | — | 96.5 | +0.0 % | — |
_2026-05-18_
<!-- /RESULTS -->

## VACASK convergence fix for large N

VACASK's DC homotopy fails for this ring starting at N = 21.  Diagnostic
analysis (`op_debug=2 homotopy_debug=1`) shows the failure sequence:

- Direct Newton diverges in 100 iterations from the zero initial condition
- `gdev` stepping: all 6 gmin levels (1e-3 → 1e2) fail in 100 iterations each
- `gshunt` stepping: makes it to gshunt ≈ 3e-4 after 100 steps, then stalls
- `src` stepping: reaches srcfact ≈ 0.116 and bounces for 100 steps before giving up

The root cause is that an odd-stage ring oscillator has only **one** DC
equilibrium (the metastable VDD/2 saddle point), and standard homotopy
continuation cannot cross the bifurcation at srcfact ≈ 0.116 from DC=0.

**Fix** (confirmed by sweep across 11 option combinations for N = 21 and N = 27):
use `icmode="uic"` on the `tran` analysis line to skip the OP solve entirely,
with all ring nodes initialised to VDD/2 = 0.6 V.  The ring oscillates
naturally from that initial condition once the 10 µA current-source pulse
kicks at t = 1 ns.

```
analysis tran1 tran ... icmode="uic" ic=[ "1"; 0.6; "2"; 0.6; ... ]
```

`vacask_gen.py` applies this automatically for N ≥ 21.  The oscillation
frequencies obtained this way (N=21: 124 MHz, N=27: 97 MHz, N=31: 84 MHz,
N=33: 79 MHz) follow 1/N scaling consistent with the N=15 reference (174 MHz)
to within < 0.3 %.

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
