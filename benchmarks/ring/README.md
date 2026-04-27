# Ring oscillator

N-stage CMOS ring oscillator (PSP103, NMOS W = 10 µm / PMOS W = 20 µm,
L = 1 µm, VDD = 1.2 V).  1 µs transient at 50 ps fixed step → 20 000
time points.  Canonical N = 9 mirrors
`/home/cdaunt/code/vacask/VACASK/benchmark/ring/`; other N values
are generated alongside via `vacask_gen.py` / `ngspice_gen.py`.

Three simulators are compared, all running the same PSP103 device
(`psp103v4.osdi`, compiled from `psp103.va` by openvaf-r 23.5.0):

- **VACASK** — reference C++ simulator (`vacask --skip-embed
  --skip-postprocess`).
- **ngspice** — ngspice 45.2 batch mode (`ngspice -b`), KLU direct
  linear solver, `pre_osdi psp103v4.osdi`.  ngspice 45.2 happily loads
  OSDI 0.4 binaries; the harness compiles `psp103v4.osdi` into the
  ngspice working directory on demand.
- **circulax** — JAX-based, calls the same `psp103v4.osdi` binary via
  the bosdi FFI shim.  Differentiable through terminal voltages.

VACASK and circulax both run a fixed 50 ps step (20 000 points over
1 µs); ngspice picks its own internal step.

## Reproduce

    pixi run python benchmarks/ring/run.py           # default sweep
    pixi run python benchmarks/ring/run.py 9 33      # specific N

## Latest results

<!-- RESULTS -->
| N | VACASK (µs/step) | ngspice (µs/step) | circulax (µs/step) | Freq VACASK (MHz) | circulax Δf |
|---|------------------|-------------------|--------------------|-------------------|-------------|
| 3 | 17.5 | 43.4 | 84.1 | 910.1 | +0.4 % |
| 9 | 49.9 | 95.8 | 166.2 | 289.6 | -0.2 % |
| 15 | 85.9 | 1290.1 | 251.2 | 173.7 | +0.1 % |
| 21 | dc_diverged | 1738.1 | 322.8 | — | — |
| 27 | dc_diverged | 2242.7 | 415.4 | — | — |
| 31 | dc_diverged | 2537.8 | 506.1 | — | — |
| 33 | dc_diverged | 2751.2 | 441.6 | — | — |
_2026-04-27_
<!-- /RESULTS -->

## Known failures at large N

VACASK's DC homotopy fails on this ring starting at N = 21 (logged as
"Homotopy failed." in stdout; surfaced as `dc_diverged` in the table).
circulax's two-phase homotopy (source-step + Gmin-step) completes
through N = 51 in prior runs; the sweep here stops at 33 so VACASK
has a row to fail in.
