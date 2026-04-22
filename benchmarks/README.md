# Circulax benchmarks

Cross-simulator comparisons on circuits VACASK already benchmarks.
Each subfolder is self-contained: `pixi run python <folder>/run.py`
reproduces the numbers and rewrites the folder's `README.md` results
table.

| Folder | Circuit | Sim window |
|--------|---------|------------|
| [rc/](rc/) | R-C low-pass with pulse train | 1 s, 1 µs step |
| [mul/](mul/) | 4-stage diode voltage multiplier | 5 ms, 10 ns step |
| [ring/](ring/) | 9-stage PSP103 ring oscillator + scaling sweep to N = 33 | 1 µs, 50 ps step |

## Methodology

- Wall time measured with `time.perf_counter()` on a single run
  (no multi-run averaging — order-of-magnitude comparisons).
- circulax numbers exclude first-call JIT compile (timed separately
  and reported under `compile_s`).
- VACASK and ngspice templates are the ones from
  `/home/cdaunt/code/vacask/VACASK/benchmark/{rc,mul,ring}/`, invoked
  in place.  Ring-scaling sweep (N ≠ 9) generates variants via
  `benchmarks/ring/{vacask_gen,ngspice_gen}.py`.
- A simulator that isn't on PATH, or can't run a specific config, is
  skipped with a status note in `results.csv`; nothing crashes the
  harness.

## Notes on the ring-oscillator scaling sweep

VACASK's DC homotopy is known to fail at N ≥ 33 on this circuit.
ngspice behaviour at large N hasn't been retested; ymmv.  circulax's
two-phase homotopy (source-step + Gmin-step) completes through N = 51.
The scaling table in `ring/README.md` makes this explicit.
