# Circulax benchmarks

Each subfolder is self-contained: `pixi run python <folder>/run.py`
reproduces the numbers (and where applicable rewrites the folder's
`README.md` results table).

## Cross-simulator wall-time comparisons

These run vacask + ngspice + circulax on identical inputs and report
µs/step.  Layout: each subfolder has `vacask/`, `ngspice/`, `circulax/`
sub-subfolders so the netlist for each simulator is discoverable in
isolation.

| Folder | Circuit | Sim window |
|--------|---------|------------|
| [rc/](rc/)     | RC low-pass with pulse train          | 1 s, 1 µs step |
| [mul/](mul/)   | 4-stage diode voltage multiplier (CW)  | 5 ms, 10 ns step |
| [ring/](ring/) | PSP103 N-stage ring oscillator (3 → 33) | 1 µs, 50 ps step |

## Waveform-accuracy comparisons (circulax vs ngspice)

Compare waveforms point-by-point and report max-error / RMS-error per
node.  Layout: top-level `run.py` orchestrates everything,
`ngspice/<name>.cir` holds the ngspice setup, circulax setup is inline.

| Folder | Circuit | Notes |
|--------|---------|-------|
| [fullwave_rect/](fullwave_rect/) | Full-wave bridge rectifier | transient |
| [diode_clipper_hb/](diode_clipper_hb/) | Diode clipper, harmonic balance | HB vs transient vs ngspice |

## Circulax-internal comparisons (no external simulator)

Sweep linear-solver backends, integrators, or solver parameters on
circulax-only setups.

| Folder | What it sweeps |
|--------|----------------|
| [lc_ladder/](lc_ladder/)             | KLU split / refactor / non-split, klu_rs, scaled to thousands of sections |
| [stiff_newton/](stiff_newton/)       | Stiff Newton convergence on LC + diode chain |
| [diode_cascade_solver/](diode_cascade_solver/) | Diode cascade internal solver knobs |

## Specialty

| Folder | Purpose |
|--------|---------|
| [ring_bsim4/](ring_bsim4/) | N=9 ring with BSIM4 (different MOSFET model than ring/'s PSP103) |

## Shared utilities

`bench_utils/` — ngspice runner, plotting helpers, accuracy metrics,
common multi-solver runner — used by the waveform-accuracy benchmarks.

## Methodology

- Wall time measured with `time.perf_counter()` on a single run after
  one warmup pass (so JIT compile isn't counted in the timed run).
- VACASK and ngspice run on templates symlinked from upstream
  `VACASK/benchmark/{rc,mul,ring}/{vacask,ngspice}/`.
- Simulators that aren't on PATH, or can't run a specific config, are
  recorded with a status note in `results.csv`; the harness never
  crashes.
