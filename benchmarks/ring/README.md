# Ring oscillator

N-stage CMOS ring oscillator (PSP103, NMOS W = 10 µm / PMOS W = 20 µm,
L = 1 µm, VDD = 1.2 V).  1 µs transient at 50 ps fixed step → 20 000
time points.  Canonical N = 9 mirrors
`/home/cdaunt/code/vacask/VACASK/benchmark/ring/`; other N values
are generated alongside via `vacask_gen.py` / `ngspice_gen.py`.

Circulax is measured on two paths:

- **OSDI** — PSP103 via the bosdi FFI path to the compiled
  `psp103v4.osdi` binary.  Full-fidelity physics, same model VACASK
  uses.  This is the production circulax path.
- **XLA** — the simplified pure-JAX MOSFET from
  `tests/fixtures/mosfet_simple.py` (softplus-smoothed square-law +
  single-piece Meyer gate cap, calibrated against PSP103 at three
  bias points).  Runs entirely inside the XLA graph — no FFI, no
  OSDI.  Device physics is ~20 % off PSP103 quantitatively, so the
  oscillation frequency shifts slightly.  Included here to isolate
  how much of circulax's per-step wall time lives at the OSDI
  boundary.

ngspice is excluded from this benchmark — see the note below.

## Reproduce

    pixi run python benchmarks/ring/run.py           # default sweep
    pixi run python benchmarks/ring/run.py 9 33      # specific N

## Latest results

<!-- RESULTS -->
| N | VACASK (s) | OSDI (s) | XLA (s) | Freq VACASK (MHz) | OSDI Δf | XLA Δf |
|---|------------|----------|---------|-------------------|---------|--------|
| 3 | 0.54 | 6.96 | 0.78 | 910.1 | +0.4 % | +14.7 % |
| 9 | 1.23 | 8.28 | 0.95 | 289.6 | -0.2 % | +1.9 % |
| 15 | 2.03 | 7.27 | 1.43 | 173.7 | +0.1 % | +1.9 % |
| 21 | dc_diverged | 8.29 | 1.67 | — | — | — |
| 27 | dc_diverged | 8.56 | 1.47 | — | — | — |
| 31 | dc_diverged | 8.85 | 1.33 | — | — | — |
| 33 | dc_diverged | 9.65 | 1.30 | — | — | — |
_2026-04-22_
<!-- /RESULTS -->

## Known failures at large N

VACASK's DC homotopy fails on this ring starting at N = 21 (logged as
"Homotopy failed." in stdout; surfaced as `dc_diverged` in the table).
circulax's two-phase homotopy (source-step + Gmin-step) completes
through N = 51 in prior runs; the sweep here stops at 33 so VACASK
has a row to fail in.

## Why ngspice is excluded

The upstream template loads `psp103v4.osdi` via `pre_osdi`, but
ngspice 45 only supports OSDI 0.3 while the `openvaf-r` binary on
this machine (Arpad Buermen's VACASK-aligned fork, v23.5.0) emits
OSDI 0.4.  No `openvaf-r` flag selects 0.3.  To add ngspice back,
install upstream `pascalkuthe/OpenVAF`, recompile `psp103v4.osdi`,
drop it in `/home/cdaunt/code/vacask/VACASK/benchmark/ring/ngspice/`,
and reinstate ngspice in the sweep loop in `run.py`.  rc and mul
don't hit this — they only use the built-in sp_* OSDI modules whose
0.3/0.4 formats happen to be compatible across both compilers.
