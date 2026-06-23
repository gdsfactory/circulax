# Diode voltage multiplier

4-stage Cockcroft-Walton multiplier driven by a 50 V / 100 kHz sine
through a 10 mΩ series R, with C = 100 nF between stages and D1N4007
diodes.  5 ms transient at 10 ns fixed step → 500 k time points.
Mirrors `/home/cdaunt/code/vacask/VACASK/benchmark/mul/`.

## Reproduce

    pixi run python benchmarks/mul/run.py

## Latest results

<!-- RESULTS -->
| Simulator | Wall (s) | Compile (s) | n_steps | µs/step | Status |
|-----------|----------|-------------|---------|---------|--------|
| vacask | 1.05 | — | 500056 | 2.08 | ok |
| ngspice | 1.06 | — | 500467 | 2.10 | ok |
| circulax | 0.68 | 0.79 | 500000 | 1.37 | ok |
_2026-06-23_
<!-- /RESULTS -->
