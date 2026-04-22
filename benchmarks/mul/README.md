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
| vacask | 1.04 | — | 500056 | 2.06 | ok |
| ngspice | 1.11 | — | 500467 | 2.20 | ok |
| circulax | 0.48 | 0.55 | 500000 | 0.96 | ok |
_2026-04-22_
<!-- /RESULTS -->
