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
| vacask | 1.02 | — | 500056 | 2.03 | ok |
| ngspice | 1.00 | — | 500467 | 2.00 | ok |
| circulax | 0.57 | 0.59 | 500000 | 1.14 | ok |
_2026-04-27_
<!-- /RESULTS -->
