# RC low-pass

Pulse source (0 → 1 V, 1 µs rise/fall, 1 ms width, 2 ms period) into a
1 kΩ resistor driving a 1 µF cap to ground.  1 s transient at 1 µs
fixed step → 1 M time points.  Mirrors
`/home/cdaunt/code/vacask/VACASK/benchmark/rc/`.

## Reproduce

    pixi run python benchmarks/rc/run.py

## Latest results

<!-- RESULTS -->
| Simulator | Wall (s) | Compile (s) | n_steps | µs/step | Status |
|-----------|----------|-------------|---------|---------|--------|
| vacask | 0.99 | — | 1005006 | 0.97 | ok |
| ngspice | 1.22 | — | 1006013 | 1.20 | ok |
| circulax | 4.04 | 0.61 | 1000000 | 4.04 | ok |
_2026-06-23_
<!-- /RESULTS -->
