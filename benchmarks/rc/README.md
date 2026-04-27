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
| vacask | 1.27 | — | 1005006 | 1.26 | ok |
| ngspice | 1.35 | — | 1006013 | 1.33 | ok |
| circulax | 6.61 | 0.53 | 1000000 | 6.61 | ok |
_2026-04-27_
<!-- /RESULTS -->
