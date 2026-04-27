# circulax ring oscillator setup

`bench_circulax.py` builds an N-stage CMOS ring oscillator in circulax
and times the transient simulation.  Two variants:

- `--variant osdi` — PSP103 device via the bosdi FFI path (production
  circulax path; full-fidelity physics).
- `--variant jax-native` — simplified pure-JAX MOSFET (no FFI, no
  OSDI); useful for isolating the cost of the OSDI boundary, with
  device physics ~20 % off PSP103.

Direct invocation:

    pixi run python benchmarks/ring/circulax/bench_circulax.py --n-stages 9 --variant osdi
