# Circulax benchmarks

Benchmarks now live under one tree: `benchmarks/`.

Use the top-level runner to discover and execute cases:

```sh
pixi run python benchmarks/run.py list
pixi run python benchmarks/run.py run ring -- 3 9
pixi run python benchmarks/run.py run release
```

The release suite is intentionally small and reproducible:

| Case | Circuit | Purpose |
|------|---------|---------|
| `rc` | R-C low-pass with pulse train | Cross-simulator timing baseline |
| `mul` | 4-stage diode voltage multiplier | Nonlinear transient comparison |
| `ring` | PSP103 ring oscillator | OSDI compact-model scaling through bosdi |

Additional OSDI smoke and accuracy cases are available as `juncap200`,
`mosvar`, and `ring-bsim4`. Older exploratory testbenches are retained under
`benchmarks/legacy/` and exposed through `legacy-*` runner names.

## OSDI device path

The production compact-model path is OSDI through bosdi. Verilog-A source is
compiled by OpenVAF to a `.osdi` shared library; bosdi loads that binary and
circulax calls it during assembly. The OSDI ABI exposes currents,
conductances, charges, and capacitances, so circulax can form Newton
Jacobians without tracing the compact-model source through JAX.

## Methodology

- Wall time is measured with `time.perf_counter()` around the timed run.
- Circulax timings report setup/JIT separately where relevant.
- VACASK and ngspice cases are skipped with status notes when their binaries
  or upstream templates are unavailable.
- Per-case `run.py` scripts own their local `results.csv` and README table.
