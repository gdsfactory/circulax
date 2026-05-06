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

## Circulax device paths

Circulax can simulate devices through three distinct execution paths,
each with different performance and differentiability characteristics:

**OSDI (bosdi)** — the production path.  Verilog-A is compiled by
OpenVAF to a `.osdi` shared library; bosdi loads it at runtime and
calls it via JAX's XLA custom-call FFI.  The OSDI ABI exposes
currents, conductances (∂I/∂V), charges, and capacitances (∂Q/∂V),
so circulax forms the Jacobian and provides `jax.grad` w.r.t. terminal
voltages analytically.  Model parameters are opaque to JAX — gradients
w.r.t. `TOXE`, `VTH0`, geometry, etc. are not available without an
additional adjoint layer.  No JIT overhead; compile-once-run-anywhere.

**VA / MIR → XLA** — the differentiable-model path.  Verilog-A source
is compiled by `openvaf-r --dump-mir` to a MIR dump, then lowered by
`circulax-va` into a pure Python/JAX module (a `@va_component` class
with physics and Jacobian expressed entirely as JAX primitives).  The
result is a native XLA kernel: no FFI boundary during simulation, full
`jax.grad` through all device parameters.  First-call JIT compile is
significant (~320 s per function for PSP103 on CPU, two functions for
DC + transient); subsequent calls hit the XLA cache.  Enables
gradient-based model calibration, sensitivity analysis, and
differentiable co-optimisation of device + circuit.

**XLA-native (simplified)** — lightweight approximation used in the
ring benchmark to isolate per-step solver cost.  A hand-written JAX
MOSFET (softplus square-law + Meyer cap) with no Verilog-A involved.
~20 % off PSP103 quantitatively but compiles in ~2 s.

| Property | OSDI (bosdi) | VA (MIR→XLA) | XLA simplified |
|----------|--------------|--------------|----------------|
| Device model | PSP103 (full) | PSP103 (full) | square-law |
| FFI call per step | ✓ | ✗ | ✗ |
| ∂/∂ terminal voltages | ✓ | ✓ | ✓ |
| ∂/∂ model parameters | ✗ | ✓ | ✓ |
| First-call JIT (CPU, PSP103 9-stage ring) | ~0 s | ~640 s | ~2 s |
| DC from zeros | ✓ | requires VDD/2 warm start | ✓ |

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
