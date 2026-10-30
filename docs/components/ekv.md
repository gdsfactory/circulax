# EKV 2.6 native MOSFET

`circulax.components.ekv.EKV26` implements the four-terminal EKV 2.6 rev.15
channel-current and terminal-charge equations in ordinary JAX, with explicit
bulk effect, reverse source/drain operation, intrinsic and overlap charges,
impact ionization, source/drain junction currents and capacitances, trap-assisted
junction tunneling, and the reference's algebraic series-resistance correction.
It requires no OSDI or bosdi runtime.

The reference is [VA-Models commit 6099b4e12e6f66ff530f59b8d01fb909094be85a](https://github.com/dwarning/VA-Models/tree/6099b4e12e6f66ff530f59b8d01fb909094be85a/code/ekv/vacode).
The model was developed by Matthias Bucher, Christophe Lallement, Christian Enz,
Fabien Theodoloz and François Krummenacher at EPFL; the Verilog-A implementation
was developed by Wladek Grabinski with Tiburon Design Automation modifications.
The original mixed license notice is preserved verbatim in the generated Python
module and vendored VA source; the accompanying LICENSE is also retained.

```python
from circulax.components.ekv import EKV26

nmos = EKV26(TYPE=1, VTO=.5, W=3e-6, L=1e-6)
pmos = EKV26(TYPE=-1, VTO=-.5, W=6e-6, L=1e-6)
f, q = pmos(d=-1, g=-1, s=0, b=0)
```

Ports are `d,g,s,b`. There are no auxiliary unknowns: intrinsic terminal charge
and junction charge are returned directly in Q, so Circulax supplies their time
derivatives. Currents are positive into terminals. **Use a negative VTO for
TYPE=-1**, as in the upstream implementation. The source has polarity-specific
mobility/charge coefficients, so complementary instances are not exact numerical
mirrors even with equal geometry and threshold magnitudes.

Important parameter groups:

| Group | Parameters |
|---|---|
| Polarity/geometry | `TYPE` (±1), `W,L,DL,DW` (m), `M,NS` (parallel/series integer factors) |
| Channel | `VTO` (V), `KP` (A/V²), `GAMMA`, `PHI`, `COX` (F/m²), `THETA,E0,UCRIT,LAMBDA`, short/narrow-channel coefficients |
| Temperature | `TEMP,TNOM` (°C), `DTEMP` (K), `_temperature` (ambient K, default 300.15) |
| Resistance | `RSH` (Ω/square), `NRS,NRD`, optional `HDIF` (m) |
| Junction geometry | `AS,AD` (m²), `PS,PD` (m) |
| Junction physics | `js,jsw,jswg`, `cj,cjsw,cjswg`, grading/potential, breakdown and tunneling coefficients |
| Overlap | `Cgso,Cgdo,Cgbo` (F/m) |
| Multiplicity | Source `M` and simulator `_mfactor` (default 1) are both preserved |

See `tests/data/va_models/ekv26/ekv26.va` for the complete parameter list and
reference bounds. The presence of `TEMP,TNOM,HDIF,AS,AD,PS,PD` is significant in
the reference. Native defaults use NaN sentinels and internally generated
presence flags to preserve `$param_given` semantics. For example, unspecified
TEMP uses `_temperature + DTEMP`, but explicit TEMP overrides DTEMP. Specifying
HDIF infers unspecified diffusion area/perimeter; explicitly setting AD/AS to zero
suppresses that inference. NRS/NRD are used when HDIF is unspecified.

The reference's default simulator `gmin=1e-12` is fixed. Aliases and simulator
warning/operating-point metadata are not exposed. The `Noise` switch remains
because the source places resistance-correction calculations inside that branch;
its deterministic consequences are preserved, while noise-source outputs are
excluded. `AF` and `KF` are retained reference parameters but there is no noise
analysis implementation. Use source-valid physical parameter ranges. Inactive
expression guards prevent JAX both-branch evaluation from overflowing; equivalence
is tested over physical bias/parameter ranges, not unphysical extremes.

**PMOS overlap caveat inherited from the reference:** the source computes overlap
charge from raw `V(g,s)`/`V(g,d)`/`V(g,b)`, then multiplies its stamp by TYPE. This
produces negative overlap capacitance for TYPE=-1 with positive overlap
coefficients. EKV26 preserves this behavior for independent reference parity.
For a generic PMOS with conventional positive overlap capacitances, use
`circulax.components.mosfet.PMOSDynamic`. Changing EKV26's overlap sign would
change the reference equations.

## Verification and generation

Run `.venv/bin/python -m pytest tests/test_ekv.py -q`. Tests cover both polarities,
weak/linear/saturated/reversed channel bias, current and charge conservation,
common-mode invariance, body effect, overlap charge, geometry/temperature,
parameter-presence semantics, width gradients, actual resistor-loaded MOS
circuit DC/AC analysis and comparison with compiled OSDI circuit responses.

Optional tests compile the vendored source with `openvaf-r` and evaluate it using
`osdi_loader`/`osdi_jax`. F/Q and the full G/C matrices are compared across six
bias vectors and eight parameter sets. These include explicit and inferred
junction geometry, finite resistance, changed mobility/mismatch coefficients,
series/parallel multiplicity and temperature. Checks use relative tolerances
2e-10 for F/Q and 2e-9 for G/C. Optional references skip when tools are absent.
Full transient-trajectory/HB parity has not been measured.

All continuous physical parameters remain differentiable JAX inputs, including
setup-dependent geometry and temperature. Discrete switches and changing whether
a parameter is present are not differentiable operations. A width derivative of
channel current is checked against finite differences.

`tests/data/va_models/ekv26/regenerate.py` records its validated bosdi source
revision. It lowers unoptimized MIR to ordinary `@component` source, preserves
runtime parameter-presence flags and all four nodes, normalizes callback/cast
representations, compacts setup-cache indices and repairs an eval-phi reference.
Run with that bosdi source on PYTHONPATH, then run `ruff check --fix` and
`ruff format` on the emitted module. Original source includes and notices are
vendored; generation and reference tests leave no binary in the source directory.

On the development CPU (JAX 0.7.2/Python 3.12.14), the default four-variable model
took **0.23 s to trace and 0.42 s to compile F/Q**, and **0.61 s to trace and
2.46 s to compile its voltage Jacobian**. Warm calls including dispatch and
synchronization averaged 250 µs for F/Q and 182 µs for the Jacobian over 100
repeats. These are component measurements, not circuit-solve timings.

```sh
.venv/bin/python -m benchmarks.sample_components circulax.components.ekv EKV26 \
  --bias '[1,1,0,0]'
```
