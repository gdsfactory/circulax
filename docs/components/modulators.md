# Composed interferometers and modulators

The factories in `circulax.components.modulators` build actual coupler/arm and
loaded-electrode circuits. Hierarchy is flattened before JIT compilation, so a
subcircuit is an API/topology boundary rather than a compilation boundary.

## MZI and loaded lumped MZM

`mach_zehnder(phase_a=0, phase_b=0, loss_a_dB=0, loss_b_dB=0,
coupling_in=0.5, coupling_out=0.5)` returns a reciprocal four-port optical circuit
with ports `in,in_aux,out,out_aux`. Two directional couplers connect independently
attenuated phase arms. Positive arm phase is `exp(+i phase)`; arm insertion losses
are power dB. Both input ports and both complementary outputs are real scattering
interfaces and need matched boundaries when measuring transmission.

For balanced couplers and incident field 1 at `in`, define
`ta=10**(-loss_a_dB/20)*exp(i*phase_a)` and similarly `tb`. The coherent outputs are
`out=(ta-tb)/2` and `out_aux=i*(ta+tb)/2`. This retains output phase, chirp and
arm imbalance instead of substituting an intensity-only cosine law.

`lumped_mzm(push_pull=True, efficiency=1, bias_phase=-pi/2,
resistance=10, capacitance=64e-15, loss_a_dB=0, loss_b_dB=0)` uses two
[loaded phase arms](phase_shifters.md). It exposes the four optical ports,
electrical input `p,n`, and internal `junction_a,junction_b`. An ideal lossless
transformer creates arm drives `(+V/2,-V/2)` for push-pull or `(V,0)` for single-arm
operation. It reflects both arm currents to the input, preserving finite input
loading. The internal electrical common reference is grounded. For single-arm
operation the second arm remains a biased optical reference. `efficiency` is
rad/V; `bias_phase` shifts the second arm. No supply/output limiting is modeled.

```python
from circulax.components.modulators import mach_zehnder, lumped_mzm

interferometer = mach_zehnder(phase_b=-1.5707963267948966, loss_b_dB=0.2)
modulator = lumped_mzm(
    push_pull=True, efficiency=1.2, resistance=25.0,
    capacitance=80e-15, loss_a_dB=0.1, loss_b_dB=0.1,
)
# Register modulator as a model in a parent netlist, drive p-n, and match optics.
```

## Traveling-wave wrapper

`traveling_wave_mzm(differential_line=..., common_line=..., length_m=2e-3,
n_sections=6, rj_product=4.7e-3, cj_per_m=320e-12,
phase_settings=None, bias_phase=-pi/2)` promotes the reference notebook's
modal-electrode/dual-arm topology. Supply existing `RLGCTransmissionLine`
objects, normally obtained with `mode_transmission_line`; their fit band/error
and discretization limitations still apply.

Each section has two half-line ladders per electrical mode, a
`DifferentialAdapter`, and two series-resistance/junction-charge branches.
Resistance scales as `rj_product/dz` (ohm m / m); capacitance scales as
`cj_per_m*dz` (F/m * m). `phase_settings` are `PNPhaseShifter` settings;
default section length and capacitance are derived from `dz`. Keep any explicit
length/capacitance overrides consistent with the electrical section geometry.
Optical sections retain real flight delay and carrier phase, with phase modulation
sampled at their midpoint. Splitter, two arms, bias element and combiner form a
real interferometer. There is no separately inserted bandwidth filter.

Ports are `d_in,d_out,c_in,c_out,optical_in,optical_out` and
`junction_a_0...junction_b_{n_sections-1}`. Source and load impedances are external
and mode dependent; the wrapper deliberately adds neither. The unused optical
coupler ports are terminated at the fixed 1-ohm optical reference.

```python
import numpy as np
from circulax.components.transmission_line import mode_transmission_line
from circulax.components.modulators import traveling_wave_mzm

f = np.r_[0.0, np.geomspace(1e6, 80e9, 80)]
d = mode_transmission_line(f, attenuation_db_per_m=250., n_eff=2.2, impedance=110.)
c = mode_transmission_line(f, attenuation_db_per_m=600., n_eff=2.7, impedance=45.)
mzm = traveling_wave_mzm(differential_line=d, common_line=c, n_sections=4)
```

The supporting `electrode_netlist` and NumPy analysis helper `loaded_group_index`
are promoted unchanged from `traveling_wave_mzm_tutorial.py` (the latter computes
Bloch group index of the loaded symmetric-T unit cell, not a fitted bandwidth).
`optical_netlist(arm_a, arm_b, quadrature_phase=-pi/2)` promotes the reference
notebook's optical connectivity, parameterizing its former global bias.
`QuadratureCombiner` promotes the older notebook's **unilateral buffered** output
constraint `(active+i*reference)/2`; it is retained solely for source compatibility
and does not claim passive reciprocal scattering. New MZI/MZM factories use actual
couplers instead. Both traveling-wave notebooks now import their library cores;
the notebook retains its PRBS, eye folding and analysis code. The tutorial helper
module exposes compatibility imports for the promoted builders.

## Independent verification and remaining scope

Tests compare compiled MZI complex outputs and complementary power against an
independent NumPy matrix product, including unequal arm losses and phases. Lumped
MZM tests cover single-arm/push-pull DC phase, both optical sideband quadratures,
RC EO response, and reflected source current against independently derived
admittance. The full compiled real-block F/Q Jacobians provide an independent dense
reference for `quadrature_response`, including both optical quadratures. The
legacy `ac_response` API refuses non-holomorphic complex EO circuits; use
`quadrature_response` for these devices.

Modal electrode tests compare compiled two- and four-section loaded circuits
against independent NumPy ABCD cascades, including RF loss, source/load mismatch,
constant junction charge, and series resistance. Delayed junction pulse tests
check full optical flight and carrier phase. The runnable [wrapper example](../../examples/photonic/library_traveling_wave_mzm.py)
checks compiled DC power and three-frequency EO response against independently
cascaded NumPy ABCD matrices plus midpoint optical propagation. Full wrapper
tests use one, two and four sections and show strictly decreasing error against
a 128-section independent reference. A second case deliberately changes RF loss,
source/load mismatch and optical/RF velocity mismatch. Complete high-speed eyes,
nonlinear large-signal section convergence and HB remain unverified here.
No experimental, BER, noise or external VA validation is claimed. No GPL reference
code or proprietary simulator is required.

A two-section representative wrapper has 105 complex unknowns. In a fresh CPU
process using complex128 and a dynamic section phase-efficiency parameter,
assembled F/Q tracing took 360 ms, XLA compilation 253 ms, warmed F/Q 70 µs,
and a full 210-column real-block Jacobian 366 µs (20 warmed evaluations;
Jacobian cold trace+compile 579 ms). These are residual/Jacobian costs with the
steady delay lookup, not Newton or history-buffer simulation timings. The
runnable example separately exercises compiled DC and frequency solves.
