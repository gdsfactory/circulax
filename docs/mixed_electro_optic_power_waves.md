# Mixed electro-optic phase-shifter power waves

This is the `phase_shifter_factory` milestone of [issue #112](https://github.com/gdsfactory/circulax/issues/112),
based on delay commit `ffb35317f483b7389c2a539c0f0220b27bce15df` and its
native-passive parent `0ba4de2edf323e9c149170c8fc0d3ad8f93ffeeb`.
It does not complete the issue. Detectors and the remaining conversion paths
are separate work.

## Audit and physical laws

The starting phase shifter already had reciprocal optical propagation and four
complex directional states. It did not use S-to-Y conversion. This migration
reuses that realization, its callable/table/junction APIs, and its single-delay
history interface. It adds shared retained power-wave construction, optical
reference normalization and public total-field observables.

For optical ports `o1`, `o2`, let `a1=a_forward`, `a2=a_reverse`. With a common
finite positive real static `r0`:

```text
potential = sqrt(r0)*(a+b)
flow = (a-b)/sqrt(r0)                       # positive into the component
F_incident = a+b-potential/sqrt(r0); Q_optical = 0
P_incident = |a|²; P_outgoing = |b|²
P_net_into_component = |a|²-|b|²
```

Amplitudes are complex envelopes in sqrt(W); powers are W. Both incident-wave
constraints remain, for arbitrary returning light and simultaneous coherent
illumination. No external matching is assumed. No inverse of `I+S`, loss
perturbation, reference perturbation, optical-ground penalty, or leakage is
introduced. Solver convergence defaults are unchanged.

The shared helper accepts optional ordered `incident_states`, allowing this
native model to retain its original directional names. Its optical terminal
row is the equivalent algebraic row used for `OpticalDelayLine`:

```text
F_optical_terminal = (2*a-potential/sqrt(r0))/sqrt(r0)
```

This is physical flow plus `F_incident/sqrt(r0)`. It has the same physical
solutions for arbitrary loads, while depending only on the current state.
Optical Q is zero. At solved samples the complete signed optical flow therefore
needs no history buffer. Residuals away from a solution change under this row
operation; do not compare old and new optical residuals element by element.

The electrical junction is unchanged:

```text
v_now = Re(V_anode-V_cathode)
length_m = length_um*1e-6
F_anode = length_m*current_per_m(v_now); F_cathode = -F_anode
Q_anode = length_m*charge_per_m(v_now);   Q_cathode = -Q_anode
```

The electrical voltage is in V, current in A, charge in C. `r0` affects only
optical coordinates. Optical absorption does not create electrical current in
this model. With `junction=None`, neither electrode receives any F or Q stamp;
electrode capacitance and resistance belong in the external circuit. Current
and charge conservation, length scaling, built-in diode/depletion laws,
capacitance integration, callable parameter merging and table APIs are retained.

## Two half-transits and midpoint modulation

```text
tau = length_um*n_group/2.99792458e14                  # seconds
h = tau/2
m_forward(t) = a1(t-h); m_reverse(t) = a2(t-h)
v_mid(t) = Re(V_anode(t-h)-V_cathode(t-h))
T(v) = 10**(-loss_dB_cm(v)*length_um/10000/20)
       * exp(-2j*pi*(neff+delta_neff(v))*length_um*1000/wavelength_nm)
b1(t) = T(v_mid(t))*m_reverse(t-h)
b2(t) = T(v_mid(t))*m_forward(t-h)
```

Thus optical transit is the full `tau`; the voltage controlling output at t
is from `t-tau/2`. Transmission multiplies a wave that has crossed the midpoint
half a transit earlier. Both directions use the same midpoint voltage and
transmission. A reflected pulse encounters this modulation twice, at different
times. At steady state `b1=T(v)*a2`, `b2=T(v)*a1`.

One `signals.at_delay(h)` snapshot supplies the voltage, incident states and
midpoint states. The two delayed midpoint equations realize the full transit
without a second distinct delay or generalized multiple-delay support. The
negative carrier phase, dB/cm attenuation, um length and nm wavelength units
are unchanged. Zero group delay reduces to instantaneous reciprocal modulation;
zero length also removes optical attenuation and phase. Exact/near-zero loss
and carrier phases 0, pi/2, pi require no numerical workaround. This migration
does not introduce new clipping or validation of voltage-dependent material
laws; callers remain responsible for their physical domains.

Transient uses accepted history with linear interpolation, including interpolation
against the current Newton trial when a step exceeds h. DC supplies constant
history. Before transient `t0`, all unknowns are held at the consistent `y0`
(or the automatically calculated time-zero DC solution). There is no arbitrary
prehistory input. Start the solve earlier for another past trajectory. `saveat`
interpolates solved states and is not the accepted history mesh. Refinement is
necessary even when a step longer than h converges, especially across many
short modulator sections. See [delay history details](bidirectional_optical_delays.md).

JAX preserves both real/imaginary blocks: voltage-dependent laws use the real
electrical voltage and are non-holomorphic. Current-state, history, length and
material-parameter derivatives remain available. Delay interpolation derivatives
are piecewise smooth between accepted-time knots. Bracket changes, zero-delay
boundaries, table knots, adaptive decisions and capacity transitions need not
have globally smooth sensitivities. Compare finite differences within a single
smooth region; the derivative of a coarse discrete solve need not equal the
continuum response. Table extrapolation/integration behavior is unchanged.

## Public use and observables

`phase_shifter_factory(..., r0=1.0)` preserves the default reference. `r0` is a
class construction coordinate, not a swept reference or an S renormalization.
Use it consistently across connected optical components, with compatible
carrier frame, phase/time convention and mode. Connect two optical terminals
per node and use physical splitters for branching; these connection rules are
not yet enforced by the compiler. Electrical ground remains electrical.

```python
from circulax import compile_circuit
from circulax.components import (
    optical_wave_source, optical_wave_termination, phase_shifter_factory,
)
from circulax.components.electronic import VoltageSource

r0 = 50.0
Phase = phase_shifter_factory(delta_neff=lambda v: -2e-4*v, r0=r0)
circuit = compile_circuit({
    "instances": {
        "PS": {"component": "phase"},
        "S": {"component": "source", "settings": {"amplitude": 0.6+0.8j}},
        "L": {"component": "load"},
        "V": {"component": "voltage", "settings": {"V": -1.0}},
        "GND": {"component": "ground"},
    },
    "connections": {
        "S,p": "PS,o1", "PS,o2": "L,p", "V,p1": "PS,anode",
        "GND,p1": ("V,p2", "PS,cathode"),
    },
}, {
    "phase": Phase, "source": optical_wave_source(r0=r0),
    "load": optical_wave_termination(r0=r0), "voltage": VoltageSource,
}, is_complex=True, backend="dense", g_leak=0.0)
y = circuit.dc()
waves = circuit.port_waves(y, "PS,o2")
assert abs(float(waves.outgoing_power)-1.0) < 1e-10
```

`Circuit.port_waves` supports phase-shifter optical terminals in DC, transient
samples (`analysis="tran", t=sol.ts`) and HB time samples
(`analysis="hb_time", t=sample_times`). Pass the same scalar parameter updates
used for the solve. `PortWaves` exposes incident/outgoing amplitudes and powers
and signed net power into the component. It reconstructs from local potential
and complete terminal F, without knowing directional-state names. A compiler
group now carries `wave_ports`: only `o1`, `o2` qualify for a phase shifter.
`PS,anode` and `PS,cathode` explicitly reject traveling-wave reconstruction,
including when junction current and charge are nonzero. A group reference alone
is insufficient. Other delayed/charged components remain unsupported unless
they declare the complete optical-flow contract.

`Circuit.port` continues to return potential. Its squared magnitude is generally
not traveling power, including at `r0=1`; the potential contains interference
between incoming and returning waves. Existing `OpticalSource` and
`OpticalSourcePulse` retain their potential-difference semantics. Use explicit
launched-wave sources for prescribed emitted power and absorbing returns.

## Analyses and compatibility

| Analysis | Supported behavior and limits |
| --- | --- |
| DC | Reciprocal constant-history scattering and original electrical I/Q laws. |
| Transient | Two half-delays, midpoint voltage, electrical dQ/dt, constant prehistory, optical port waves at saved samples. |
| `small_signal` | Full real-block voltage/field quadratures, optical perturbation delay tau, voltage-to-field delay tau/2, original junction conductance and capacitance. Independent optical quadrature drives are supported. |
| `ac_response` | Remains explicitly rejected for this non-holomorphic mixed optical model. Use `small_signal`. |
| HB | Periodic full-complex FFT shift by h, preserving positive and negative envelope harmonics, nonlinear midpoint modulation and electrical Q. Read powers from time samples. Harmonic truncation/aliasing still requires convergence checks. |

Small-signal RF phasor phase and optical quadrature are separate coordinates.
`SmallSignalResponse.power(..., field_unit="sqrt(W)")` is a normalized-field
intensity derivative `2*(x_dc*dx+y_dc*dy)`. It is not a general port-power API.
Declared wave-component terminal names (including mixed electrodes) now reject
this helper because terminal potential is not an incident/outgoing wave.
For legacy/custom field aliases callers still explicitly assert normalization
and matching through `field_unit`; aliases do not establish physical matching.
Under reflection, or at a non-unit optical reference, do not apply this helper
to potential aliases. Quadrature/voltage access remains supported. AC traveling
power reconstruction is deferred; `Circuit.port_waves` continues to reject AC
perturbations and HB coefficient labels.

The port order `(anode,cathode,o1,o2)` and state order
`(a_forward,a_reverse,m_forward,m_reverse)` are unchanged. At r0=1 the physical
unknown values, carrier response, propagation and electrical equations are
compatible. Incident constraint signs and optical terminal rows change, so
custom residual consumers must update. At other references optical potentials
scale with sqrt(r0), flows with its inverse; directional states remain in
sqrt(W). Recompile circuits for the new metadata and rebuild raw guesses/history
when changing reference. Prefer public observables to raw states. A supplied
raw transient `y0` must include consistent electrical, incident and midpoint
states. This does not make the non-holomorphic delayed component SAX-exportable.

## Detector audit and separate follow-up

There are no detector implementations in `circulax/components` in this starting
tree. `examples/photonic/photodetectors.ipynb` defines `IdealPhotodetector`,
`Photodiode`, `CarrierPhotodiode` and their `optical_conversion` helper.
The helper uses `(1-reflection)/(1+reflection)` as optical admittance and
`field/(1+reflection)` to reconstruct incident light at r0=1. It correctly
bases generation on absorbed power `(1-|reflection|²)*|incident|²`, with
absorbed-power responsivity, but that conversion is singular at reflection=-1.
The notebook-local two-terminal `OpticalCW` source is a unit-reference launched
boundary with an optical return node. None of these definitions is migrated here.

Concrete follow-up work:

- Replace detector optical admittance/division with a retained incident constraint,
  `b=reflection*a` and shared reference normalization, while retaining absorbed
  power and avoiding double-counting external responsivity/reflection loss.
- Expose only `optical_in` as optical. Preserve anode/cathode signs, dark I/V,
  depletion charge, carrier populations, screening, recombination and collection
  laws, along with full real derivatives and explicit analysis limits.
- Replace notebook-local source boundaries with explicit one-terminal sources;
  validate reference invariance, reflective endpoints and carrier/charge/energy
  conservation through public APIs before migrating the examples.
- Audit potential-squared matched-resistor detector proxies in `mzm.ipynb` and
  `examples/inverse_design/03_photonic_cmz_demux.ipynb`, plus postprocessed
  responsivity/EO power in `traveling_wave_mzm.ipynb`. Establish observation-plane
  matching and use waves for reflected fields; a resistor proxy is not a physical
  carrier detector. Keep notebook execution/migration as its own reviewed task.

Rational-model conversion paths, FIR, generalized multiple delays, other mixed
models, and PR #98 are outside this milestone. PR #98 remains separate and opt-in.

## Validation

`tests/test_phase_shifter_power_waves.py` uses independent reciprocal scattering,
round-trip and midpoint transmission laws, electrical polynomial/PN derivatives,
and full real-block finite differences. Analytic optical networks use zero
leakage and no optical ground. The suite makes `s_to_y` raise if a migrated path
calls it. Fresh CPU/float64 validation on Python 3.12 (2026-11-11):


| Disjoint test selection | Fresh result |
| --- | --- |
| 45 new phase-shifter acceptance tests, existing `test_phase_shifter.py`, `test_small_signal.py`, `test_ac_response.py`, `test_ac_sweep.py`, `test_harmonic_balance.py` | 179 passed |
| Additional raw SAX port-name guard and reflected small-signal midpoint/power checks | 4 passed |
| `test_native_power_waves.py`, `test_power_wave_sax.py`, `test_sax_component.py`, `test_optical_wave_boundaries.py`, `test_optical_delay_power_waves.py`, `test_delay.py`, `test_delay_history.py` | 334 passed |
| `test_dc.py`, `test_dc_checked.py`, `test_transient.py`, `test_waveform_sources.py`, `test_delay_contract.py`, `test_compiler.py`, `test_circuit.py`, `test_external_port_boundary.py`, `test_sp_wavelength.py`, `test_sensitivity.py` | 159 passed, 9 skipped |
| Total | **676 passed, 9 skipped, no test failures** |

The first two selections together include all **49 new** tests in
`test_phase_shifter_power_waves.py` and the **21 existing** phase-shifter tests.
Earlier debugging/rerun counts are not added. These are fresh runs against this
milestone, independent of the preceding delay milestone's historical counts.
The nine optional sensitivity skips report `bosdi/osdi_jax not available`.
Existing SAX holomorphic/conjugation, KLU `free_numeric`, and `Circuit.ac()`
deprecation warnings remain. Sparse/KLU DC and dense DC/transient/small-signal/HB
paths are exercised. The migration example executes and its relative links resolve.

Scoped repository pre-commit hooks pass (trailing whitespace, end-of-file,
merge-conflict checks); YAML, TOML and notebook hooks skip as nonapplicable.
`git diff --check` passes. Ruff passes for `phase_shifter.py`, `power_waves.py`,
`small_signal.py` and the new tests. Full touched-file Ruff still reports the
same 10 findings in `circuit.py` and 2 in `compiler.py`, verified by comparing
codes/messages against the starting commit; there are no new findings.
Formatting checks pass for five changed Python files; `compiler.py` retains its
starting-version multiline `configured` comprehension, which the installed
formatter would collapse. That unrelated formatting difference is preserved.
Thus the full touched-file lint/format commands retain these baseline failures;
repository hooks and simulation tests pass.

This is scoped verification, not the full repository suite. GPU execution,
complete notebooks and a documentation-site build were not run. Detector
migration, general AC traveling-power observables, rational conversions, FIR,
generalized multiple delays and opt-in PR #98 work remain separate.
