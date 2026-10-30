# Rings and ring modulation

Import `AllPassRing`, `AddDropRing`, `AllPassRingEO`, `loaded_ring`,
`RingModulator`, `RingModulatorEO`, and `RingModSetup` from
`circulax.components.resonators`. See [common conventions](photonic_conventions.md)
for the carrier, port and power definitions.

## Passive single-resonance models

`AllPassRing(p1,p2)` has a through path and independent forward/reverse modes.
`AddDropRing(p1,p2,p3,p4)` has bus pairs p1/p2 and p3/p4: p1 excitation exits
through p2 and drops at p4; reversing p1/p4 reverses the corresponding path.
They are reciprocal, matched scattering devices; no input is source-clamped.
Internal `a1`...`a4` are incident power waves in sqrt(W), while
`c_forward/c_reverse` have units sqrt(J). There is no backscattering between
cavity directions, polarization coupling or multimode physics.

The independent temporal coupled-mode realization is

```
dc/dt = (i*Delta-G)*c + sum(sqrt(2*gamma_k)*a_k)
b_k = a_k - sqrt(2*gamma_k)*c
G = gamma_i + sum(gamma_k)
Delta = 2*pi*(f_res-f_ref)
```

The appropriate direct paths connect the paired ports. The detuning is in
rad/s; the rotating-frame carrier convention is exp(+i omega_ref t).
`AllPassRing` uses `gamma_e`; `AddDropRing` uses `gamma_1/gamma_2`.
Invalid rate/impedance domains are rejected with JAX-safe runtime checks.
All rates are **amplitude** decay rates in 1/s and must be nonnegative with
G>0; `z0` is positive, real and defaults to 1. Energy decays at 2G, the
energy lifetime is 1/(2G), intensity linewidth FWHM is G/pi Hz, and
Q=omega_res/(2G). Zero couplings are exact; total zero decay is outside the
steady-state model domain. Avoid differentiating sqrt(gamma) exactly at zero
coupling. No loss or endpoint clipping is added by the implementation.

The model is a narrowband approximation around one resonance, with constant
rates and detuning. Use it where linewidth and modulation bandwidth are small
relative to the cavity free spectral range. There is no multi-FSR or broadband
round-trip propagation prediction. All rates/detuning are physical inputs;
no device geometry or experimental calibration is inferred.

For all-pass:

```
H(s) = 1 - 2*gamma_e/(s+G-i*Delta)
```

Critical coupling occurs at gamma_e=gamma_i. In an add-drop ring the drop
coefficient is `-2*sqrt(gamma_1*gamma_2)/(s+G-i*Delta)` and through is
`1-2*gamma_1/(s+G-i*Delta)`. Both quadratures and interference phases matter.
The exact instantaneous identity is

```
sum(|a|²-|b|²) = d(|c_forward|²+|c_reverse|²)/dt
                + 2*gamma_i*(|c_forward|²+|c_reverse|²)
```

It includes storage, so instantaneous input power need not equal output power.

## Electrical tuning and composition

`AllPassRingEO` adds `v_e`, whose real voltage shifts resonance by
`tuning*v_e` in rad/s (tuning in rad/s/V). It draws zero physical real current.
A unit imaginary restoring conductance constrains the artificial complex
extension of the electrical node. It does not represent an electrical loss
or a physical admittance. The physical drive remains a real voltage.

`loaded_ring(resistance=50, capacitance=100e-15, **ring_settings)` composes
that core with a series resistor and shunt capacitor, returning a reusable
`Circuit` subcircuit. Exposed ports are p1/p2 optical, drive electrical,
gnd electrical reference, and junction internal voltage. Connect gnd explicitly
in a parent circuit. All electrical loading is in that R/C network; no extra
bandwidth pole is hidden in the control model. Physical absorption heating is
not modeled.

## Promoted notebook models

The existing definitions in `examples/photonic/ring_modulator.ipynb` cells
4/6/18 were promoted without changing their deterministic equations or public
parameters; those cells now import the package models. Their conventions
remain deliberately distinct:

- `RingModulator` and `RingModulatorEO` are **directed behavioral blocks**:
  p1 draws no current, p2 is constrained to an output field. They are not
  reciprocal scattering devices and do not infer incident power from a
  reflected optical node. Their fields are directed example amplitudes.
- Legacy detuning is `2*pi*(f_operating-f_resonance)+v_to_wr*V`, the opposite
  frequency ordering to the new passive models. The voltage coefficient is
  retained as a legacy detuning coefficient, not reinterpreted as a positive
  physical resonance shift.
- Legacy `tau` values are amplitude lifetimes in the implemented ODE.
  Round-trip gamma/alpha-to-rate mapping is retained as the notebook's
  small-loss CMT approximation. `RingModulatorEO` is high impedance and
  expects a connected real electrical driver or load to constrain its voltage.
- `RingModSetup` is the existing illustrative `.setup` example. Its phase
  expression `2*pi*neff*L` contains no wavelength and is not a dimensionally
  complete physical propagation law when L is in metres. Its current stamps
  are behavioral, not a passive scattering interface. It has no cavity
  storage or envelope dynamics. It is retained for example compatibility,
  not recommended for quantitative resonator design.

The external GPL Verilog-A repository was not used to implement these models.
The new cores are independently derived from the stated energy balance and
linear coupled-mode equations; the legacy cores come from this repository's
own current examples.

## Verification and analyses

Focused tests in `tests/test_optical_resonators.py` compare compiled matched
Circulax DC/AC circuits to independent complex CMT equations across under,
critical and over coupling, positive/negative detuning and zero external
coupling. Full four-input add-drop scattering is checked for reciprocity,
unitarity when intrinsic loss is zero, and absorbed power when lossy. An
independent instantaneous identity checks storage and internal dissipation for
simultaneous excitation in both directions. Tests distinguish amplitude from
energy decay rates and check JIT/vmap and forward/reverse parameter AD.

A composed loaded EO ring is compared against independent CMT+RC upper and
lower sidebands through `quadrature_response`, including a check that the
electrical quadrature remains zero. Ringdown and filter transients are compared
against exact exponential solutions with timestep refinement. The promoted
legacy cores are checked against independently restated notebook equations,
including gradients through `.setup`.

The passive cores are holomorphic and support DC/AC/transient; the EO core is
non-holomorphic and requires real-block/quadrature analysis. Legacy model
flags remain conservative. A small three-harmonic linear ring HB comparison checks independently
computed upper/lower sidebands and absence of spurious higher harmonics.
Full nonlinear ring HB accuracy has not been verified here. No external
simulator, measured-device calibration, experimental speed, BER or noise
validation is claimed.

The promoted notebook's cells 0–18 execute successfully, including the original
pulse transient and `.setup` demonstration. The simulated settled through-field
magnitude is 0.925, matching its independent analytic 0.925; the setup phase
gradient at kappa=0.3 remains -0.140432 rad per unit kappa. Later notebook RF/HB/
NRZ sweeps were preserved but not executed as part of this promotion check.

## Local compilation measurements

A fresh CPU process with float64/complex128 measured dynamic continuous rate,
coupling and detuning parameters. Inputs use separate real/imaginary blocks;
F and Q remain separate output blocks. Tracing uses `jax.jit(fn).lower(x,p)`,
XLA compilation uses `.compile()`, and 1000 warmed calls block until ready.
The Jacobian is `jax.jacfwd(fn,0)` over the full real state vector. All figures
are milliseconds except the warmed column, which is microseconds.

| Model | Complex variables | Operation | Trace ms | Compile ms | Warm us |
|---|---:|---|---:|---:|---:|
| AllPassRing | 6 | F/Q | 30.8 | 46.1 | 23.39 |
| AllPassRing | 6 | real Jacobian | 31.5 | 50.0 | 16.54 |
| AddDropRing | 10 | F/Q | 28.0 | 59.8 | 18.09 |
| AddDropRing | 10 | real Jacobian | 40.6 | 80.8 | 39.03 |
| AllPassRingEO | 7 | F/Q | 20.6 | 42.3 | 14.22 |
| AllPassRingEO | 7 | real Jacobian | 63.5 | 63.7 | 25.61 |

These final-model measurements include runtime checks of dynamic parameter
domains. Shared CPU load and callback checks affect warmed timing; these are
observed timings, not guaranteed device-independent costs.

A representative matched source/resistor/AllPassRing/termination circuit has
9 complex unknowns (18 real). In an ordinary fresh-process API measurement,
setup took 0.452 s, its first DC solve 0.464 s, and ten repeated ordinary
`Circuit.dc()` calls averaged 462 ms each. Those API calls were not enclosed
in a persistent outer `jax.jit`; the local compilation table above measures
explicitly compiled kernels separately. Hierarchy is flattened and does not
create independent JIT compilation boundaries.
