# Optical passive primitives and delayed waveguide

These native models follow [the common conventions](photonic_conventions.md).
Ports are complex optical pseudo-voltages; reference impedance `z0` is real,
positive and defaults to 1. Power waves satisfy
`a=(v+z0*i)/(2*sqrt(z0))`, `b=(v-z0*i)/(2*sqrt(z0))`;
incident power is `abs(a)**2` watts. Current points into the component.

## Models

Import from `circulax.components.optical_passive`:

| Model | Ports | Parameters and behavior |
|---|---|---|
| `OpticalAttenuator` | p1, p2 | Matched reciprocal transmission `10**(-loss_dB/20)` |
| `OpticalPhaseShifter` | p1, p2 | Matched reciprocal transmission `10**(-loss_dB/20)*exp(+i*phase)`; phase in radians |
| `OpticalTermination` | p1 | Matched absorber `i=v/z0`; no ground port required |
| `OpticalInterface` | p1, p2 | Power reflectivity R, uniform loss in dB, common phase, differential reflection phase |
| `NonlinearOpticalWaveguide` | p1, p2 | Bidirectional delayed propagation, linear attenuation and independent per-direction two-photon absorption |

The first four are independent implementations of standard scattering relations.
`OpticalPhaseShifter` retains the positive phase convention of the example-local
`PhaseBias` in `traveling_wave_mzm_reference.ipynb`; it does not add electrical
loading or envelope delay. Two-port primitives use incident power-wave states
`a1/a2` and exact constraints `v=sqrt(z0)*(a+S@a)`. This avoids inversion of
`I+S`, so ideal through, full reflection and zero phase remain exact without
clipping, artificial loss or leakage.

For the interface, with `delta=reflection_phase`,

```
S = 10**(-loss_dB/20)*exp(i*phase) *
    [[sqrt(R)*exp(i*delta), sqrt(1-R)],
     [sqrt(1-R), -sqrt(R)*exp(-i*delta)]]
```

The two reflection coefficients cannot be chosen independently while retaining
reciprocity and losslessness. This matrix is symmetric and unitary before the
uniform attenuation. Its loss represents absorption into unmodeled channels.
Valid domains: `0<=R<=1`, `loss_dB>=0`, positive `z0`; angles are unrestricted.
Invalid passive loss, reflectivity and impedance domains are rejected with
JAX-safe runtime checks rather than clipped. Square-root
parameter derivatives are physically singular at R=0/1; differentiate an angle
parameterization for optimization exactly at those endpoints.

## Example promotion and waveguide assumptions

`NonlinearOpticalWaveguide` is promoted from code cell 3 in
`examples/photonic/photonics_transient.ipynb`. The notebook now imports it.
The equations preserve the original behavior at z0=1. Its states `a1/a2`
retain the original **incident pseudo-voltage** convention, unlike the power
waves of the static primitives: incident power is `abs(a)**2/z0` W.
The TPA denominator now uses that power for arbitrary z0. The zero-loss
integrated attenuation has a safe analytic limit with finite derivatives.

For length L in cm, power loss alpha in 1/cm and input power P,

```
Leff = (1-exp(-alpha*L))/alpha   (limit L at alpha=0)
Eout(t) = exp(-alpha*L/2-i*beta*L) * Ein(t-tau)
          / sqrt(1+tpa_coeff*P(t-tau)*Leff)
tau = length_um*n_group/2.99792458e14
```

`tpa_coeff` is in 1/(W cm), wavelength in nm and length in micrometres.
Length, loss, TPA coefficient and group index are nonnegative; wavelength and
z0 are positive. Each direction propagates independently; cross-TPA between
counterpropagating fields, Kerr phase, free carriers and heating are excluded.
Carrier phase is applied once with negative sign; the explicit delay acts on
the envelope. The analytic TPA law is a memoryless distributed-loss solution
along each delayed characteristic, not a carrier-recovery model.

## Existing passive audit

`DirectionalCoupler`, `Splitter`, `OpticalWaveguide` and `Grating` in
`photonic.py` retain their public ports/parameters and now use exact wave
constraints. Their internal state count increases by one wave per port.
`DirectionalCoupler` supports exact coupling=0 and coupling=1 without clipping;
its full four-port S matrix is reciprocal/unitary. `Grating` preserves unit
transmission at its ideal center instead of limiting it to 0.9999.
`OpticalWaveguide` supports zero length/loss and carrier phases crossing
singular Y representations. Its wavelength dispersion slope does not implement
an envelope propagation delay.

`Splitter` is **contractive, not unitary**: the actual three-port matrix has
singular values (1,1,0). It preserves power for excitation only at p1, but
absorbs a coherent dark combination incident at p2/p3. That existing behavior
is preserved and the previous misleading losslessness claim is corrected.
`TunableBeamSplitter` remains a directed buffered field mixer, with no claim of
reciprocal passive scattering under arbitrary terminations. `OpticalSource`
continues to prescribe voltage; its `power` does not imply calibrated incident
power under reflection. A voltage source of amplitude `2*sqrt(P*z0)` followed
by series resistance z0 provides a matched incident-power boundary.

## Verification and supported analyses

`tests/test_optical_passive.py` checks independent matched KCL reconstruction
of complete S matrices, multi-input power conservation/loss, reciprocity,
phase and exact endpoints. It checks the promoted model in actual matched
Circulax DC/TPA circuits, zero length, finite derivatives at zero loss,
JIT/vmap and agreement of forward/reverse parameter AD. A nonlinear delayed
pulse is checked against the independently evaluated shifted propagation law
at two fixed timesteps; refining 2 ps to 1 ps reduces the error. Static phase
has a compiled AC circuit check. No external VA, measured or experimental
validation is claimed; no GPL reference code is used.

Static primitives support DC, AC, transient and algebraic HB equations.
The delayed nonlinear model supports DC and transient and requires
non-holomorphic real-block analysis for AC/HB. The existing `ac_response` API
correctly refuses it even when TPA happens to be zero; the class flag cannot
vary with traced parameters. `quadrature_response` is independently checked
at finite TPA bias for both amplitude and phase quadratures, including the
full envelope delay phase and distinct real/imaginary incremental gains. Forward and reverse AD of the compiled quadrature response with respect
to `length_um` are independently checked against the exact analytic
attenuation, TPA-compression and envelope-delay derivative. Matched incident
DC states are fixed for that derivative because they are independent of length.
Delayed HB and full transient delay-parameter gradients have not been verified
here; transient delay-parameter adjoints have separate solver coverage.
AC results are already complex arrays and should be indexed directly by
`circuit.port_map[name]`; `circuit.port()` currently expects real-block DC/
transient arrays.

## Compilation measurement

CPU, float64/complex128, fresh process, eight real inputs (two complex optical
ports and two complex wave states). Continuous phase/loss or length/loss/TPA
parameters are dynamic arguments, not constants. `jax.jit(fn).lower(x,p)`
measures tracing, `.compile()` separately measures XLA compilation; warmed
calls block until ready, averaged over 1000 calls. The real Jacobian has
shape 8x8. Final-model timings include JAX-safe dynamic parameter checks. No delay interpolation/history cost is included in this local
constitutive benchmark; complete transient timing is not inferred from it.

| Model | Operation | Trace ms | Compile ms | Warm microseconds |
|---|---|---:|---:|---:|
| Phase shifter | F/Q | 22.6 | 51.3 | 23.70 |
| Phase shifter | real Jacobian | 45.3 | 49.9 | 17.14 |
| Nonlinear waveguide | F/Q | 46.7 | 54.1 | 16.11 |
| Nonlinear waveguide | real Jacobian | 44.8 | 85.2 | 15.02 |

For reproduction, construct a real-block function that combines the first and
last halves of dynamic `x` into complex `y`, calls
`Model.solver_call(0., y, Model(...dynamic p...))`, and concatenates real and
imaginary F/Q. Use `jax.jacfwd(fn, 0)` for the state Jacobian. A representative
matched source/series resistor/waveguide/termination circuit has seven complex
unknowns (14 real), including ground. Timings vary by hardware and are local
component measurements, not solver benchmarks.

The complete promoted `photonics_transient.ipynb` code executes successfully
with its existing assertions. At 5 mm/15 mm, measured differential envelope
delay is 133.4 ps (analytic 133.4 ps); maximum envelope errors are 1.12e-7
and 1.02e-7. Settled powers are 0.1918 mW and 0.0619 mW, matching the notebook's
independent analytic TPA/loss estimates. Its full circuit has 14 complex unknowns
after exact splitter wave states are allocated.
