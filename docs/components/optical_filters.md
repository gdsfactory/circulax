# Causal optical envelope filter

`OpticalFilter` in `circulax.components.optical_filters` is a matched,
reciprocal two-port envelope filter with one stable state in each propagation
direction. It follows [common optical conventions](photonic_conventions.md).

```
k = 2*pi*bandwidth_hz
H(s) = transmission*k/(s+k-i*detuning)
dc/dt = (-k+i*detuning)*c + sqrt(k)*a
b = transmission*sqrt(k)*c
```

`bandwidth_hz>0` is the power half-width at half maximum around envelope
frequency `detuning/(2*pi)` Hz. The full intensity FWHM is twice that value.
`transmission` is a real amplitude in [0,1], `detuning=omega_center-omega_ref`
is in rad/s, and reference impedance `z0` is real and positive (default 1).
The pole is `-k+i*detuning`, so positive detuning moves the spectral center
to positive envelope frequency under the exp(+i omega_ref t) convention.
States `a1/a2` are incident waves in sqrt(W); `c1/c2` are in sqrt(J).

The two optical ports are genuine scattering interfaces. For transmission=1,
the state normalization gives
`Pin-Pout-d|c|²/dt = |a-sqrt(k)*c|² >= 0` per direction. Lower transmission
adds nonnegative internal dissipation. The associated hidden loss channels
are unmodeled; this generic filter is not a full cavity with a through output.
The reference-frequency detuning can be changed dynamically through component
parameters; no carrier oscillation is resolved. There are no electrical,
thermal or stochastic states.

The independent tests in `tests/test_optical_resonators.py` compare actual
matched compiled circuits with the complex rational transfer at positive and
negative detuning, check the -3 dB bandwidth and phase, and compare a loaded
step to `1-exp(-k*t)` with timestep refinement. State-space and frequency
response use the same documented physical pole but independently evaluated
reference formulas. DC, AC and transient are verified. A three-harmonic HB comparison checks both independent sidebands and
absence of higher harmonics. The holomorphic F/Q stamp preserves the full
complex envelope rather than discarding its imaginary quadrature.
No Verilog-A, fitted data or experimental validation is used or claimed.

A higher-order prototype is deferred: the requested small causal filter is
already supplied by one pole, and larger filters can be composed from these
native stages when their loss/phase specification is defined.

Fresh-process CPU measurements with dynamic bandwidth/detuning/transmission
and full real/imaginary input blocks: F/Q tracing 26.7 ms, compilation 42.9 ms,
warmed evaluation 14.81 us; real-state Jacobian tracing 25.1 ms, compilation
59.3 ms, warmed evaluation 13.59 us. These include dynamic parameter-domain
runtime checks. Six complex variables represent two
ports, two incident waves and two storage states. Methodology matches the
[ring measurements](resonators.md#local-compilation-measurements); these are
local constitutive costs, not complete solver timings.

Invalid bandwidth, transmission and reference-impedance domains are rejected
with JAX-safe runtime checks, including under compilation.
