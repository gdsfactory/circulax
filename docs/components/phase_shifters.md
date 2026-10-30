# Electrical phase shifters

`ElectricalPhaseShifter` is a reciprocal matched optical two-port controlled by
the real electrical voltage `anode-cathode`. It multiplies each incident optical
field by

```
t = 10**(-(loss_dB + loss_per_v2*(V-bias)**2)/20)
    * exp(+1j*(phase + efficiency*(V-bias)))
```

`efficiency` is rad/V, `phase` is rad, and attenuation is power dB. Use positive
`z0`, nonnegative `loss_dB` and `loss_per_v2`; this model introduces no delay,
junction current or loading. Electrical terminals must be connected to a real
source/load network; they are ideal control probes. Optical states `a1,a2` are
incident pseudo-voltages, so incident power is `abs(a)**2/z0` W. Both directions
have the same transmission. Imaginary electrical control does not affect optical
phase; physical electrical networks must constrain that quadrature.

`loaded_phase_shifter(resistance=..., capacitance=..., **settings)` composes this
core with an actual series resistor and constant junction capacitor. External
ports are `anode,cathode,o1,o2,junction`; `junction` exposes the internal loaded
control voltage. Resistance is in ohms and capacitance in farads. The capacitor
stores `Q=C*V`, which is an integrated constant capacitance law. This first model
does not claim a voltage-dependent depletion charge or diode conduction.

```python
from circulax.components.phase_shifters import loaded_phase_shifter

phase_arm = loaded_phase_shifter(
    resistance=25.0, capacitance=80e-15,
    efficiency=1.2, bias=-1.0, phase=0.0, loss_dB=0.1,
)
# Use phase_arm as a subcircuit model; drive anode/cathode and match o1/o2.
```

## Promoted traveling-wave junctions

`PNJunctionPhaseShifter` is promoted from cell 7 of
`examples/photonic/traveling_wave_mzm.ipynb`, retaining its ports, aliases,
parameters and equations. `PNPhaseShifter` is the reference notebook's variant,
retaining its parameter conventions. Both use four optical wave/delay states,
bidirectional propagation, constant junction charge and a midpoint quadratic
phase law:

```
phi = length_mm * (a1*(V-bias) + a2*(V**2-bias**2))
```

Here `a1` is rad/(V mm), `a2` is rad/(V² mm). The control is sampled at half the
optical flight time, and two half-delay stages give the full envelope delay.
The carrier factor is `exp(-i*2*pi*neff*length_um/wavelength_um)`; it is applied
once in total. Propagation attenuation is in dB/cm. Group delay is
`length_um*n_group/2.99792458e14` s. Zero group delay is supported by the native
delay mechanism. There is no cosmetic bandwidth filter.

| API | Length / wavelength | Index / loss | Electrical charge |
|---|---|---|---|
| `PNJunctionPhaseShifter` | `length`, `wl`, both µm | `neff0`, `loss_dB_cm` | `Cj0_per_um_fF * length * 1e-15 * V` |
| `PNPhaseShifter` | `length_um`, `wavelength_um` | `neff`, `loss_db_cm` | `capacitance * V` (F) |

Both default to `n_group=4.05`, `a1=1.2`, `a2=0.15`, `bias=-1` V.
`PNJunctionPhaseShifter` retains configurable `z0` and aliases `anode_0,cathode_0`;
`PNPhaseShifter` retains the reference notebook's fixed 1-ohm normalization.
Their electrical ports need an external bias reference and can be placed behind
series resistance. The original examples now import these implementations.
No external GPL code was copied: these are promotions of repository examples.

## Verification and limits

Focused tests check the promoted quadratic law, charge magnitude/sign,
full-complex electrical charge, JIT/vmap, forward/reverse AD and dynamic parameter
gradients. Compiled matched circuits check loaded DC phase, independent RC
frequency response using the complete real-block F/Q Jacobians, RC charging
against the exponential solution, and pulse arrival through both promoted
junction APIs. Pulse tests compare complex envelope amplitude and carrier phase
against independently calculated attenuation and delayed ramps.

These models support DC and native transient. **Public `ac_response` currently
rejects non-holomorphic complex EO circuits**, correctly requiring a sideband
contract; it must not be bypassed by declaring these models holomorphic. The new `quadrature_response` utility supports explicit I/Q phasors and
sidebands; tests compare it with independently assembled real-block equations.
HB and delay-parameter gradients have not been verified for these promoted
models. No measured-device calibration, experimental validation or VA/OSDI
comparison is claimed. Nonlinear depletion capacitance and diffusion charge are
outside these simple junction models.

CPU measurements with complex128, dynamic voltage and continuous phase parameter
(JAX CPU, 2026-10-27; dispatch-inclusive warm mean, 200 iterations):

| Model | Complex local unknowns | Trace / XLA compile (ms) | Warm F/Q / real-block Jacobian (µs) |
|---|---:|---:|---:|
| `ElectricalPhaseShifter` | 6 | 9.8 / 27.2 | 2.9 / 10.2 |
| `PNPhaseShifter` | 8 | 11.3 / 31.4 | 2.9 / 5.4 |
| `PNJunctionPhaseShifter` | 8 | 9.8 / 31.6 | 2.9 / 4.7 |

These local measurements include both real/imaginary variables; they are not
full-circuit solve costs. Jacobian cold trace+compile was 67–100 ms. Local
benchmark calls use the DC form of the delay lookup; transient history management
cost is separate.
