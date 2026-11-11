# Notebook traveling-power observations

This is the notebook observation milestone of [issue #112](https://github.com/gdsfactory/circulax/issues/112),
on `codex/issue-112-power-waves`, starting at `35479effdfb4487220e75c798538e96897226a5d`.
It builds on the [boundary](optical_wave_boundaries.md), [passive](native_passive_power_waves.md),
[delay](bidirectional_optical_delays.md), [phase-shifter](mixed_electro_optic_power_waves.md)
and [detector](detector_power_waves.md) milestones. It does not complete #112.

## Observation-plane audit and migration

| Notebook/path | Original meaning and boundary | Migrated observation |
| --- | --- | --- |
| `examples/photonic/mzm.ipynb`, `Detector` | A 1-ohm resistor to optical ground, with no electrical output, responsivity, charge or carrier population: an ideal matched power monitor at the output grating's fiber plane. | `OpticalWaveTermination`; incident power in W from `port_waves(..., "Detector,p")`. Plot remains logarithmic, now explicitly dB relative to 1 W. |
| `examples/inverse_design/03_photonic_cmz_demux.ipynb`, `Det_1` through `Det_4` | Four 1-ohm matched resistor monitors. Power fractions feed the original row-normalized routing objective. Three unused coupler inputs were grounded, hence perfect shorts, **not** matched dumps. | Four matched wave terminations; preserve the three shorts as independent one-port SAX boundaries with reflection -1. Measure monitor incident powers with the same scalar runtime parameters as each solve. Preserve row denominator `sum(power)+1e-12`, all 20 loss wavelengths, target ordering and MSE. |
| `examples/photonic/traveling_wave_mzm.ipynb`, static output, dumps and received output | 1-ohm matched resistor monitors; no electrical detector circuit. Output gratings and all couplers are reflectionless at their terminals; dumps absorb their own incident waves. | Explicit matched optical terminations, incident powers and total incident envelopes. Static phase-bias elements use the reusable SAX retained-wave construction. |
| Same notebook, MZM output | Plane before the output grating, previously measured through an alias to terminal potential. | Incident wave on `GRATING_OUT,waveguide`, oriented into the grating. The after-grating plane is `OPTICAL_LOAD,p`. The grating is a reciprocal matched attenuator, so its loss is applied once between these planes. |
| Same notebook, ideal detected S31 | A virtual, nonloading receiver at the MZM-output plane, with 0.8 A/W and a 50-ohm matched electrical receiving load. It was never a connected photodiode or a second optical branch. | Preserve that calibrated measurement conversion and label it virtual. Use the traveling-power derivative for its input. No extra detector branch, junction loading, reflectivity or carrier law is added. |

All three notebooks replace the legacy ideal potential source with
`OpticalWaveSource(amplitude=sqrt(P)*exp(j*phase))`. At the authored reference 1,
the old source and new source give the same forward launch because the audited
networks return no wave to the laser. In CMZ, the shorted unused inputs receive
no backward excitation from the matched forward outputs; their reflection is
nevertheless preserved. This equivalence is specific to these networks, not a
general reinterpretation of `OpticalSource.power` under reflection.

Physical arm splitting, loss, carrier phase, electrical source/load impedances,
junction bias, common/differential conventions, modulation amplitude and the
inverse-design objective are preserved. Removing optical return wires replaces
matched resistors with their equivalent one-terminal absorbing boundaries;
electrical ground and electrical loads remain electrical. No convergence
default, leakage, optical-ground penalty, reference perturbation, artificial
loss or delay algorithm changes are introduced.

The full CMZ training run exposed a separate domain error: unconstrained Adam
can propose `coupling < 0` or `coupling > 1`, rejected by the existing passive
model. Training now represents the seven couplings by logits and maps them
through sigmoid before the same physical objective. Initial physical couplings,
length coordinates, eight starts, 2000 steps, learning rate and reported physical
parameter histories are preserved. This changes optimizer trajectories; it is
an explicit domain correction, not a claim of identical optimization history.
There is no new finite coupling floor, endpoint clipping or optical attenuation.

## Small-signal traveling-power derivative

`Circuit.port_power_response(response, "Instance,terminal", params=...)` returns
`PortPowerResponse(incident, outgoing)` with a signed `net` property. Each value
is a **complex RF peak phasor in W**, with the response's frequency axis. The net
sign is positive into the selected component. This is not a total field or a
nonnegative total power.

The method differentiates the public total-field `port_waves` powers about
`response.y_dc`, with the same circuit and scalar settings used to build the
small-signal model. Both optical coordinates are real state coordinates:

```text
D(P)(dy_RF) = D(P)(Re(dy_RF)) + j D(P)(Im(dy_RF))
dP_a = 2*(Re(a_dc)*da_x + Im(a_dc)*da_y)
dP_b = 2*(Re(b_dc)*db_x + Im(b_dc)*db_y)
dP_net = dP_a-dP_b
```

The x/y perturbations in these equations are independently complex RF phasors.
There is no `abs(AC field)**2` and no conflation of RF phase with optical phase.
The solve uses the supported full real-block `small_signal` linearization.

```python
# response = circuit.small_signal(y_dc=op, params=settings)(
#     freqs=frequencies, sources={"VD": source_peak})
# plane3 = circuit.port_power_response(
#     response, "GRATING_OUT,waveguide", params=settings)
# h31 = plane3.incident / source_peak  # W/V, before the output grating
```

Supported terminals need complete instantaneous optical F flow and zero optical
Q, the same contract as total-field `port_waves`. This includes phase-shifter
and delay optical terminals with declared algebraic flow; their solved response
already includes the history/frequency transfer. Electrical terminals,
non-wave states, generic unsupported delay models and charged optical terminals
remain rejected. A frequency-domain component's local terminal flow cannot be
reconstructed by differentiating its DC operator, so this method explicitly
rejects that terminal, even at zero frequency. Its connected instantaneous
boundary can be observed, using the full frequency-dependent network response.
Independent source terminals are also rejected because their forcing derivative
is not contained in a state-only derivative. Observe the attached device
terminal. Directly batched responses require external `jax.vmap` around the
measurement. No unsupported-analysis guard is bypassed.

`SmallSignalResponse.power(field_unit="sqrt(W)")` remains the older explicit
normalized-field helper; it still rejects declared component terminal potentials.
`Circuit.port_waves` still rejects AC perturbation and harmonic-coefficient
analysis labels. DC and transient observations use total envelopes; HB powers
are computed after reconstructing total time samples, including interference.

For the virtual receiver, reflection is zero and responsivity converts incident
power to ideal generated/collected current once:

```text
dI_peak = R_PD*dP_plane3                 [A]
V3_peak = Z3*dI_peak                    [V]
a1_peak = V_source_peak/(2*sqrt(Z1))     [sqrt(W)]
b3_peak = V3_peak/sqrt(Z3)               [sqrt(W)], a3=0; RF peak convention
S31 = b3_peak/a1_peak
P_electrical_average = |V3_peak|²/(2*Z3) [W, for positive-frequency sinusoidal drive]
```

These electrical RF wave phasors use peak voltage, so their squared magnitudes
carry a factor of two relative to average sinusoidal power; the common factor
cancels in S31. Optical carrier envelopes above use |a|² directly in W.

The reusable physical detectors instead define responsivity per **absorbed**
optical W: `I_generated=R_abs*(P_incident-P_reflected)`. Their full/partial
reflection controls verify that loss is applied once, and that current flows
from cathode to anode internally and into the external anode load. The ideal
model has no stored charge and generated equals collected current. Photodiode
junction charge and carrier-model stored populations/collection remain distinct,
as described in the detector guide; this notebook milestone does not change them.

## Remaining-path inventory

The following concrete paths remain after this focused migration:

- `examples/photonic/photonics_transient.ipynb`: notebook-local
  `NonlinearOpticalWaveguide` has incident states, bidirectional delayed
  intensity-dependent TPA, and a configurable `z0`, but lacks the shared wave
  metadata/observable contract. Its legacy potential pulse source, resistor
  monitors and field-squared observations still need an independent delayed
  nonlinear derivative, boundary and normalization audit.
- `examples/photonic/ring_modulator.ipynb`: `RingModulator`, `RingModSetup` and
  `RingModulatorEO`, plus local potential pulse/step/sinusoidal/NRZ sources.
  Audit resonator-state energy normalization, through-port signed flow,
  electro-optic sidebands, transient/HB powers and electrical RC behavior before
  migrating their potential-based observations. The circulating-state magnitude
  must not be relabeled as terminal traveling power.
- `examples/photonic/tfln_traveling_wave_mzm.ipynb`: its `sectioned_mzm` uses a
  legacy potential source and resistor boundaries, and squares returned fields
  for comparison with the analytical TFLN response. Audit reference planes,
  electrode filtering and section-dependent propagation separately.
- `circulax/components/rational.py:rational_delay_component`: re-embeds delays
  through `Y_deembed -> S_deembed -> P S P -> s_to_y(S_full)` in `_fast_physics`.
  This remaining photonic-capable realization conversion is unchanged.
  `rational_component` and `rational_fdomain_component` represent supplied
  admittance state-space models; a future scattering-state realization needs an
  explicit model/artifact contract rather than reinterpreting existing Y data.
- `circulax/fitting/sparam.py:fit_with_delay`, and
  `circulax/fitting/surface.py:initialize_surface`: S-to-Y fitting/data-conversion
  paths remain. General `s_to_y` utilities remain legitimate conversion APIs;
  `scattering_state_space_to_admittance_jax` and its scalar/batched wrappers
  also eliminate incident waves using `(I+D)^-1`; these conversion-based
  realizations remain separate. Fitting an optical model through singular admittance data needs a separate
  scattering-realization migration. Electrical modal-line fitting in the
  traveling-wave notebook remains outside the optical conversion prohibition.
- General frequency-domain terminal-flow power derivatives, independent-source
  forcing-aware observables, discipline enforcement and the full #112 remaining
  acceptance audit are still open. No FIR, generalized multiple delays, PR #98,
  rational realization migration, remote issue update or publication is included.

## Fresh validation

CPU / Python 3.12 / JAX float64 validation on 2026-11-11:

- A completed focused run passed all **37** new tests. These exercise the public
  power derivative, matched/partial/full reflection, optical phases 0/pi/2/pi,
  independently complex RF quadratures, references 0.01/1/50, detector absorbed
  power/current signs, finite differences, delay phase, JIT/vmap, frequency-flow
  rejection with supported boundary observations, independent MZM/CMZ network
  references, CMZ raw power balance and all eleven objective gradients.
- All Python cells of `mzm.ipynb` and `traveling_wave_mzm.ipynb` executed in order
  with Agg plots and optical realization conversions patched to raise. The latter
  includes DC, full real-block EO, HB, NRZ and both PAM4 records. No outputs or
  metadata were written to the notebook files.
- Traveling-wave HB refinement from two to four harmonics changed OMA by
  **0.138709%** and extinction ratio by **0.0044453 dB**. Halving the NRZ
  integration timestep changed received power by at most **0.031603%** of the
  full-scale received power. A second timestep halving was interrupted; no
  result is claimed for it.
- The complete CMZ training calculation ran all eight starts and 2000 steps;
  the winner had **84.76%** mean in-band routing contrast and **87.8%** mean
  center-channel contrast (initial center-channel contrast 6.9%). The first
  run's final animation cell was interrupted. A fresh complete rerun passed all
  nine Python cells, including eight starts × 2000 steps and all **44**
  animation wavelength sweeps/frames. The GIF was saved outside the repository;
  no generated animation or notebook outputs are part of the change.
- Analytic wave-network tests use zero leakage and no optical ground.
  Physical detector fixtures ground only the electrical load return.
  S-to-Y conversion is patched to raise for migrated optical realization paths.
- Scoped hooks passed whitespace, EOF, conflict and notebook stripping checks;
  YAML and TOML checks skipped as nonapplicable. Four changed Python files pass
  formatting. New tests, `power_waves.py` and `small_signal.py` pass Ruff.
  `circuit.py` retains seven diagnostics, with codes/messages identical to the
  starting commit under the installed Ruff. Notebook lint similarly retains
  zero MZM, six traveling-wave and ten CMZ findings, with no new messages.
  Notebook outputs, metadata and cell IDs are unchanged.

An early test-harness run selected a Markdown cell as Python; that selection was
corrected. Full CMZ execution also exposed the optimizer domain error described
above. These failures are resolved. An interrupted broad regression run and its
temporary logs did not produce a completed result. A parallel pytest command
was rejected before collection because pytest-xdist is unavailable; the rerun
uses ordinary pytest processes in disjoint batches. A fresh disjoint rerun is
recorded below; historical counts in earlier guides are not substituted for it.


Final disjoint regression rerun on the committed source:

| Selection | Fresh result |
| --- | --- |
| `test_notebook_power_waves.py`, detector and both phase-shifter suites | **204 passed**, 1 inherited warning |
| Native passives, SAX adapter, boundaries, optical delays and delay/history suites | **334 passed**, 1 inherited warning |
| Delay contract, sources, compiler/circuit, small-signal, AC, HB, DC, transient, external boundaries, wavelength and sensitivity suites | **281 passed**, 24 inherited warnings |
| Total, without double-counting earlier focused reruns | **819 passed, 0 skipped, 0 failed** |

The total includes the 37 new acceptance cases. The selected sensitivity tests
passed in this environment; the nine optional skips in older guides are
historical, not a result of this rerun. Warnings are the existing KLU
`free_numeric` and `Circuit.ac()` deprecations and SAX holomorphic/conjugation
warning. This is scoped CPU verification, not the full repository suite, GPU
execution or a documentation-site build.
