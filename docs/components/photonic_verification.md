# Photonic library verification record

Verification on 2026-10-27 used the local `.venv` with Python 3.12.14,
JAX 0.7.2 CPU, float64/complex128. These are numerical/analytical checks of
illustrative compact models, not measured-device calibration or experimental
validation. The [catalog](photonics.md) links equations, units and per-family limits.

## Focused checks

The final integrated run passed **122 tests in 201.96 s** across these eight files
(six existing KLU deprecation warnings):

```bash
python -m pytest \
  tests/test_optical_passive.py tests/test_optical_resonators.py \
  tests/test_phase_shifters.py tests/test_photodetectors.py \
  tests/test_photonic_modulators.py tests/test_optical_analysis.py \
  tests/test_photonic_link.py tests/test_thermo_optic.py -q
```

Before the final integrated run, guarded passive/ring/filter tests passed as
72 tests plus one additional analytical delay-length gradient test; the final
phase/detection/thermal/IQ/link batch passed 40 tests in 83.38 s. The nine
modulator tests passed as part of the 16-test phase/modulator batch in 80.21 s.
These counts refer to distinct test families; repeated runs are not added.

The independent comparisons include:

- Complete scattering matrices, reciprocity, singular values and simultaneous
  multi-input power identities, including ideal coupling/reflection endpoints.
- Shifted nonlinear optical pulses, carrier phase and envelope flight time;
  finite-TPA I/Q frequency response and forward/reverse derivatives of delay
  length against an analytical derivative with carrier phase disabled.
- Coherent MZI/MZM fields and complementary outputs from NumPy matrix products;
  reflected electrical loading, independent RC response and finite-drive link
  transient at two timesteps.
- All-pass/add-drop CMT response, intrinsic absorption, detuning signs, linewidth,
  cavity amplitude versus energy decay, full bidirectional energy identities,
  ringdown timestep convergence and loaded EO upper/lower sidebands.
- Filter independent transfer/ODE/step reference and cavity storage/dissipation.
- Actual loaded detector I/V, power under reflection and optical phase changes,
  junction charge derivatives, RC/transit response including extraction feedback,
  pulse charge at two timesteps and small-signal HB versus SP.
- Joule power, DC temperature and phase, temperature-dependent heater resistance,
  heating/cooling time constants and mixed-domain Jacobians.
- Independent modal-electrode ABCD responses with loss/mismatch, full distributed
  EO conversion with optical flight, and decreasing one/two/four-section errors
  against a 128-section independent reference; a severe velocity mismatch case.
- Linear ring/filter HB positive/negative sidebands with no spurious harmonics;
  dynamic parameters, JIT/vmap and forward/reverse AD in representative models.

The independent reviews were assigned across family boundaries. They confirmed
power/cavity/IQ identities and identified domain gaps, which were fixed with
JAX-safe runtime checks. Perfect reflection is outside the absorbing detector's
admittance domain and is explicitly rejected; `OpticalInterface` handles that
endpoint with wave constraints. No epsilon loss was introduced.

Abrupt inconsistent source initialization can excite the existing default
trapezoidal solver's algebraic ringing. Thermal step and composed-link tests
use consistent pre-step DC states and explicitly selected BDF2; no shared solver
implementation was changed. Imaginary electrical residues are checked against
the actual finite-ground/solver precision, rather than an unattainable exact zero.

## Existing regressions

The integration run exercised these existing files alongside the new tests:

```bash
python -m pytest tests/test_components.py tests/test_subcircuit.py \
  tests/test_sp_wavelength.py tests/test_ac_response.py \
  tests/test_delay_contract.py tests/test_sax_component.py -q
python -m pytest tests/test_distribution.py tests/test_external_port_boundary.py \
  tests/test_circuit.py tests/test_dc.py -q
python -m pytest tests/test_sample_component_runtime.py tests/test_delay.py -q
```

The first group contributed 115 passing existing checks in the initial combined
189-test run. That run also caught an overly strict imaginary-voltage bound in a
new phase test, corrected and rerun in the passing 40-test batch above. The second
group contributed 70 passing existing checks in a 107-test run that also included
37 ring/filter checks (78.53 s). The last group passed 9 tests in 45.66 s,
including existing fixed/adaptive transient delay-length gradients against finite
differences and electronic sample runtime checks. Thus **194 distinct existing
regression checks passed**; overlapping new-model checks are not counted again.
The entire repository test suite, optional proprietary flows and all heavy
notebook sweeps were not run.

## Examples and packaging

Executed all three new scripts with their embedded independent assertions:

```bash
python examples/photonic/library_components.py
python examples/photonic/library_link.py
python examples/photonic/library_traveling_wave_mzm.py
```

The link has 30 complex unknowns and produces 0.04558172 V at its default bias;
construction plus first DC measured 1.91 s on the final run. The distributed
example uses two sections / 105 complex unknowns and checks three-frequency EO
response against an independent ABCD/midpoint propagation calculation.
The promoted nonlinear-waveguide notebook executed end-to-end: differential delay
133.4 ps, envelope errors 1.12e-7/1.02e-7 and steady powers 0.1918/0.0619 mW.
The original ring notebook's baseline cells 0–18 executed with simulated and
analytical steady transmission both 0.925. Other promoted notebook definitions
have frozen-equation regression checks; their complete heavy sweeps/eyes were
not rerun. All photonic notebooks remain valid JSON.

All new model/test/example/analysis files and the extended benchmark pass Ruff
lint and formatting. The modified pre-existing `photonic.py` retains its existing
copyright/argument-count lint exceptions. Local Markdown links in the new model
pages/catalog were checked.

`uv build --wheel --out-dir /tmp/circulax-photonic-wheel` succeeded. A fresh process
outside the checkout imported the wheel using its ZIP path while an import finder
blocked `bosdi`, `osdi_jax` and `openvaf`; **23 new/promoted native cores evaluated
with finite F/Q**, and six zero-argument subcircuit factories constructed. The
parameterized traveling-wave factory was executed separately in its example.
The wheel includes all seven new component modules and `optical_analysis.py`.
No runtime reference checkout, proprietary tool or undeclared measurement file
is needed. No generated reference binaries were added.

## Performance and validation boundaries

The [catalog](photonics.md) records a reproducible complex-state benchmark command
and final measured core timings including domain guards. Per-family pages give
additional tracing/compilation/warmed F/Q and full real-block Jacobian timings.
For the 105-complex-variable wrapper, dynamic local phase parameters measured
trace 360 ms, XLA compile 253 ms, warm F/Q 70 us and full real-block Jacobian
366 us; parameter partitions and history management affect actual solve cost.

No external VA/OSDI comparison or data calibration was performed. The GPL external
reference was not copied or translated, and the cited primary-paper DOI was
inaccessible. Nonlinear ring HB, full nonlinear high-speed section convergence,
experimental BER/noise/eyes and full transient delay gradients for the newly
promoted nonlinear/PN models remain unverified. Existing delay-engine gradient
regressions do not establish every new model's transient sensitivity.
