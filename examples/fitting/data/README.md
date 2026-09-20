# Example data

The single [vector-fitting tutorial](../vector_fitting.ipynb) covers the three
scikit-rf networks below, with a clickable contents section and an engineering introduction.
It starts with `skrf.data.ring_slot`, the smooth passive two-port bundled with
scikit-rf, which needs no separately vendored Touchstone file. The resonant
four-port and active transmitter then explore different modeling assumptions.

- Upstream example: <https://scikit-rf.readthedocs.io/en/latest/examples/vectorfitting/vectorfitting_ex1_ringslot.html>
- License: BSD-3-Clause; see `SCIKIT_RF_LICENSE.txt` in this directory.

`Agilent_E5071B.s4p` is copied from the scikit-rf test data and is used by the
vector-fitting example notebook:

- Source: <https://github.com/scikit-rf/scikit-rf/blob/master/skrf/tests/Agilent_E5071B.s4p>
- Upstream example: <https://scikit-rf.readthedocs.io/en/latest/examples/vectorfitting/vectorfitting_ex3_Agilent_E5071B.html>
- License: BSD-3-Clause; see `SCIKIT_RF_LICENSE.txt` in this directory.

The original Touchstone header identifies the measurement as made with an
Agilent Technologies E5071B network analyzer on 2012-04-05.

`190ghz_tx_measured.s2p` is the measured active two-port used by scikit-rf's
190 GHz vector-fitting example:

- Source: <https://github.com/scikit-rf/scikit-rf/blob/master/doc/source/examples/vectorfitting/190ghz_tx_measured.S2P>
- Upstream example: <https://scikit-rf.readthedocs.io/en/latest/examples/vectorfitting/vectorfitting_ex2_190ghz_active.html>
- License: BSD-3-Clause; see `SCIKIT_RF_LICENSE.txt` in this directory.

The Touchstone header identifies the device as an active transmitter measured
from 140 to 220 GHz on 2018-06-13.

## Synthetic noisy cable

`noisy_transmission_line.s2p` is generated in the
[time-delay notebook](../../electrical/time_delay.ipynb); it is not measured or
third-party data. It uses a matched 50 Ω, skin-effect-inspired model:

```text
S21(s) = S12(s) = A0 * exp(-s*tau - k*sqrt(s))
S11 = S22 = 0
A0 = 10**(-0.1/20)
tau = 5e-9 seconds
k = (11.9 * ln(10) / 20) / sqrt(pi * 40e9)
```

The principal square root supplies attenuation and the associated phase lag.
Insertion loss is `0.1 + 11.9*sqrt(f/40e9)` dB: 0.1595 dB at 1 MHz, about
1.982 dB at 1 GHz, 6.05 dB at 10 GHz, and 12 dB at 40 GHz. This simplified
model omits dielectric loss, impedance variation, and the low-frequency
transition from skin effect to DC conductor behavior.

The 4,201 frequencies are the sorted unique union of
`geomspace(1e6, 1e9, 301)` and `linspace(1e9, 40e9, 3901)`. Extra low-frequency
samples resolve the loss curvature. Complex Gaussian noise has RMS magnitude
0.001 per independent S entry, using NumPy's `default_rng(2026)`. Real and
imaginary standard deviations are `0.001/sqrt(2)`. Transmission noise is shared
to preserve reciprocity; reflection noises are independent.

Holdout samples have zero-based indices `0, 5, 10, ...`. The notebook supplies
the known 5 ns propagation delay, split equally between ports. It fits the
remaining attenuation and dispersion using only training samples and reports
held-out errors. Targets are 0.8% NRMSE and 0.005 maximum absolute S error.
An independent time-domain convolution checks the fitted pulse against the
analytic cable response. The dispersive phase is not treated as a pure-delay
estimate.
