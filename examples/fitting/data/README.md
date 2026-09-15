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

## Synthetic noisy transmission line

`noisy_transmission_line.s2p` is generated in the
[time-delay notebook](../../electrical/time_delay.ipynb); it is not measured or
third-party data. It contains 501 points from 1 MHz to 2 GHz for a matched 50 Ω
line with a 1 ns transmission delay and voltage-wave attenuation of 0.9.

Complex Gaussian noise has RMS magnitude 0.001 per independent S entry, using
NumPy's `default_rng(2026)`. Real and imaginary standard deviations are
`0.001 / sqrt(2)`. The transmissions share the same noise to preserve reciprocity;
reflection noises are independent. Holdout samples have zero-based indices
`0, 5, 10, ...`. The notebook estimates delay using only the remaining samples,
compares ordinary and delay-aware fits, and runs the fitted model in transient.
