"""Waveform comparison metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WaveformComparison:
    """Accuracy metrics between a reference waveform and a test waveform."""

    node: str
    max_abs_error: float  # V
    rms_error: float  # V
    rel_rms_pct: float  # % of signal swing

    def print(self) -> None:
        print(
            f"  [{self.node}]  max|err|={self.max_abs_error * 1e3:.4f} mV  "
            f"RMS={self.rms_error * 1e3:.4f} mV  rel={self.rel_rms_pct:.4f}%"
        )


def compare_waveforms(
    ref_time: np.ndarray,
    ref_signal: np.ndarray,
    test_time: np.ndarray,
    test_signal: np.ndarray,
    node: str = "",
) -> WaveformComparison:
    """Interpolate *ref_signal* onto *test_time* and compute error metrics.

    Parameters
    ----------
    ref_time, ref_signal:
        Reference simulator (e.g. NGSpice) time and voltage arrays.
    test_time, test_signal:
        Simulator under test (e.g. Circulax) time and voltage arrays.
    node:
        Label for the comparison (e.g. ``"v(2)"``).

    """
    ref_interp = np.interp(test_time, ref_time, ref_signal)
    err = test_signal - ref_interp
    swing = float(np.max(ref_signal) - np.min(ref_signal))
    return WaveformComparison(
        node=node,
        max_abs_error=float(np.max(np.abs(err))),
        rms_error=float(np.sqrt(np.mean(err**2))),
        rel_rms_pct=float(np.sqrt(np.mean(err**2)) / (swing + 1e-30) * 100),
    )
