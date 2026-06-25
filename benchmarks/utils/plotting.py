"""Benchmark comparison plotting utilities."""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(
    ref_label: str,
    test_label: str,
    time_scale: float,
    time_unit: str,
    panels: list[dict],
    output_path: str | pathlib.Path,
) -> None:
    """Save a multi-panel waveform comparison plot.

    Parameters
    ----------
    ref_label:
        Name of the reference simulator (e.g. ``"NGSpice"``).
    test_label:
        Name of the simulator under test (e.g. ``"Circulax"``).
    time_scale:
        Multiplier applied to time axes for display (e.g. ``1e3`` for ms).
    time_unit:
        Label for the time axis (e.g. ``"ms"``).
    panels:
        List of panel dicts, each with keys:

        - ``"title"``  (str)
        - ``"ref_time"``   (np.ndarray)
        - ``"ref_signal"`` (np.ndarray)
        - ``"test_time"``  (np.ndarray)
        - ``"test_signal"`` (np.ndarray)
        - ``"ylabel"`` (str, optional, default ``"Voltage (V)"``)
        - ``"show_error"`` (bool, optional) — if True this panel is an error panel

    output_path:
        Where to save the PNG.

    """
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        ref_t = np.asarray(panel["ref_time"]) * time_scale
        ref_s = np.asarray(panel["ref_signal"])
        tst_t = np.asarray(panel["test_time"]) * time_scale
        tst_s = np.asarray(panel["test_signal"])
        ylabel = panel.get("ylabel", "Voltage (V)")
        show_error = panel.get("show_error", False)

        if show_error:
            ax.plot(tst_t, tst_s * 1e3, color="red", lw=0.8)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_ylabel("Error (mV)")
        else:
            ax.plot(ref_t, ref_s, label=ref_label, lw=1, alpha=0.75)
            ax.plot(tst_t, tst_s, "--", label=test_label, lw=1)
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)

        ax.set_title(panel["title"])

    axes[-1].set_xlabel(f"Time ({time_unit})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"  Plot saved → {output_path}")
    plt.close(fig)
