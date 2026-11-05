"""Plot measured multiplier waveforms after the sequential comparisons finish."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent


def main():
    sources = [
        ("reduced-1.npz", "Circulax reduced", "-"),
        ("full-1.npz", "Circulax full", "--"),
        ("native-upstream-1.npz", "VACASK adaptive trap", ":"),
        ("native-bdf2-capped-1.npz", "VACASK adaptive BDF2, cap 2 ps", "-."),
    ]
    waves = [(np.load(ROOT / name), label, style) for name, label, style in sources]
    reference = waves[0][0]
    native = waves[2][0]
    native_interpolated = np.column_stack(
        [np.interp(reference["time"], native["time"], native["voltages"][:, i]) for i in range(32, 64)]
    )
    worst_bit = int(np.argmax(np.max(abs(native_interpolated - reference["voltages"][:, 32:]), axis=0)))
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for ax, bit in zip(axes.flat, [0, worst_bit, 31], strict=False):
        for color, (wave, label, style) in enumerate(waves):
            ax.plot(wave["time"] * 1e9, wave["voltages"][:, 32 + bit], style, label=label, linewidth=1.2, color=f"C{color}")
        ax.set(
            title=f"Output p{bit}" + (" (largest native discrepancy)" if bit == worst_bit else ""),
            xlabel="Time (ns)",
            ylabel="Voltage (V)",
            xlim=(0, 2),
        )
        ax.grid(alpha=0.25)
    reference = waves[0][0]
    ax = axes[1, 1]
    for color, (wave, label, style) in enumerate(waves[1:], 1):
        interpolated = np.column_stack([np.interp(reference["time"], wave["time"], wave["voltages"][:, i]) for i in range(32, 64)])
        error = np.max(abs(interpolated - reference["voltages"][:, 32:]), axis=1)
        ax.semilogy(reference["time"] * 1e9, np.maximum(error, 1e-16), style, label=label, linewidth=1.2, color=f"C{color}")
    ax.set(
        title="Maximum output difference from reduced Circulax", xlabel="Time (ns)", ylabel="Absolute difference (V)", xlim=(0, 2)
    )
    ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2)
    figure.savefig(ROOT / "collapse-waveforms.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
