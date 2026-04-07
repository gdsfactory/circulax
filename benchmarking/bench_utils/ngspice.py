"""NGSpice batch runner.

Usage::

    from bench_utils.ngspice import run_ngspice

    time, voltages = run_ngspice(
        cir_file="circuits/rc_pulse.cir",
        node_names=["v(1)", "v(2)"],
    )
    # voltages["v(1)"] → np.ndarray
"""

from __future__ import annotations

import pathlib
import subprocess
import tempfile

import numpy as np


def run_ngspice(
    cir_file: str | pathlib.Path,
    node_names: list[str],
    *,
    output_path: str | pathlib.Path | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run an NGSpice netlist in batch mode and return waveform data.

    The netlist **must** contain a ``.control`` block that calls::

        wrdata <output_path> <node_names...>

    If *output_path* is ``None`` a temporary file is used.

    Parameters
    ----------
    cir_file:
        Path to the ``.cir`` netlist.  The file must contain a ``.control``
        block with ``wrdata`` writing to *output_path*.
    node_names:
        Ordered list of node names as written in the ``wrdata`` line
        (e.g. ``["v(1)", "v(2)"]``).  These become the keys of the returned
        ``voltages`` dict.
    output_path:
        Where NGSpice writes the ASCII data file.  If ``None``, a temporary
        file is created and the netlist's ``wrdata`` path must use the same
        default (``/tmp/ngspice_out.dat``).

    Returns
    -------
    time : np.ndarray, shape (N,)
    voltages : dict mapping node_name → np.ndarray of shape (N,)

    """
    cir_file = pathlib.Path(cir_file)

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".dat", delete=False)
        output_path = pathlib.Path(tmp.name)
        tmp.close()
    else:
        output_path = pathlib.Path(output_path)

    if output_path.exists():
        output_path.unlink()

    result = subprocess.run(
        ["ngspice", "-b", str(cir_file)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"NGSpice exited with code {result.returncode}:\n{result.stderr}")

    if not output_path.exists():
        raise FileNotFoundError(
            f"NGSpice did not produce output at {output_path}.\nCheck that the .cir wrdata path matches.\nstderr:\n{result.stderr}"
        )

    # NGSpice wrdata format: each variable gets its OWN time column, so N variables
    # produce 2N columns: [time0  v0  time1  v1  ...  timeN-1  vN-1]
    # All time columns are identical; we use column 0 as the time axis.
    data = np.loadtxt(output_path, comments="%")
    time = data[:, 0]
    voltages = {name: data[:, 2 * i + 1] for i, name in enumerate(node_names)}
    return time, voltages
