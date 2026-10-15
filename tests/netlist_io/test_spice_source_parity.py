"""Independent sources agree with ngspice operating points and waveforms."""

import shutil
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax.components.electronic import WaveformCurrentSource, WaveformVoltageSource
from circulax.netlist_io import parse_source


@pytest.mark.parametrize(("component", "card"), [(WaveformVoltageSource, "V1 out 0"), (WaveformCurrentSource, "I1 0 out")])
@pytest.mark.parametrize(
    "spec",
    [
        "SIN(0.9 1 1k)",
        "DC 0.5 SIN(0.9 1 1k)",
        "DC 0 SIN(0.9 1 1k)",
        "SIN(1 2 1k 100u 300 30)",
        "SIN(1 2)",
        "SIN(1 2 0)",
        "PULSE(2 3)",
        "PULSE(2 3 100u 0 0 200u 500u)",
        "PULSE(2 3 100u 50u 50u 200u 500u)",
        "PULSE(2 3 100u 0 0 0 0)",
        "PWL(0 2 200u 3 400u 1)",
        "PWL(0 2 200u 3 400u 1) r=200u td=100u",
        "PWL(0 2 200u 3 400u 1) r=0",
    ],
)
def test_ngspice_source_parity(component: type, card: str, spec: str, tmp_path: Path) -> None:
    """Use ngspice's actual time points, including its initialization and edges."""
    executable = shutil.which("ngspice")
    if executable is None:
        pytest.skip("ngspice is not installed")
    tstep, tstop = 10e-6, 1e-3
    # ngspice supports repeating PWL only for voltage sources. Validate the
    # current-source extension against the equivalent voltage waveform.
    if component is WaveformCurrentSource and "r=" in spec:
        card = "V1 out 0"
    netlist = tmp_path / "source.cir"
    netlist.write_text(
        f"""Source parity
{card} {spec}
R1 out 0 1
.control
set numdgt=16
op
wrdata op.txt v(out)
tran {tstep} {tstop}
wrdata tran.txt v(out)
quit
.endc
.end
"""
    )
    result = subprocess.run(  # noqa: S603 -- explicit ngspice executable; no shell
        [executable, "-b", str(netlist)], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    dc = np.loadtxt(tmp_path / "op.txt").reshape(-1, 2)[0, 1]
    reference = np.loadtxt(tmp_path / "tran.txt")
    settings = parse_source(spec, tstep=tstep, tstop=tstop)
    source = component(**settings)

    def value(source: WaveformVoltageSource | WaveformCurrentSource, t: float | jax.Array) -> jax.Array:
        f, _ = source(t=t)
        return -f["i_src"] if component is WaveformVoltageSource else f["p1"]

    assert float(value(source, 0.0)) == pytest.approx(dc, abs=1e-10)
    waveform = component(**settings, source_mode=1.0)
    actual = jax.vmap(lambda t: value(waveform, t))(jnp.asarray(reference[:, 0]))
    np.testing.assert_allclose(actual, reference[:, 1], rtol=1e-9, atol=1e-9)
