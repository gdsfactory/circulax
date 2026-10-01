"""Public native analyses select mode-specific physics without delayed slots."""

import shutil
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax import compile_circuit


@pytest.fixture(scope="module")
def binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("openvaf-r")
    if compiler is None:
        pytest.skip("openvaf-r is not installed")
    root = tmp_path_factory.mktemp("native-analysis")
    source = root / "analysis.va"
    source.write_text("""`include "disciplines.vams"
module native_modes(p, n);
  inout p, n;
  electrical p, n;
  parameter real r = 1000;
  parameter real c = 1e-9;
  real v;
  analog begin
    v = $limit(V(p,n), "pnjlim", 0.026, 0.6);
    if (analysis("ac")) I(p,n) <+ 9*v/r;
    else I(p,n) <+ v/r;
    I(p,n) <+ ddt(c*V(p,n));
  end
endmodule
""")
    target = root / "analysis.osdi"
    subprocess.run([compiler, str(source), "-o", str(target)], check=True, capture_output=True)  # noqa: S603 -- explicit compiler; no shell
    return target


@pytest.mark.parametrize("initial_mode", ["dc", "ac", "tran"])
def test_public_native_analyses(binary: Path, initial_mode: str) -> None:
    osdi_component = pytest.importorskip("bosdi.circulax").osdi_component
    descriptor = osdi_component(str(binary), ("p", "n"), analysis=initial_mode, state_policy="limiting_only")
    netlist = {
        "instances": {"r": {"component": "native"}, "gnd": {"component": "ground"}},
        "connections": {"r,n": "gnd,p1"},
        "ports": {"out": "r,p"},
    }
    circuit = compile_circuit(
        netlist, {"native": descriptor, "ground": lambda: 0}, backend="dense", is_complex=False, g_leak=0, rtol=1e-8, atol=1e-12
    )
    np.testing.assert_allclose(jax.jit(circuit.dc)(), 0, atol=1e-12)
    frequencies = jnp.array([1e3, 1e6])
    # Conductance must come from DC (1/r), even though AC reports 9/r.
    expected = 2 / (1 + 50 * (1 / 1000 + 2j * np.pi * frequencies * 1e-9)) - 1
    np.testing.assert_allclose(
        jax.jit(lambda freqs: circuit.sp(ports="out", freqs=freqs))(frequencies)[:, 0, 0], expected, atol=1e-10
    )
    updated = jax.jit(lambda resistance: circuit.sp(ports="out", freqs=frequencies, params={"r.r": resistance, "r.c": 2e-9}))(
        2000.0
    )
    expected_updated = 2 / (1 + 50 * (1 / 2000 + 2j * np.pi * frequencies * 2e-9)) - 1
    np.testing.assert_allclose(updated[:, 0, 0], expected_updated, atol=1e-10)
    global_updated = jax.jit(lambda resistance: circuit.sp(ports="out", freqs=frequencies, r=resistance, c=2e-9))(2000.0)
    np.testing.assert_allclose(global_updated, updated, atol=1e-10)
    with pytest.raises(NotImplementedError, match="harmonic balance"):
        circuit.hb(freq=1e3)
    y0 = circuit.dc().at[circuit.port_map["r,p"]].set(1.0)
    times = jnp.linspace(0, 1e-6, 11)
    solution = circuit.transient(t0=0, t1=1e-6, dt0=1e-9, y0=y0, saveat=times, max_steps=10000)
    # The physical capacitor is solver-managed Q; no delayed ABI state is needed.
    np.testing.assert_allclose(circuit.port(solution.ys, "out"), np.exp(-np.asarray(times) / 1e-6), rtol=2e-3, atol=1e-4)
