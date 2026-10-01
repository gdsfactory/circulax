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
  real factor;
  analog begin
    @(initial_step) factor = $simparam("setup_scale", 1);
    v = $limit(V(p,n), "pnjlim", 0.026, 0.6);
    if (analysis("ac")) I(p,n) <+ 9*factor*$simparam("scale", 1)*v/r;
    else I(p,n) <+ factor*$simparam("scale", 1)*v/r;
    I(p,n) <+ ddt($simparam("charge_scale", 1)*c*V(p,n));
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


@pytest.mark.parametrize("initial_mode", ["dc", "ac", "tran"])
def test_circuit_simparams_survive_all_analyses(binary: Path, initial_mode: str) -> None:
    osdi_component = pytest.importorskip("bosdi.circulax").osdi_component
    descriptor = osdi_component(
        str(binary),
        ("p", "n"),
        analysis=initial_mode,
        state_policy="limiting_only",
        simparams={"setup_scale": 2, "scale": 99},
    )
    settings = {"scale": 3, "charge_scale": 4}
    netlist = {
        "instances": {"r": {"component": "native"}, "gnd": {"component": "ground"}},
        "connections": {"r,n": "gnd,p1"},
        "ports": {"out": "r,p"},
    }
    circuit = compile_circuit(
        netlist,
        {"native": descriptor},
        simparams=settings,
        backend="dense",
        is_complex=False,
        g_leak=0,
        rtol=1e-8,
        atol=1e-12,
    )
    settings["scale"] = 99
    assert dict(descriptor.model.simparams)["scale"] == 99
    native = circuit.source_models["native"]
    assert dict(native.model.simparams) == {"setup_scale": 2, "scale": 3, "charge_scale": 4}
    for mode in ["dc", "ac", "tran"]:
        assert native.with_analysis(mode).model.simparams == native.model.simparams
    frequencies = jnp.array([1e3, 1e6])
    expected = 2 / (1 + 50 * (6 / 1000 + 2j * np.pi * frequencies * 4e-9)) - 1
    np.testing.assert_allclose(
        jax.jit(lambda f: circuit.sp(ports="out", freqs=f))(frequencies)[:, 0, 0],
        expected,
        atol=1e-10,
    )
    updated = jax.jit(lambda r: circuit.sp(ports="out", freqs=frequencies, params={"r.r": r, "r.c": 2e-9}))(2000.0)
    expected_updated = 2 / (1 + 50 * (6 / 2000 + 2j * np.pi * frequencies * 8e-9)) - 1
    np.testing.assert_allclose(updated[:, 0, 0], expected_updated, atol=1e-10)
    y0 = circuit.dc().at[circuit.port_map["r,p"]].set(1.0)
    times = jnp.linspace(0, 1e-6, 11)
    solution = circuit.transient(t0=0, t1=1e-6, dt0=1e-9, y0=y0, saveat=times, max_steps=10000)
    np.testing.assert_allclose(circuit.port(solution.ys, "out"), np.exp(-np.asarray(times) / (4e-9 / 0.006)), rtol=2e-3, atol=1e-4)


def test_resolved_compile_forwards_simparams(binary: Path, tmp_path: Path) -> None:
    from circulax.netlist_io import Library

    card = tmp_path / "settings.lib"
    card.write_text(f'load "{binary}"\nmodel rm native_modes\nr1 (out 0) rm\n')
    circuit = Library.from_file(card).resolve().compile(state_policy="limiting_only", simparams={"scale": 3})
    descriptor = circuit.source_models["native_modes"]
    assert dict(descriptor.model.simparams) == {"scale": 3}
    result = circuit.sp(ports="out", freqs=jnp.array([1e3]))
    expected = 2 / (1 + 50 * (3e-3 + 2j * np.pi * 1e3 * 1e-9)) - 1
    np.testing.assert_allclose(result[0, 0, 0], expected, atol=1e-10)
