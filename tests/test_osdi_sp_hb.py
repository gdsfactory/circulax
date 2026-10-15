"""Regression tests for GitHub issue #65.

``.sp()`` crashed with ``AttributeError: 'OsdiComponentGroup' object has no
attribute 'has_delay'`` and ``.hb()`` crashed with ``ValueError: The FFI call
to 'OsdiResidualEvalHandleCpu' cannot be differentiated`` whenever a netlist
contained an OSDI device. Both are compared against the equivalent circuit
built from the built-in :class:`~circulax.components.electronic.Capacitor`,
which was unaffected and gives an independent ground truth.

Circuit under test: an AC voltage source driving an R-C divider, where the
capacitor is an OSDI device loaded from ``tests/data/va/capacitor.osdi``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest

if TYPE_CHECKING:
    from circulax import Circuit

jax.config.update("jax_enable_x64", True)  # noqa: FBT003 -- JAX configuration API

_DATA_DIR = Path(__file__).parent / "data" / "va"
CAPACITOR_OSDI = str(_DATA_DIR / "capacitor.osdi")

F0 = 1e9
HARMONICS = 3


def _bosdi_available() -> bool:
    if not Path(CAPACITOR_OSDI).exists():
        return False  # run `pixi run compile_osdi` to enable these tests
    try:
        from bosdi.circulax import OsdiComponentGroup  # noqa: F401
        from osdi_jax import osdi_residual_eval_with_handle  # noqa: F401
        from osdi_loader import load_osdi_model

        load_osdi_model(CAPACITOR_OSDI)
    except (ImportError, RuntimeError, OSError):
        return False
    else:
        return True


pytestmark = pytest.mark.skipif(not _bosdi_available(), reason="bosdi/osdi_jax not available")


def _rc_divider_circuit(cap_model: object, cap_ports: tuple[str, str], *, backend: str) -> Circuit:
    from circulax import compile_circuit
    from circulax.components.electronic import Resistor, VoltageSourceAC

    net = {
        "instances": {
            "Vs": {"component": "vac", "settings": {"V": 1.0, "freq": F0}},
            "R1": {"component": "resistor", "settings": {"R": 50.0}},
            "C1": {"component": "ocap", "settings": {}},
            "GND": {"component": "ground"},
        },
        "connections": {
            "Vs,p1": "R1,p1",
            "Vs,p2": "GND,p1",
            "R1,p2": (f"C1,{cap_ports[0]}",),
            f"C1,{cap_ports[1]}": "GND,p1",
        },
        "ports": {"out": "R1,p2"},
    }
    models = {
        "vac": VoltageSourceAC,
        "resistor": Resistor,
        "ocap": cap_model,
        "ground": lambda: 0,
    }
    return compile_circuit(net, models, backend=backend)


def _osdi_capacitor(*, analysis: str = "ac", osdi_path: str = CAPACITOR_OSDI) -> tuple[object, tuple[str, str]]:
    from circulax import osdi_component

    return osdi_component(
        osdi_path=osdi_path,
        ports=("p", "n"),
        default_params={"$mfactor": 1.0, "c": 1e-12},
        analysis=analysis,
    ), ("p", "n")


def _builtin_capacitor() -> tuple[object, tuple[str, str]]:
    from circulax.components.electronic import Capacitor

    return Capacitor, ("p1", "p2")


@pytest.mark.parametrize("backend", ["dense", "klu_split"])
def test_sp_osdi_matches_builtin_capacitor(backend: str) -> None:
    osdi_cap, osdi_ports = _osdi_capacitor()
    builtin_cap, builtin_ports = _builtin_capacitor()

    c_osdi = _rc_divider_circuit(osdi_cap, osdi_ports, backend=backend)
    c_builtin = _rc_divider_circuit(builtin_cap, builtin_ports, backend=backend)

    y_dc_osdi = c_osdi.dc()
    y_dc_builtin = c_builtin.dc()

    sp_osdi = np.asarray(c_osdi.sp(ports=["out"], freqs=np.array([F0]), z0=50.0, y_dc=y_dc_osdi))
    sp_builtin = np.asarray(c_builtin.sp(ports=["out"], freqs=np.array([F0]), z0=50.0, y_dc=y_dc_builtin))

    np.testing.assert_allclose(sp_osdi, sp_builtin, rtol=1e-6)


@pytest.mark.parametrize("backend", ["dense", "klu_split"])
def test_hb_osdi_matches_builtin_capacitor(backend: str) -> None:
    osdi_cap, osdi_ports = _osdi_capacitor()
    builtin_cap, builtin_ports = _builtin_capacitor()

    c_osdi = _rc_divider_circuit(osdi_cap, osdi_ports, backend=backend)
    c_builtin = _rc_divider_circuit(builtin_cap, builtin_ports, backend=backend)

    y_dc_osdi = c_osdi.dc()
    y_dc_builtin = c_builtin.dc()

    K = 2 * HARMONICS + 1
    _, freq_osdi = c_osdi.hb(freq=F0, harmonics=HARMONICS, y0=y_dc_osdi, y_flat_init=jnp.tile(y_dc_osdi, K))
    _, freq_builtin = c_builtin.hb(freq=F0, harmonics=HARMONICS, y0=y_dc_builtin, y_flat_init=jnp.tile(y_dc_builtin, K))

    amp_osdi = abs(c_osdi.port(freq_osdi[1], "out"))
    amp_builtin = abs(c_builtin.port(freq_builtin[1], "out"))

    np.testing.assert_allclose(amp_osdi, amp_builtin, rtol=1e-6)
    # Analytic value for this R-C divider (see issue #65).
    np.testing.assert_allclose(2.0 * amp_osdi, 0.954028, atol=5e-6)


@pytest.fixture(scope="module")
def mode_capacitor(tmp_path_factory: pytest.TempPathFactory) -> str:
    compiler = shutil.which("openvaf-r")
    if compiler is None:
        pytest.skip("openvaf-r is not installed")
    root = tmp_path_factory.mktemp("hb-mode-capacitor")
    source = root / "capacitor.va"
    source.write_text("""`include "disciplines.vams"
module mode_capacitor(p, n);
  inout p, n;
  electrical p, n;
  parameter real c = 1e-12;
  analog begin
    if (analysis("ac")) I(p,n) <+ ddt(9*c*V(p,n));
    else if (analysis("tran")) I(p,n) <+ ddt(c*V(p,n));
  end
endmodule
""")
    target = root / "capacitor.osdi"
    subprocess.run([compiler, str(source), "-o", str(target)], check=True, capture_output=True)  # noqa: S603 -- explicit compiler; no shell
    return str(target)


@pytest.mark.parametrize("backend", ["dense", "klu_split"])
@pytest.mark.parametrize("initial_mode", ["dc", "ac", "tran"])
def test_hb_selects_transient_native_physics(mode_capacitor: str, backend: str, initial_mode: str) -> None:
    capacitor, ports = _osdi_capacitor(analysis=initial_mode, osdi_path=mode_capacitor)
    circuit = _rc_divider_circuit(capacitor, ports, backend=backend)
    # Exercise automatic DC initialization and native registration under JIT.
    _, spectrum = jax.jit(lambda: circuit.hb(freq=F0, harmonics=HARMONICS))()
    amplitude = 2 * abs(circuit.port(spectrum[1], "out"))
    expected = 1 / np.sqrt(1 + (2 * np.pi * F0 * 50 * 1e-12) ** 2)
    np.testing.assert_allclose(amplitude, expected, rtol=1e-6)
