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

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

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
        return True
    except (ImportError, RuntimeError, OSError):
        return False


pytestmark = pytest.mark.skipif(not _bosdi_available(), reason="bosdi/osdi_jax not available")


def _rc_divider_circuit(cap_model, cap_ports, *, backend):
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


def _osdi_capacitor():
    from circulax import osdi_component

    return osdi_component(
        osdi_path=CAPACITOR_OSDI,
        ports=("p", "n"),
        default_params={"$mfactor": 1.0, "c": 1e-12},
    ), ("p", "n")


def _builtin_capacitor():
    from circulax.components.electronic import Capacitor

    return Capacitor, ("p1", "p2")


@pytest.mark.parametrize("backend", ["dense", "klu_split"])
def test_sp_osdi_matches_builtin_capacitor(backend):
    osdi_cap, osdi_ports = _osdi_capacitor()
    builtin_cap, builtin_ports = _builtin_capacitor()

    c_osdi = _rc_divider_circuit(osdi_cap, osdi_ports, backend=backend)
    c_builtin = _rc_divider_circuit(builtin_cap, builtin_ports, backend=backend)

    y_dc_osdi = c_osdi.dc()
    y_dc_builtin = c_builtin.dc()

    sp_osdi = np.asarray(c_osdi.sp(ports=["out"], freqs=np.array([F0]), z0=50.0, y_dc=y_dc_osdi))
    sp_builtin = np.asarray(
        c_builtin.sp(ports=["out"], freqs=np.array([F0]), z0=50.0, y_dc=y_dc_builtin)
    )

    np.testing.assert_allclose(sp_osdi, sp_builtin, rtol=1e-6)


@pytest.mark.parametrize("backend", ["dense", "klu_split"])
def test_hb_osdi_matches_builtin_capacitor(backend):
    osdi_cap, osdi_ports = _osdi_capacitor()
    builtin_cap, builtin_ports = _builtin_capacitor()

    c_osdi = _rc_divider_circuit(osdi_cap, osdi_ports, backend=backend)
    c_builtin = _rc_divider_circuit(builtin_cap, builtin_ports, backend=backend)

    y_dc_osdi = c_osdi.dc()
    y_dc_builtin = c_builtin.dc()

    K = 2 * HARMONICS + 1
    _, freq_osdi = c_osdi.hb(
        freq=F0, harmonics=HARMONICS, y0=y_dc_osdi, y_flat_init=jnp.tile(y_dc_osdi, K)
    )
    _, freq_builtin = c_builtin.hb(
        freq=F0, harmonics=HARMONICS, y0=y_dc_builtin, y_flat_init=jnp.tile(y_dc_builtin, K)
    )

    port_osdi = c_osdi._resolve_port_node("out")
    port_builtin = c_builtin._resolve_port_node("out")

    amp_osdi = abs(freq_osdi[1, port_osdi])
    amp_builtin = abs(freq_builtin[1, port_builtin])

    np.testing.assert_allclose(amp_osdi, amp_builtin, rtol=1e-6)
    # Analytic value for this R-C divider (see issue #65).
    np.testing.assert_allclose(2.0 * amp_osdi, 0.954028, atol=5e-6)
