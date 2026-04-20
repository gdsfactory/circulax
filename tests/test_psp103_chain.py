"""PSP103 verification ladder — stage 5.

3-stage inverter chain DC operating point.  Exercises the 2-phase homotopy
(source-step + Gmin-step) on a non-trivial PSP103 circuit before graduating
to the 9-stage ring oscillator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import pytest

from circulax.components.osdi import _BOSDI_AVAILABLE

pytestmark = pytest.mark.skipif(
    not _BOSDI_AVAILABLE, reason="bosdi package not available"
)

if _BOSDI_AVAILABLE:
    _TESTS_DIR = Path(__file__).resolve().parent
    if str(_TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(_TESTS_DIR))
    from fixtures.psp103_models import geom_settings, make_psp103_descriptors


def _build_chain_netlist(n_stages: int, vin: float, vdd: float = 1.2) -> dict:
    """Chain of ``n_stages`` CMOS inverters driven by an explicit Vin source."""
    mos_n = geom_settings(10e-6, 1e-6)
    mos_p = geom_settings(20e-6, 1e-6)

    instances: dict = {
        "GND": {"component": "ground"},
        "Vdd": {"component": "vsrc", "settings": {"V": vdd}},
        "Vin": {"component": "vsrc", "settings": {"V": vin}},
    }
    connections: dict = {
        "Vdd,p1": "vdd,p1",
        "Vdd,p2": "GND,p1",
        "Vin,p1": "n0,p1",
        "Vin,p2": "GND,p1",
    }

    for stage in range(n_stages):
        in_node = f"n{stage}"
        out_node = f"n{stage + 1}"
        mn = f"mn{stage}"
        mp = f"mp{stage}"
        cl = f"cl{stage}"

        instances[mn] = {"component": "nmos", "settings": mos_n}
        instances[mp] = {"component": "pmos", "settings": mos_p}
        instances[cl] = {"component": "cap",  "settings": {"C": 50e-15}}

        connections[f"{mn},D"] = f"{out_node},p1"
        connections[f"{mn},G"] = f"{in_node},p1"
        connections[f"{mn},S"] = "GND,p1"
        connections[f"{mn},B"] = "GND,p1"

        connections[f"{mp},D"] = f"{out_node},p1"
        connections[f"{mp},G"] = f"{in_node},p1"
        connections[f"{mp},S"] = "vdd,p1"
        connections[f"{mp},B"] = "vdd,p1"

        connections[f"{cl},p1"] = f"{out_node},p1"
        connections[f"{cl},p2"] = "GND,p1"

    return {"instances": instances, "connections": connections}


def _solve_chain_dc(vin: float, vdd: float = 1.2):
    from circulax import compile_netlist
    from circulax.components.electronic import Capacitor, VoltageSource
    from circulax.solvers import analyze_circuit

    psp103n, psp103p = make_psp103_descriptors()
    models = {
        "nmos": psp103n,
        "pmos": psp103p,
        "vsrc": VoltageSource,
        "cap":  Capacitor,
    }

    groups, sys_size, port_map = compile_netlist(
        _build_chain_netlist(3, vin=vin, vdd=vdd), models
    )
    solver = analyze_circuit(groups, sys_size)

    G_HOMOTOPY = 1e-2
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, G_HOMOTOPY)
    y_source = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y = solver.solve_dc_gmin(groups, y_source, g_start=G_HOMOTOPY, n_steps=30)
    return y, port_map


class TestStage5InverterChain:
    """3-stage chain: driven by an explicit Vin, each stage alternates logic."""

    def test_chain_dc_input_low(self):
        """Vin=0 → n1 ≈ VDD (inv1), n2 ≈ 0 (inv2), n3 ≈ VDD (inv3)."""
        VDD = 1.2
        y, port_map = _solve_chain_dc(vin=0.0, vdd=VDD)

        v1 = float(y[port_map["n1,p1"]])
        v2 = float(y[port_map["n2,p1"]])
        v3 = float(y[port_map["n3,p1"]])
        print(f"\nChain DC, Vin=0:  n1={v1:.3f}  n2={v2:.3f}  n3={v3:.3f}")

        assert v1 > 0.75 * VDD, f"n1 = {v1:.3f} V, expected high"
        assert v2 < 0.25 * VDD, f"n2 = {v2:.3f} V, expected low"
        assert v3 > 0.75 * VDD, f"n3 = {v3:.3f} V, expected high"

    def test_chain_dc_input_high(self):
        """Vin=VDD → n1 ≈ 0, n2 ≈ VDD, n3 ≈ 0."""
        VDD = 1.2
        y, port_map = _solve_chain_dc(vin=VDD, vdd=VDD)

        v1 = float(y[port_map["n1,p1"]])
        v2 = float(y[port_map["n2,p1"]])
        v3 = float(y[port_map["n3,p1"]])
        print(f"\nChain DC, Vin=VDD: n1={v1:.3f}  n2={v2:.3f}  n3={v3:.3f}")

        assert v1 < 0.25 * VDD, f"n1 = {v1:.3f} V, expected low"
        assert v2 > 0.75 * VDD, f"n2 = {v2:.3f} V, expected high"
        assert v3 < 0.25 * VDD, f"n3 = {v3:.3f} V, expected low"
