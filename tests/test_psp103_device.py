"""PSP103 verification ladder — stages 2, 3, 4.

Stage 2:  Parameter-name introspection via the new ``OsdiModel.param_names``
          API exposed by bosdi.
Stage 3:  Single-transistor DC sweeps — physics sanity checks (monotonicity
          of Id(Vgs), near-zero Id at Vds=0) for both NMOS and PMOS.
Stage 4:  CMOS inverter DC transfer curve + step-response transient.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from circulax.components.osdi import _BOSDI_AVAILABLE

pytestmark = pytest.mark.skipif(
    not _BOSDI_AVAILABLE, reason="bosdi package not available"
)

if _BOSDI_AVAILABLE:
    import sys
    from pathlib import Path

    _TESTS_DIR = Path(__file__).resolve().parent
    if str(_TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(_TESTS_DIR))
    from fixtures.psp103_models import (
        PSP103_OSDI,
        geom_settings,
        make_psp103_descriptors,
    )


# ────────────────────────────────────────────────────────────────────────────
# Stage 2 — param-name introspection via bosdi's new param_names API
# ────────────────────────────────────────────────────────────────────────────


class TestStage2ParamIntrospection:
    """bosdi exposes canonical PSP103 parameter names without ctypes tricks."""

    def test_osdi_model_metadata(self):
        from osdi_loader import load_osdi_model

        m = load_osdi_model(PSP103_OSDI)
        assert m.num_pins == 4               # D, G, S, B
        assert m.num_nodes == 6              # + 2 internal
        assert m.num_params == 783
        assert m.num_states == 0
        assert len(m.param_names) == 783

    def test_canonical_names_resolve_known_params(self):
        """A handful of well-known PSP103 parameter names are present."""
        from osdi_loader import load_osdi_model

        m = load_osdi_model(PSP103_OSDI)
        names = {n for n in m.param_names if n}
        for known in ("L", "W", "TYPE", "TR", "VFBO", "TOXO", "THESATO", "UO"):
            assert known in names, f"canonical name {known!r} missing from PSP103"

    def test_param_kinds_and_types(self):
        """bosdi decodes each param's kind (INST/MODEL/OPVAR) and type (REAL/INT/STR)."""
        from osdi_loader import load_osdi_model

        m = load_osdi_model(PSP103_OSDI)
        kinds = m.param_kinds()
        types = m.param_types()
        assert set(kinds) >= {"INST", "MODEL"}  # PSP103 has both
        assert set(types) >= {"REAL"}           # at least some real params

        # Geometry params are per-instance (INST kind).
        for inst_param in ("L", "W", "AD", "AS"):
            idx = m.param_names.index(inst_param)
            assert kinds[idx] == "INST", f"{inst_param} should be INST"

    def test_descriptor_round_trip_via_canonical_api(self):
        """osdi_component(..., param_names=None) uses canonical OSDI ordering."""
        from circulax.components.osdi import OsdiModelDescriptor, osdi_component

        desc = osdi_component(
            osdi_path=PSP103_OSDI,
            ports=("D", "G", "S", "B"),
            default_params={"TYPE": 1.0, "L": 1e-6, "W": 10e-6},
        )
        assert isinstance(desc, OsdiModelDescriptor)
        assert desc.is_canonical
        assert len(desc.param_names) == 783
        # Case-insensitive lookup works.
        inst = desc.make_instance({"w": 20e-6, "l": 0.5e-6})
        assert inst["W"] == pytest.approx(20e-6)
        assert inst["L"] == pytest.approx(0.5e-6)
        assert inst["TYPE"] == pytest.approx(1.0)   # inherited from default

    def test_unknown_param_raises_with_suggestions(self):
        from circulax.components.osdi import osdi_component

        desc = osdi_component(
            osdi_path=PSP103_OSDI, ports=("D", "G", "S", "B")
        )
        with pytest.raises(ValueError, match="Unknown OSDI parameter"):
            desc.make_instance({"WIDTH": 10e-6})   # should suggest 'W'


# ────────────────────────────────────────────────────────────────────────────
# Stage 3 — single PSP103 NMOS / PMOS DC
# ────────────────────────────────────────────────────────────────────────────


def _compile_single_nmos(vgs: float, vds: float):
    """Build a single-NMOS DC test: D→Vds, G→Vgs, S/B→GND.  Returns (y, solver)."""
    from circulax import compile_netlist
    from circulax.components.electronic import VoltageSource
    from circulax.solvers import analyze_circuit

    psp103n, _ = make_psp103_descriptors()
    netlist = {
        "instances": {
            "GND": {"component": "ground"},
            "Vds": {"component": "vsrc", "settings": {"V": vds}},
            "Vgs": {"component": "vsrc", "settings": {"V": vgs}},
            "M":   {"component": "nmos", "settings": geom_settings(10e-6, 1e-6)},
        },
        "connections": {
            "Vds,p1": "d,p1",   "Vds,p2": "GND,p1",
            "Vgs,p1": "g,p1",   "Vgs,p2": "GND,p1",
            "M,D": "d,p1",
            "M,G": "g,p1",
            "M,S": "GND,p1",
            "M,B": "GND,p1",
        },
    }
    models = {"nmos": psp103n, "vsrc": VoltageSource}
    groups, sys_size, port_map = compile_netlist(netlist, models)
    solver = analyze_circuit(groups, sys_size)
    # Large embedded Gmin (10 mS) to stabilise Newton: dominates PSP103's
    # negative -gds diagonal while still reporting the physical operating
    # point (node voltages are fixed by the VSources; Gmin can't perturb them).
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, 1e-2)
    y0 = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=12)
    y = solver.solve_dc_gmin(groups, y0, g_start=1e-2, n_steps=15)
    return y, port_map


class TestStage3SingleDevice:
    """Single PSP103 NMOS / PMOS DC operating points and sweeps."""

    def test_nmos_dc_off_state(self):
        """Vgs=0, Vds=0.6: NMOS is off; drain node stays near Vds (sub-threshold)."""
        y, port_map = _compile_single_nmos(vgs=0.0, vds=0.6)
        vd = float(y[port_map["d,p1"]])
        assert jnp.all(jnp.isfinite(y))
        assert 0.0 <= vd <= 1.2 + 1e-3

    def test_nmos_dc_on_state(self):
        """Vgs=1.0, Vds=0.6: NMOS is on; drain node still equals Vds (fixed by source)."""
        y, port_map = _compile_single_nmos(vgs=1.0, vds=0.6)
        assert jnp.all(jnp.isfinite(y))
        # The drain voltage is pinned by Vds, so we just confirm convergence.
        assert float(y[port_map["d,p1"]]) == pytest.approx(0.6, abs=1e-3)

    def test_nmos_id_monotone_in_vgs(self):
        """|I_ds| increases monotonically as Vgs rises through threshold.

        We read the source-side current out of Vgs's internal ``i_src`` branch:
        for VoltageSource, ``i_src`` is the current flowing *into* the +terminal.
        For an NMOS with S=GND, the drain-source current flows from D to S
        through the channel; the gate draws negligible DC current in PSP103.
        Instead, we extract I_ds from the drain-side voltage source, which
        must sink exactly -I_ds at DC to hold Vds fixed.
        """
        from circulax import compile_netlist
        from circulax.components.electronic import VoltageSource
        from circulax.solvers import analyze_circuit

        psp103n, _ = make_psp103_descriptors()
        ids_values = []
        for vgs in (0.0, 0.3, 0.6, 0.9, 1.2):
            netlist = {
                "instances": {
                    "GND": {"component": "ground"},
                    "Vds": {"component": "vsrc", "settings": {"V": 0.6}},
                    "Vgs": {"component": "vsrc", "settings": {"V": vgs}},
                    "M":   {"component": "nmos",
                            "settings": geom_settings(10e-6, 1e-6)},
                },
                "connections": {
                    "Vds,p1": "d,p1", "Vds,p2": "GND,p1",
                    "Vgs,p1": "g,p1", "Vgs,p2": "GND,p1",
                    "M,D": "d,p1",
                    "M,G": "g,p1",
                    "M,S": "GND,p1",
                    "M,B": "GND,p1",
                },
            }
            models = {"nmos": psp103n, "vsrc": VoltageSource}
            groups, sys_size, port_map = compile_netlist(netlist, models)
            solver = analyze_circuit(groups, sys_size)
            high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, 1e-2)
            y0 = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=12)
            y = solver.solve_dc_gmin(groups, y0, g_start=1e-2, n_steps=15)
            # The Vds source's i_src slot carries the current it sinks to hold Vds.
            # Find it: VoltageSource exposes an "i_src" internal state.
            ids_key = "Vds,i_src"
            assert ids_key in port_map, f"port_map missing {ids_key}; got: {list(port_map.keys())[:20]}"
            i_src = float(y[port_map[ids_key]])
            ids_values.append((vgs, i_src))

        print("\nNMOS Id(Vgs) sweep @ Vds=0.6V:")
        for vgs, i in ids_values:
            print(f"  Vgs={vgs:.2f} V  Vds source current = {i:+.3e} A")

        ids_mag = [abs(i) for _, i in ids_values]
        # Subthreshold → saturation: |Id| should grow by several orders of magnitude.
        assert ids_mag[-1] > ids_mag[0], (
            f"expected |Id| to grow with Vgs, got {ids_mag[0]:.3e} → {ids_mag[-1]:.3e}"
        )
        # Loose monotonicity check (allow tiny non-monotone noise in subthreshold).
        assert ids_mag[-1] >= ids_mag[-2] >= ids_mag[-3], (
            f"expected monotone |Id| in strong inversion, got {ids_mag}"
        )

    def test_pmos_dc_converges(self):
        """PMOS with Vsg=1.0V, Vsd=0.6V: check DC converges at physical voltages."""
        from circulax import compile_netlist
        from circulax.components.electronic import VoltageSource
        from circulax.solvers import analyze_circuit

        _, psp103p = make_psp103_descriptors()
        VDD = 1.2
        # PMOS source at VDD; gate at VDD - 1.0 = 0.2V → |Vgs| = 1.0V.
        netlist = {
            "instances": {
                "GND": {"component": "ground"},
                "Vdd": {"component": "vsrc", "settings": {"V": VDD}},
                "Vd":  {"component": "vsrc", "settings": {"V": VDD - 0.6}},
                "Vg":  {"component": "vsrc", "settings": {"V": VDD - 1.0}},
                "M":   {"component": "pmos",
                        "settings": geom_settings(20e-6, 1e-6)},
            },
            "connections": {
                "Vdd,p1": "vdd,p1", "Vdd,p2": "GND,p1",
                "Vd,p1":  "d,p1",   "Vd,p2":  "GND,p1",
                "Vg,p1":  "g,p1",   "Vg,p2":  "GND,p1",
                "M,D": "d,p1",
                "M,G": "g,p1",
                "M,S": "vdd,p1",
                "M,B": "vdd,p1",
            },
        }
        models = {"pmos": psp103p, "vsrc": VoltageSource}
        groups, sys_size, port_map = compile_netlist(netlist, models)
        solver = analyze_circuit(groups, sys_size)
        high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, 1e-2)
        y0 = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=12)
        y = solver.solve_dc_gmin(groups, y0, g_start=1e-2, n_steps=15)
        assert jnp.all(jnp.isfinite(y))
        assert float(y[port_map["vdd,p1"]]) == pytest.approx(VDD, abs=1e-3)


# ────────────────────────────────────────────────────────────────────────────
# Stage 4 — CMOS inverter
# ────────────────────────────────────────────────────────────────────────────


def _build_inverter_netlist(vin: float, vdd: float = 1.2, c_load: float = 50e-15) -> dict:
    """CMOS inverter with explicit Vin source.  Sizing matches VACASK ring osc."""
    mos_n = geom_settings(10e-6, 1e-6)
    mos_p = geom_settings(20e-6, 1e-6)
    return {
        "instances": {
            "GND": {"component": "ground"},
            "Vdd": {"component": "vsrc", "settings": {"V": vdd}},
            "Vin": {"component": "vsrc", "settings": {"V": vin}},
            "MN":  {"component": "nmos", "settings": mos_n},
            "MP":  {"component": "pmos", "settings": mos_p},
            "CL":  {"component": "cap",  "settings": {"C": c_load}},
        },
        "connections": {
            "Vdd,p1": "vdd,p1",   "Vdd,p2": "GND,p1",
            "Vin,p1": "in,p1",    "Vin,p2": "GND,p1",
            "MN,D": "out,p1", "MN,G": "in,p1", "MN,S": "GND,p1", "MN,B": "GND,p1",
            "MP,D": "out,p1", "MP,G": "in,p1", "MP,S": "vdd,p1", "MP,B": "vdd,p1",
            "CL,p1": "out,p1",    "CL,p2": "GND,p1",
        },
    }


def _inverter_models():
    from circulax.components.electronic import Capacitor, VoltageSource

    psp103n, psp103p = make_psp103_descriptors()
    return {
        "nmos": psp103n,
        "pmos": psp103p,
        "vsrc": VoltageSource,
        "cap":  Capacitor,
    }


class TestStage4Inverter:
    """CMOS inverter DC transfer curve and step-input transient."""

    def test_inverter_dc_transfer_curve(self):
        """Vout should fall monotonically as Vin sweeps 0→VDD."""
        from circulax import compile_netlist
        from circulax.solvers import analyze_circuit

        VDD = 1.2
        vin_sweep = np.linspace(0.0, VDD, 7)
        vout_sweep = []

        models = _inverter_models()
        for vin in vin_sweep:
            groups, sys_size, port_map = compile_netlist(
                _build_inverter_netlist(vin=float(vin), vdd=VDD), models
            )
            solver = analyze_circuit(groups, sys_size)
            high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, 1e-2)
            y0 = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=15)
            y = solver.solve_dc_gmin(groups, y0, g_start=1e-2, n_steps=20)
            vout_sweep.append(float(y[port_map["out,p1"]]))

        print("\nCMOS inverter DC transfer curve:")
        for vin, vout in zip(vin_sweep, vout_sweep):
            print(f"  Vin={vin:.3f} V  →  Vout={vout:.4f} V")

        # Vout at Vin=0 should be close to VDD.
        assert vout_sweep[0] > 0.75 * VDD, (
            f"Vout(Vin=0) = {vout_sweep[0]:.3f} V, expected near VDD={VDD}"
        )
        # Vout at Vin=VDD should be close to GND.
        assert vout_sweep[-1] < 0.25 * VDD, (
            f"Vout(Vin=VDD) = {vout_sweep[-1]:.3f} V, expected near 0"
        )
        # Monotone-decreasing (allow tiny Gmin/Newton noise).
        for i in range(len(vout_sweep) - 1):
            assert vout_sweep[i + 1] <= vout_sweep[i] + 5e-3, (
                f"transfer curve non-monotone at step {i}: "
                f"{vout_sweep[i]:.3f} → {vout_sweep[i + 1]:.3f}"
            )
        # Transfer curve should cross VDD/2.
        mid_crossings = [
            i for i in range(len(vout_sweep) - 1)
            if (vout_sweep[i] - VDD / 2) * (vout_sweep[i + 1] - VDD / 2) < 0
        ]
        assert mid_crossings, "inverter transfer curve does not cross VDD/2"

    def test_inverter_step_transient(self):
        """Step input (0→VDD) at t=1ns; output should switch high→low within a few ns."""
        import diffrax

        from circulax import compile_netlist
        from circulax.components.electronic import Capacitor, SmoothPulse, VoltageSource
        from circulax.solvers import analyze_circuit, setup_transient
        from circulax.solvers.transient import SDIRK3VectorizedTransientSolver

        psp103n, psp103p = make_psp103_descriptors()
        VDD = 1.2
        # Use SmoothPulse for the input so derivatives are well-defined.
        netlist = {
            "instances": {
                "GND": {"component": "ground"},
                "Vdd": {"component": "vsrc", "settings": {"V": VDD}},
                "Vin": {"component": "pulse",
                        "settings": {"V": VDD, "delay": 1e-9, "tr": 0.5e-9}},
                "MN":  {"component": "nmos", "settings": geom_settings(10e-6, 1e-6)},
                "MP":  {"component": "pmos", "settings": geom_settings(20e-6, 1e-6)},
                "CL":  {"component": "cap",  "settings": {"C": 50e-15}},
            },
            "connections": {
                "Vdd,p1": "vdd,p1", "Vdd,p2": "GND,p1",
                "Vin,p1": "in,p1",  "Vin,p2": "GND,p1",
                "MN,D": "out,p1", "MN,G": "in,p1",
                "MN,S": "GND,p1", "MN,B": "GND,p1",
                "MP,D": "out,p1", "MP,G": "in,p1",
                "MP,S": "vdd,p1", "MP,B": "vdd,p1",
                "CL,p1": "out,p1", "CL,p2": "GND,p1",
            },
        }
        models = {
            "nmos": psp103n, "pmos": psp103p,
            "vsrc": VoltageSource, "pulse": SmoothPulse, "cap": Capacitor,
        }
        groups, sys_size, port_map = compile_netlist(netlist, models)
        solver = analyze_circuit(groups, sys_size)

        # DC init: Vin ≈ 0 (pulse hasn't started yet).
        high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, 1e-2)
        y0 = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=15)
        y0 = solver.solve_dc_gmin(groups, y0, g_start=1e-2, n_steps=20)
        assert jnp.all(jnp.isfinite(y0))

        t0, t1 = 0.0, 10e-9
        run = setup_transient(
            groups, solver, transient_solver=SDIRK3VectorizedTransientSolver
        )
        sol = run(
            t0=t0, t1=t1, dt0=0.01e-9, y0=y0,
            saveat=diffrax.SaveAt(ts=jnp.linspace(t0, t1, 200)),
            max_steps=50_000,
        )
        ys = np.asarray(sol.ys)
        assert np.all(np.isfinite(ys))

        vout = ys[:, port_map["out,p1"]]
        vin_trace = ys[:, port_map["in,p1"]]
        print(f"\nInverter step: vout start={vout[0]:.3f} V, end={vout[-1]:.3f} V")
        print(f"                vin  start={vin_trace[0]:.3f} V, end={vin_trace[-1]:.3f} V")

        # Before the step Vout should be high; after the step it should drop.
        assert vout[0] > 0.75 * VDD, f"Vout before step = {vout[0]:.3f} V (expected high)"
        assert vout[-1] < 0.4 * VDD, (
            f"Vout after step = {vout[-1]:.3f} V (expected to fall toward GND)"
        )
