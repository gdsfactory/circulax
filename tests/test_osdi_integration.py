"""Integration tests for OSDI device model components.

Tests that OpenVAF-compiled .osdi binaries can be loaded, compiled into
a netlist, and solved at the DC operating point — and that the results
match the equivalent pure-JAX component implementation.

Requires bosdi to be installed or its src/ directory on PYTHONPATH.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from circulax.components.osdi import _BOSDI_AVAILABLE

pytestmark = pytest.mark.skipif(not _BOSDI_AVAILABLE, reason="bosdi package not available")

_OSDI_DIR = Path(__file__).parent / "osdi"
OSDI_RESISTOR = str(_OSDI_DIR / "resistor_va.osdi")
OSDI_CAPACITOR = str(_OSDI_DIR / "capacitor_va.osdi")


@pytest.fixture
def osdi_capacitor():
    """Load the OSDI capacitor descriptor (ports: P, N; params: mfactor, C, m).

    OSDI param ordering for capacitor_va.osdi: [$mfactor(INST), C(MODEL), m(MODEL)]
    where mfactor is the global SPICE multiplicity, C is capacitance, and m is
    the Verilog-A parallel factor.
    """
    from circulax import osdi_component
    return osdi_component(
        osdi_path=OSDI_CAPACITOR,
        ports=("P", "N"),
        param_names=("mfactor", "C", "m"),
        default_params={"mfactor": 1.0, "C": 1e-9, "m": 1.0},
    )


@pytest.fixture
def osdi_resistor():
    """Load the OSDI resistor descriptor (ports: A, B; params: m, R).

    OSDI param ordering for resistor_va.osdi: [$mfactor(INST), R(MODEL)]
    where m is the SPICE multiplicity factor and R is the resistance.
    """
    from circulax import osdi_component
    return osdi_component(
        osdi_path=OSDI_RESISTOR,
        ports=("A", "B"),
        param_names=("m", "R"),
        default_params={"m": 1.0, "R": 1000.0},
    )


def test_osdi_component_loads(osdi_resistor):
    """osdi_component() should load the binary and report correct metadata."""
    assert osdi_resistor.model.num_pins == 2
    assert osdi_resistor.model.num_params == 2
    assert osdi_resistor.model.num_states == 0
    assert osdi_resistor.ports == ("A", "B")
    assert osdi_resistor.param_names == ("m", "R")


def test_osdi_compile_netlist(osdi_resistor):
    """compile_netlist should produce an OsdiComponentGroup for OSDI instances."""
    from circulax import compile_netlist
    from circulax.components.electronic import VoltageSource
    from circulax.components.osdi import OsdiComponentGroup

    net = {
        "instances": {
            "GND": {"component": "ground"},
            "Vs": {"component": "vsrc", "settings": {"V": 1.0}},
            "R1": {"component": "res", "settings": {"R": 100.0, "m": 1.0}},
        },
        "connections": {
            "Vs,p1": "R1,A",
            "Vs,p2": "GND,p1",
            "R1,B": "GND,p1",
        },
        "ports": {"out": "R1,A"},
    }
    models = {"vsrc": VoltageSource, "res": osdi_resistor}
    groups, _sys_size, _ = compile_netlist(net, models)

    assert "res" in groups
    assert isinstance(groups["res"], OsdiComponentGroup)
    assert groups["res"].num_pins == 2
    assert groups["res"].params.shape == (1, 2)   # N=1, num_params=2
    # param order: [m=1.0, R=100.0]; R is at index 1
    assert float(groups["res"].params[0, 1]) == pytest.approx(100.0)


def test_osdi_dc_single_resistor(osdi_resistor):
    """DC operating point of a single OSDI resistor should match Ohm's law.

    Circuit: Vs=1V --- R1=100Ω --- GND
    Expected: node voltage = 1V, current = 10mA
    """
    from circulax import compile_netlist
    from circulax.components.electronic import VoltageSource
    from circulax.solvers import analyze_circuit

    net = {
        "instances": {
            "GND": {"component": "ground"},
            "Vs": {"component": "vsrc", "settings": {"V": 1.0}},
            "R1": {"component": "res", "settings": {"R": 100.0, "m": 1.0}},
        },
        "connections": {
            "Vs,p1": "R1,A",
            "Vs,p2": "GND,p1",
            "R1,B": "GND,p1",
        },
        "ports": {"out": "R1,A"},
    }
    models = {"vsrc": VoltageSource, "res": osdi_resistor}
    groups, sys_size, port_map = compile_netlist(net, models)

    y0 = jnp.zeros(sys_size)
    solver = analyze_circuit(groups, sys_size)
    y = solver.solve_dc(groups, y0)

    node_out = port_map["R1,A"]
    v_out = float(y[node_out])
    assert v_out == pytest.approx(1.0, rel=1e-4)


def test_osdi_dc_resistor_ladder_matches_jax(osdi_resistor):
    """OSDI resistor ladder DC solution must match the pure-JAX Resistor.

    3-node voltage divider: Vs=4V --- R1 --- R2 --- R3 --- GND
    With equal R=1kΩ the node voltages should be 3V, 2V, 1V.
    """
    from circulax import compile_netlist
    from circulax.components.electronic import Resistor, VoltageSource
    from circulax.solvers import analyze_circuit

    R_val = 1000.0

    def _make_ladder_netlist(res_component_name):
        return {
            "instances": {
                "GND": {"component": "ground"},
                "Vs": {"component": "vsrc", "settings": {"V": 4.0}},
                "R1": {"component": res_component_name, "settings": {"R": R_val, "m": 1.0}},
                "R2": {"component": res_component_name, "settings": {"R": R_val, "m": 1.0}},
                "R3": {"component": res_component_name, "settings": {"R": R_val, "m": 1.0}},
            },
            "connections": {
                "Vs,p1": "R1,A",
                "Vs,p2": "GND,p1",
                "R1,B": "R2,A",
                "R2,B": "R3,A",
                "R3,B": "GND,p1",
            },
            "ports": {"n1": "R1,A", "n2": "R1,B", "n3": "R2,B"},
        }

    # --- OSDI solver ---
    osdi_net = _make_ladder_netlist("osdi_res")
    osdi_models = {"vsrc": VoltageSource, "osdi_res": osdi_resistor}
    osdi_groups, osdi_size, osdi_pmap = compile_netlist(osdi_net, osdi_models)
    osdi_solver = analyze_circuit(osdi_groups, osdi_size)
    y_osdi = osdi_solver.solve_dc(osdi_groups, jnp.zeros(osdi_size))

    v_osdi = [float(y_osdi[osdi_pmap[k]]) for k in ("R1,A", "R1,B", "R2,B")]

    # --- Pure-JAX Resistor solver (uses p1/p2 port names) ---
    jax_net = {
        "instances": {
            "GND": {"component": "ground"},
            "Vs": {"component": "vsrc", "settings": {"V": 4.0}},
            "R1": {"component": "jres", "settings": {"R": R_val}},
            "R2": {"component": "jres", "settings": {"R": R_val}},
            "R3": {"component": "jres", "settings": {"R": R_val}},
        },
        "connections": {
            "Vs,p1": "R1,p1",
            "Vs,p2": "GND,p1",
            "R1,p2": "R2,p1",
            "R2,p2": "R3,p1",
            "R3,p2": "GND,p1",
        },
        "ports": {"n1": "R1,p1", "n2": "R1,p2", "n3": "R2,p2"},
    }
    jax_models = {"vsrc": VoltageSource, "jres": Resistor}
    jax_groups, jax_size, jax_pmap = compile_netlist(jax_net, jax_models)
    jax_solver = analyze_circuit(jax_groups, jax_size)
    y_jax = jax_solver.solve_dc(jax_groups, jnp.zeros(jax_size))

    v_jax = [float(y_jax[jax_pmap[k]]) for k in ("R1,p1", "R1,p2", "R2,p2")]

    # R1,A = source node = 4V; divider nodes are 4*(2/3) and 4*(1/3)
    np.testing.assert_allclose(v_osdi, [4.0, 8.0 / 3.0, 4.0 / 3.0], rtol=1e-4, err_msg="OSDI ladder node voltages wrong")
    np.testing.assert_allclose(v_osdi, v_jax, rtol=1e-4, err_msg="OSDI and JAX ladder solutions differ")


def test_osdi_batched_instances(osdi_resistor):
    """Multiple OSDI instances in one group should be batched into a single osdi_eval call."""
    from circulax import compile_netlist
    from circulax.components.electronic import VoltageSource
    from circulax.components.osdi import OsdiComponentGroup
    from circulax.solvers import analyze_circuit

    net = {
        "instances": {
            "GND": {"component": "ground"},
            "Vs": {"component": "vsrc", "settings": {"V": 2.0}},
            "R1": {"component": "res", "settings": {"R": 100.0, "m": 1.0}},
            "R2": {"component": "res", "settings": {"R": 200.0, "m": 1.0}},
        },
        "connections": {
            "Vs,p1": "R1,A",
            "R1,B": "R2,A",
            "R2,B": "GND,p1",
            "Vs,p2": "GND,p1",
        },
        "ports": {"mid": "R1,B"},
    }
    models = {"vsrc": VoltageSource, "res": osdi_resistor}
    groups, sys_size, port_map = compile_netlist(net, models)

    # Both resistors should be batched into one OsdiComponentGroup of N=2
    assert isinstance(groups["res"], OsdiComponentGroup)
    assert groups["res"].params.shape == (2, 2)

    solver = analyze_circuit(groups, sys_size)
    y = solver.solve_dc(groups, jnp.zeros(sys_size))

    # Voltage divider: 2V * 200/(100+200) = 4/3 V at midpoint
    v_mid = float(y[port_map["R1,B"]])
    assert v_mid == pytest.approx(4.0 / 3.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Capacitor tests
# ---------------------------------------------------------------------------


def test_osdi_capacitor_loads(osdi_capacitor):
    """osdi_component() should load the capacitor binary and report correct metadata."""
    assert osdi_capacitor.model.num_pins == 2
    assert osdi_capacitor.model.num_params == 3  # C, m, tnom
    assert osdi_capacitor.model.num_states == 0
    assert osdi_capacitor.ports == ("P", "N")
    assert osdi_capacitor.param_names == ("mfactor", "C", "m")


def test_osdi_capacitor_dc_open_circuit(osdi_capacitor):
    """A capacitor is open circuit at DC: node voltage equals source voltage.

    Circuit: Vs=2V --- R=1kΩ --- C --- GND
    At DC steady state the capacitor carries no current, so there is no
    voltage drop across the resistor and V_cap = Vs = 2V.
    """
    from circulax import compile_netlist
    from circulax.components.electronic import Resistor, VoltageSource
    from circulax.solvers import analyze_circuit

    net = {
        "instances": {
            "GND": {"component": "ground"},
            "Vs": {"component": "vsrc", "settings": {"V": 2.0}},
            "R1": {"component": "res", "settings": {"R": 1000.0}},
            "C1": {"component": "cap", "settings": {"mfactor": 1.0, "C": 1e-6, "m": 1.0}},
        },
        "connections": {
            "Vs,p1": "R1,p1",
            "Vs,p2": "GND,p1",
            "R1,p2": "C1,P",
            "C1,N": "GND,p1",
        },
        "ports": {"cap_top": "C1,P"},
    }
    models = {"vsrc": VoltageSource, "res": Resistor, "cap": osdi_capacitor}
    groups, sys_size, port_map = compile_netlist(net, models)

    y0 = jnp.zeros(sys_size)
    solver = analyze_circuit(groups, sys_size)
    y = solver.solve_dc(groups, y0)

    v_cap = float(y[port_map["C1,P"]])
    assert v_cap == pytest.approx(2.0, rel=1e-4)


def test_osdi_capacitor_transient_rc_matches_jax(osdi_capacitor):
    """OSDI capacitor transient RC charging should match the pure-JAX Capacitor.

    Circuit: Vs=1V, R=1kΩ, C=1µF  →  RC = 1ms.
    Starting from y0=0 (uncharged), voltage follows V(t) = Vs*(1 - exp(-t/RC)).
    Verified at t=RC and t=3RC against both the analytical formula and the
    pure-JAX Capacitor.
    """
    import diffrax

    from circulax import compile_netlist
    from circulax.components.electronic import Capacitor, Resistor, VoltageSource
    from circulax.solvers import analyze_circuit, setup_transient

    R_val = 1000.0
    C_val = 1e-6
    Vs_val = 1.0
    RC = R_val * C_val  # 1e-3 s

    def _run_rc(cap_component_name, cap_port_p, cap_port_n, models, cap_settings):
        net = {
            "instances": {
                "GND": {"component": "ground"},
                "Vs": {"component": "vsrc", "settings": {"V": Vs_val}},
                "R1": {"component": "res", "settings": {"R": R_val}},
                "C1": {"component": cap_component_name, "settings": cap_settings},
            },
            "connections": {
                "Vs,p1": "R1,p1",
                "Vs,p2": "GND,p1",
                "R1,p2": f"C1,{cap_port_p}",
                f"C1,{cap_port_n}": "GND,p1",
            },
            "ports": {"cap_top": f"C1,{cap_port_p}"},
        }
        groups, sys_size, port_map = compile_netlist(net, models)
        linear_strat = analyze_circuit(groups, sys_size)

        t_end = 3.0 * RC
        ts = jnp.array([RC, 3.0 * RC])
        run = setup_transient(groups, linear_strat)
        sol = run(t0=0.0, t1=t_end, dt0=RC * 1e-3, y0=jnp.zeros(sys_size),
                  saveat=diffrax.SaveAt(ts=ts), max_steps=100_000)

        cap_node = port_map[f"C1,{cap_port_p}"]
        assert sol.ys is not None
        return sol.ys[:, cap_node]

    # OSDI capacitor
    osdi_models = {"vsrc": VoltageSource, "res": Resistor, "cap": osdi_capacitor}
    v_osdi = _run_rc("cap", "P", "N", osdi_models, {"mfactor": 1.0, "C": C_val, "m": 1.0})

    # Pure-JAX capacitor
    jax_models = {"vsrc": VoltageSource, "res": Resistor, "cap": Capacitor}
    v_jax = _run_rc("cap", "p1", "p2", jax_models, {"C": C_val})

    # Analytical: V(t) = Vs * (1 - exp(-t/RC))
    v_analytical = jnp.array([Vs_val * (1.0 - jnp.exp(-1.0)),   # t = RC
                               Vs_val * (1.0 - jnp.exp(-3.0))])  # t = 3RC

    np.testing.assert_allclose(v_osdi, v_analytical, rtol=1e-3,
                               err_msg="OSDI RC charging does not match analytical")
    np.testing.assert_allclose(v_osdi, v_jax, rtol=1e-3,
                               err_msg="OSDI and JAX-Capacitor RC trajectories differ")
