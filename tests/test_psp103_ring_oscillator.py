"""9-stage CMOS ring oscillator using PSP103 OSDI models.

Translated from the vacask reference circuit in:
  circulax/components/osdi/vacask/runme.sim + models.inc

Circuit
-------
  VDD = 1.2 V
  9 CMOS inverters in a ring (nodes n1–n9)
  Each inverter: PMOS(D=out, G=in, S=vdd, B=vdd) + NMOS(D=out, G=in, S=gnd, B=gnd)
  NMOS: W = 10 µm, L = 1 µm
  PMOS: W = 20 µm, L = 1 µm  (pfact=2, matching vacask reference)
  Kickstart: SmoothPulse voltage source + 100 kΩ on node n1 to break symmetry
"""

import sys
import re
import struct
import ctypes
from pathlib import Path

import equinox as eqx
import pytest
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, "/home/cdaunt/code/bosdi/src")

from circulax.components.osdi import _BOSDI_AVAILABLE, osdi_component

pytestmark = pytest.mark.skipif(not _BOSDI_AVAILABLE, reason="bosdi package not available")

# ── Paths ────────────────────────────────────────────────────────────────────
_OSDI_DIR  = Path(__file__).parent.parent / "circulax" / "components" / "osdi" / "compiled"
_PSP103    = str(_OSDI_DIR / "psp103v4_psp103.osdi")
_VA_DIR    = Path(__file__).parent.parent / "circulax" / "components" / "osdi" / "psp103v4"
_MODELS_INC = Path(__file__).parent.parent / "circulax" / "components" / "osdi" / "vacask" / "models.inc"


# ── Extract parameter names from binary ──────────────────────────────────────

def _read_psp103_param_names() -> list[str]:
    """Read the 783 PSP103 parameter names in binary order via the OSDI descriptor."""
    import osdi_shim_nb

    osdi_shim_nb.load_osdi_library(_PSP103, 4)
    lib = ctypes.CDLL(_PSP103)

    desc = ctypes.addressof(ctypes.c_uint8.in_dll(lib, "OSDI_DESCRIPTORS"))
    raw  = bytes(bytearray((ctypes.c_uint8 * 128).from_address(desc)))
    num_params      = struct.unpack_from("<I", raw, 76)[0]
    param_opvar_ptr = struct.unpack_from("<Q", raw, 88)[0]

    with open("/proc/self/maps") as f:
        maps = [
            (int(s, 16), int(e, 16))
            for line in f
            for s, e in [line.split()[0].split("-")]
            if "r" in line.split()[1]
        ]

    def readable(a):
        return any(s <= a < e for s, e in maps)

    STRIDE = 40
    block = bytes(bytearray((ctypes.c_uint8 * (num_params * STRIDE)).from_address(param_opvar_ptr)))

    names = []
    for i in range(num_params):
        off    = i * STRIDE
        np_ptr = struct.unpack_from("<Q", block, off)[0]
        if not np_ptr or not readable(np_ptr):
            names.append(None)
            continue
        raw_np  = bytes(bytearray((ctypes.c_uint8 * 8).from_address(np_ptr)))
        fn_ptr  = struct.unpack_from("<Q", raw_np)[0]
        if not fn_ptr or not readable(fn_ptr):
            names.append(None)
            continue
        names.append(ctypes.string_at(fn_ptr).decode("ascii"))

    return names


# ── Extract default values from VA source ────────────────────────────────────

_PARAM_RE = re.compile(r"`(?:MPR|MPI|IPR|IPI)\w*\(\s*(\w+)\s*,\s*([^,\)]+)")


def _to_float(s: str) -> float:
    s = s.strip().replace("inf", "1e30").replace("-inf", "-1e30")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _extract_va_defaults(filename: str, seen: set | None = None) -> dict[str, float]:
    if seen is None:
        seen = set()
    result: dict[str, float] = {}
    for line in (_VA_DIR / filename).read_text().splitlines():
        inc = re.match(r'\s*`include\s+"([^"]+)"', line)
        if inc:
            for k, v in _extract_va_defaults(inc.group(1), seen).items():
                result.setdefault(k, v)
            continue
        m = _PARAM_RE.search(line)
        if m and m.group(1) not in seen:
            seen.add(m.group(1))
            result[m.group(1)] = _to_float(m.group(2))
    return result


def _parse_models_inc_block(block: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for line in block.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            result[k.strip().upper()] = _to_float(v.strip())
    return result


# ── Build OSDI descriptors (cached at module level) ──────────────────────────

def _build_psp103_descriptors():
    param_names   = _read_psp103_param_names()
    va_defaults   = _extract_va_defaults("PSP103_module.include")
    models_text   = _MODELS_INC.read_text()

    n_overrides = _parse_models_inc_block(
        re.search(r"model psp103n psp103va \((.*?)\)", models_text, re.DOTALL).group(1)
    )
    p_overrides = _parse_models_inc_block(
        re.search(r"model psp103p psp103va \((.*?)\)", models_text, re.DOTALL).group(1)
    )

    n_defaults: dict[str, float] = {}
    p_defaults: dict[str, float] = {}

    for name in param_names:
        if name is None:
            continue
        base = 1.0 if name == "$mfactor" else va_defaults.get(name, 0.0)
        n_defaults[name] = n_overrides.get(name, base)
        p_defaults[name] = p_overrides.get(name, base)

    # Filter to non-None names only (all 783 are named for PSP103)
    named = [n for n in param_names if n is not None]
    ports = ("D", "G", "S", "B")

    psp103n = osdi_component(
        osdi_path=_PSP103,
        ports=ports,
        param_names=tuple(named),
        default_params=n_defaults,
    )
    psp103p = osdi_component(
        osdi_path=_PSP103,
        ports=ports,
        param_names=tuple(named),
        default_params=p_defaults,
    )
    return psp103n, psp103p


# ── Instance parameter helper ─────────────────────────────────────────────────

def _mos_settings(w: float, l: float, ld: float = 0.5e-6, ls: float = 0.5e-6) -> dict:
    """Compute geometry-derived instance parameters matching the vacask subckt."""
    return {
        "W":  w,
        "L":  l,
        "AD": w * ld,
        "AS": w * ls,
        "PD": 2.0 * (w + ld),
        "PS": 2.0 * (w + ls),
    }


# ── Build netlist ─────────────────────────────────────────────────────────────

def _build_ring_oscillator_netlist(
    w_n: float = 10e-6,
    w_p: float = 20e-6,
    l: float = 1e-6,
    c_load: float = 50e-15,
) -> dict:
    """9-stage CMOS ring oscillator netlist in circulax/SAX format.

    w_n: NMOS width (default 10 µm, matching vacask reference).
    w_p: PMOS width (default 20 µm = pfact=2, matching vacask reference).
    c_load: lumped load capacitance per stage output (default 50 fF).

    Why 50 fF?  bosdi's terminal Jacobian for PSP103 only includes currents that
    flow directly between external terminals (D, G, S, B).  The main channel
    current IDS flows between the *internal* nodes di→si, whose collapsed
    Jacobian contributions appear at cond[D,D] ≈ −2 mS and cond[D,D] ≈ −4 mS
    for NMOS and PMOS respectively, giving a net −6 mS diagonal at each ring
    node.  With SDIRK3 (α ≈ 2.29) the load cap must satisfy
    α × C_load / dt > 6 mS → C_load > 26 fF at dt = 10 ps.  50 fF provides
    a comfortable margin and is physically realistic (total node capacitance
    including gate loading of the next stage + metal wiring in a 1 µm process).
    """
    mos_n = _mos_settings(w_n, l)
    mos_p = _mos_settings(w_p, l)

    instances: dict = {}
    connections: dict = {}

    # VDD supply
    instances["Vvdd"] = {"component": "vsrc", "settings": {"V": 1.2}}
    connections["Vvdd,p1"] = "vdd,p1"
    connections["Vvdd,p2"] = "GND,p1"

    # Kickstart: SmoothPulse (≈ 10 µA current pulse via 100 kΩ) on node n1
    # SmoothPulse from GND→kick_n, then Resistor kick_n→n1
    instances["Vkick"] = {"component": "kick", "settings": {"V": 1.0, "delay": 1e-9, "tr": 1e-9}}
    instances["Rkick"] = {"component": "r_kick", "settings": {"R": 1e5}}
    connections["Vkick,p1"] = "kick_n,p1"
    connections["Vkick,p2"] = "GND,p1"
    connections["Rkick,p1"] = "kick_n,p1"
    connections["Rkick,p2"] = "n1,p1"

    # 9 inverter stages: node n{i} drives n{i%9+1}
    for stage in range(1, 10):
        in_node  = f"n{stage}"
        out_node = f"n{stage % 9 + 1}"

        mn = f"mn{stage}"
        mp = f"mp{stage}"

        # NMOS: D=out, G=in, S=GND, B=GND
        instances[mn] = {"component": "nmos", "settings": mos_n}
        connections[f"{mn},D"] = f"{out_node},p1"
        connections[f"{mn},G"] = f"{in_node},p1"
        connections[f"{mn},S"] = "GND,p1"
        connections[f"{mn},B"] = "GND,p1"

        # PMOS: D=out, G=in, S=vdd, B=vdd (pfact=2: W=20µm for stronger pull-up)
        instances[mp] = {"component": "pmos", "settings": mos_p}
        connections[f"{mp},D"] = f"{out_node},p1"
        connections[f"{mp},G"] = f"{in_node},p1"
        connections[f"{mp},S"] = "vdd,p1"
        connections[f"{mp},B"] = "vdd,p1"

        # Lumped load capacitor: output node → GND
        cl = f"CL{stage}"
        instances[cl] = {"component": "cload", "settings": {"C": c_load}}
        connections[f"{cl},p1"] = f"{out_node},p1"
        connections[f"{cl},p2"] = "GND,p1"

    return {
        "instances":   instances,
        "connections": connections,
        "ports":       {"out": "n1,p1"},
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def psp103_models():
    return _build_psp103_descriptors()


@pytest.fixture(scope="module")
def ring_compiled(psp103_models):
    from circulax import compile_netlist
    from circulax.components.electronic import VoltageSource, Resistor, SmoothPulse
    from circulax.solvers import analyze_circuit

    from circulax.components.electronic import Capacitor
    psp103n, psp103p = psp103_models
    models = {
        "nmos":   psp103n,
        "pmos":   psp103p,
        "vsrc":   VoltageSource,
        "kick":   SmoothPulse,
        "r_kick": Resistor,
        "cload":  Capacitor,
    }
    netlist = _build_ring_oscillator_netlist()
    groups, sys_size, port_map = compile_netlist(netlist, models)
    solver = analyze_circuit(groups, sys_size)
    return groups, sys_size, port_map, solver


def test_psp103_descriptors_load(psp103_models):
    """PSP103 OSDI binary loads and returns valid N/P descriptors."""
    psp103n, psp103p = psp103_models
    from circulax.components.osdi import OsdiModelDescriptor
    assert isinstance(psp103n, OsdiModelDescriptor)
    assert isinstance(psp103p, OsdiModelDescriptor)
    assert psp103n.model.num_pins   == 4
    assert psp103n.model.num_params == 783
    assert psp103n.model.num_states == 0
    assert psp103n.default_params["TYPE"] == pytest.approx(1.0)
    assert psp103p.default_params["TYPE"] == pytest.approx(-1.0)


def test_ring_oscillator_dc(ring_compiled):
    """Ring oscillator converges to the metastable DC operating point.

    A 9-stage ring oscillator has no stable DC equilibrium (odd number of
    inverter stages), so the Newton solver converges to a numerically-stable
    but physically metastable fixed point where all ring nodes sit near VDD/2.

    Plain Newton from zeros diverges to an unphysical fixed point because
    PSP103 has negative diagonal Jacobian entries (outward-current convention,
    G[D,D] = -gds).  The two-phase homotopy approach avoids this:

    1. GMIN stepping from zeros: large g_leak (10 mS) dominates the negative
       -gds diagonal and makes Newton converge.  g_leak reduces logarithmically
       to the solver's baseline (1 nS).
    2. Source stepping from the GMIN result: VDD ramps from 10% to 100%,
       each step warm-started from the previous solution.

    We assert that all ring nodes land within [0, VDD].
    """
    groups, sys_size, port_map, solver = ring_compiled

    VDD = 1.2
    # Two-phase homotopy for circuits with negative diagonal Jacobians (PSP103
    # G[D,D]=-gds under outward-current convention):
    #
    # Phase 1 — source stepping with large embedded Gmin (10 mS):
    #   At VDD=0.12V (scale=0.1) all transistors are deep in subthreshold;
    #   10 mS easily dominates any negative diagonal.  Newton converges to ~0 V.
    #   Stepping to VDD=1.2V with the warm start keeps Newton in the physical basin.
    #   Gmin=10 mS >> gds even at full VDD, so the metastable point is near VDD/2.
    #
    # Phase 2 — GMIN stepping from the source result (10 mS → solver baseline):
    #   Now VDD is fixed at 1.2V.  Gmin reduces logarithmically; each step is
    #   warm-started from the previous solution.  As Gmin → 1 nS, Newton tracks
    #   the true metastable equilibrium without ever leaving the physical basin.
    G_HOMOTOPY = 1e-2  # 10 mS — must dominate max |gds| across all transistors
    high_gmin_solver = eqx.tree_at(lambda s: s.g_leak, solver, G_HOMOTOPY)
    y_source = high_gmin_solver.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y = solver.solve_dc_gmin(groups, y_source, g_start=G_HOMOTOPY, n_steps=30)

    assert jnp.all(jnp.isfinite(y)), "DC solver returned non-finite values"

    print("\nDC operating point:")
    inv_map = {v: k for k, v in port_map.items()}
    for i in range(sys_size):
        print(f"  [{i:2d}] {inv_map.get(i,'?'):20s}: {float(y[i]):.4f} V")

    for stage in range(1, 10):
        k = f"n{stage},p1"
        if k in port_map:
            v = float(y[port_map[k]])
            assert 0.0 <= v <= VDD, f"Ring node {k} = {v:.3f} V is outside [0, VDD]"


def test_ring_oscillator_transient(ring_compiled):
    """Ring oscillator produces oscillation in transient simulation.

    Initialisation strategy: all ring nodes start at VDD/2 (the metastable
    equilibrium for an odd-stage ring) and VDD is set to 1.2 V.  The
    SmoothPulse kickstart source then breaks symmetry at t ≈ 1 ns, triggering
    self-sustaining oscillation.  We assert a voltage swing > 100 mV on n1
    over a 100 ns window, which should hold for any physically correct
    PSP103 parameterisation.
    """
    import diffrax
    from circulax.solvers import setup_transient
    from circulax.solvers.transient import SDIRK3VectorizedTransientSolver

    groups, sys_size, port_map, solver = ring_compiled

    # Phase 1: two-phase homotopy DC solve (same approach as test_ring_oscillator_dc).
    # Source stepping with large embedded Gmin (10 mS) followed by GMIN stepping
    # to the solver baseline.  This correctly initialises all internal OSDI nodes
    # (si, di for each transistor) at the metastable VDD/2 operating point.
    G_HOMOTOPY = 1e-2
    high_gmin_solver = eqx.tree_at(lambda s: s.g_leak, solver, G_HOMOTOPY)
    y_source = high_gmin_solver.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y0 = solver.solve_dc_gmin(groups, y_source, g_start=G_HOMOTOPY, n_steps=30)
    assert jnp.all(jnp.isfinite(y0)), "DC solve for transient IC produced non-finite values"

    t0, t1 = 0.0, 100e-9   # 0–100 ns (ample time for oscillation to appear)
    dt     = 0.01e-9        # 10 ps — ensures C_load/dt >> |PSP103 terminal leakage Jacobian|

    run = setup_transient(groups, solver, transient_solver=SDIRK3VectorizedTransientSolver)
    sol = run(
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=y0,
        saveat=diffrax.SaveAt(ts=jnp.linspace(t0, t1, 500)),
        max_steps=200_000,
    )

    ys = np.asarray(sol.ys)
    assert np.all(np.isfinite(ys)), "Transient solution contains non-finite values"

    # Locate n1 in the port_map — it may be stored as "n1,p1" or "n1"
    n1_key = "n1,p1" if "n1,p1" in port_map else "n1"
    n1_idx = port_map[n1_key]
    v_n1   = ys[:, n1_idx]
    v_swing = float(np.max(v_n1) - np.min(v_n1))

    print(f"\nTransient: {ys.shape[0]} points over {t1*1e9:.0f} ns")
    print(f"  n1 swing: {v_swing*1e3:.1f} mV  (min={np.min(v_n1):.3f} V, max={np.max(v_n1):.3f} V)")
    assert v_swing > 0.1, f"Expected oscillation, got only {v_swing*1e3:.1f} mV swing on n1"
