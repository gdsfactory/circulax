"""9-stage CMOS ring oscillator using PSP103 OSDI models (stages 6 & 7).

Translated from the VACASK reference at
``/home/cdaunt/code/vacask/VACASK/benchmark/ring/vacask/{runme.sim,models.inc}``.

Circuit
-------
  VDD = 1.2 V
  9 CMOS inverters in a ring (nodes n1–n9)
  Each inverter: PMOS(D=out, G=in, S=vdd, B=vdd) + NMOS(D=out, G=in, S=gnd, B=gnd)
  NMOS: W = 10 µm, L = 1 µm
  PMOS: W = 20 µm, L = 1 µm  (pfact=2, matching VACASK)
  Kickstart: SmoothPulse + 100 kΩ on node n1 (≈10 µA current pulse, mirroring
             VACASK's ``i0`` pulse source).

To regenerate the VACASK reference waveform (optional for the frequency
comparison in ``test_ring_oscillator_vs_vacask_reference``)::

    cd /home/cdaunt/code/vacask/VACASK/benchmark/ring/vacask
    vacask runme.sim                       # produces tran1.raw
    python -c 'from rawfile import rawread; import numpy as np; \
        r = rawread("tran1.raw").get(); \
        np.savez("tests/fixtures/vacask_ring_ref.npz", t=r["time"], v1=r["1"])'

The test is skipped when that ``.npz`` fixture is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax.components.osdi import _BOSDI_AVAILABLE

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.skipif(
    not _BOSDI_AVAILABLE, reason="bosdi package not available"
)

if _BOSDI_AVAILABLE:
    _TESTS_DIR = Path(__file__).resolve().parent
    if str(_TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(_TESTS_DIR))
    from fixtures.psp103_models import geom_settings, make_psp103_descriptors


_VACASK_REF_FIXTURE = Path(__file__).parent / "fixtures" / "vacask_ring_ref.npz"


# ── Netlist builder ──────────────────────────────────────────────────────────


def _build_ring_oscillator_netlist(
    w_n: float = 10e-6,
    w_p: float = 20e-6,
    length: float = 1e-6,
    c_load: float = 50e-15,
) -> dict:
    """9-stage CMOS ring oscillator netlist in circulax/SAX format.

    ``c_load`` (50 fF by default) stabilises the implicit-integrator Jacobian.
    bosdi's terminal Jacobian for PSP103 only exposes currents flowing between
    the external terminals (D, G, S, B); the main channel current IDS flows
    between internal nodes di→si whose collapsed Jacobian contributions
    appear at ``cond[D,D]`` ≈ −2 mS and ≈ −4 mS for NMOS/PMOS respectively
    (net −6 mS at each ring node).  For SDIRK3 (α ≈ 2.29) we need
    ``α × C_load / dt > 6 mS`` → ``C_load > 26 fF`` at dt = 10 ps, so
    50 fF provides a comfortable margin and is physically realistic for a
    1 µm process (gate loading + metal wiring).
    """
    mos_n = geom_settings(w_n, length)
    mos_p = geom_settings(w_p, length)

    instances: dict = {}
    connections: dict = {}

    # VDD supply
    instances["Vvdd"] = {"component": "vsrc", "settings": {"V": 1.2}}
    connections["Vvdd,p1"] = "vdd,p1"
    connections["Vvdd,p2"] = "GND,p1"

    # Kickstart: SmoothPulse (≈ 10 µA via 100 kΩ) on node n1
    instances["Vkick"] = {
        "component": "kick",
        "settings": {"V": 1.0, "delay": 1e-9, "tr": 1e-9},
    }
    instances["Rkick"] = {"component": "r_kick", "settings": {"R": 1e5}}
    connections["Vkick,p1"] = "kick_n,p1"
    connections["Vkick,p2"] = "GND,p1"
    connections["Rkick,p1"] = "kick_n,p1"
    connections["Rkick,p2"] = "n1,p1"

    # 9 inverter stages: n{i} drives n{i%9+1}.
    for stage in range(1, 10):
        in_node  = f"n{stage}"
        out_node = f"n{stage % 9 + 1}"
        mn = f"mn{stage}"
        mp = f"mp{stage}"
        cl = f"CL{stage}"

        instances[mn] = {"component": "nmos", "settings": mos_n}
        instances[mp] = {"component": "pmos", "settings": mos_p}
        instances[cl] = {"component": "cload", "settings": {"C": c_load}}

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

    return {
        "instances":   instances,
        "connections": connections,
        "ports":       {"out": "n1,p1"},
    }


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def psp103_models():
    return make_psp103_descriptors()


@pytest.fixture(scope="module")
def ring_compiled(psp103_models):
    from circulax import compile_netlist
    from circulax.components.electronic import (
        Capacitor,
        Resistor,
        SmoothPulse,
        VoltageSource,
    )
    from circulax.solvers import analyze_circuit

    psp103n, psp103p = psp103_models
    models = {
        "nmos":   psp103n,
        "pmos":   psp103p,
        "vsrc":   VoltageSource,
        "kick":   SmoothPulse,
        "r_kick": Resistor,
        "cload":  Capacitor,
    }
    # No external load cap — trap doesn't need the SDIRK3-stability padding.
    netlist = _build_ring_oscillator_netlist(c_load=0.0)
    groups, sys_size, port_map = compile_netlist(netlist, models)
    solver = analyze_circuit(groups, sys_size, backend="klu_split")
    return groups, sys_size, port_map, solver


# ── Tests ────────────────────────────────────────────────────────────────────


def test_psp103_descriptors_load(psp103_models):
    """Canonical-mode descriptors report valid PSP103 metadata."""
    from circulax.components.osdi import OsdiModelDescriptor

    psp103n, psp103p = psp103_models
    assert isinstance(psp103n, OsdiModelDescriptor)
    assert isinstance(psp103p, OsdiModelDescriptor)
    assert psp103n.model.num_pins == 4
    assert psp103n.model.num_params == 783
    assert psp103n.model.num_states == 0
    assert psp103n.default_params["TYPE"] == pytest.approx(1.0)
    assert psp103p.default_params["TYPE"] == pytest.approx(-1.0)


# ── Stage 6 — ring oscillator DC operating point ────────────────────────────


def test_ring_oscillator_dc(ring_compiled):
    """9-stage ring converges to the metastable DC operating point.

    Odd-stage rings have no stable DC equilibrium; the Newton solver finds a
    numerically stable (but physically metastable) fixed point where all
    ring nodes sit near VDD/2.  Plain Newton from zeros diverges because
    PSP103 has negative diagonal Jacobian entries (outward-current
    convention, ``G[D,D] = -gds``).  The two-phase homotopy below avoids it:

    Phase 1 — source stepping with large embedded Gmin (10 mS) that
       dominates any ``-gds``; Newton converges in the physical basin as
       VDD ramps from ~0.12 V to 1.2 V.
    Phase 2 — Gmin stepping from the source result back down to the solver
       baseline (1 nS), each step warm-started from the previous solution.
    """
    groups, sys_size, port_map, solver = ring_compiled

    VDD = 1.2
    G_HOMOTOPY = 1e-2  # 10 mS
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, G_HOMOTOPY)
    y_source = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y = solver.solve_dc_gmin(groups, y_source, g_start=G_HOMOTOPY, n_steps=30)
    assert jnp.all(jnp.isfinite(y)), "DC solver returned non-finite values"

    for stage in range(1, 10):
        key = f"n{stage},p1"
        if key in port_map:
            v = float(y[port_map[key]])
            assert 0.0 <= v <= VDD, f"ring node {key} = {v:.3f} V is outside [0, VDD]"


# ── Stage 7 — ring oscillator transient ─────────────────────────────────────


def _run_ring_transient(ring_compiled, t1=100e-9, n_save=500):
    """Run transient and return ``(t, v_n1)`` numpy arrays.

    Uses the trapezoidal integrator (zero numerical damping, A-stable) at
    a 5 ps fixed step.  Trap matches VACASK's tran_method="trap" exactly
    on the open-loop tests and tracks the limit-cycle frequency cleanly.
    """
    import diffrax

    from circulax.solvers import setup_transient
    from circulax.solvers.transient import TrapRefactoringTransientSolver

    groups, sys_size, port_map, solver = ring_compiled

    G_HOMOTOPY = 1e-2
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, G_HOMOTOPY)
    y_source = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=20)
    y0 = solver.solve_dc_gmin(groups, y_source, g_start=G_HOMOTOPY, n_steps=30)
    assert jnp.all(jnp.isfinite(y0)), "DC solve for transient IC produced non-finite values"

    # PSP103 is strongly nonlinear — use the refactor variant so Newton
    # gets a fresh Jacobian factorisation per iteration (full quadratic
    # convergence) instead of the cheaper frozen-Jac inner loop.
    run = setup_transient(
        groups, solver, transient_solver=TrapRefactoringTransientSolver,
    )
    sol = run(
        t0=0.0,
        t1=t1,
        dt0=5e-12,   # 5 ps fixed step — matches VACASK's runme.sim
        y0=y0,
        saveat=diffrax.SaveAt(ts=jnp.linspace(0.0, t1, n_save)),
        max_steps=2_000_000,
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    t = np.asarray(sol.ts)
    ys = np.asarray(sol.ys)
    assert np.all(np.isfinite(ys)), "transient contained non-finite values"

    n1_key = "n1,p1" if "n1,p1" in port_map else "n1"
    v_n1 = ys[:, port_map[n1_key]]
    return t, v_n1


def _dominant_frequency(t: np.ndarray, signal: np.ndarray) -> float:
    """Return the fundamental frequency (Hz) of a (near-)square-wave signal.

    Ring-oscillator outputs are square-wave-like, with strong odd harmonics
    from the sharp transitions.  An FFT-peak detector can easily pick the
    3rd or 5th harmonic instead of the fundamental.  We use rising
    mid-supply crossings of the (DC-removed) signal as a robust period
    estimator: count the time between consecutive rising crossings of zero
    and take the median.  This is invariant to harmonic content and works
    even for non-uniformly-sampled inputs (it only uses ``t``).

    Falls back to FFT peak if fewer than 3 rising crossings are found
    (e.g. for very short capture windows).
    """
    centered = signal - signal.mean()
    sign_changes = np.diff(np.sign(centered))
    rising = np.where(sign_changes > 0)[0]
    if len(rising) >= 3:
        # Skip the first crossing (initial transient) and use the rest.
        rising = rising[1:]
        # Linear-interpolate the actual zero-crossing time per index.
        cross_t = []
        for i in rising:
            x0, x1 = float(centered[i]), float(centered[i + 1])
            t0, t1 = float(t[i]), float(t[i + 1])
            cross_t.append(t0 - x0 * (t1 - t0) / (x1 - x0))
        cross_t = np.asarray(cross_t)
        if len(cross_t) >= 2:
            periods = np.diff(cross_t)
            return float(1.0 / np.median(periods))

    # Fallback: FFT peak.
    dt = float(t[1] - t[0])
    freqs = np.fft.rfftfreq(len(centered), d=dt)
    power = np.abs(np.fft.rfft(centered))
    power[0] = 0.0
    return float(freqs[int(np.argmax(power))])


def test_ring_oscillator_transient(ring_compiled):
    """Ring produces self-sustaining oscillation (> 100 mV swing on n1)."""
    t, v_n1 = _run_ring_transient(ring_compiled, t1=100e-9, n_save=500)
    swing = float(v_n1.max() - v_n1.min())
    print(
        f"\nRing transient: {len(t)} points / {t[-1] * 1e9:.0f} ns | "
        f"n1 swing = {swing * 1e3:.1f} mV "
        f"(min={v_n1.min():.3f} V, max={v_n1.max():.3f} V)"
    )
    assert swing > 0.1, f"expected oscillation, got only {swing * 1e3:.1f} mV swing"


@pytest.mark.skipif(
    not _VACASK_REF_FIXTURE.exists(),
    reason=(
        f"VACASK reference fixture missing: {_VACASK_REF_FIXTURE}. "
        "See this test module's docstring for the regen recipe."
    ),
)
def test_ring_oscillator_vs_vacask_reference(ring_compiled):
    """Compare oscillation frequency against a VACASK reference run.

    Tolerance: ±10 %.  Period is extracted from rising mid-supply
    zero-crossings (see :func:`_dominant_frequency`) rather than the FFT
    peak — ring outputs are square-wave-like, and an FFT peak detector
    can land on the 3rd harmonic, producing a spurious "3× faster"
    artefact (this exact bug temporarily mis-localised the ring-osc
    investigation; see ``docs/bosdi_psp103_ring_oscillator_issue.md``).
    """
    ref = np.load(_VACASK_REF_FIXTURE)
    t_ref = np.asarray(ref["t"])
    v_ref = np.asarray(ref["v1"])
    # Use the long, settled tail of the VACASK trace (skip startup).
    mask = t_ref > 100e-9
    f_ref = _dominant_frequency(t_ref[mask], v_ref[mask])

    t, v_n1 = _run_ring_transient(ring_compiled, t1=100e-9, n_save=2000)
    f_sim = _dominant_frequency(t, v_n1)

    print(f"\nVACASK freq   = {f_ref / 1e6:7.2f} MHz  ({1e9 / f_ref:5.2f} ns period)")
    print(f"circulax freq = {f_sim / 1e6:7.2f} MHz  ({1e9 / f_sim:5.2f} ns period)")
    ratio = f_sim / f_ref if f_ref > 0 else float("inf")
    assert 0.9 <= ratio <= 1.1, (
        f"ring oscillator frequency mismatch exceeds ±10 % tolerance: "
        f"circulax {f_sim / 1e6:.2f} MHz vs VACASK {f_ref / 1e6:.2f} MHz "
        f"(ratio {ratio:.3f})"
    )
