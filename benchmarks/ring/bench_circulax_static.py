"""PSP103 N=9 ring oscillator benchmark — all-static variant only.

Runs the full DC + transient pipeline using the all-static lowering
(every numeric .va default + VACASK card + geometry baked into the
lowered class) to measure per-step performance compared to the
TYPE-only baseline.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import sys
import tempfile
import time
from pathlib import Path

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests"))

from circulax.va.emitter import emit_source  # noqa: E402
from circulax.va.va_defaults import parse_va_defaults_expanded  # noqa: E402
from fixtures.psp103_models import (  # noqa: E402
    PSP103N_DEFAULTS,
    PSP103P_DEFAULTS,
    geom_settings,
)

from circulax import compile_netlist  # noqa: E402
from circulax.components.electronic import (  # noqa: E402
    Capacitor,
    Resistor,
    SmoothPulse,
    VoltageSource,
)
from circulax.solvers import analyze_circuit, setup_transient  # noqa: E402
from circulax.solvers.transient import TrapFactorizedTransientSolver  # noqa: E402
from circulax.va import compile_va, lower  # noqa: E402

VA = REPO / "tests" / "data" / "va" / "psp103v4" / "psp103.va"


def _all_static(card, geom):
    defs = parse_va_defaults_expanded(VA)
    out = {}
    for n, spec in defs.items():
        v = spec.default
        if isinstance(v, str):
            try: v = float(v)
            except: continue
        if isinstance(v,(int,float)) and not isinstance(v,bool): out[n]=float(v)
    for k,v in card.items():
        if isinstance(v,(int,float)) and not isinstance(v,bool): out[k]=float(v)
    for k,v in geom.items(): out[k]=float(v)
    return out


def main():
    import os
    mode = os.environ.get("MODE", "all_static")
    nmos_geom = geom_settings(10e-6, 1e-6)
    pmos_geom = geom_settings(20e-6, 1e-6)
    if mode == "type_only":
        sn = {"TYPE": 1}; sp = {"TYPE": -1}
    else:
        sn = _all_static(PSP103N_DEFAULTS, nmos_geom); sn["TYPE"] = 1
        sp = _all_static(PSP103P_DEFAULTS, pmos_geom); sp["TYPE"] = -1
    print(f"mode: {mode}", flush=True)

    t0 = time.perf_counter()
    mod = compile_va(str(VA)).modules[0]
    defs = parse_va_defaults_expanded(VA)
    dev_n = lower(mod, va_defaults=defs, collapse_nodes=True,
                  static_params=sn, class_name="PSP103N")
    dev_p = lower(mod, va_defaults=defs, collapse_nodes=True,
                  static_params=sp, class_name="PSP103P")
    src = emit_source([dev_n, dev_p])
    out = Path(tempfile.mkdtemp()) / "psp_static.py"
    out.write_text(src)
    spec = importlib.util.spec_from_file_location("psp_static", out)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    print(f"emit/lower: {time.perf_counter()-t0:.1f}s, {len(src.splitlines())} lines", flush=True)

    def _fields(cls):
        return {f.name: f for f in dataclasses.fields(cls)
                if f._field_type.name == "_FIELD"}
    def _coerce(by_name, name, value):
        return int(float(value)) if by_name[name].type == "int" else float(value)
    fn = _fields(m.PSP103N); fp = _fields(m.PSP103P)
    nmos_p = {k: _coerce(fn, k, v) for k, v in PSP103N_DEFAULTS.items() if k in fn}
    nmos_p.update({k: _coerce(fn, k, v) for k, v in nmos_geom.items() if k in fn})
    pmos_p = {k: _coerce(fp, k, v) for k, v in PSP103P_DEFAULTS.items() if k in fp}
    pmos_p.update({k: _coerce(fp, k, v) for k, v in pmos_geom.items() if k in fp})

    from test_psp103_ring_oscillator import _build_ring_oscillator_netlist
    netlist = _build_ring_oscillator_netlist(c_load=0.0)
    for stage in range(1, 10):
        netlist["instances"][f"mn{stage}"]["settings"] = nmos_p
        netlist["instances"][f"mp{stage}"]["settings"] = pmos_p
    models = {"nmos": m.PSP103N, "pmos": m.PSP103P, "vsrc": VoltageSource,
              "kick": SmoothPulse, "r_kick": Resistor, "cload": Capacitor}
    groups, sys_size, port_map = compile_netlist(netlist, models)
    print(f"sys_size: {sys_size}", flush=True)

    solver = analyze_circuit(groups, sys_size, backend="klu_split")

    print("DC homotopy with VDD/2 warm-start...", flush=True)
    from test_psp103_ring_oscillator import _va_ring_warm_start
    y_warm = _va_ring_warm_start(sys_size, port_map, vdd=1.2)
    t0 = time.perf_counter()
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, 1e-2)
    # Skip source-stepping (the warm-start already places y near the
    # physical operating point).  Just gmin-step from the warm-start.
    y0 = solver.solve_dc_gmin(groups, y_warm, g_start=1e-2, n_steps=30)
    dc_s = time.perf_counter() - t0
    finite = bool(jnp.all(jnp.isfinite(y0)))
    print(f"  DC: {dc_s:.1f}s  finite={finite}  max|y|={float(jnp.max(jnp.abs(y0))):.3e}", flush=True)
    if not finite:
        return

    # Transient at the same T1 / DT the official benchmark uses, so per-
    # step numbers compare directly.
    T1 = 1e-6  # 1 µs
    DT = 5e-11
    N_STEPS = int(T1 / DT)
    print(f"Transient setup (T1={T1*1e6}us, dt={DT*1e12}ps, n_steps={N_STEPS})...", flush=True)

    run_fn = setup_transient(groups, solver,
                             transient_solver=TrapFactorizedTransientSolver)
    saveat = diffrax.SaveAt(ts=jnp.linspace(0.0, T1, 2000))
    saveat_warm = diffrax.SaveAt(ts=jnp.linspace(0.0, 2*DT, 2000))
    controller = diffrax.ConstantStepSize()
    max_steps = int(2 * N_STEPS)

    print("warmup (first call compiles)...", flush=True)
    t_w = time.perf_counter()
    _ = run_fn(t0=0.0, t1=2*DT, dt0=DT, y0=y0, saveat=saveat_warm,
               max_steps=max_steps,
               stepsize_controller=controller).ys.block_until_ready()
    warm_s = time.perf_counter() - t_w
    print(f"  warmup: {warm_s:.1f}s", flush=True)

    print("transient run...", flush=True)
    t_r = time.perf_counter()
    sol = run_fn(t0=0.0, t1=T1, dt0=DT, y0=y0, saveat=saveat,
                 max_steps=max_steps, stepsize_controller=controller)
    sol.ys.block_until_ready()
    wall = time.perf_counter() - t_r

    finite = bool(np.all(np.isfinite(np.asarray(sol.ys))))
    us_per_step = wall / N_STEPS * 1e6
    print(f"  trans: {wall:.2f}s  ({us_per_step:.1f} us/step)  finite={finite}", flush=True)

    # Detect oscillation frequency at n1 (first ring node)
    ts = np.asarray(sol.ts)
    ys = np.asarray(sol.ys)
    if "n1,p1" in port_map:
        v1 = ys[:, port_map["n1,p1"]]
        mask = ts > 100e-9
        if mask.any():
            v = v1[mask]; t = ts[mask]
            centered = v - v.mean()
            rising = np.where(np.diff(np.sign(centered)) > 0)[0]
            if len(rising) >= 3:
                rising = rising[1:]
                xings = []
                for i in rising:
                    x0, x1 = float(centered[i]), float(centered[i+1])
                    t0, t1 = float(t[i]), float(t[i+1])
                    xings.append(t0 - x0 * (t1 - t0) / (x1 - x0))
                if len(xings) >= 2:
                    period = float(np.median(np.diff(np.asarray(xings))))
                    print(f"  freq: {1.0/period/1e6:.1f} MHz  (n1 swing: min={v.min():.3f}V max={v.max():.3f}V)", flush=True)
                else:
                    print(f"  no oscillation detected (n1 swing: min={v.min():.3f}V max={v.max():.3f}V)", flush=True)
            else:
                print(f"  no oscillation: <3 zero crossings (n1 swing: min={v.min():.3f}V max={v.max():.3f}V)", flush=True)


if __name__ == "__main__":
    main()
