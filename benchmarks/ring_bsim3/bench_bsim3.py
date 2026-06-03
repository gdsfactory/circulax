"""BSIM3v3 ring oscillator — OSDI vs VA, various ring sizes.

Uses the VACASK c6288 0.35 µm process model card.
Ring: N-stage CMOS, VDD=1.8V, W_n=2µm W_p=4µm L=0.35µm.
Transient: 1 µs at 50 ps fixed step (20 000 steps).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_REPO = Path(__file__).resolve().parents[2]
_VA   = Path("/home/cdaunt/code/vacask/VACASK/devices/bsim3v3.va")
_OSDI = "/home/cdaunt/code/vacask/VACASK/build/devices/bsim3v3.osdi"

sys.path.insert(0, str(_REPO / "tests"))

T1      = 1e-6
DT      = 5e-11
N_STEPS = int(T1 / DT)
N_SAVE  = 4000

# VACASK c6288 0.35 µm model card (from models.inc)
_MC_N = dict(
    type=1.0,
    tnom=27.0, nch=2.498e17, tox=9e-9, xj=1e-7,
    lint=9.36e-8, wint=1.47e-7,
    vth0=0.6322,  k1=0.756,    k2=-3.83e-2,  k3=-2.612,
    dvt0=2.812,   dvt1=0.462,  dvt2=-9.17e-2,
    nlx=3.52291e-8, w0=1.163e-6, k3b=2.233,
    vsat=86301.58,  ua=6.47e-9,  ub=4.23e-18, uc=-4.706281e-11,
    rdsw=650.0, u0=388.3203, wr=1.0,
    a0=0.3496967, ags=0.1, b0=0.546, b1=1.0,
    dwg=-6e-9, dwb=-3.56e-9, prwb=-0.213,
    keta=-3.605872e-2, a1=2.778747e-2, a2=0.9,
    voff=-6.735529e-2, nfactor=1.139926, cit=1.622527e-4,
    cdsc=-2.147181e-5, eta0=1.0281729e-2, etab=-5.042203e-3,
    dsub=0.31871233, pclm=1.114846,
    pdiblc1=2.45357e-3, pdiblc2=6.406289e-3,
    drout=0.31871233, pscbe1=5e6, pscbe2=5e-9,
    pdiblcb=-0.234, pvag=0.0, delta=0.01,
    ww=-1.420242e-9, wwn=0.2613948, ll=1.300902e-10, lln=0.316394,
    kt1=-0.3, kt2=-0.051, at=22400.0, ute=-1.48,
    ua1=3.31e-10, ub1=2.61e-19, uc1=-3.42e-10, kt1l=0.0, prt=764.3,
    kf=3.2e-8, af=1.0, rsh=7.0,
    w=2e-6, l=0.35e-6, ad=1e-12, pd=2.5e-6,
    **{"as": 1e-12, "ps": 2.5e-6},
)
_MC_P = dict(
    type=-1.0,
    tnom=27.0, nch=3.533024e17, tox=9e-9, xj=1e-7,
    lint=6.23e-8, wint=1.22e-7,
    vth0=-0.6732829, k1=0.8362093, k2=-8.606622e-2, k3=1.82,
    dvt0=1.903801, dvt1=0.5333922, dvt2=-0.1862677,
    nlx=1.28e-8, w0=2.1e-6, k3b=-0.24,
    vsat=103503.2, ua=1.39995e-9, ub=1e-19, uc=-2.73e-11,
    rdsw=460.0, u0=138.7609,
    a0=0.4716551, ags=0.12,
    keta=-1.871516e-3, a1=0.3417965, a2=0.83,
    voff=-0.074182, nfactor=1.54389, cit=-1.015667e-3,
    cdsc=8.937517e-4, cdscb=1.45e-4, cdscd=1.04e-4,
    dvt0w=0.232, dvt1w=4.5e6, dvt2w=-0.0023,
    eta0=6.024776e-2, etab=-4.64593e-3,
    dsub=0.23222404, pclm=0.989,
    pdiblc1=2.07418e-2, pdiblc2=1.33813e-3,
    drout=0.3222404, pscbe1=118000.0, pscbe2=1e-9, pvag=0.0,
    kt1=-0.25, kt2=-0.032, prt=64.5, at=33000.0, ute=-1.5,
    ua1=4.312e-9, ub1=6.65e-19, uc1=0.0, kt1l=0.0,
    kf=3.2e-8, af=1.0, rsh=7.0,
    w=4e-6, l=0.35e-6, ad=2e-12, pd=4.5e-6,
    **{"as": 2e-12, "ps": 4.5e-6},
)


def _dom_freq(t: np.ndarray, v: np.ndarray) -> float:
    centered = v - v.mean()
    rising = np.where(np.diff(np.sign(centered)) > 0)[0]
    if len(rising) < 3:
        return float("nan")
    rising = rising[1:]
    times = []
    for i in rising:
        x0, x1 = float(centered[i]), float(centered[i + 1])
        t0, t1 = float(t[i]), float(t[i + 1])
        times.append(t0 - x0 * (t1 - t0) / (x1 - x0))
    if len(times) < 2:
        return float("nan")
    return float(1.0 / np.median(np.diff(np.asarray(times))))


def _make_va_classes():
    """Compile BSIM3v3 VA → Python component classes (cached after first call)."""
    import dataclasses
    import importlib.util
    import tempfile

    from circulax.va.emitter import emit_source
    from circulax.va.va_defaults import parse_va_defaults_expanded

    from circulax.va import compile_va, lower

    defs = parse_va_defaults_expanded(_VA)
    dump = compile_va(str(_VA))

    int_static = {n: int(s.default) for n, s in defs.items() if s.type_ == "int"}

    dev_n = lower(dump.modules[0], va_defaults=defs, collapse_nodes=True,
                  static_params={**int_static, "type": 1,
                                 "nqsmod": 0, "acnqsmod": 0, "elm": 3, "capmod": 3},
                  differentiable_params=tuple(k for k in _MC_N
                                              if isinstance(_MC_N[k], float)),
                  class_name="BSIM3N")
    dev_p = lower(dump.modules[0], va_defaults=defs, collapse_nodes=True,
                  static_params={**int_static, "type": -1,
                                 "nqsmod": 0, "acnqsmod": 0, "elm": 3, "capmod": 3},
                  differentiable_params=tuple(k for k in _MC_P
                                              if isinstance(_MC_P[k], float)),
                  class_name="BSIM3P")

    tmp = tempfile.mkdtemp()
    out = Path(tmp) / "bsim3_va.py"
    out.write_text(emit_source([dev_n, dev_p]))
    spec = importlib.util.spec_from_file_location("bsim3_va", out)
    mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    def _params(cls, mc, defs_map):
        fn = {f.name for f in dataclasses.fields(cls) if f._field_type.name == "_FIELD"}
        p = {k: float(defs_map[k].default)
             for k in fn if k in defs_map and defs_map[k].type_ == "float"}
        p.update({k: float(v) for k, v in mc.items() if k in fn})
        p["_simparam_gmin"] = 0.0
        return p

    return mod.BSIM3N, _params(mod.BSIM3N, _MC_N, defs), \
           mod.BSIM3P, _params(mod.BSIM3P, _MC_P, defs)


_VA_CLASSES = None   # cached across run() calls


def _build_ring(n_stages: int, variant: str):
    from circulax import compile_netlist
    from circulax.components.electronic import Resistor, SmoothPulse, VoltageSource

    instances = {
        "Vvdd":  {"component": "vsrc",  "settings": {"V": 1.8}},
        "Vkick": {"component": "kick",  "settings": {"V": 1.0, "delay": 1e-9, "tr": 1e-9}},
        "Rkick": {"component": "r_kick","settings": {"R": 1e5}},
    }
    connections = {
        "Vvdd,p1": "vdd,p1", "Vvdd,p2": "GND,p1",
        "Vkick,p1": "kick_n,p1", "Vkick,p2": "GND,p1",
        "Rkick,p1": "kick_n,p1", "Rkick,p2": "n1,p1",
    }
    # OSDI ports are uppercase (D/G/S/B); VA-lowered ports are lowercase (d/g/s/b).
    D, G, S, B = ("D", "G", "S", "B") if variant == "osdi" else ("d", "g", "s", "b")
    for s in range(1, n_stages + 1):
        in_n, out_n = f"n{s}", f"n{s % n_stages + 1}"
        mn, mp = f"mn{s}", f"mp{s}"
        instances[mn] = {"component": "nmos", "settings": dict(_MC_N)}
        instances[mp] = {"component": "pmos", "settings": dict(_MC_P)}
        connections.update({
            f"{mn},{D}": f"{out_n},p1", f"{mn},{G}": f"{in_n},p1",
            f"{mn},{S}": "GND,p1",      f"{mn},{B}": "GND,p1",
            f"{mp},{D}": f"{out_n},p1", f"{mp},{G}": f"{in_n},p1",
            f"{mp},{S}": "vdd,p1",      f"{mp},{B}": "vdd,p1",
        })

    if variant == "osdi":
        # BSIM3v3 has 5 NQS charge-partition states.  We bypass circulax's
        # stateful-OSDI guard and initialise them to zero — acceptable for a
        # DC + transient benchmark where the capmod states converge quickly.
        from circulax.components.osdi import OsdiModelDescriptor
        from osdi_loader import load_osdi_model
        _model = load_osdi_model(_OSDI)
        nmos = OsdiModelDescriptor(model=_model, ports=("D","G","S","B"),
                                   param_names=None,
                                   default_params=dict(_MC_N),
                                   use_schur_reduction=False)
        pmos = OsdiModelDescriptor(model=_model, ports=("D","G","S","B"),
                                   param_names=None,
                                   default_params=dict(_MC_P),
                                   use_schur_reduction=False)
        models = {"nmos": nmos, "pmos": pmos,
                  "vsrc": VoltageSource, "kick": SmoothPulse, "r_kick": Resistor}

    elif variant == "va":
        global _VA_CLASSES
        if _VA_CLASSES is None:
            _VA_CLASSES = _make_va_classes()
        ClsN, params_n, ClsP, params_p = _VA_CLASSES

        # Update per-stage instance settings with full params
        for s in range(1, n_stages + 1):
            instances[f"mn{s}"]["settings"] = params_n
            instances[f"mp{s}"]["settings"] = params_p

        models = {"nmos": ClsN, "pmos": ClsP,
                  "vsrc": VoltageSource, "kick": SmoothPulse, "r_kick": Resistor}
    else:
        raise ValueError(variant)

    return compile_netlist(
        {"instances": instances, "connections": connections, "ports": {"out": "n1,p1"}},
        models,
    )


def run(n_stages: int = 3, variant: str = "osdi") -> dict:
    from circulax.solvers import analyze_circuit, setup_transient
    from circulax.solvers.transient import TrapFactorizedTransientSolver

    groups, sys_size, port_map = _build_ring(n_stages, variant)
    solver = analyze_circuit(groups, sys_size, backend="klu_split")

    t_dc    = time.perf_counter()
    hi_gmin = eqx.tree_at(lambda s: s.g_leak, solver, 1e-2)
    # Seed ring nodes at VDD/2 (switching threshold) for faster convergence to
    # the unstable equilibrium.  Zero-vector start can diverge for N≥5 rings.
    y_init  = jnp.zeros(sys_size).at[port_map["vdd,p1"]].set(1.8)
    for k in (f"n{s},p1" for s in range(1, n_stages + 1)):
        if k in port_map:
            y_init = y_init.at[port_map[k]].set(0.9)
    y_src   = hi_gmin.solve_dc_source(groups, y_init, n_steps=20)
    y0      = solver.solve_dc_gmin(groups, y_src, g_start=1e-2, n_steps=30)
    dc_s    = time.perf_counter() - t_dc

    if not bool(jnp.all(jnp.isfinite(y0))):
        return {"status": "dc_diverged", "n_stages": n_stages, "variant": variant}

    ring_nodes = [f"n{s},p1" for s in range(1, n_stages + 1)]
    dc_vols    = {k: round(float(y0[port_map[k]]), 3)
                  for k in ring_nodes if k in port_map}

    run_fn    = setup_transient(groups, solver,
                                transient_solver=TrapFactorizedTransientSolver)
    saveat    = diffrax.SaveAt(ts=jnp.linspace(0.0, T1, N_SAVE))
    ctrl      = diffrax.ConstantStepSize()
    max_steps = int(2 * N_STEPS)

    saveat_w  = diffrax.SaveAt(ts=jnp.linspace(0.0, 2 * DT, N_SAVE))
    t_compile = time.perf_counter()
    _ = run_fn(t0=0.0, t1=2 * DT, dt0=DT, y0=y0,
               saveat=saveat_w, max_steps=max_steps,
               stepsize_controller=ctrl).ys.block_until_ready()
    compile_s = time.perf_counter() - t_compile

    t0  = time.perf_counter()
    sol = run_fn(t0=0.0, t1=T1, dt0=DT, y0=y0,
                 saveat=saveat, max_steps=max_steps,
                 stepsize_controller=ctrl)
    sol.ys.block_until_ready()
    wall = time.perf_counter() - t0

    ys = np.asarray(sol.ys)
    ts = np.asarray(sol.ts)

    if not np.all(np.isfinite(ys)):
        return {"status": "tran_nan", "n_stages": n_stages, "variant": variant,
                "sys_size": sys_size, "dc_vols": dc_vols,
                "compile_s": compile_s, "dc_s": dc_s, "wall_s": wall}

    v    = ys[:, port_map["n1,p1"]]
    mask = ts > 100e-9
    freq = _dom_freq(ts[mask], v[mask]) if mask.any() else float("nan")

    return {
        "status":      "ok",
        "n_stages":    n_stages,
        "variant":     variant,
        "sys_size":    sys_size,
        "dc_vols":     dc_vols,
        "compile_s":   compile_s,
        "dc_s":        dc_s,
        "wall_s":      wall,
        "us_per_step": wall / N_STEPS * 1e6,
        "freq_MHz":    freq / 1e6 if np.isfinite(freq) else None,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-stages", type=int, default=3)
    p.add_argument("--variant",  choices=("osdi", "va"), default="osdi")
    args = p.parse_args()
    r = run(args.n_stages, args.variant)
    for k, v in r.items():
        print(f"  {k}: {v}")
