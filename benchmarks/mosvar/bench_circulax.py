"""IHP mosvar DC sweep — circulax OSDI vs VA paths.

Tier 1 of the IHP-PDK accuracy ladder. mosvar is the IHP MOS varactor
(3 ports g/bi/b, 4 internal nodes, gate-tunneling-dominated DC). Test:
sweep V_g with bulk(s) tied to ground, compare gate current.

Variants:
  - ``osdi`` — IHP mosvar.va compiled to .osdi via openvaf-r,
    via ``circulax.components.osdi.osdi_component``.
  - ``va``   — IHP mosvar.va lowered through ``compile_va_unopt`` +
    ``circulax.va.lower()`` into a JAX-traceable ``@va_component``.
    Uses unopt-MIR for nested-conditional fidelity (per the
    Tier-0 fix).

Testbench:
  Vsrc(V_g) → g, GND → bi, GND → b. DC sweep V_g over [-2, 2] V step 0.5.

Usage:
    pixi run python benchmarks/mosvar/bench_circulax.py --variant osdi
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import os
import subprocess
import tempfile
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_REPO = Path(__file__).resolve().parents[2]
_IHP_MOSVAR = Path(
    "/home/cdaunt/code/gdsfactory/pdks/IHP-Open-PDK/ihp-sg13g2/libs.tech/verilog-a/mosvar"
)
_VA_SOURCE = _IHP_MOSVAR / "mosvar.va"
_OSDI_DIR = _REPO / "circulax" / "components" / "osdi" / "compiled"
_OSDI_PATH = _OSDI_DIR / "mosvar_ihp.osdi"

# DC bias points (V_g, with V_bi = V_b = 0).
_BIAS = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)


def _ensure_osdi_compiled() -> Path:
    if _OSDI_PATH.exists():
        return _OSDI_PATH
    _OSDI_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["openvaf-r", str(_VA_SOURCE), "-o", str(_OSDI_PATH)],
        check=True, cwd=_IHP_MOSVAR, capture_output=True, text=True,
    )
    return _OSDI_PATH


def _default_params() -> dict[str, float]:
    from circulax.va.va_defaults import parse_va_defaults_expanded
    defs = parse_va_defaults_expanded(_VA_SOURCE)
    return {k: float(v.default) for k, v in defs.items() if v.type_ == "float"}


def _build_osdi_descriptor():
    from circulax.components.osdi import osdi_component
    return osdi_component(
        osdi_path=str(_ensure_osdi_compiled()),
        ports=("g", "bi", "b"),
        default_params=_default_params(),
    )


def _build_va_descriptor():
    """Lower IHP mosvar.va via the unopt-MIR path."""
    from bosdi.va.ir_client import compile_va_unopt
    from circulax.va.emitter import emit_source
    from circulax.va.va_defaults import parse_va_defaults_expanded

    from circulax.va import lower

    cwd = os.getcwd()
    try:
        os.chdir(_IHP_MOSVAR)
        dump = compile_va_unopt(str(_VA_SOURCE.name))
    finally:
        os.chdir(cwd)
    defs = parse_va_defaults_expanded(_VA_SOURCE)
    int_static = {k: int(v.default) for k, v in defs.items() if v.type_ == "int"}
    dev = lower(
        dump.modules[0],
        va_defaults=defs,
        collapse_nodes=True,
        static_params=int_static,
        class_name="MOSVAR",
    )
    tmpd = Path(tempfile.mkdtemp(prefix="mosvar_va_"))
    src = tmpd / "mosvar.py"
    src.write_text(emit_source([dev]))
    spec = importlib.util.spec_from_file_location("mosvar_va_mod", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MOSVAR, defs


def _va_instance(cls, defs):
    fields = {f.name for f in dataclasses.fields(cls) if f._field_type.name == "_FIELD"}
    p = {k: float(defs[k].default) for k in fields if k in defs and defs[k].type_ == "float"}
    if "_simparam_gmin" in fields:
        p["_simparam_gmin"] = 0.0
    return cls(**p)


def _dc_one_bias_osdi(v_g: float, mv) -> float:
    """DC operating point for a single bias V_g via OSDI. Returns I_g [A]."""
    from circulax import compile_netlist
    from circulax.components.electronic import VoltageSource
    from circulax.solvers import analyze_circuit

    netlist = {
        "instances": {
            "GND": {"component": "ground"},
            "VS":  {"component": "vsrc", "settings": {"V": v_g}},
            "M":   {"component": "mosvar", "settings": {}},
        },
        "connections": {
            "VS,p1": "g,p1",  "VS,p2": "GND,p1",
            "M,g":   "g,p1",
            "M,bi":  "GND,p1",
            "M,b":   "GND,p1",
        },
    }
    models = {"vsrc": VoltageSource, "mosvar": mv}
    groups, sys_size, port_map = compile_netlist(netlist, models)
    solver = analyze_circuit(groups, sys_size)
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, 1e-3)
    y0 = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=8)
    y = solver.solve_dc_gmin(groups, y0, g_start=1e-3, n_steps=10)
    return float(y[port_map["VS,i_src"]])


def _dc_sweep_va_static(inst) -> dict[float, float]:
    """For 3-port mosvar with internal nodes, we need a netlist DC solve too —
    static eval doesn't give the operating point of internal nodes.
    """
    raise NotImplementedError("VA path uses _dc_one_bias_va via netlist DC solve")


def _dc_one_bias_va(v_g: float, va_cls) -> float:
    """DC operating point via the VA-lowered class through compile_netlist."""
    from circulax import compile_netlist
    from circulax.components.electronic import VoltageSource
    from circulax.solvers import analyze_circuit

    netlist = {
        "instances": {
            "GND": {"component": "ground"},
            "VS":  {"component": "vsrc", "settings": {"V": v_g}},
            "M":   {"component": "mosvar", "settings": {}},
        },
        "connections": {
            "VS,p1": "g,p1",  "VS,p2": "GND,p1",
            "M,g":   "g,p1",
            "M,bi":  "GND,p1",
            "M,b":   "GND,p1",
        },
    }
    models = {"vsrc": VoltageSource, "mosvar": va_cls}
    groups, sys_size, port_map = compile_netlist(netlist, models)
    solver = analyze_circuit(groups, sys_size)
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, 1e-3)
    y0 = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=8)
    y = solver.solve_dc_gmin(groups, y0, g_start=1e-3, n_steps=10)
    return float(y[port_map["VS,i_src"]])


def run(variant: str) -> dict:
    print(f"=== mosvar DC sweep — variant={variant} ===")
    if variant == "osdi":
        mv = _build_osdi_descriptor()
        print(f"OSDI: {len(mv.param_names)} params, ports={mv.ports}")
        results = {}
        for v in _BIAS:
            try:
                i_g = _dc_one_bias_osdi(v, mv)
                results[v] = i_g
            except Exception as exc:
                results[v] = f"ERROR: {type(exc).__name__}: {exc}"
        return {"variant": variant, "iv": results}
    if variant == "va":
        cls, _defs = _build_va_descriptor()
        print(f"VA-unopt class built: {cls.__name__}")
        results = {}
        for v in _BIAS:
            try:
                i_g = _dc_one_bias_va(v, cls)
                results[v] = i_g
            except Exception as exc:
                results[v] = f"ERROR: {type(exc).__name__}: {exc}"
        return {"variant": variant, "iv": results}
    raise SystemExit(f"unknown variant: {variant}")


def _print_table(res: dict) -> None:
    print()
    print(f"{'V_g [V]':>10} | {'I_g [A]':>16}")
    print("-" * 32)
    for v, i in res["iv"].items():
        if isinstance(i, float):
            print(f"{v:>10.3f} | {i:>16.6e}")
        else:
            print(f"{v:>10.3f} | {i}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", choices=("osdi", "va"), default="osdi")
    args = ap.parse_args()
    _print_table(run(args.variant))
