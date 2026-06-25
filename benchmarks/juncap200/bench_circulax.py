"""Single-junction DC sweep of IHP juncap200 — circulax OSDI vs VA paths.

Tier 0 of the IHP-PDK accuracy ladder. JUNCAP200 is a 2-terminal junction
diode used as PSP103's source/drain junction; standalone, no surface-potential
ψs solve. If the VA-lowering path can't match OSDI on this model, the bug
manifests in a much smaller cascade than full PSP103.

Variants:
  - ``osdi`` — IHP juncap200.va compiled to .osdi via openvaf-r, loaded via
    ``circulax.components.osdi.osdi_component`` (canonical-name mode).
  - ``va``   — IHP juncap200.va lowered through ``circulax.va.lower()``
    into a JAX-traceable ``@component``. Currently broken — returns I=0.0
    at every bias point with default IHP params (smoke-tested 2026-05-03);
    use this bench to localize the lowering bug before promoting to Tier 1.

Testbench:
  Vsrc(V_AK) → A(juncap)K → GND
  DC sweep V_AK over [-1.0, -0.5, 0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9] V

Usage:
    pixi run python benchmarks/juncap200/bench_circulax.py --variant osdi
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

# Resolve repo + VA source paths.
_REPO = Path(__file__).resolve().parents[2]
_BENCH_DIR = Path(__file__).resolve().parent
_VA_SOURCE = _BENCH_DIR / "va_source" / "juncap200.va"

# OSDI binary lives next to other compiled models in circulax/components/osdi/compiled/.
# Compiled lazily on first run if missing — needs openvaf-r in PATH.
_OSDI_DIR = _REPO / "circulax" / "components" / "osdi" / "compiled"
_OSDI_PATH = _OSDI_DIR / "juncap200_ihp.osdi"

# DC bias points (Volts).
_BIAS = (-1.0, -0.5, 0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9)


def _ensure_osdi_compiled() -> Path:
    """Compile juncap200_ihp.osdi from IHP source if missing."""
    if _OSDI_PATH.exists():
        return _OSDI_PATH
    _OSDI_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["openvaf-r", str(_VA_SOURCE), "-o", str(_OSDI_PATH)],
        check=True,
        cwd=str(_BENCH_DIR / "va_source"),
        capture_output=True,
        text=True,
    )
    return _OSDI_PATH


def _default_params() -> dict[str, float]:
    """Float-typed default param dict from IHP juncap200.va `parameter` decls."""
    from circulax.va.va_defaults import parse_va_defaults_expanded

    defs = parse_va_defaults_expanded(_VA_SOURCE)
    return {k: float(v.default) for k, v in defs.items() if v.type_ == "float"}


def _build_osdi_descriptor():
    """Return a canonical-mode OsdiModelDescriptor for juncap200."""
    from circulax.components.osdi import osdi_component

    return osdi_component(
        osdi_path=str(_ensure_osdi_compiled()),
        ports=("A", "K"),
        default_params=_default_params(),
    )


def _build_va_descriptor():
    """Lower IHP juncap200.va to a JAX-traceable ``@component`` class.

    Uses the **unopt-MIR** ingestion path (``compile_va_unopt`` on
    ``openvaf-r --dump-unopt-mir``) instead of the default JSON path.
    The JSON path's MIR-optimization pass collapses 3-level nested
    if-blocks into single 4-edge phis, which the lowering's diamond
    detector mishandles for the conditional-init pattern that
    juncap200's JuncapExpressInit3 uses for ISATFOR2/MFOR2/ISATREV.
    The unopt path preserves the chain of 2-edge phis and emits a
    correct chain of ``jnp.where`` calls. Trade-off: per-instance
    hoisting is sacrificed (cache recomputed each Newton call); see
    ``bosdi.va.ir_client.compile_va_unopt`` docstring.
    """
    from bosdi.va.ir_client import compile_va_unopt
    from circulax.va.emitter import emit_source
    from circulax.va.va_defaults import parse_va_defaults_expanded

    from circulax.va import lower

    cwd = os.getcwd()
    try:
        os.chdir(_BENCH_DIR / "va_source")  # `include directives are resolved from cwd
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
        class_name="JUNCAP200",
    )
    tmpd = Path(tempfile.mkdtemp(prefix="juncap200_va_"))
    src = tmpd / "juncap200.py"
    src.write_text(emit_source([dev]))
    spec = importlib.util.spec_from_file_location("juncap200_va_mod", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.JUNCAP200, defs


def _va_instance(cls, defs):
    """Instantiate the VA-lowered class with IHP defaults.

    The unopt-MIR ingestion path may emit a different param set than the
    JSON path (e.g. it doesn't inject ``_simparam_gmin`` because the
    partially-optimized function uses different sim-param plumbing).
    Pass kwargs only for fields the class actually declares.
    """
    fields = {f.name for f in dataclasses.fields(cls) if f._field_type.name == "_FIELD"}
    p = {k: float(defs[k].default) for k in fields if k in defs and defs[k].type_ == "float"}
    if "_simparam_gmin" in fields:
        p["_simparam_gmin"] = 0.0
    return cls(**p)


def _dc_one_bias_osdi(v_ak: float, junc) -> float:
    """DC operating point for a single bias V_AK using the OSDI path. Returns I_A [A]."""
    from circulax import compile_netlist
    from circulax.components.electronic import VoltageSource
    from circulax.solvers import analyze_circuit

    netlist = {
        "instances": {
            "GND": {"component": "ground"},
            "VS":  {"component": "vsrc", "settings": {"V": v_ak}},
            "J":   {"component": "juncap", "settings": {}},
        },
        "connections": {
            "VS,p1": "a,p1",  "VS,p2": "GND,p1",
            "J,A":   "a,p1",
            "J,K":   "GND,p1",
        },
    }
    models = {"vsrc": VoltageSource, "juncap": junc, "ground": None}
    # `ground` is a reserved instance name resolved by build_net_map; no model needed.
    models = {k: v for k, v in models.items() if v is not None}
    groups, sys_size, port_map = compile_netlist(netlist, models)
    solver = analyze_circuit(groups, sys_size)
    # Embedded Gmin to stabilise Newton on hard reverse-bias points.
    high_gmin = eqx.tree_at(lambda s: s.g_leak, solver, 1e-3)
    y0 = high_gmin.solve_dc_source(groups, jnp.zeros(sys_size), n_steps=8)
    y = solver.solve_dc_gmin(groups, y0, g_start=1e-3, n_steps=10)
    # The Vsrc current sinker carries -I_anode (current flowing out of the V+ node).
    i_src_key = "VS,i_src"
    idx = port_map[i_src_key]
    return float(y[idx])


def _dc_sweep_va_static(inst) -> dict[float, float]:
    """Stand-alone VA-only single-device eval (no netlist). Static — no
    DC homotopy — just a residual eval at the imposed V_AK / 0V bias.

    For a 2-terminal device with no internal nodes, this is exact: the
    voltage state is fully determined by the external sources, so the
    residual = -I_anode (KCL at node A).
    """
    out: dict[float, float] = {}
    for v_ak in _BIAS:
        y = jnp.array([v_ak, 0.0])
        f, _q = inst.solver_call(0.0, y, inst)
        out[v_ak] = float(f[0])
    return out


def run(variant: str) -> dict:
    print(f"=== juncap200 DC sweep — variant={variant} ===")
    if variant == "osdi":
        junc = _build_osdi_descriptor()
        print(f"OSDI: {len(junc.param_names)} params, ports={junc.ports}")
        results = {}
        for v in _BIAS:
            try:
                i_a = _dc_one_bias_osdi(v, junc)
                results[v] = i_a
            except Exception as exc:
                results[v] = f"ERROR: {type(exc).__name__}: {exc}"
        return {"variant": variant, "iv": results}
    if variant == "va":
        cls, defs = _build_va_descriptor()
        inst = _va_instance(cls, defs)
        results = _dc_sweep_va_static(inst)
        return {"variant": variant, "iv": results}
    raise SystemExit(f"unknown variant: {variant}")


def _print_table(res: dict) -> None:
    print()
    print(f"{'V_AK [V]':>10} | {'I_A [A]':>16}")
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
