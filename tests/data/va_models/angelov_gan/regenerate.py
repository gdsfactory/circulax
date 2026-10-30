"""Regenerate native Angelov GaN with bosdi source 159907ddf29a0225614cf1b7999214401b0b5a2d.

Requires bosdi VA lowering and openvaf-r with --dump-unopt-mir.
The emitted component uses ordinary JAX with no bosdi/OSDI runtime.
"""

# ruff: noqa: CPY001, C901, PLR0912, PLR0915, ISC003

import os
import re
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

from bosdi.va import lower
from bosdi.va.emitter import _live_init_cache_slots, emit_source
from bosdi.va.ir_client import compile_va_unopt
from bosdi.va.mir import Constant, CurrentKind, ParamGivenRef, ParamRef
from bosdi.va.va_defaults import ParamSpec, parse_va_defaults_expanded

SOURCE = Path(__file__).with_name("angelov_gan.va").resolve()
OUTPUT = SOURCE.parents[4] / "circulax/components/angelov_gan.py"


def main() -> None:
    defaults = parse_va_defaults_expanded(SOURCE)
    previous = Path.cwd()
    try:
        with tempfile.TemporaryDirectory() as scratch:
            os.chdir(scratch)
            for original in SOURCE.parent.iterdir():
                if original.suffix in {".va", ".h", ".vams"}:
                    shutil.copy2(original, Path(scratch) / original.name)
            dump = compile_va_unopt(str(Path(scratch) / SOURCE.name))
    finally:
        os.chdir(previous)
    module = dump.modules[0]
    node_names = dict(zip(module.port_nodes, module.ports, strict=True))
    for interner in (module.init_interner, module.eval_interner):
        for kind in interner.parameters.values():
            if hasattr(kind, "hi_node") and kind.hi_node:
                node_names[kind.hi_node] = kind.hi
            if hasattr(kind, "lo_node") and kind.lo_node:
                node_names[kind.lo_node] = kind.lo
    branches = {}
    for sim, node in list(module.dae.unknowns.items()):
        match = re.fullmatch(r"br\[Unnamed \{ hi: (node\d+), lo: Some\((node\d+)\) \}\]", node)
        if match:
            bid = int(sim.removeprefix("sim_node"))
            branches[(node_names[match[1]], node_names[match[2]])] = bid
            module.dae.unknowns[sim] = f"br[Branch(BranchId({bid}))]"
    for interner in (module.init_interner, module.eval_interner, module.setup_interner):
        for name, kind in list(interner.parameters.items()):
            if isinstance(kind, CurrentKind) and kind.branch_id is None and (kind.hi, kind.lo) in branches:
                interner.parameters[name] = replace(kind, branch_id=branches[(kind.hi, kind.lo)])
    given_parameters = set()
    for interner in (module.setup_interner, module.init_interner, module.eval_interner):
        for value, kind in list(interner.parameters.items()):
            if isinstance(kind, ParamGivenRef) and kind.name in {"Temp", "Tnom", "Pg", "Ne"}:
                given_parameters.add(kind.name)
                interner.parameters[value] = ParamRef(name="given_" + kind.name, param_id=1000 + kind.param_id)
    defaults["Ldc"] = ParamSpec("float", "0.1")
    original_defaults = {name: defaults[name].default for name in given_parameters}
    for name in given_parameters:
        defaults["given_" + name] = ParamSpec("int", "0")

    # Normalize the MIR integer-to-boolean cast explicitly; older lowerers
    # otherwise silently select a fallback incoming phi value.
    for function in (module.setup_fn, module.init_fn, module.eval_fn):
        for callback in function.call_decls:
            callback.raw = callback.raw.replace("%simparam_opt", "%SimParamOpt")
        function.constants.append(Constant(name="gan_zero", kind="iconst", iconst=0))
        for block in function.blocks:
            for instruction in block.insts:
                if instruction.opcode == "ibcast":
                    instruction.opcode = "ine"
                    instruction.operands = [instruction.operands[0], "gan_zero"]
    device = lower(
        dump.modules[0],
        va_defaults=defaults,
        collapse_nodes=False,
        static_params={"Noise": 0, "Noimod": 0},
        class_name="AngelovGaN",
    )
    device.jacobian_resist = {}
    device.jacobian_react = {}
    device.init_cache_refs = _live_init_cache_slots(device)
    device.phi_resolutions = {}
    # Disabled noise coordinates have no physical coupling. Remove their
    # empty native rows; reference tests project the corresponding raw OSDI
    # coordinates out (OSDI internally grounds them).
    for noise_node in ("v_ia", "v_ib"):
        device.states.remove(noise_node)
        device.f_expressions.pop(noise_node, None)
        device.q_expressions.pop(noise_node, None)
    source = emit_source([device])
    # Python bool bitwise inversion differs from logical negation in eager setup.
    source = re.sub(r"~(i_v\d+|v\d+)", r"jnp.logical_not(\1)", source)
    for name in sorted(given_parameters):
        source = re.sub(r"\b" + name + r": float = [^,)]+", name + ': float = float("nan")', source)
        source = re.sub(r", given_" + name + r": int = 0", "", source)
    setup_preamble = '    """Evaluate native Angelov GaN reference equations with parameter-presence flags."""\n' + "".join(
        f"    given_{name} = jnp.logical_not(jnp.isnan({name}))\n"
        f"    {name} = jnp.where(given_{name}, {name}, {original_defaults[name]})\n"
        for name in sorted(given_parameters)
    )
    source = re.sub(r"(def (?:_AngelovGaN_setup|AngelovGaN)\([^\n]+\n)", lambda m: m[1] + setup_preamble, source)
    assigned = set(re.findall(r"^    ((?:i_)?v\d+) =", source, re.MULTILINE))
    for missing in set(re.findall(r"\bi_v\d+\b", source)) - assigned:
        evaluated = missing.removeprefix("i_")
        if evaluated in assigned:
            source = re.sub(r"\b" + missing + r"\b", evaluated, source)
    helpers = (SOURCE.parent / "native_helpers.py").read_text()
    helpers = helpers[helpers.index("def _gan_limexp") :]
    source = source.replace("def _AngelovGaN_setup", helpers + "\n\ndef _AngelovGaN_setup", 1)
    source = source.replace(
        "    v941 = v856 * v940",
        """    v941, _mode_tanh1 = _gan_channel(mode=Idsmod, vgs=s.v_gii-s.v_si,
        vgd=s.v_gdi-s.v_di, vds=s.v_di-s.v_si, vrf=signals.rf-s.v_si,
        ipk=v815, vpk=v654, p1=v478, p2=P2, p3=v491, alphar=Alphar,
        alphas=Alphas, lam=Lambda, lvg=Lvg, lsb=v877, vtr=v619, ebd=Ebd, krfdc=KRFDC)""",
    )
    source = source.replace("    v775 = 1.0 + v774", "    v775 = _mode_tanh1")
    source = source.replace(
        "    v21957 = v21894 + v21901",
        """    _mode_qgs, v21957 = _gan_charge(mode=Capmod, vgs=s.v_gsi-s.v_si,
        vgd=s.v_gdi-s.v_di, vds=s.v_di-s.v_si, delta=v80, cgspi=Cgspi,
        cgs0=Cgs0, cgdpi=Cgdpi, cgd0=Cgd0, p10=P10, p11=P11, p20=P20,
        p21=P21, p30=P30, p31=P31, p40=P40, p41=P41, p111=P111,
        p222=P222, p10pk=P10pk, m=m, tccgs0=Tccgs0, tccgd0=Tccgd0, tcp10=Tcp10)""",
    )
    source = source.replace("    v21958 = v21898 + v21905", "    v21958 = _mode_qgs")
    notice = SOURCE.read_text().split("*/", 1)[0].removeprefix("/*").strip()
    header = (
        '"""Native JAX Angelov GaN channel, charge and thermal/dispersion equations.\n\n'
        + (
            "Generated from VA-Models commit 6099b4e12e6f66ff530f59b8d01fb909094be85a,\n"
            "code/angelov/vacode/angelov_gan.va. Regenerate using tests/data/va_models/angelov_gan/regenerate.py.\n"
            "No OSDI or bosdi runtime is required. Original notice follows verbatim.\n\n"
        )
        + notice
        + '\n"""'
    )
    lint = "\n\n# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, CPY001, E501, E712, E402, F841, FBT003, PLR0915, PLR0917\n"
    surface = (SOURCE.parent / "native_surface.py").read_text()
    surface = surface[surface.index("import inspect") :]
    OUTPUT.write_text(header + lint + source[source.index("\n\nfrom __future__") :] + "\n" + surface)


if __name__ == "__main__":
    main()
