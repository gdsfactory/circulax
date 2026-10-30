"""Regenerate native EKV26 with bosdi source 159907ddf29a0225614cf1b7999214401b0b5a2d.

Requires bosdi VA lowering and openvaf-r with --dump-unopt-mir.
The emitted component uses ordinary JAX with no bosdi/OSDI runtime.
"""

# ruff: noqa: CPY001, C901, PLR0912, ISC003

import os
import re
import shutil
import tempfile
from pathlib import Path

from bosdi.va import lower
from bosdi.va.emitter import _live_init_cache_slots, emit_source
from bosdi.va.ir_client import compile_va_unopt
from bosdi.va.mir import Constant, ParamGivenRef, ParamRef
from bosdi.va.va_defaults import ParamSpec, parse_va_defaults_expanded

SOURCE = Path(__file__).with_name("ekv26.va").resolve()
OUTPUT = SOURCE.parents[4] / "circulax/components/ekv.py"


def main() -> None:
    defaults = parse_va_defaults_expanded(SOURCE)
    previous = Path.cwd()
    try:
        with tempfile.TemporaryDirectory() as scratch:
            os.chdir(scratch)
            for original in SOURCE.parent.iterdir():
                if original.suffix in {".va", ".h"}:
                    shutil.copy2(original, Path(scratch) / original.name)
            dump = compile_va_unopt(str(Path(scratch) / SOURCE.name))
    finally:
        os.chdir(previous)
    module = dump.modules[0]
    given_parameters = set()
    for interner in (module.setup_interner, module.init_interner, module.eval_interner):
        for value, kind in list(interner.parameters.items()):
            if isinstance(kind, ParamGivenRef):
                given_parameters.add(kind.name)
                interner.parameters[value] = ParamRef(name="given_" + kind.name, param_id=1000 + kind.param_id)
    original_defaults = {name: defaults[name].default for name in given_parameters}
    for name in given_parameters:
        defaults["given_" + name] = ParamSpec("int", "0")

    # Normalize the MIR integer-to-boolean cast explicitly; older lowerers
    # otherwise silently select a fallback incoming phi value.
    for function in (module.setup_fn, module.init_fn, module.eval_fn):
        for callback in function.call_decls:
            callback.raw = callback.raw.replace("%simparam_opt", "%SimParamOpt")
        function.constants.append(Constant(name="ekv_zero", kind="iconst", iconst=0))
        for block in function.blocks:
            for instruction in block.insts:
                if instruction.opcode == "ibcast":
                    instruction.opcode = "ine"
                    instruction.operands = [instruction.operands[0], "ekv_zero"]
    device = lower(
        dump.modules[0],
        va_defaults=defaults,
        collapse_nodes=False,
        static_params={},
        class_name="EKV26",
    )
    device.jacobian_resist = {}
    device.jacobian_react = {}
    device.init_cache_refs = _live_init_cache_slots(device)
    source = emit_source([device])
    # Python bool bitwise inversion differs from logical negation in eager setup.
    source = re.sub(r"~(i_v\d+|v\d+)", r"jnp.logical_not(\1)", source)
    for name in sorted(given_parameters):
        source = re.sub(r"\b" + name + r": float = [^,)]+", name + ': float = float("nan")', source)
        source = re.sub(r", given_" + name + r": int = 0", "", source)
    setup_preamble = '    """Evaluate native EKV reference equations with parameter-presence flags."""\n' + "".join(
        f"    given_{name} = jnp.logical_not(jnp.isnan({name}))\n"
        f"    {name} = jnp.where(given_{name}, {name}, {original_defaults[name]})\n"
        for name in sorted(given_parameters)
    )
    source = re.sub(r"(def (?:_EKV26_setup|EKV26)\([^\n]+\n)", lambda m: m[1] + setup_preamble, source)
    assigned = set(re.findall(r"^    ((?:i_)?v\d+) =", source, re.MULTILINE))
    for missing in set(re.findall(r"\bi_v\d+\b", source)) - assigned:
        evaluated = missing.removeprefix("i_")
        if evaluated in assigned:
            source = re.sub(r"\b" + missing + r"\b", evaluated, source)
    notice = SOURCE.read_text().split("*/", 1)[0].removeprefix("/*").strip()
    header = (
        '"""Native JAX EKV 2.6 rev.15 current and charge equations.\n\n'
        + (
            "Generated from VA-Models commit 6099b4e12e6f66ff530f59b8d01fb909094be85a,\n"
            "code/ekv/vacode/ekv26.va. Regenerate using tests/data/va_models/ekv26/regenerate.py.\n"
            "No OSDI or bosdi runtime is required. Original notice follows verbatim.\n\n"
        )
        + notice
        + '\n"""'
    )
    lint = "\n\n# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, CPY001, E501, E712, FBT003, PLR0915, PLR0917\n"
    OUTPUT.write_text(header + lint + source[source.index("\n\nfrom __future__") :])


if __name__ == "__main__":
    main()
