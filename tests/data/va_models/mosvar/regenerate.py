"""Regenerate native MOSVAR with bosdi source 159907ddf29a0225614cf1b7999214401b0b5a2d.

Requires bosdi VA lowering and openvaf-r with --dump-unopt-mir.
The emitted component uses ordinary JAX with no bosdi/OSDI runtime.
"""

# ruff: noqa: CPY001

import os
import re
import shutil
import tempfile
from pathlib import Path

from bosdi.va import lower
from bosdi.va.emitter import _live_init_cache_slots, emit_source
from bosdi.va.ir_client import compile_va_unopt
from bosdi.va.mir import Constant
from bosdi.va.va_defaults import parse_va_defaults_expanded

SOURCE = Path(__file__).with_name("mosvar.va").resolve()
OUTPUT = SOURCE.parents[4] / "circulax/components/mosvar.py"


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
    # Normalize the MIR integer-to-boolean cast explicitly; older lowerers
    # otherwise silently select a fallback incoming phi value.
    for function in (module.setup_fn, module.init_fn, module.eval_fn):
        function.constants.append(Constant(name="mosvar_zero", kind="iconst", iconst=0))
        for block in function.blocks:
            for instruction in block.insts:
                if instruction.opcode == "ibcast":
                    instruction.opcode = "ine"
                    instruction.operands = [instruction.operands[0], "mosvar_zero"]
    device = lower(
        dump.modules[0],
        va_defaults=defaults,
        collapse_nodes=False,
        static_params={"LEVEL": 1000, "SWRES": 1},
        class_name="MOSVAR",
    )
    device.jacobian_resist = {}
    device.jacobian_react = {}
    device.init_cache_refs = _live_init_cache_slots(device)
    source = emit_source([device])
    # Python bool bitwise inversion differs from logical negation in eager setup.
    source = re.sub(r"~(i_v\d+|v\d+)", r"jnp.logical_not(\1)", source)
    original = SOURCE.read_text()
    notice = original[original.index("//  (c) Copyright") : original.index("// Further information")]
    notice = "\n".join(line.removeprefix("//").strip() for line in notice.splitlines())
    header = (
        '''"""Native JAX MOSVAR 1.3 current, charge and internal relaxation equations.

Generated from VA-Models commit 6099b4e12e6f66ff530f59b8d01fb909094be85a,
code/MOSVAR/vacode/mosvar.va and its includes. Regenerate with
tests/data/va_models/mosvar/regenerate.py using the bosdi version recorded
there. LEVEL=1000 and SWRES=1 are specialized; all finite parasitic resistances
and internal charge states are retained. Noise is excluded.
No bosdi or OSDI runtime is needed.

'''
        + notice
        + '\n"""'
    )
    lint = "\n\n# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, CPY001, FBT003, PLR0915, PLR0917\n"
    OUTPUT.write_text(header + lint + source[source.index("\n\nfrom __future__") :])


if __name__ == "__main__":
    main()
