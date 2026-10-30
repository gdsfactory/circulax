# ruff: noqa: E402, FLY002
# Copyright (c) 2026 Circulax contributors
"""Regenerate the native model; normal imports require neither bosdi nor OpenVAF."""

import os
from pathlib import Path

from bosdi.va import lower
from bosdi.va.emitter import emit_source
from bosdi.va.ir_client import compile_va_unopt
from bosdi.va.va_defaults import ParamSpec, parse_va_defaults_expanded

repo = Path(__file__).resolve().parents[4]
p = repo / "tests/data/va_models/r3_cmc/r3_cmc.va"
# OpenVAF writes a scratch .osdi in its cwd during MIR dumping.
import tempfile

_scratch = tempfile.TemporaryDirectory()
os.chdir(_scratch.name)
defs = parse_va_defaults_expanded(p)
defs.update({k: ParamSpec("float", v) for k, v in {"scale": "1.0", "rthresh": "1e-3", "imax": "1.0"}.items()})
print("defaults", len(defs), flush=True)
dump = compile_va_unopt(str(p))
print("compiled", flush=True)
from bosdi.va.mir import Constant

for fn in (dump.modules[0].eval_fn, dump.modules[0].init_fn):
    fn.constants.append(Constant(name="r3_zero", kind="iconst", iconst=0))
    for block in fn.blocks:
        for inst in block.insts:
            if inst.opcode == "call" and inst.call_target == "inst1":
                inst.opcode = "optbarrier"
                inst.operands = [inst.operands[1]]
            if inst.opcode == "ibcast":
                inst.opcode = "ine"
                inst.operands = [inst.operands[0], "r3_zero"]
dev = lower(dump.modules[0], va_defaults=defs, collapse_nodes=False, class_name="R3_CMC")
print("lowered", len(dev.params), flush=True)
dev.jacobian_resist = {}
dev.jacobian_react = {}
import re

source = emit_source([dev])
slots = re.search(r"return jnp.array\(\[([^\]]+)\]\)", source).group(1).split(", ")
slot_map = {name: i for i, name in enumerate(slots)}
source = re.sub(r"(    (i_v\d+) = init\[)\d+(\])", lambda m: f"{m[1]}{slot_map[m[2]]}{m[3]}", source)
# OpenVAF integer casts of comparisons need arithmetic promotion in JAX.
source = re.sub(r"~(i_v\d+|v\d+)", r"jnp.logical_not(\1)", source)
source = source.replace(
    "i_v230 = i_v225 + i_v228", "i_v230 = jnp.asarray(i_v225, dtype=jnp.int32) + jnp.asarray(i_v228, dtype=jnp.int32)"
)
header = "\n".join(
    [
        "# Copyright (c) 2023 Silicon Integration Initiative, Inc.",
        "# Licensed under the Educational Community License, Version 2.0.",
        "# https://opensource.org/licenses/ECL-2.0",
        "# Distributed on an AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.",
        "# See tests/data/va_models/r3_cmc/LICENSE.txt and NOTICE.txt.",
        "# Generated from VA-Models commit 6099b4e12e6f66ff530f59b8d01fb909094be85a.",
        "# ruff: noqa: A002, ANN001, ANN002, ANN003, ANN202, FBT003, PLR0915, PLR0917",
        "",
    ]
)
source = source.replace(
    "Regenerate with: ``python -m circulax.va <path/to/device.va>``",
    "Regenerate with tests/data/va_models/r3_cmc/generate.py (bosdi + OpenVAF required only for regeneration).",
)
(repo / "circulax/components/r3.py").write_text(header + source)
