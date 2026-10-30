"""Regenerate DIODE_CMC; requires OpenVAF-r and bosdi development source.

Validated generator commit: 159907ddf29a0225614cf1b7999214401b0b5a2d.
Native runtime does not require these tools. SSA selectors are pinned to this
exact upstream VA and generator combination; parity tests guard their meaning.
"""

# ruff: noqa: ANN001, ANN201, CPY001
import ast
from contextlib import chdir
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

from bosdi.va import lower
from bosdi.va.emitter import _live_init_cache_slots, emit_source
from bosdi.va.ir_client import compile_va_unopt
from bosdi.va.va_defaults import parse_va_defaults_expanded

p = Path(__file__).with_name("diode_cmc.va").resolve()
defs = parse_va_defaults_expanded(p)
print("defaults", len(defs), flush=True)
with TemporaryDirectory(prefix="diode_cmc_mir_") as scratch:
    for dependency in p.parent.iterdir():
        if dependency.suffix in {".va", ".include", ".h", ".vams"}:
            copy2(dependency, Path(scratch) / dependency.name)
    with chdir(scratch):
        dump = compile_va_unopt(p.name)
print("compiled", flush=True)
dev = lower(
    dump.modules[0],
    va_defaults=defs,
    collapse_nodes=False,
    static_params={"LEVEL": 2002.0, "VERSION": 2.0, "SUBVERSION": 0.0, "REVISION": 0.0},
    class_name="DIODE_CMC",
)
print("lowered ports", dev.ports, "states", dev.states, "parameters", len(dev.params), flush=True)
dev.jacobian_resist = {}
dev.jacobian_react = {}
# Restore conditional node-collapse semantics with explicit inactive constraints.
f, q = dev.f_expressions, dev.q_expressions
active = "i_v374520"
f["K"] = f"jnp.where({active}, {f['K']}, -({f['A']}))"
q["K"] = f"jnp.where({active}, 0.0, -({q['A']}))"
f["v_AIK"] = f"jnp.where({active}, {f['v_AIK']}, s.v_AIK - signals.K)"
q["v_AIK"] = f"jnp.where({active}, {q['v_AIK']}, 0.0)"
for state, cond in (("v_charge_A", "i_v374608"), ("v_charge_K", "i_v374608"), ("v_depl_A", "i_v390134")):
    f[state] = f"jnp.where({cond}, {f[state]}, s.{state})"
dev.init_cache_refs = _live_init_cache_slots(dev)
source = emit_source([dev])
# VA boolean inversion must remain logical inversion even for eager Python bools.


class LogicalNot(ast.NodeTransformer):
    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Invert):
            return ast.copy_location(
                ast.Call(
                    func=ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr="logical_not", ctx=ast.Load()),
                    args=[node.operand],
                    keywords=[],
                ),
                node,
            )
        return node


source = ast.unparse(ast.fix_missing_locations(LogicalNot().visit(ast.parse(source)))) + "\n"
notice = p.read_text().split("`include", 1)[0]
notice = "\n".join("# " + line.removeprefix("//").strip() for line in notice.splitlines())
header = "# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, CPY001, E501, FBT003, PLR0915, PLR0917\n"
header += "# Native equations generated from VA-Models 6099b4e12e6f66ff530f59b8d01fb909094be85a.\n"
header += "# Generator: bosdi 159907ddf29a0225614cf1b7999214401b0b5a2d, unoptimized MIR.\n"
output = Path(__file__).resolve().parents[4] / "circulax/components/diode_cmc.py"
output.write_text(header + notice + "\n" + source)
