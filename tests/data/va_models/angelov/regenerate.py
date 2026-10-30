"""Regenerate native Angelov from the pinned source; no runtime compiler needed.

Requires bosdi source159907ddf29a0225614cf1b7999214401b0b5a2d and OpenVAF-r.
Repairs branch IDs, boolean casts, parameter presence, init cache indices,
stale phi metadata, and known multi-way switch lowering mistakes. The four
SSA repairs are validated against independent OSDI fixtures for every mode.
"""

# ruff: noqa: ANN001, ANN201, CPY001, E501, E402
import re
from contextlib import chdir
from dataclasses import replace
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

from bosdi.va import lower
from bosdi.va.emitter import _live_init_cache_slots, emit_source
from bosdi.va.ir_client import compile_va_unopt
from bosdi.va.mir import Constant, CurrentKind, ParamGivenRef, ParamRef
from bosdi.va.va_defaults import ParamSpec, parse_va_defaults_expanded

p = Path(__file__).with_name("angelov.va").resolve()
with TemporaryDirectory(prefix="angelov_mir_") as scratch:
    for dependency in p.parent.iterdir():
        if dependency.suffix in {".va", ".include", ".h", ".vams"}:
            copy2(dependency, Path(scratch) / dependency.name)
    with chdir(scratch):
        dump = compile_va_unopt(p.name)
m = dump.modules[0]
for fn in (m.init_fn, m.eval_fn):
    fn.constants.append(Constant(name="angelov_zero", kind="iconst", iconst=0))
    for b in fn.blocks:
        for i in b.insts:
            if i.opcode == "ibcast":
                i.opcode = "ine"
                i.operands = [i.operands[0], "angelov_zero"]
# Unoptimized text MIR omits branch IDs; restore by endpoint identity.
node_names = dict(zip(m.port_nodes, m.ports, strict=True))
for interner in (m.init_interner, m.eval_interner):
    for kind in interner.parameters.values():
        if hasattr(kind, "hi_node") and kind.hi_node:
            node_names[kind.hi_node] = kind.hi
        if hasattr(kind, "lo_node") and kind.lo_node:
            node_names[kind.lo_node] = kind.lo
branches = {}
for sim, node in list(m.dae.unknowns.items()):
    match = re.fullmatch(r"br\[Unnamed \{ hi: (node\d+), lo: Some\((node\d+)\) \}\]", node)
    if match:
        bid = int(sim.removeprefix("sim_node"))
        branches[(node_names[match[1]], node_names[match[2]])] = bid
        m.dae.unknowns[sim] = f"br[Branch(BranchId({bid}))]"
defs = parse_va_defaults_expanded(p)
for interner in (m.init_interner, m.eval_interner, m.setup_interner):
    for name, kind in list(interner.parameters.items()):
        if isinstance(kind, CurrentKind) and kind.branch_id is None and (kind.hi, kind.lo) in branches:
            bid = branches[(kind.hi, kind.lo)]
            interner.parameters[name] = replace(kind, branch_id=bid)
        if isinstance(kind, ParamGivenRef) and kind.name in {"Temp", "Tnom", "Pg", "Ne"}:
            flag = "_given_" + kind.name
            interner.parameters[name] = ParamRef(name=flag, param_id=1000 + kind.param_id)
            defs[flag] = ParamSpec("float", "0.0")
dev = lower(m, va_defaults=defs, collapse_nodes=False, static_params={"Noise": 0, "Noimod": 0}, class_name="Angelov")
print("ports", dev.ports, "states", dev.states, "params", dev.params, flush=True)
dev.jacobian_resist = {}
dev.jacobian_react = {}
for name in ("v_ia", "v_ib"):
    dev.states.remove(name)
    dev.f_expressions.pop(name, None)
    dev.q_expressions.pop(name, None)
dev.init_cache_refs = _live_init_cache_slots(dev)
dev.phi_resolutions = {}
import ast


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


# Optional VA parameters use NaN as "not given"; infer presence inside
# physics so circuit parameter updates preserve $param_given semantics.
optional = {"Temp": "25.0", "Tnom": "25.0", "Pg": "15.0", "Ne": "1.0"}
dev.params = [
    (name, ty, "float('nan')" if name in optional else default)
    for name, ty, default in dev.params
    if not name.startswith("_given_")
]
source = emit_source([dev])
# The setup and physics functions both need sanitized local parameter values.
tree = ast.parse(source)
for fn in tree.body:
    if isinstance(fn, ast.FunctionDef) and fn.name in {"_Angelov_setup", "Angelov"}:
        preamble = []
        for name, default in optional.items():
            preamble.extend(
                ast.parse(f"_given_{name} = ~jnp.isnan({name})\n{name} = jnp.where(_given_{name}, {name}, {default})").body
            )
        offset = int(bool(fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant)))
        fn.body[offset:offset] = preamble
source = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
# Repair the reference's multi-way case phis: the unoptimized lowerer picks
# a default branch for these switches. These short formulas preserve the
# source's exact channel and capacitance modes without duplicating the network.
source = source.replace(
    "    v359 = 1.0 + v358",
    "    _psi_nested = v236 * (v320 + P2 * v322 + P3 * v320 * v322)\n    v359 = jnp.where(Idsmod >= 2, 1.0 + jnp.tanh(0.5 * (_limexp(_psi_nested) - _limexp(-_psi_nested))), 1.0 + v358)",
)
source = source.replace(
    "    v442 = v399 * v441",
    "    v442 = _channel_current(Idsmod, v399 * v441, s.v_di-s.v_si, s.v_gi-s.v_di, v291, v236, P2, P3, v391, v335, v359, Alphar, Alphas, Lambda, Lvg, v414, v286)",
)
source = source.replace(
    "    v13457 = v13389 + v13395",
    "    v13457 = jnp.where(Capmod == 1, (Cgdpi + v1908 * (v1681 * (1.0+jnp.tanh(v1776)) + 2.0*P111)) * (s.v_gdi-s.v_di), v13389 + v13395)",
)
source = source.replace(
    "    v13458 = v13392 + v13398",
    "    v13458 = jnp.where(Capmod == 1, (Cgspi + v1817 * (1.0+jnp.tanh(v1658)) * v1671) * (s.v_gsi-s.v_si), v13392 + v13398)",
)
source = ast.unparse(ast.fix_missing_locations(LogicalNot().visit(ast.parse(source)))) + "\n"
source += Path(__file__).with_name("channel_helpers.txt").read_text()
source += Path(__file__).with_name("constructor.txt").read_text()
notice = p.read_text().split("*/", 1)[0].removeprefix("/*")
notice = "\n".join("# " + line for line in notice.splitlines())
header = (
    "# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, B010, CPY001, E402, E501, EM102, FBT003, PLR0913, PLR0915, PLR0917\n"
)
header += "# Native Angelov: upstream6099b4e12e6f66ff530f59b8d01fb909094be85a; bosdi159907ddf29a0225614cf1b7999214401b0b5a2d.\n"
output = Path(__file__).resolve().parents[4] / "circulax/components/angelov.py"
output.write_text(header + notice + "\n" + source)
