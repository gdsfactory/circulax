"""Scoped model-card loading and hierarchical VACASK circuit elaboration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from circulax.netlist_io.expressions import Scope, evaluate
from circulax.netlist_io.syntax import NetlistError, Statement, children, parameters, parse_file

_KINDS = {
    "ParamStatement",
    "Model",
    "Subckt",
    "SubcktCall",
    "IfBlock",
    "IncludeStatement",
    "LibInclude",
    "LibStatement",
    "HDLStatement",
    "Global",
}


@dataclass(frozen=True)
class ResolvedInstance:
    """A leaf compact model with evaluated model and instance parameters."""

    name: str
    module: str
    nodes: tuple[str, ...]
    parameters: dict[str, float]


@dataclass
class ResolvedCircuit:
    """Simulator-independent leaf topology and the modules used by its cards."""

    instances: list[ResolvedInstance]
    loads: list[tuple[str, Path]]
    ports: dict[str, str] = field(default_factory=dict)
    temperature_c: float = 27.0

    def compile(
        self,
        *,
        module_paths: tuple[Path, ...] = (),
        compiler: str | None = None,
        cache_dir: Path | None = None,
        backend: str = "dense",
    ) -> Any:
        """Compile the resolved topology using bosdi OSDI descriptors."""
        from circulax.netlist_io.osdi import compile_resolved

        return compile_resolved(self, module_paths=module_paths, compiler=compiler, cache_dir=cache_dir, backend=backend)


class _Frame:
    def __init__(self, scope: Scope, parent: _Frame | None = None) -> None:
        self.scope = scope
        self.parent = parent
        self.models: dict[str, tuple[Any, Scope]] = {}
        self.subcircuits: dict[str, tuple[Statement, _Frame]] = {}
        self.calls: list[Statement] = []

    def find(self, name: str, table: str) -> Any:
        values = getattr(self, table)
        if name in values:
            return values[name]
        if self.parent is not None:
            return self.parent.find(name, table)
        return None


def _body(statement: Statement) -> list[Statement]:
    aliases = {"Parameters": "ParamStatement", "Instance": "SubcktCall", "Include": "IncludeStatement"}
    return [
        Statement(aliases.get(n.kind, n.kind), n, statement.path)
        for n in children(statement.node)
        if n.kind not in {"Keyword", "Notation", "Identifier", "SubcktNodes", "HierarchialNode", "Parameter", "Condition"}
    ]


def _nodes(node: Any) -> list[str]:
    spice = children(node, "HierarchialNode")
    if spice:
        return [n.text.strip("'\"") for n in spice]
    lists = children(node, "SubcktNodes") or children(node, "SNodeList")
    return [n.text.strip("'\"") for container in lists for n in children(container, "SNode")] or [
        n.text.strip("'\"") for n in children(node, "SNode")
    ]


class Library:
    """Load existing model libraries and elaborate their public subcircuits.

    Parameters and model cards are evaluated separately for each instance.
    Libraries retain their lexical scopes, local nodes and conditional branches.
    """

    def __init__(self, *, temperature_c: float = 27.0, include_paths: tuple[Path, ...] = (), dialect: str = "vacask") -> None:
        """Create an empty model library at the requested card temperature."""
        if not math.isfinite(temperature_c) or temperature_c <= -273.15:
            msg = "temperature_c must be finite and above absolute zero"
            raise NetlistError(msg)
        self.temperature_c = temperature_c
        self.include_paths = tuple(Path(p) for p in include_paths)
        self.dialect = dialect
        self.loads: list[tuple[str, Path]] = []
        self.globals = {"0"}
        self.grounds = {"0"}
        self.scope = Scope()
        self.scope.dialect = dialect
        self.scope.bindings["$temp"] = temperature_c
        self.frame = _Frame(self.scope)
        self._parsed: dict[Path, list[Statement]] = {}

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        section: str | None = None,
        temperature_c: float = 27.0,
        include_paths: tuple[Path, ...] = (),
        dialect: str = "vacask",
    ) -> Library:
        """Read a library and select a named corner section when requested."""
        library = cls(temperature_c=temperature_c, include_paths=include_paths, dialect=dialect)
        library.include(path, section=section)
        return library

    def include(self, path: str | Path, *, section: str | None = None) -> None:
        """Add a library to the same root scope, preserving declaration order."""
        statements = self._read(Path(path).resolve(), section, (), set())
        self._populate(self.frame, statements)

    def _read(
        self, path: Path, section: str | None, active: tuple[Path, ...], seen: set[tuple[Path, str | None]]
    ) -> list[Statement]:
        if path in active:
            msg = f"include cycle: {' -> '.join(map(str, (*active, path)))}"
            raise NetlistError(msg)
        if (path, section) in seen:
            return []
        seen.add((path, section))
        if path not in self._parsed:
            self._parsed[path] = parse_file(path, self.dialect)
        statements = self._parsed[path]
        sections = [s for s in statements if s.kind == "LibStatement"]
        if section is not None and not any(children(s.node, "Identifier")[0].text == section for s in sections):
            msg = f"{path}: missing section {section!r}"
            raise NetlistError(msg)
        output = []
        for statement in statements:
            if statement.kind == "LibStatement":
                if children(statement.node, "Identifier")[0].text == section:
                    output.extend(self._expand(_body(statement), (*active, path), seen))
            else:
                output.extend(self._expand([statement], (*active, path), seen))
        return output

    def _expand(self, statements: list[Statement], active: tuple[Path, ...], seen: set[tuple[Path, str | None]]) -> list[Statement]:
        output = []
        for statement in statements:
            if statement.kind in {"IncludeStatement", "LibInclude"}:
                reference = children(statement.node, "StringLiteral")[0].text.strip('"')
                candidates = [statement.path.parent / reference, *(p / reference for p in self.include_paths)]
                path = next((p.resolve() for p in candidates if p.is_file()), None)
                if path is None:
                    msg = f"{statement.path}: include {reference!r} not found"
                    raise NetlistError(msg)
                section_nodes = children(statement.node, "IncludeSection")
                section = (
                    children(section_nodes[0], "Identifier")[0].text
                    if section_nodes
                    else children(statement.node, "Identifier")[0].text
                    if statement.kind == "LibInclude"
                    else None
                )
                output.extend(self._read(path, section, active, seen))
            else:
                output.append(statement)
        return output

    def _populate(self, frame: _Frame, statements: list[Statement], overrides: dict[str, float] | None = None) -> None:  # noqa: C901, PLR0912 -- statement dispatch
        # Collect all local bindings before evaluation, allowing forward references.
        for statement in statements:
            if statement.kind == "ParamStatement":
                frame.scope.bindings.update(parameters(statement.node))
        if overrides:
            unknown = overrides.keys() - frame.scope.bindings.keys()
            if unknown:
                msg = f"unknown subcircuit parameters: {sorted(unknown)}"
                raise NetlistError(msg)
            frame.scope.bindings.update(overrides)
        for statement in statements:
            node = statement.node
            kind = statement.kind
            if kind == "ParamStatement":
                continue
            if kind == "Model":
                name = (_nodes(node) or [n.text for n in children(node, "Identifier")])[0]
                frame.models[name] = (node, frame.scope)
            elif kind == "Subckt":
                name = children(node, "Identifier")[0].text
                frame.subcircuits[name] = (statement, frame)
            elif kind == "SubcktCall":
                frame.calls.append(statement)
            elif kind == "HDLStatement":
                reference = children(node, "StringLiteral")[0].text.strip('"')
                load = (reference, statement.path.parent)
                if load not in self.loads:
                    self.loads.append(load)
            elif kind == "Global":
                if node.text.lstrip().startswith("ground "):
                    self.grounds.update(_nodes(node))
                else:
                    self.globals.update(_nodes(node))
            elif kind == "IfBlock":
                for case in children(node, "IfElseCase"):
                    condition = children(case, "Condition")
                    if not condition or evaluate(condition[0], frame.scope):
                        self._populate(frame, self._expand(_body(Statement(case.kind, case, statement.path)), (), set()))
                        break
            elif kind in {"IncludeStatement", "LibInclude"}:
                self._populate(frame, self._expand([statement], (), set()))
            else:
                msg = f"{statement.path}: unsupported statement {kind}"
                raise NetlistError(msg)

    def instantiate(
        self, subcircuit: str, nodes: tuple[str, ...] | None = None, settings: dict[str, float] | None = None, *, name: str = "X1"
    ) -> ResolvedCircuit:
        """Expand a library subcircuit without loading executable OSDI modules."""
        found = self.frame.find(subcircuit, "subcircuits")
        if found is None:
            msg = f"unknown subcircuit {subcircuit!r}"
            raise NetlistError(msg)
        ports = tuple(_nodes(found[0].node))
        actual_nodes = nodes if nodes is not None else ports
        instances: list[ResolvedInstance] = []
        self._instantiate(found, actual_nodes, settings or {}, name, instances, ())
        return ResolvedCircuit(instances, list(self.loads), dict(zip(ports, actual_nodes, strict=True)), self.temperature_c)

    def resolve(self) -> ResolvedCircuit:
        """Expand instances declared at the top level of a loaded circuit file."""
        instances: list[ResolvedInstance] = []
        self._calls(self.frame, {}, "", instances, ())
        return ResolvedCircuit(instances, list(self.loads), temperature_c=self.temperature_c)

    def _instantiate(
        self,
        definition: tuple[Statement, _Frame],
        nodes: tuple[str, ...],
        settings: dict[str, float],
        name: str,
        instances: list[ResolvedInstance],
        active: tuple[int, ...],
    ) -> None:
        statement, parent = definition
        token = id(statement.node)
        if token in active:
            msg = f"subcircuit recursion at {name}"
            raise NetlistError(msg)
        ports = _nodes(statement.node)
        if len(ports) != len(nodes):
            msg = f"{name}: expected {len(ports)} terminals, got {len(nodes)}"
            raise NetlistError(msg)
        frame = _Frame(Scope(parent.scope), parent)
        frame.scope.bindings.update(parameters(statement.node))
        self._populate(frame, _body(statement), settings)
        self._calls(frame, dict(zip(ports, nodes, strict=True)), name, instances, (*active, token))

    def _calls(
        self, frame: _Frame, terminals: dict[str, str], prefix: str, instances: list[ResolvedInstance], active: tuple[int, ...]
    ) -> None:
        for statement in frame.calls:
            symbols = _nodes(statement.node)
            if children(statement.node, "SNodeList"):
                call_name, master = [n.text for n in children(statement.node, "Identifier")]
                nodes = symbols
            else:
                call_name, *nodes, master = symbols
            name = f"{prefix}/{call_name}" if prefix else call_name
            mapped = tuple(
                "0" if n in self.grounds else terminals.get(n, n if n in self.globals or not prefix else f"{prefix}/{n}")
                for n in nodes
            )
            settings = {k: evaluate(v, frame.scope) for k, v in parameters(statement.node).items()}
            lookup = frame
            model = None
            subcircuit = None
            while lookup is not None:
                model = lookup.models.get(master)
                subcircuit = lookup.subcircuits.get(master) if model is None else None
                if model is not None or subcircuit is not None:
                    break
                lookup = lookup.parent
            if subcircuit is not None:
                self._instantiate(subcircuit, mapped, settings, name, instances, active)
                continue
            if model is not None:
                model_node, model_scope = model
                module = children(model_node, "Identifier")[-1].text
                card = {k: evaluate(v, model_scope) for k, v in parameters(model_node).items()}
                card.update(settings)
            elif master in {"vsource", "isource"}:
                module = master
                card = settings
            else:
                msg = f"{statement.path}: unresolved model/subcircuit {master!r} for {name}"
                raise NetlistError(msg)
            instances.append(ResolvedInstance(name, module, mapped, card))
