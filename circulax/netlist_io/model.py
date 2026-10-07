"""Reusable model-library registrations for schematic instance elaboration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import kfnetlist as kfnl

from circulax.netlist_io.expressions import evaluate_source
from circulax.netlist_io.library import Library
from circulax.netlist_io.osdi import build_resolved
from circulax.netlist_io.units import convert_unit


@dataclass(frozen=True)
class NetlistDefinition:
    """Canonical topology and shared leaf models, before solver compilation."""

    source_netlist: kfnl.Netlist
    source_models: dict[str, Any]


class LibraryModel:
    """Register one library wrapper, binding settings separately per instance.

    This object is deliberately non-callable. ``compile_circuit`` elaborates
    wrapper instances to kfnetlist and lets kfnetlist flatten their topology.
    OSDI descriptor definitions are shared; card and instance values populate
    individual leaf settings and therefore rows in the native device batch.

    ``params_map`` maps schematic setting names to library parameter names.
    ``parameter_units`` explicitly declares (source, target) units for those
    schematic settings. Defaults use the same source units. ``expressions``
    maps library parameter names to safe expressions over converted settings.
    Unspecified defaults remain owned by the original library cards.
    """

    _is_circulax_library_model = True

    def __init__(
        self,
        library: Library | None,
        subcircuit: str,
        *,
        defaults: Mapping[str, Any] | None = None,
        params_map: Mapping[str, str] | None = None,
        port_map: Mapping[str, str] | None = None,
        parameter_units: Mapping[str, tuple[str, str]] | None = None,
        parameter_values: Mapping[str, Mapping[Any, float]] | None = None,
        expressions: Mapping[str, str] | None = None,
        module_paths: tuple[Path, ...] = (),
        osdi_modules: tuple[Path, ...] = (),
        compiler: str | None = None,
        cache_dir: Path | None = None,
        analysis: str = "dc",
        state_policy: str = "reject",
        simparams: Mapping[str, float] | None = None,
    ) -> None:
        """Declare schematic-to-library mappings and native provisioning.

        @tags circulax-simulation
        """
        self.library = library
        self.subcircuit = subcircuit
        self.defaults = dict(defaults or {})
        self.params_map = dict(params_map or {})
        self.port_map = dict(port_map or {})
        self.parameter_units = dict(parameter_units or {})
        self.parameter_values = {name: dict(mapping) for name, mapping in (parameter_values or {}).items()}
        self.expressions = dict(expressions or {})
        self.options = {
            "module_paths": module_paths,
            "osdi_modules": osdi_modules,
            "compiler": compiler,
            "cache_dir": cache_dir,
            "analysis": analysis,
            "state_policy": state_policy,
            "simparams": simparams,
        }
        self._modules = {}
        self._source = None
        self._libraries = {}

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        subcircuit: str,
        *,
        section: str | None = None,
        sections: tuple[str, ...] = (),
        dialect: str = "ngspice",
        temperature_c: float = 27.0,
        include_paths: tuple[Path, ...] = (),
        **options: Any,
    ) -> LibraryModel:
        """Register lazily; parse the selected corner only when instantiated.

        @tags circulax-simulation
        """
        model = cls(None, subcircuit, **options)
        model._source = (Path(path), section, tuple(sections), dialect, temperature_c, include_paths)
        return model

    def _select_library(self, values: dict[str, Any]) -> Library:
        """Resolve and cache the explicitly selected corner library.

        @tags circulax-simulation
        """
        library = self.library
        if self._source is not None:
            path, section, sections, dialect, temperature_c, include_paths = self._source
            selected = values.pop("corner", section)
            if sections and selected not in sections:
                msg = f"unsupported corner {selected!r}; expected one of {sections!r}"
                raise ValueError(msg)
            if selected not in self._libraries:
                self._libraries[selected] = Library.from_file(
                    path,
                    section=selected,
                    dialect=dialect,
                    temperature_c=temperature_c,
                    include_paths=include_paths,
                )
            library = self._libraries[selected]
        if library is None:
            msg = "LibraryModel requires a library or file registration"
            raise ValueError(msg)
        return library

    def instantiate(self, settings: Mapping[str, Any] | None = None) -> NetlistDefinition:
        """Evaluate this instance's cards and retain reusable native models.

        @tags circulax-simulation
        """
        values = {**self.defaults, **(settings or {})}
        library = self._select_library(values)
        for name, translation in self.parameter_values.items():
            if name in values:
                if values[name] not in translation:
                    msg = f"unsupported value {values[name]!r} for {name!r}; expected one of {tuple(translation)!r}"
                    raise ValueError(msg)
                values[name] = translation[values[name]]
        for name, (source, target) in self.parameter_units.items():
            if name in values:
                values[name] = convert_unit(values[name], source, target)
        mapped = {
            self.params_map.get(name, name): value
            for name, value in values.items()
            if not self.expressions or name in self.params_map
        }
        mapped.update({name: evaluate_source(expression, values) for name, expression in self.expressions.items()})
        resolved = library.instantiate(self.subcircuit, settings=mapped)
        if self.port_map:
            unknown = set(self.port_map.values()) - resolved.ports.keys()
            if unknown:
                msg = f"unknown library terminals {sorted(unknown)}"
                raise ValueError(msg)
            ports = {name: node for name, node in resolved.ports.items() if name not in self.port_map.values()}
            ports.update({alias: resolved.ports[terminal] for alias, terminal in self.port_map.items()})
            resolved.ports = ports
        netlist, models, modules = build_resolved(
            resolved,
            **self.options,
            modules=self._modules.get(id(library)),
            expose_internal_nodes=False,
        )
        self._modules[id(library)] = modules
        return NetlistDefinition(netlist, models)
