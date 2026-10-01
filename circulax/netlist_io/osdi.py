"""Provision OpenVAF modules and compile an elaborated library topology."""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from circulax.netlist_io.syntax import NetlistError

if TYPE_CHECKING:
    from circulax.circuit import Circuit
    from circulax.netlist_io.library import ResolvedCircuit, ResolvedInstance


def compile_va(source: Path, *, compiler: str | None = None, cache_dir: Path | None = None) -> Path:
    """Compile into a content-addressed cache, including preprocessor dependencies."""
    executable = shutil.which(compiler or "openvaf-r")
    if executable is None:
        msg = "OpenVAF compiler not found; supply compiler= or put openvaf-r on PATH"
        raise FileNotFoundError(msg)
    source = source.resolve()
    digest = hashlib.sha256()
    version = subprocess.check_output([executable, "--version"])  # noqa: S603 -- explicit compiler, no shell
    digest.update(version)
    digest.update(f"{platform.system()}/{platform.machine()}".encode())
    visited: set[Path] = set()

    def include(path: Path) -> None:
        path = path.resolve()
        if path in visited:
            return
        visited.add(path)
        data = path.read_bytes()
        digest.update(str(path).encode())
        digest.update(data)
        for reference in re.findall(rb'`include\s+"([^"]+)"', data):
            dependency = path.parent / os.fsdecode(reference)
            # OpenVAF supplies constants.vams/disciplines.vams itself.
            if dependency.is_file():
                include(dependency)
            elif reference not in {b"constants.vams", b"disciplines.vams", b"constants.h", b"discipline.h"}:
                msg = f"{path}: unresolved Verilog-A include {reference!r}"
                raise FileNotFoundError(msg)

    include(source)
    cache = Path(cache_dir or Path.home() / ".cache/circulax/osdi")
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{source.stem}-{digest.hexdigest()}.osdi"
    if not target.is_file():
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            output = Path(temporary) / "model.osdi"
            run = subprocess.run(  # noqa: S603 -- explicit compiler, no shell
                [executable, str(source), "-o", str(output)], capture_output=True, text=True, check=False
            )
            if run.returncode or not output.is_file():
                msg = f"OpenVAF failed for {source}:\n{run.stdout}\n{run.stderr}"
                raise NetlistError(msg)
            output.replace(target)
    return target


class _Parameter(ctypes.Structure):
    _fields_ = [
        ("names", ctypes.POINTER(ctypes.c_char_p)),
        ("aliases", ctypes.c_uint32),
        ("description", ctypes.c_char_p),
        ("units", ctypes.c_char_p),
        ("flags", ctypes.c_uint32),
        ("length", ctypes.c_uint32),
    ]


def module_metadata(path: Path) -> tuple[str, dict[str, str]]:
    """Read canonical names and declared aliases from the OSDI 0.4 ABI."""
    if ctypes.sizeof(ctypes.c_void_p) != 8:
        msg = "OSDI metadata inspection currently requires a 64-bit platform"
        raise NetlistError(msg)
    library = ctypes.CDLL(str(path))
    version = tuple(ctypes.c_uint32.in_dll(library, f"OSDI_VERSION_{part}").value for part in ("MAJOR", "MINOR"))
    if version != (0, 4):
        msg = f"{path}: unsupported OSDI ABI {version}; bosdi requires 0.4"
        raise NetlistError(msg)
    if ctypes.c_uint32.in_dll(library, "OSDI_NUM_DESCRIPTORS").value != 1:
        msg = f"{path}: expected one OSDI module per binary"
        raise NetlistError(msg)
    name = ctypes.c_char_p.in_dll(library, "OSDI_DESCRIPTORS").value.decode()
    base = ctypes.addressof(ctypes.c_char_p.in_dll(library, "OSDI_DESCRIPTORS"))
    count = ctypes.c_uint32.from_address(base + 76).value
    address = ctypes.c_void_p.from_address(base + 88).value
    table = ctypes.cast(address, ctypes.POINTER(_Parameter))
    aliases = {}
    for index in range(count):
        parameter = table[index]
        canonical = parameter.names[0].decode()
        for alias in range(parameter.aliases + 1):
            aliases[parameter.names[alias].decode().lower()] = canonical
    return name, aliases


def _native_source(instance: ResolvedInstance) -> tuple[str, Any, dict[str, float]]:
    """Translate the supported native DC and pulse source parameters."""
    from circulax.components.electronic import CurrentSource, PulseVoltageSource, VoltageSource

    module = instance.module.lower()
    settings = instance.parameters
    unknown = settings.keys() - {"dc", "mag", "phase", "type", "val0", "val1", "delay", "rise", "fall", "width", "period"}
    if unknown:
        msg = f"{instance.name}: unsupported native source parameters {sorted(unknown)}"
        raise NotImplementedError(msg)
    model = VoltageSource if module == "vsource" else CurrentSource
    component = module
    if settings.get("type", "dc") == "pulse":
        if module != "vsource" or not {"width", "period"} <= settings.keys():
            msg = f"{instance.name}: pulse requires voltage source, width and period"
            raise NotImplementedError(msg)
        if settings.get("dc", settings.get("val0", 0.0)) != settings.get("val0", 0.0):
            msg = "separate pulse DC override is not supported"
            raise NotImplementedError(msg)
        model = PulseVoltageSource
        component = "pulse_vsource"
        translation = {
            "val0": "v1",
            "val1": "v2",
            "delay": "td",
            "rise": "tr",
            "fall": "tf",
            "width": "pw",
            "period": "per",
        }
        settings = {translation[k]: v for k, v in settings.items() if k in translation}
    elif settings.get("type", "dc") == "dc":
        unknown_waveform = settings.keys() - {"dc", "mag", "phase", "type"}
        if unknown_waveform:
            msg = f"{instance.name}: waveform parameters on a DC source"
            raise NetlistError(msg)
        settings = {"V" if module == "vsource" else "I": settings.get("dc", 0.0)}
    else:
        msg = f"{instance.name}: unsupported source type {settings['type']!r}"
        raise NotImplementedError(msg)
    return component, model, settings


def _provision_modules(
    resolved: ResolvedCircuit, module_paths: tuple[Path, ...], compiler: str | None, cache_dir: Path | None
) -> dict[str, tuple[Path, dict[str, str]]]:
    """Resolve and provision the module declarations belonging to the libraries."""
    modules = {}
    for reference, directory in resolved.loads:
        candidates = [directory / reference, *(Path(p) / reference for p in module_paths)]
        path = next((p.resolve() for p in candidates if p.is_file()), None)
        if path is None:
            msg = f"OSDI load {reference!r} not found in {candidates}"
            raise FileNotFoundError(msg)
        if path.suffix == ".va":
            path = compile_va(path, compiler=compiler, cache_dir=cache_dir)
        name, aliases = module_metadata(path)
        if name.lower() in modules and modules[name.lower()][0] != path:
            msg = f"duplicate OSDI module {name!r}"
            raise NetlistError(msg)
        modules[name.lower()] = (path, aliases)

    return modules


def compile_resolved(  # noqa: C901, PLR0912 -- topology and terminal validation
    resolved: ResolvedCircuit,
    *,
    module_paths: tuple[Path, ...] = (),
    compiler: str | None = None,
    cache_dir: Path | None = None,
    backend: str = "dense",
    analysis: str = "dc",
    state_policy: str = "reject",
    simparams: Mapping[str, float] | None = None,
) -> Circuit:
    """Compile original model cards; reject runtime features bosdi cannot represent."""
    from bosdi.circulax import osdi_component

    from circulax.circuit import compile_circuit

    modules = _provision_modules(resolved, module_paths, compiler, cache_dir)

    instances = {"GND": {"component": "ground"}}
    models = {}
    nodes: dict[str, list[str]] = {"0": ["GND,p1"]}
    ports = {}
    for index, instance in enumerate(resolved.instances):
        key = f"device{index}"
        module = instance.module.lower()
        settings = instance.parameters
        if module in {"vsource", "isource"}:
            component, model, settings = _native_source(instance)
            models[component] = model
            names = ("p1", "p2")
        else:
            if module not in modules:
                msg = f"{instance.name}: no loaded OSDI module named {instance.module!r}"
                raise NetlistError(msg)
            path, aliases = modules[module]
            component = module
            names = tuple(f"p{i}" for i in range(len(instance.nodes)))
            if component not in models:
                models[component] = osdi_component(
                    str(path),
                    ports=names,
                    temperature=resolved.temperature_c + 273.15,
                    analysis=analysis,
                    state_policy=state_policy,
                    simparams=simparams,
                )
            elif models[component].ports != names:
                msg = f"{instance.name}: inconsistent terminal count for {module}"
                raise NetlistError(msg)
            canonical = {}
            for name, value in settings.items():
                if name.lower() not in aliases:
                    msg = f"{instance.name}: unknown parameter {name!r} for {module}"
                    raise NetlistError(msg)
                canonical[aliases[name.lower()]] = value
            settings = canonical
        if len(names) != len(instance.nodes):
            msg = f"{instance.name}: expected {len(names)} terminals"
            raise NetlistError(msg)
        instances[key] = {"component": component, "settings": settings}
        for name, node in zip(names, instance.nodes, strict=True):
            nodes.setdefault(node, []).append(f"{key},{name}")
    connections = {members[0]: tuple(members[1:]) for members in nodes.values() if len(members) > 1}
    # Expose all flattened nodes, plus the public wrapper's formal port aliases.
    for node, members in nodes.items():
        ports[node] = members[0]
    for port, node in resolved.ports.items():
        if node not in nodes:
            msg = f"unconnected public terminal {port!r}"
            raise NetlistError(msg)
        ports[port] = nodes[node][0]
    return compile_circuit(
        {"instances": instances, "connections": connections, "ports": ports},
        models,
        backend=backend,
        is_complex=False,
        g_leak=0.0,
        rtol=1e-8,
        atol=1e-12,
    )
