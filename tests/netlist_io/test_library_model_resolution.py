"""Capability inspection elaborates cards without provisioning native models."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from circulax.netlist_io import Library, LibraryModel, ResolvedCircuit
from circulax.netlist_io.syntax import NetlistError


def test_resolve_preserves_corner_units_parameters_and_topology_without_building(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read-only resolution has the same settings contract as instantiation."""
    path = tmp_path / "corners.lib"
    path.write_text("""* native modules need not exist for discovery
.lib tt
.param factor=1
.endl tt
.lib ff
.param factor=3
.endl ff
.subckt device p n
.param w=1u
.model core unavailable_native width=w value='w*factor'
N1 p mid core
N2 mid n core
.ends
""")
    model = LibraryModel.from_file(
        path,
        "device",
        section="tt",
        sections=("tt", "ff"),
        defaults={"width": 2},
        params_map={"width": "w"},
        parameter_units={"width": ("um", "m")},
        port_map={"P": "p", "N": "n"},
        compiler="missing-compiler",
        osdi_modules=(tmp_path / "missing.osdi",),
    )

    def forbidden(*_args: object, **_kwargs: object) -> None:
        """Native provisioning cannot participate in capability inspection."""
        pytest.fail("resolve must not build, provision, compile or load native modules")

    monkeypatch.setattr("circulax.netlist_io.model.build_resolved", forbidden)
    monkeypatch.setattr("circulax.netlist_io.osdi._provision_modules", forbidden)
    monkeypatch.setattr("circulax.netlist_io.osdi.osdi_component", forbidden)
    monkeypatch.setattr("circulax.netlist_io.osdi.compile_va", forbidden)
    settings = {"corner": "ff", "width": 4}
    resolved = model.resolve(settings)
    tt_sources, ff_sources = model.source_digests(), model.source_digests(settings)
    assert [r["path"] for r in tt_sources] == [r["path"] for r in ff_sources] == [str(path)]
    assert settings == {"corner": "ff", "width": 4}
    assert resolved.ports == {"P": "p", "N": "n"}
    assert len(resolved.instances) == 2
    assert resolved.instances[0].nodes[1] == resolved.instances[1].nodes[0]
    for leaf in resolved.instances:
        assert leaf.module == "unavailable_native"
        assert leaf.parameters["width"] == pytest.approx(4e-6)
        assert leaf.parameters["value"] == pytest.approx(12e-6)
    assert model._modules == {}  # noqa: SLF001 -- ensure discovery leaves provisioning cache untouched
    assert model.resolve().instances[0].parameters["value"] == pytest.approx(2e-6)
    with pytest.raises(ValueError, match="unsupported corner"):
        model.resolve({"corner": "bad"})


def test_instantiate_uses_public_resolver_before_building(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Execution consumes the same mapped topology exposed to discovery."""
    path = tmp_path / "model.lib"
    path.write_text("* card\n.subckt device p n\n.model vs vsource dc=1\nN1 p n vs\n.ends\n")
    model = LibraryModel.from_file(path, "device")
    resolve = model.resolve
    seen = []

    def inspect(settings: Mapping[str, Any] | None = None) -> ResolvedCircuit:
        """Track the shared resolver's returned topology."""
        resolved = resolve(settings)
        seen.append(resolved)
        return resolved

    monkeypatch.setattr(model, "resolve", inspect)
    definition = model.instantiate()
    assert len(seen) == 1
    assert seen[0].instances[0].parameters["dc"] == 1
    assert definition.source_netlist.instances["device0"].settings["V"] == 1


def test_temperature_is_reserved_and_native_build_receives_selected_temperature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run temperature reparses cards without becoming a device parameter."""
    path = tmp_path / "temperature.lib"
    path.write_text("* card\n.subckt device p n\n.model vs vsource dc=1\nN1 p n vs\n.ends\n")
    model = LibraryModel.from_file(path, "device", temperature_c=27.0)
    default = model.resolve()
    hot = model.resolve({"temperature_c": 80.0})
    assert default.temperature_c == 27.0
    assert hot.temperature_c == 80.0
    assert hot.instances[0].parameters == default.instances[0].parameters == {"dc": 1.0}
    seen = []

    def build(resolved: ResolvedCircuit, **_options: Any) -> tuple[Any, dict, dict]:
        """Capture the temperature passed to native topology construction."""
        seen.append(resolved.temperature_c)
        return None, {}, {}

    monkeypatch.setattr("circulax.netlist_io.model.build_resolved", build)
    model.instantiate({"temperature_c": 80.0})
    model.instantiate()
    assert seen == [80.0, 27.0]
    assert len(model._libraries) == 2  # noqa: SLF001 -- temperature/corner cache isolation
    model.resolve({"temperature_c": 80.0})
    assert len(model._libraries) == 2  # noqa: SLF001 -- repeated runs reuse parsed cards


def test_base_library_temperature_override_reparses_available_sources(tmp_path: Path) -> None:
    """Preparsed registration supports overrides without mutating its defaults."""
    path = tmp_path / "temperature.lib"
    path.write_text("* card\n.subckt device p n\n.model vs vsource dc=1\nN1 p n vs\n.ends\n")
    library = Library.from_file(path)
    model = LibraryModel(library, "device")
    assert model.resolve({"temperature_c": 80}).temperature_c == 80
    assert library.temperature_c == model.resolve().temperature_c == 27


def test_temperature_override_without_source_provenance_is_explicitly_rejected() -> None:
    """In-memory cards cannot silently ignore an unsupported temperature change."""
    model = LibraryModel(Library(), "device")
    with pytest.raises(ValueError, match="source"):
        model.resolve({"temperature_c": 80.0})


def test_card_and_include_edits_refresh_resolved_values_and_source_digests(tmp_path: Path) -> None:
    """Source revisions invalidate parsed cards while unchanged sources reuse them."""
    include = tmp_path / "parameters.lib"
    include.write_text("* parameter\n.param base=2\n")
    card = tmp_path / "card.lib"
    template = (
        '* card\n.include "parameters.lib"\n.subckt device p n\n.param value={}\n'
        ".model vs vsource dc='value+base'\nN1 p n vs\n.ends\n"
    )
    card.write_text(template.format(1))
    model = LibraryModel.from_file(card, "device")
    first = model.resolve()
    digests = model.source_digests()
    cached = tuple(model._libraries.values())  # noqa: SLF001 -- verify unchanged parsed library reuse
    assert first.instances[0].parameters["dc"] == 3
    assert model.source_digests() == digests
    assert tuple(model._libraries.values()) == cached  # noqa: SLF001 -- no redundant reparsing
    assert {record["path"] for record in digests} == {str(card), str(include)}
    card.write_text(template.format(4))
    assert model.resolve().instances[0].parameters["dc"] == 6
    changed = model.source_digests()
    assert changed != digests
    include.write_text("* parameter\n.param base=5\n")
    assert model.resolve().instances[0].parameters["dc"] == 9
    assert model.source_digests() != changed


_CARD = "* card\n.subckt device p n\n.model vs vsource dc=1\nN1 p n vs\n.ends\n"


def _age(*paths: Path) -> None:
    """Make files look long-unchanged so the (mtime, size) fast path may trust them."""
    old = 1_000_000_000_000_000_000
    for path in paths:
        os.utime(path, ns=(old, old))


def test_native_artifact_edits_invalidate_cached_provisioning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A rebuilt artifact is re-provisioned; unchanged artifacts keep the cached modules."""
    card = tmp_path / "card.lib"
    card.write_text(_CARD)
    artifact = tmp_path / "native.osdi"
    artifact.write_bytes(b"first-build")
    model = LibraryModel.from_file(card, "device", osdi_modules=(artifact,))
    seen = []

    def build(_resolved: ResolvedCircuit, *, modules: Any = None, **_options: Any) -> tuple[Any, dict, dict]:
        """Record the provisioning cache offered to each build."""
        seen.append(modules)
        return None, {}, {"native": (artifact, {})}

    monkeypatch.setattr("circulax.netlist_io.model.build_resolved", build)
    model.instantiate()
    model.instantiate()
    assert seen[0] is None
    assert seen[1] is not None
    artifact.write_bytes(b"second-build")
    model.instantiate()
    assert seen[2] is None
    model.instantiate()
    assert seen[3] is not None


def test_native_module_search_paths_invalidate_cached_provisioning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A module that appears earlier on the load search path is noticed."""
    card = tmp_path / "card.lib"
    card.write_text(_CARD.replace("* card", '* card\n.hdl "native.osdi"'))
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    (second / "native.osdi").write_bytes(b"later")
    model = LibraryModel.from_file(card, "device", module_paths=(first, second))
    seen = []

    def build(_resolved: ResolvedCircuit, *, modules: Any = None, **_options: Any) -> tuple[Any, dict, dict]:
        """Record the provisioning cache offered to each build."""
        seen.append(modules)
        return None, {}, {"native": (second / "native.osdi", {})}

    monkeypatch.setattr("circulax.netlist_io.model.build_resolved", build)
    model.instantiate()
    model.instantiate()
    (first / "native.osdi").write_bytes(b"earlier")
    model.instantiate()
    assert [modules is None for modules in seen] == [True, False, True]


def test_deleted_include_is_a_netlist_error(tmp_path: Path) -> None:
    """Removing a dependency reports a clear error instead of a raw FileNotFoundError."""
    include = tmp_path / "parameters.lib"
    include.write_text("* parameter\n.param base=2\n")
    card = tmp_path / "card.lib"
    card.write_text('* card\n.include "parameters.lib"\n' + _CARD.split("\n", 1)[1])
    model = LibraryModel.from_file(card, "device")
    model.resolve()
    include.unlink()
    with pytest.raises(NetlistError, match="not found"):
        model.resolve()
    card.unlink()
    with pytest.raises(NetlistError, match="not found"):
        model.resolve()


def test_new_file_shadowing_an_earlier_include_path_is_noticed(tmp_path: Path) -> None:
    """A file created earlier on the include search order replaces the previously found one."""
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (second / "parameters.lib").write_text("* parameter\n.param base=2\n")
    card = tmp_path / "card.lib"
    card.write_text('* card\n.include "parameters.lib"\n.subckt device p n\n.model vs vsource dc=base\nN1 p n vs\n.ends\n')
    model = LibraryModel.from_file(card, "device", include_paths=(first, second))
    assert model.resolve().instances[0].parameters["dc"] == 2
    (first / "parameters.lib").write_text("* parameter\n.param base=7\n")
    assert model.resolve().instances[0].parameters["dc"] == 7


def test_unchanged_sources_are_not_reread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated calls stat dependencies and hash contents only when (mtime, size) change."""
    include = tmp_path / "parameters.lib"
    include.write_text("* parameter\n.param base=2\n")
    card = tmp_path / "card.lib"
    card.write_text('* card\n.include "parameters.lib"\n.subckt device p n\n.model vs vsource dc=base\nN1 p n vs\n.ends\n')
    _age(card, include)
    model = LibraryModel.from_file(card, "device")
    assert model.resolve().instances[0].parameters["dc"] == 2
    reads: list[Path] = []
    original = Path.read_bytes

    def counting(self: Path) -> bytes:
        """Count content reads."""
        reads.append(self)
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", counting)
    for _ in range(3):
        assert model.resolve().instances[0].parameters["dc"] == 2
    assert reads == []
    include.write_text("* parameter\n.param base=50\n")
    assert model.resolve().instances[0].parameters["dc"] == 50
    assert include in reads


def test_library_revisions_are_unique() -> None:
    """Provisioning caches are keyed by a counter that is never reused, not by object identity."""
    revisions = [Library().revision for _ in range(5)]
    assert revisions == sorted(set(revisions))


_TEMPERATURE_CARD = "* card\n.subckt device p n\n.param value=0\n.model vs vsource dc=value\nN1 p n vs\n.ends\n"


def test_temperature_setting_remains_visible_to_expressions(tmp_path: Path) -> None:
    """The run temperature selects the card corner and is still an expression input."""
    card = tmp_path / "card.lib"
    card.write_text(_TEMPERATURE_CARD)
    model = LibraryModel.from_file(card, "device", expressions={"value": "temperature_c * 2"})
    resolved = model.resolve({"temperature_c": 80.0})
    assert resolved.temperature_c == 80.0
    assert resolved.instances[0].parameters["dc"] == 160.0
    assert model.resolve({"temperature_c": 27.0}).instances[0].parameters["dc"] == 54.0


def test_temperature_setting_reaches_the_library_only_when_explicitly_mapped(tmp_path: Path) -> None:
    """An explicit params_map entry forwards it; otherwise it is not a library parameter."""
    card = tmp_path / "card.lib"
    card.write_text(_TEMPERATURE_CARD)
    mapped = LibraryModel.from_file(card, "device", params_map={"temperature_c": "value"})
    resolved = mapped.resolve({"temperature_c": 80.0})
    assert resolved.temperature_c == 80.0
    assert resolved.instances[0].parameters["dc"] == 80.0
    plain = LibraryModel.from_file(card, "device")
    assert plain.resolve({"temperature_c": 80.0}).instances[0].parameters["dc"] == 0
