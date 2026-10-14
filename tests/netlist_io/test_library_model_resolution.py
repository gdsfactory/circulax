"""Capability inspection elaborates cards without provisioning native models."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from circulax.netlist_io import Library, LibraryModel, ResolvedCircuit


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
    assert model.inspection_identity()["resolved"]["corner"] == "tt"
    assert model.inspection_identity(settings)["resolved"]["corner"] == "ff"
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


def test_card_and_include_edits_refresh_resolved_values_and_inspection_identity(tmp_path: Path) -> None:
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
    identity = model.inspection_identity()
    cached = tuple(model._libraries.values())  # noqa: SLF001 -- verify unchanged parsed library reuse
    assert first.instances[0].parameters["dc"] == 3
    assert model.inspection_identity() == identity
    assert tuple(model._libraries.values()) == cached  # noqa: SLF001 -- no redundant reparsing
    assert {record["path"] for record in identity["sources"]} == {str(card), str(include)}
    card.write_text(template.format(4))
    assert model.resolve().instances[0].parameters["dc"] == 6
    changed = model.inspection_identity()
    assert changed["fingerprint"] != identity["fingerprint"]
    include.write_text("* parameter\n.param base=5\n")
    assert model.resolve().instances[0].parameters["dc"] == 9
    assert model.inspection_identity()["fingerprint"] != changed["fingerprint"]


def test_inspection_identity_tracks_available_artifacts_without_native_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Artifact provenance is reported without proving ABI readiness or compiling."""
    card = tmp_path / "card.lib"
    card.write_text("* card\n.subckt device p n\n.model vs vsource dc=1\nN1 p n vs\n.ends\n")
    artifact = tmp_path / "native.osdi"
    artifact.write_bytes(b"provenance-only-not-a-native-module")
    model = LibraryModel.from_file(card, "device", osdi_modules=(artifact,))

    def forbidden(*_args: object, **_kwargs: object) -> None:
        """Discovery must not load, validate or provision native modules."""
        pytest.fail("identity inspection cannot load native artifacts")

    monkeypatch.setattr("circulax.netlist_io.model.build_resolved", forbidden)
    monkeypatch.setattr("circulax.netlist_io.osdi.module_metadata", forbidden)
    first = model.inspection_identity()
    artifact.write_bytes(b"changed-provenance-still-not-an-osdi-module")
    second = model.inspection_identity()
    assert first["fingerprint"] != second["fingerprint"]
    assert second["resolved"]["temperature_c"] == 27
    assert second["native_artifacts"][0]["path"] == str(artifact)
