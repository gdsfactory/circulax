"""Reusable schematic registrations preserve instance settings and leaf sharing."""

import os
from pathlib import Path
from types import SimpleNamespace

import kfnetlist as kfnl
import pytest

from circulax import compile_circuit
from circulax.netlist_io import Library, LibraryModel
from circulax.netlist_io.expressions import evaluate_source
from circulax.netlist_io.library import ResolvedCircuit, ResolvedInstance
from circulax.netlist_io.osdi import _provision_modules
from circulax.netlist_io.units import convert_unit


def test_explicit_units_and_current_expression_parser() -> None:
    """Unit declarations and expressions have stable numeric meanings.

    @tags circulax-simulation
    """
    assert convert_unit(2, "um", "m") == pytest.approx(2e-6)
    assert convert_unit(3, "um^2", "m^2") == pytest.approx(3e-12)
    assert evaluate_source("width * 1e-6", {"width": 2}) == pytest.approx(2e-6)
    with pytest.raises(ValueError, match="incompatible"):
        convert_unit(2, "um", "s")
    with pytest.raises(ValueError, match="unsupported"):
        convert_unit(2, "unknown", "m")
    assert convert_unit(2e-6, "m", "um") == pytest.approx(2)


def test_reusable_wrapper_binds_units_before_cards_and_shares_descriptor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Instance settings differ while descriptor identity remains shared.

    @tags circulax-simulation
    """
    card = tmp_path / "model.lib"
    card.write_text("""* card
.subckt device p n
.param w=1u
.model core native w=w r='w*1e6'
N1 p mid core
N2 mid n core
.ends
""")
    descriptor = SimpleNamespace(ports=("p0", "p1"))
    creations = []

    def create_descriptor(*args: object, **_kwargs: object) -> SimpleNamespace:
        """Count shared native definitions.

        @tags circulax-simulation
        """
        creations.append(args)
        return descriptor

    monkeypatch.setattr(
        "circulax.netlist_io.osdi._provision_modules",
        lambda *_a: {"native": (Path("native.osdi"), {"w": "w", "r": "r"})},
    )
    monkeypatch.setattr("circulax.netlist_io.osdi.osdi_component", create_descriptor)
    model = LibraryModel(
        Library.from_file(card),
        "device",
        params_map={"width": "w"},
        defaults={"width": 1},
        parameter_units={"width": ("um", "m")},
    )
    first = model.instantiate({"width": 2})
    second = model.instantiate({"width": 4})
    assert not callable(model)
    assert len(creations) == 1
    assert first.source_models["native"] is second.source_models["native"] is descriptor
    assert [i.settings["w"] for i in first.source_netlist.instances.values() if i.component == "native"] == [2e-6, 2e-6]
    assert [i.settings["r"] for i in second.source_netlist.instances.values() if i.component == "native"] == [4, 4]
    assert {p.name for p in first.source_netlist.ports} == {"p", "n"}
    default = model.instantiate()
    assert [i.settings["r"] for i in default.source_netlist.instances.values() if i.component == "native"] == [1, 1]


def test_lazy_registration_selects_corner_and_expression_targets(tmp_path: Path) -> None:
    """Select per-instance corners without forwarding source-only settings.

    @tags circulax-simulation
    """
    path = tmp_path / "corners.lib"
    model = LibraryModel.from_file(
        path,
        "device",
        section="tt",
        sections=("tt", "ff"),
        defaults={"amplitude": 1},
        expressions={"value": "amplitude * 2"},
    )
    # Registration performs no parsing or native provisioning.
    path.write_text("""* cornered wrapper
.lib tt
.param factor=1
.endl tt
.lib ff
.param factor=3
.endl ff
.subckt device p n
.param value=1
.model vs vsource dc='value*factor'
N1 p n vs
.ends
""")
    first = model.instantiate()
    second = model.instantiate({"corner": "ff", "amplitude": 2, "unused_factory_setting": 99})
    assert first.source_netlist.instances["device0"].settings["V"] == 2
    assert second.source_netlist.instances["device0"].settings["V"] == 12
    assert first.source_models["vsource"] is second.source_models["vsource"]
    with pytest.raises(ValueError, match="unsupported corner"):
        model.instantiate({"corner": "bad"})


def test_categorical_parameters_are_bound_per_instance(tmp_path: Path) -> None:
    """Translate schematic enum values before evaluating library parameters.

    @tags circulax-simulation
    """
    path = tmp_path / "shape.lib"
    path.write_text("* model\n.subckt device p n\n.param value=1\n.model vs vsource dc=value\nN1 p n vs\n.ends\n")
    model = LibraryModel.from_file(
        path,
        "device",
        defaults={"shape": "octagon"},
        parameter_values={"shape": {"octagon": 0, "square": 1, "circle": 2}},
        expressions={"value": "shape + 1"},
    )
    assert model.instantiate().source_netlist.instances["device0"].settings["V"] == 1
    assert model.instantiate({"shape": "circle"}).source_netlist.instances["device0"].settings["V"] == 3
    with pytest.raises(ValueError, match="unsupported value"):
        model.instantiate({"shape": "triangle"})


def test_rf_wrapper_preserves_inline_inductor_and_internal_nodes(tmp_path: Path) -> None:
    """RF wrappers preserve native R/L/C leaves and evaluated instance values.

    @tags circulax-simulation
    """
    path = tmp_path / "rf.lib"
    path.write_text("""* RF wrapper
.subckt rf p n
.param scale=2
L1 p mid L='scale*1n'
R1 mid inner R=10
C1 inner n C='scale*1p'
.ends
""")
    resolved = Library.from_file(path).instantiate("rf", settings={"scale": 3})
    assert [i.module for i in resolved.instances] == ["l", "r", "c"]
    assert resolved.instances[0].parameters["L"] == pytest.approx(3e-9)
    assert resolved.instances[2].parameters["C"] == pytest.approx(3e-12)
    assert resolved.instances[0].nodes[1] == resolved.instances[1].nodes[0]
    assert resolved.instances[1].nodes[1] == resolved.instances[2].nodes[0]


def test_original_latin1_library_comments_and_utf8_includes(tmp_path: Path) -> None:
    """Read original card encodings without altering parameter semantics.

    @tags circulax-simulation
    """
    included = tmp_path / "original.lib"
    included.write_bytes(
        (
            "* Original model width in µm\n.subckt device p n\n.param width=2u\n"
            ".model vs vsource dc='width*1e6'\nN1 p n vs\n.ends\n"
        ).encode("latin-1"),
    )
    entry = tmp_path / "entry.lib"
    entry.write_text('* UTF-8 µm comment\n.include "original.lib"\n', encoding="utf-8")
    resolved = Library.from_file(entry).instantiate("device")
    assert resolved.instances[0].parameters["dc"] == pytest.approx(2)


def test_compile_wrapper_instances_flattens_to_shared_batched_leaves(tmp_path: Path) -> None:
    """Canonical flatten preserves topology, aliases, settings and batching.

    @tags circulax-simulation
    """
    card = tmp_path / "source.lib"
    card.write_text("""// series source
subckt source(p n)
parameters amplitude=1
A (p mid) vsource dc=amplitude
B (mid n) vsource dc=amplitude*2
ends source
""")
    model = LibraryModel(Library.from_file(card, dialect="spectre"), "source", port_map={"P": "p", "N": "n"})
    parent = kfnl.Netlist()
    parent.create_inst(name="first", kcl="", component="source", settings={"amplitude": 1})
    parent.create_inst(name="second", kcl="", component="source", settings={"amplitude": 3})
    parent.create_inst(name="GND", kcl="", component="ground")
    parent.create_port("out")
    parent.create_net(kfnl.NetlistPort("out"), kfnl.PortRef(instance="first", port="P"))
    parent.create_net(kfnl.PortRef(instance="first", port="N"), kfnl.PortRef(instance="second", port="P"))
    parent.create_net(kfnl.PortRef(instance="second", port="N"), kfnl.PortRef(instance="GND", port="p1"))
    before = parent.to_dict()
    circuit = compile_circuit(parent, {"source": model}, backend="dense", g_leak=0)
    assert parent.to_dict() == before
    leaves = [i for i in circuit.source_netlist.instances.values() if i.component == "vsource"]
    assert sorted(i.settings["V"] for i in leaves) == [1, 2, 3, 6]
    assert len(circuit.groups) == 1
    assert next(iter(circuit.groups.values())).var_indices.shape[0] == 4
    assert float(circuit.port(circuit.dc(), "out")) == pytest.approx(12)


def test_supplied_modules_skip_unneeded_library_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compatible supplied modules avoid provisioning unrelated library loads.

    @tags circulax-simulation
    """
    supplied = tmp_path / "compatible.osdi"
    monkeypatch.setattr("circulax.netlist_io.osdi.module_metadata", lambda _path: ("sp_resistor", {"r": "r"}))
    resolved = ResolvedCircuit(
        [ResolvedInstance("r", "r", ("p", "n"), {"r": 1})],
        [("unavailable.osdi", tmp_path)],
    )
    modules = _provision_modules(resolved, (), (supplied,), None, None)
    assert modules["sp_resistor"][0] == supplied.resolve()


def test_native_library_registration_provisions_and_batches(tmp_path: Path) -> None:
    """Use the complete native provisioning path with a compatible binary.

    @tags circulax-simulation
    """
    binary = os.environ.get("CIRCULAX_TEST_OSDI_RESISTOR")
    if not binary:
        pytest.skip("set CIRCULAX_TEST_OSDI_RESISTOR to a compatible sp_resistor binary")
    card = tmp_path / "resistor.lib"
    card.write_text("""* shared model
.subckt wrapper p n
.param value=100
.model shared sp_resistor r=value
N1 p mid shared
N2 mid n shared
.ends
""")
    model = LibraryModel.from_file(card, "wrapper", osdi_modules=(Path(binary),))
    circuit = compile_circuit(
        {
            "instances": {
                "one": {"component": "wrapper", "settings": {"value": 100}},
                "two": {"component": "wrapper", "settings": {"value": 200}},
                "GND": {"component": "ground"},
            },
            "connections": {"one,n": "two,p", "two,n": "GND,p1"},
            "ports": {"out": "one,p"},
        },
        {"wrapper": model},
        backend="dense",
        is_complex=False,
    )
    assert len(circuit.groups) == 1
    group = next(iter(circuit.groups.values()))
    assert group.params.shape[0] == 4
    assert len(group.index_map) == 4
    leaves = [i for i in circuit.source_netlist.instances.values() if i.component == "sp_resistor"]
    assert sorted(i.settings["resistance"] for i in leaves) == [100, 100, 200, 200]
    assert float(circuit.port(circuit.dc(), "out")) == pytest.approx(0)
