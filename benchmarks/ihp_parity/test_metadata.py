"""Optional tests against a real IHP checkout; no geometry imports needed."""

import ast
import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def pdk() -> tuple[Path, ModuleType]:
    root = os.environ.get("IHP_PDK_ROOT")
    if not root:
        pytest.skip("set IHP_PDK_ROOT to test a real IHP checkout")
    parser = pytest.importorskip("netlist_parser")
    if not hasattr(parser, "parse_spectre"):
        pytest.skip("NetlistParse Python Spectre binding is required")
    root = Path(root)
    spec = importlib.util.spec_from_file_location("ihp_circulax_adapter", root / "ihp/models/circulax.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return root, adapter


def model_list(root: Path, filename: str, function: str) -> list[dict]:
    tree = ast.parse((root / "ihp/cells" / filename).read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function)
    assignment = next(
        node
        for node in ast.walk(definition)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.List)
        and any(
            isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant) and target.slice.value == "models"
            for target in node.targets
        )
    )
    return ast.literal_eval(assignment.value)


def test_nmos_metadata_resolves_geometry_and_pin_order(pdk: tuple[Path, ModuleType]) -> None:
    root, adapter = pdk
    models = adapter.with_circulax_models(model_list(root, "fet_transistors.py", "nmos_schematic"))
    for corner in ["mos_tt", "mos_ss", "mos_ff", "mos_sf", "mos_fs"]:
        resolved = adapter.resolve_component(
            models, {"width": 2, "length": 0.13, "nf": 2, "m": 3}, {"D": "out", "G": "in", "S": "0", "B": "0"}, corner=corner
        )
        assert resolved.temperature_c == 27.0
        device = resolved.instances[0]
        assert device.nodes == ("out", "in", "0", "0")
        assert device.parameters["w"] == 2e-6
        assert device.parameters["l"] == pytest.approx(0.13e-6)
        assert device.parameters["nf"] == 2
        assert device.parameters["mult"] == 3


def test_rf_capacitor_metadata_preserves_composite_topology(pdk: tuple[Path, ModuleType]) -> None:
    root, adapter = pdk
    models = adapter.with_circulax_models(model_list(root, "capacitors.py", "rfcmim_schematic"))
    resolved = adapter.resolve_component(models, {"width": 10, "length": 10}, {"PLUS": "p", "MINUS": "n", "BN": "0"})
    assert len(resolved.instances) == 12
    assert {device.module for device in resolved.instances} == {"sp_capacitor", "sp_inductor", "sp_resistor"}


def test_invalid_corner_fails_instead_of_falling_back(pdk: tuple[Path, ModuleType]) -> None:
    root, adapter = pdk
    models = adapter.with_circulax_models(model_list(root, "fet_transistors.py", "nmos_schematic"))
    with pytest.raises(ValueError, match="corner"):
        adapter.resolve_component(models, {}, {}, corner="typo")
