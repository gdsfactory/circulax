# Copyright (c) 2026 Circulax contributors
"""Single-source device prose, deterministic generation and executable examples."""

from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.device_docs import (
    GUIDE,
    NOTEBOOK,
    Device,
    collect_devices,
    execute_examples,
    generate,
    make_notebook,
    notebook_matches,
)

ROOT = Path(__file__).resolve().parents[1]
DEVICES = collect_devices()


@pytest.fixture
def source_tree(tmp_path: Path) -> Path:
    """A source-only tree whose module would fail if imported during generation."""
    directory = tmp_path / "circulax/components"
    directory.mkdir(parents=True)
    reference = tmp_path / "docs/components/model.md"
    reference.parent.mkdir(parents=True)
    reference.write_text("# Model\n", encoding="utf-8")
    (directory / "sample.py").write_text(
        '''raise RuntimeError("must not import")

def Sample():
    """An example with Unicode: sqrt(W), π.

    Device group:
        Photonic / Test

    Device reference:
        model.md

    Examples:
        ```python
        assert 2 + 2 == 4
        ```
    """
''',
        encoding="utf-8",
    )
    return tmp_path


def test_static_generation_and_repeatability(source_tree: Path) -> None:
    generate(source_tree)
    generate(source_tree, check=True)
    initial = (source_tree / NOTEBOOK).read_bytes()
    generate(source_tree)
    assert (source_tree / NOTEBOOK).read_bytes() == initial
    assert "An example with Unicode" in (source_tree / GUIDE).read_text()


def test_source_change_rejects_stale_docs_and_discards_outputs(source_tree: Path) -> None:
    generate(source_tree)
    path = source_tree / NOTEBOOK
    notebook = json.loads(path.read_text())
    notebook["cells"][-1]["outputs"] = [{"output_type": "stream", "name": "stdout", "text": "old output"}]
    notebook["cells"][-1]["execution_count"] = 1
    path.write_text(json.dumps(notebook))
    generate(source_tree)  # Kernel outputs alone do not invalidate source cells.
    generate(source_tree, check=True)
    source = source_tree / "circulax/components/sample.py"
    source.write_text(source.read_text().replace("2 + 2 == 4", "3 + 3 == 6"))
    with pytest.raises(ValueError, match="Stale device documentation"):
        generate(source_tree, check=True)
    generate(source_tree)
    regenerated = json.loads(path.read_text())
    assert regenerated["cells"][-1]["outputs"] == []
    assert "old output" not in (source_tree / "docs/examples/device_guide.md").read_text()


def test_notebook_execution_metadata_allowed_but_source_edits_rejected() -> None:
    expected = make_notebook(DEVICES[:2])
    actual = json.loads(json.dumps(expected))
    actual["metadata"]["papermill"] = {"duration": 1.0}
    actual["cells"][0]["metadata"]["execution"] = {"time": "today"}
    assert notebook_matches(actual, expected)
    actual["cells"][0]["source"] += "Edited source"
    assert not notebook_matches(actual, expected)


@pytest.mark.parametrize("replacement", ["    Examples:\n", "    Device reference:\n"])
def test_missing_sections_report_source_location(source_tree: Path, replacement: str) -> None:
    source = source_tree / "circulax/components/sample.py"
    source.write_text(source.read_text().replace(replacement, replacement.replace(":", " removed:")))
    with pytest.raises(ValueError, match=r"sample.py:.*Sample"):
        collect_devices(source_tree)


def test_examples_are_isolated_and_errors_identify_device() -> None:
    device = Device("Sample", "example", "Photonic / Test", "model.md", "Example", ("x = 1\n", "assert x == 1\n"), "sample.py", 4)
    with pytest.raises(RuntimeError, match="Sample example 2"):
        execute_examples([device])
    assert execute_examples([replace(device, examples=("assert 1 + 1 == 2\n",))]) == 1


def test_photonic_component_coverage_and_runtime_docstring() -> None:
    names = {device.name for device in DEVICES}
    modules = (
        "optical_passive",
        "photonic",
        "phase_shifters",
        "modulators",
        "resonators",
        "optical_filters",
        "photodetectors",
        "thermo_optic",
    )
    expected = set()
    for module in modules:
        tree = ast.parse((ROOT / f"circulax/components/{module}.py").read_text())
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                continue
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                if isinstance(target, ast.Name) and target.id in {"component", "source", "fdomain_component"}:
                    expected.add(node.name)
    assert expected <= names
    from circulax.components import OpticalFilter

    assert "Examples:" in OpticalFilter.__doc__
    assert "Device group:" in OpticalFilter.__doc__


def test_generated_device_artifacts_current() -> None:
    generate(check=True)


@pytest.mark.parametrize("device", DEVICES, ids=lambda device: device.name)
def test_device_docstring_example(device: Device) -> None:
    import importlib

    runtime_object = getattr(importlib.import_module(device.module), device.name)
    assert "Device group:" in runtime_object.__doc__
    assert "Examples:" in runtime_object.__doc__
    assert execute_examples([device]) == len(device.examples)


def test_physics_only_change_invalidates_saved_outputs(source_tree: Path) -> None:
    generate(source_tree)
    path = source_tree / NOTEBOOK
    saved = json.loads(path.read_text())
    saved["cells"][-1]["outputs"] = [{"output_type": "stream", "name": "stdout", "text": "old physics"}]
    path.write_text(json.dumps(saved))
    source = source_tree / "circulax/components/sample.py"
    source.write_text(source.read_text() + "\nPHYSICS_COEFFICIENT = 2.0\n")
    with pytest.raises(ValueError, match="Stale device documentation"):
        generate(source_tree, check=True)
    generate(source_tree)
    assert json.loads(path.read_text())["cells"][-1]["outputs"] == []


def test_mkdocs_hook_renders_current_source(monkeypatch: pytest.MonkeyPatch) -> None:
    import contextlib
    import io
    import runpy
    import sys
    from collections.abc import Iterator
    from types import SimpleNamespace

    from scripts.device_docs import render_guide

    rendered = {}

    @contextlib.contextmanager
    def virtual_open(path: Path, mode: str = "w") -> Iterator[io.BytesIO | io.StringIO]:
        buffer = io.BytesIO() if "b" in mode else io.StringIO()
        yield buffer
        rendered[Path(path).as_posix()] = buffer.getvalue()

    monkeypatch.setitem(
        sys.modules,
        "mkdocs_gen_files",
        SimpleNamespace(
            open=virtual_open,
            set_edit_path=lambda *_args: None,
        ),
    )
    runpy.run_path(str(ROOT / "docs/gen_doc_stubs.py"))
    assert rendered["components/devices.md"] == render_guide(collect_devices())
    assert "examples/device_guide.md" in rendered
    assert "references/components/optical_filters.md" in rendered


def test_library_revision_normalizes_checkout_line_endings(source_tree: Path) -> None:
    before = collect_devices(source_tree)[0].revision
    path = source_tree / "circulax/components/sample.py"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert collect_devices(source_tree)[0].revision == before


@pytest.mark.parametrize(
    "name",
    [
        "Angelov",
        "AngelovGaN",
        "EKV26",
        "JUNCAP200",
        "DIODE_CMC",
        "MOSVAR",
        "R2CMC",
        "R2ETCMC",
        "R3_CMC",
    ],
)
def test_va_models_cite_implementation_and_validation(name: str) -> None:
    import re

    device = next(d for d in DEVICES if d.name == name)
    assert "**Implementation source:**" in device.prose
    assert "**Validation reference:**" in device.prose
    citations = re.findall(r"https://github.com/dwarning/VA-Models/blob/[0-9a-f]{40}/code/[^)\s]+", device.prose)
    assert len(citations) >= 2
    assert any((ROOT / "tests/data/va_models").rglob(Path(citations[0]).name))
    assert "not measured-device calibration" in device.prose


def test_validation_citation_does_not_imply_a_port(source_tree: Path) -> None:
    from scripts.device_docs import render_guide

    source = source_tree / "circulax/components/sample.py"
    text = source.read_text(encoding="utf-8").replace(
        "    Device group:",
        "    **Validation reference:** Independent comparison against a pinned Verilog-A source.\n\n    Device group:",
    )
    source.write_text(text, encoding="utf-8")
    rendered = render_guide(collect_devices(source_tree))
    assert "**Validation reference:**" in rendered
    assert "**Implementation source:**" not in rendered
