# Copyright (c) 2026 Circulax contributors
"""Native sample exports must work without the optional Verilog-A toolchain."""

import subprocess
import sys
from pathlib import Path


def test_native_components_import_without_verilog_a_runtime() -> None:
    # Test in a clean interpreter so another reference test cannot preload bosdi.
    program = """
import importlib.abc
import sys

class NoVerilogA(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"bosdi", "osdi_loader", "osdi_jax", "openvaf_py"}:
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, NoVerilogA())
from circulax import components
for name in (
    "BJT_PNP", "BJT_PNP_Dynamic", "R2CMC", "R2ETCMC", "R3_CMC", "JUNCAP200",
    "MOSVAR", "DIODE_CMC", "EKV26", "PMOSDynamic", "OpAmp", "Comparator",
    "Angelov", "AngelovGaN",
):
    model = getattr(components, name)()
    assert model.ports, name
assert not any(name == "bosdi" or name.startswith("bosdi.") for name in sys.modules)
"""
    result = subprocess.run(  # noqa: S603 -- fixed Python program, no shell
        [sys.executable, "-c", program],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
