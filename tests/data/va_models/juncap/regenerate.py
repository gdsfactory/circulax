"""Regenerate the native model; requires bosdi VA lowering and openvaf-r.

Validated with bosdi source 159907ddf29a0225614cf1b7999214401b0b5a2d
(unoptimized MIR path), not released bosdi 0.1.8.
The generated file itself has no bosdi or OpenVAF runtime dependency.
"""

# ruff: noqa: CPY001

from contextlib import chdir
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

from bosdi.va import lower
from bosdi.va.emitter import _live_init_cache_slots, emit_source
from bosdi.va.ir_client import compile_va_unopt
from bosdi.va.va_defaults import parse_va_defaults_expanded

SOURCE = Path(__file__).with_name("juncap200.va").resolve()
OUTPUT = Path(__file__).resolve().parents[4] / "circulax/components/juncap.py"
HEADER = '''"""Native JAX JUNCAP200 200.6.2 junction current and charge equations.

Generated from VA-Models commit 6099b4e12e6f66ff530f59b8d01fb909094be85a,
code/psp103/vacode/juncap200.va and its includes. Original equations copyright
NXP Semiconductors, CEA-Leti, Delft University of Technology and Arizona State
University; see retained source notices in tests/data/va_models/juncap.

Regenerate using tests/data/va_models/juncap/regenerate.py. Both full and
express branches remain selectable; LEVEL is fixed at 200. Noise and operating
point metadata are excluded. No OSDI or bosdi runtime is required.
"""'''


def main() -> None:
    defaults = parse_va_defaults_expanded(SOURCE)
    with TemporaryDirectory(prefix="juncap_mir_") as scratch:
        for dependency in SOURCE.parent.iterdir():
            if dependency.suffix in {".va", ".include", ".h", ".vams"}:
                copy2(dependency, Path(scratch) / dependency.name)
        with chdir(scratch):
            dump = compile_va_unopt(SOURCE.name)
    device = lower(
        dump.modules[0],
        va_defaults=defaults,
        collapse_nodes=True,
        static_params={"LEVEL": 200},
        class_name="JUNCAP200",
    )
    # Let Circulax/JAX differentiate F,Q; avoid the optional bosdi decorator.
    device.jacobian_resist = {}
    device.jacobian_react = {}
    # Compact references before emitting setup and eval: the older ordinary
    # component emitter otherwise used pre-compaction lookup indices.
    device.init_cache_refs = _live_init_cache_slots(device)
    source = emit_source([device])
    lint = "\n\n# ruff: noqa: ANN001, ANN002, ANN003, ANN201, ANN202, CPY001, E501, FBT003, PLR0915, PLR0917\n"
    source = HEADER + lint + source[source.index("\n\nfrom __future__") :]
    OUTPUT.write_text(source)


if __name__ == "__main__":
    main()
