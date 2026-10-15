"""Integration checks for ngspice raw vectors used by the parity harness."""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from benchmarks.ihp_parity.run import vacask_testbench
from benchmarks.utils.vacask_reference import run_ngspice, run_vacask


@pytest.mark.parametrize("analysis", ["op", "dc v1 0 1 0.25", "ac dec 3 1k 1meg", "tran 1n 10n"])
def test_ngspice_batch_raw_vectors(analysis: str) -> None:
    pytest.importorskip("InSpice")
    if shutil.which("ngspice") is None:
        pytest.skip("ngspice is not installed")
    deck = (
        "Resistive divider\nV1 in 0 DC 1 AC 1\nR1 in out 1k\nR2 out 0 1k\n"
        f".control\n{analysis}\nwrite op1.raw all\nquit\n.endc\n.end\n"
    )
    vectors = run_ngspice(deck).vectors
    np.testing.assert_allclose(vectors["out"], vectors["in"] / 2, atol=1e-12)
    np.testing.assert_allclose(vectors["v1:flow(br)"], -vectors["in"] / 2000, atol=1e-12)
    if analysis.startswith("dc"):
        np.testing.assert_allclose(vectors["vin"], np.linspace(0, 1, 5))
    elif analysis.startswith("ac"):
        assert len(vectors["frequency"]) == 10
        assert np.iscomplexobj(vectors["out"])
    elif analysis.startswith("tran"):
        assert vectors["time"][-1] == pytest.approx(1e-8)


def test_vacask_testbench_uses_converted_library_and_native_controls() -> None:
    text = (
        '.lib "/pdk/native models/corner.lib" mos_tt\nV1 in 0 PULSE(0 1 1n 1p 1p 2n 4n)\nX1 out in 0 0 device w=1u\nR1 in out 1k\n'
    )
    rendered = vacask_testbench(text, model_root=Path("/pdk/converted models"))
    assert 'include "/pdk/converted models/corner.lib" section=mos_tt' in rendered
    assert 'V1 (in 0) vs type="pulse" val0=0 val1=1 delay=1n rise=1p fall=1p width=2n period=4n' in rendered
    assert "X1 (out in 0 0) device w=1u" in rendered
    assert "R1 (in out) parity_r r=1k" in rendered


@pytest.mark.parametrize(("choose", "expected"), [(0, 1 / 3), (1, 0.4)])
def test_vacask_converted_wrapper(tmp_path: Path, choose: int, expected: float) -> None:
    """Check scoped converted includes, both branches and multiplicity."""
    pytest.importorskip("InSpice")
    binary = os.environ.get("VACASK_EXECUTABLE")
    modules = os.environ.get("VACASK_MODULE_PATH")
    if not binary or not modules:
        pytest.skip("set VACASK_EXECUTABLE and VACASK_MODULE_PATH for reference tests")
    module_path = Path(modules)
    (tmp_path / "local.lib").write_text("parameters base=1k\nmodel local sp_resistor\n")
    library = tmp_path / "wrapper.lib"
    library.write_text(
        'subckt wrapper(in out)\nparameters choose=0 m=1\ninclude "local.lib"\n'
        "n1 (in out) local r=base $mfactor=m\nr2 (out 0) local r=1k $mfactor=m\n"
        "@if (choose==0)\nr3 (out 0) local r=1k $mfactor=m\n@else\n"
        "r3 (out 0) local r=2k $mfactor=m\n@end\nends\n"
    )
    result = run_vacask(
        "Converted wrapper regression\nground 0\nmodel vs vsource\n"
        f'include "{library}"\nv1 (in 0) vs dc=1\nx1 (in out) wrapper choose={choose} m=2\n'
        'control\nabort always\noptions rawfile="binary"\nanalysis op1 op\nendc\n',
        binary=binary,
        module_paths=(module_path,),
        shared_library_paths=tuple(
            Path(path) for path in os.environ.get("VACASK_SHARED_LIBRARY_PATH", "").split(os.pathsep) if path
        ),
        osdi_modules=(module_path / "sp_resistor.osdi",),
    )
    np.testing.assert_allclose(result.vectors["out"], expected, atol=1e-12)
    np.testing.assert_allclose(result.vectors["v1:flow(br)"], -2 * (1 - expected) / 1000, atol=1e-12)
