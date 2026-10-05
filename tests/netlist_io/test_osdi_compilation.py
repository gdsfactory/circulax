"""Optional native tests using a configured VACASK module installation."""

import os
from pathlib import Path

import pytest

from circulax.netlist_io import Library


def test_card_aliases_share_an_osdi_batch(tmp_path: Path) -> None:
    module_path = os.environ.get("VACASK_MODULE_PATH")
    if not module_path:
        pytest.skip("set VACASK_MODULE_PATH for native OSDI compilation tests")
    pytest.importorskip("bosdi")
    card = tmp_path / "aliases.lib"
    card.write_text(""".model first_r r
.model second_r r
V1 in 0 DC 1
R1 in 0 first_r r=1k
R2 in 0 second_r r=2k
""")
    circuit = Library.from_file(card).resolve().compile(osdi_modules=(Path(module_path) / "spice/resistor.osdi",))
    group = next(group for group in circuit.groups.values() if group.name.startswith("sp_resistor"))
    assert group.params.shape[0] == 2
    assert len(group.index_map) == 2
    assert float(circuit.port(circuit.dc(), "in")) == pytest.approx(1.0)


def test_invalid_temperature_is_rejected(tmp_path: Path) -> None:
    pytest.importorskip("bosdi")
    card = tmp_path / "empty.lib"
    card.write_text(".param value=1\n")
    for temperature in [-273.15, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="temperature"):
            Library.from_file(card, temperature_c=temperature)


def test_native_device_multiplier_is_forwarded(tmp_path: Path) -> None:
    module_path = os.environ.get("VACASK_MODULE_PATH")
    if not module_path:
        pytest.skip("set VACASK_MODULE_PATH for native OSDI compilation tests")
    pytest.importorskip("bosdi")
    card = tmp_path / "multiplier.lib"
    card.write_text(".model rm r r=1k\nV1 out 0 1\nN1 out 0 rm m=3\n")
    circuit = Library.from_file(card).resolve().compile(osdi_modules=(Path(module_path) / "spice/resistor.osdi",))
    source = next(group for group in circuit.groups.values() if group.name == "vsource")
    current = circuit.dc()[source.var_indices[source.index_map["device0"], -1]]
    assert float(current) == pytest.approx(-0.003)
