from pathlib import Path

import pytest

from circulax.netlist_io import Library, NetlistError


def test_nested_wrapper_keeps_geometry_and_scoped_model(tmp_path: Path) -> None:
    card = tmp_path / "device.lib"
    card.write_text("""parameters gain=2
subckt device(d g s b)
parameters w=1u l=0.13u ng=1 m=1
model core psp103va (type=1 toxo=2n*gain)
@if (ng==1)
  m1 (d g s b) core w=w l=l nf=ng mult=m ad=w*0.34u
@else
  m1 (d g s b) core w=w l=l nf=ng mult=m ad=w*0.38u
@end
ends
""")
    library = Library.from_file(card)
    resolved = library.instantiate("device", ("out", "gate", "0", "0"), {"w": 2e-6, "ng": 2, "m": 3})
    assert len(resolved.instances) == 1
    device = resolved.instances[0]
    assert device.nodes == ("out", "gate", "0", "0")
    assert device.module == "psp103va"
    assert device.parameters["w"] == 2e-6
    assert device.parameters["ad"] == pytest.approx(2e-6 * 0.38e-6)
    assert device.parameters["toxo"] == pytest.approx(4e-9)
    assert device.parameters["nf"] == 2
    assert device.parameters["mult"] == 3


def test_sections_shared_includes_and_local_nodes(tmp_path: Path) -> None:
    (tmp_path / "wrapper.lib").write_text("""subckt device(p n)
parameters r=10
model rm sp_resistor r=r*factor
r1 (p mid) rm
r2 (mid n) rm
ends
""")
    card = tmp_path / "corner.lib"
    card.write_text("""parameters common=1
section tt
parameters factor=1
include "wrapper.lib"
endsection
section ff
parameters factor=2
include "wrapper.lib"
endsection
""")
    library = Library.from_file(card, section="ff")
    resolved = library.instantiate("device", ("p", "0"), name="X1")
    assert [i.parameters["r"] for i in resolved.instances] == [20, 20]
    assert resolved.instances[0].nodes == ("p", "X1/mid")
    assert resolved.instances[1].nodes == ("X1/mid", "0")
    with pytest.raises(NetlistError, match="section"):
        Library.from_file(card, section="absent")


def test_expression_semantics_and_lazy_condition(tmp_path: Path) -> None:
    card = tmp_path / "model.lib"
    card.write_text("""parameters a=2 b=a*3
subckt device(p n)
parameters x=b
model rm sp_resistor r=(x>0?max(x,1k):missing)
r1 (p n) rm
ends
""")
    instance = Library.from_file(card).instantiate("device", ("p", "0"), {"x": 2000}).instances[0]
    assert instance.parameters["r"] == 2000


def test_parameter_cycles_unknown_settings_and_recursion(tmp_path: Path) -> None:
    card = tmp_path / "bad.lib"
    card.write_text("""subckt device(p n)
parameters a=b b=a
model rm sp_resistor r=a
r1 (p n) rm
ends
""")
    library = Library.from_file(card)
    with pytest.raises(NetlistError, match="cycle"):
        library.instantiate("device", ("p", "0"))
    with pytest.raises(NetlistError, match="unknown"):
        library.instantiate("device", ("p", "0"), {"typo": 1})
    card.write_text('include "bad.lib"\n')
    with pytest.raises(NetlistError, match="cycle"):
        Library.from_file(card)


def test_temperature_and_mfactor_are_not_python_eval(tmp_path: Path) -> None:
    card = tmp_path / "model.lib"
    card.write_text("""subckt device(p n)
parameters m=1
model rm sp_resistor r=($temp+273)/300
r1 (p n) rm $mfactor=m
ends
""")
    instance = Library.from_file(card, temperature_c=77).instantiate("device", ("p", "0"), {"m": 2}).instances[0]
    assert instance.parameters["r"] == pytest.approx(350 / 300)
    assert instance.parameters["$mfactor"] == 2


def test_invalid_syntax_and_unsupported_control_fail(tmp_path: Path) -> None:
    card = tmp_path / "bad.lib"
    card.write_text("parameters r=1+\n")
    with pytest.raises(NetlistError):
        Library.from_file(card)
    card.write_text("control\nanalysis op1 op\nendc\n")
    with pytest.raises(NetlistError, match="control"):
        Library.from_file(card)


def test_local_model_shadows_public_wrapper_name(tmp_path: Path) -> None:
    card = tmp_path / "varactor.lib"
    card.write_text("""subckt device(p n)
model device mosvar c=1p
m1 (p n) device
ends
""")
    assert Library.from_file(card).instantiate("device").instances[0].module == "mosvar"


def test_spectre_direct_and_case_sensitive_suffixes(tmp_path: Path) -> None:
    card = tmp_path / "spectre.lib"
    card.write_text("""subckt device(p n)
parameters a=1M b=1m
model rm resistor r=a*b
r1 (p n) rm
ends
""")
    assert Library.from_file(card, dialect="spectre").instantiate("device").instances[0].parameters["r"] == 1000


def test_nested_unsupported_statement_is_rejected(tmp_path: Path) -> None:
    card = tmp_path / "analysis.lib"
    card.write_text("subckt device(p n)\nac1 ac start=1 stop=1k\nends\n")
    with pytest.raises(NetlistError, match="unsupported"):
        Library.from_file(card).instantiate("device")
