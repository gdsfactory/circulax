from pathlib import Path

import pytest

from circulax.netlist_io import Library, NetlistError


def test_nested_wrapper_keeps_geometry_and_scoped_model(tmp_path: Path) -> None:
    card = tmp_path / "device.lib"
    card.write_text("""* device
.param gain=2
.subckt device d g s b
.param w=1u l=0.13u ng=1 m=1
.model core psp103va type=1 toxo='2n*gain'
.if (ng==1)
  N1 d g s b core w=w l=l nf=ng mult=m ad='w*0.34u'
.else
  N1 d g s b core w=w l=l nf=ng mult=m ad='w*0.38u'
.endif
.ends
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
    (tmp_path / "wrapper.lib").write_text("""* wrapper
.subckt device p n
.param r=10
.model rm sp_resistor r='r*factor'
N1 p mid rm
N2 mid n rm
.ends
""")
    card = tmp_path / "corner.lib"
    card.write_text("""* corner
.param common=1
.lib tt
.param factor=1
.include "wrapper.lib"
.endl tt
.lib ff
.param factor=2
.include "wrapper.lib"
.endl ff
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
    card.write_text("""* model
.param a=2 b='a*3'
.subckt device p n
.param x=b
.model rm sp_resistor r='(x>0) ? max(x,1k) : missing'
N1 p n rm
.ends
""")
    instance = Library.from_file(card).instantiate("device", ("p", "0"), {"x": 2000}).instances[0]
    assert instance.parameters["r"] == 2000


def test_parameter_cycles_unknown_settings_and_recursion(tmp_path: Path) -> None:
    card = tmp_path / "bad.lib"
    card.write_text("""* bad
.subckt device p n
.param a=b b=a
.model rm sp_resistor r=a
N1 p n rm
.ends
""")
    library = Library.from_file(card)
    with pytest.raises(NetlistError, match="cycle"):
        library.instantiate("device", ("p", "0"))
    with pytest.raises(NetlistError, match="unknown"):
        library.instantiate("device", ("p", "0"), {"typo": 1})
    card.write_text('* bad\n.include "bad.lib"\n')
    with pytest.raises(NetlistError, match="cycle"):
        Library.from_file(card)


def test_temperature_and_mfactor_are_not_python_eval(tmp_path: Path) -> None:
    card = tmp_path / "model.lib"
    card.write_text("""* model
.subckt device p n
.param m=1
.model rm sp_resistor r='m*2'
N1 p n rm
.ends
""")
    library = Library.from_file(card, temperature_c=77)
    assert library.scope.bindings["$temp"] == 77
    instance = library.instantiate("device", ("p", "0"), {"m": 2}).instances[0]
    assert instance.parameters["r"] == pytest.approx(4)


def test_invalid_syntax_and_unsupported_control_fail(tmp_path: Path) -> None:
    card = tmp_path / "bad.lib"
    card.write_text("* bad\n.param r=1+\n")
    with pytest.raises(NetlistError):
        Library.from_file(card)
    card.write_text("* bad\ncontrol\nanalysis op1 op\nendc\n")
    with pytest.raises(NetlistError, match="control"):
        Library.from_file(card)


def test_local_model_shadows_public_wrapper_name(tmp_path: Path) -> None:
    card = tmp_path / "varactor.lib"
    card.write_text("""* varactor
.subckt device p n
.model device mosvar c=1p
N1 p n device
.ends
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
    card.write_text("* analysis\n.subckt device p n\n.ac dec 10 1 1meg\n.ends\n")
    with pytest.raises(NetlistError, match="unsupported"):
        Library.from_file(card).instantiate("device")


def test_spice_quoted_model_expressions(tmp_path: Path) -> None:
    """SPICE primes are expressions, including nested scope references and SI units."""
    card = tmp_path / "quoted.lib"
    card.write_text("""* Quoted SPICE model card
.param flag=1 scale='(flag==0)*2 + (flag==1)*3'
.model rm r r='scale*1k'
""")
    from circulax.netlist_io.expressions import evaluate
    from circulax.netlist_io.syntax import parameters

    library = Library.from_file(card)
    model, scope = library.frame.models["rm"]
    assert evaluate(parameters(model)["r"], scope) == 3000


def test_literal_parameter_names_are_preserved(tmp_path: Path) -> None:
    card = tmp_path / "names.lib"
    card.write_text("* names\n.model rm sp_resistor r=1k __cx_mfactor=7\nN1 p 0 rm\n")
    instance = Library.from_file(card).resolve().instances[0]
    assert instance.parameters == {"r": 1000, "__cx_mfactor": 7}
