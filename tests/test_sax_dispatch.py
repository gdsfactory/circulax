"""Tests for SAX dispatch — detection, equivalence with the nodal solve, and the API.

The equivalence tests are the load-bearing ones: dispatching to SAX is only
sound if the S-matrix it composes is the same one circulax's nodal solve
produces for the same netlist. Every structural test below is in service of
knowing *when* that swap is legitimate.
"""

import jax
import jax.numpy as jnp
import pytest
import sax
from sax.models import coupler, straight

from circulax import compile_circuit
from circulax.s_transforms import sax_component
from circulax.sax_dispatch import build_sax_circuit, check_sax_dispatch

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def mzi():
    """A Mach-Zehnder interferometer built entirely from SAX models."""
    net = {
        "instances": {
            "lft": {"component": "coupler", "settings": {"coupling": 0.5}},
            "top": {"component": "straight", "settings": {"length": 25.0, "loss_dB_cm": 1.0}},
            "btm": {"component": "straight", "settings": {"length": 15.0, "loss_dB_cm": 1.0}},
            "rgt": {"component": "coupler", "settings": {"coupling": 0.5}},
        },
        "connections": {
            "lft,out0": "top,in0",
            "lft,out1": "btm,in0",
            "top,out0": "rgt,in0",
            "btm,out0": "rgt,in1",
        },
        "ports": {"in0": "lft,in0", "in1": "lft,in1", "out0": "rgt,out0", "out1": "rgt,out1"},
    }
    return net, {"coupler": coupler, "straight": straight}


# --- Detection -------------------------------------------------------------


def test_all_sax_models_are_dispatchable(mzi) -> None:
    net, models = mzi
    verdict = check_sax_dispatch(net, models)
    assert verdict
    assert verdict.reasons == ()


def test_nonlinear_component_blocks_dispatch(mzi) -> None:
    """One non-SAX model makes the whole netlist a DAE system — no S-matrix to compose."""
    from circulax.components.electronic import Diode

    net, models = mzi
    net = {**net, "instances": {**net["instances"], "D1": {"component": "diode"}}}
    models = {**models, "diode": Diode}

    verdict = check_sax_dispatch(net, models)
    assert not verdict
    assert any("D1" in reason and "diode" in reason for reason in verdict.reasons)


def test_ground_blocks_dispatch(mzi) -> None:
    """A GND instance means a driven test bench, which SAX has no notion of."""
    net, models = mzi
    net = {**net, "instances": {**net["instances"], "GND": {"component": "ground"}}}
    models = {**models, "ground": lambda: 0}

    verdict = check_sax_dispatch(net, models)
    assert not verdict
    assert any("GND" in reason for reason in verdict.reasons)


def test_missing_ports_block_dispatch(mzi) -> None:
    net, models = mzi
    net = {k: v for k, v in net.items() if k != "ports"}

    verdict = check_sax_dispatch(net, models)
    assert not verdict
    assert any("ports" in reason for reason in verdict.reasons)


def test_lrc_netlist_is_not_dispatchable(simple_lrc_netlist) -> None:
    net, models = simple_lrc_netlist
    assert not check_sax_dispatch(net, models)


def test_unused_nonlinear_model_does_not_block(mzi) -> None:
    """Only instantiated models matter; an unused entry is not a blocker."""
    from circulax.components.electronic import Diode

    net, models = mzi
    assert check_sax_dispatch(net, {**models, "diode": Diode})


def test_wrapped_sax_component_is_recognised(mzi) -> None:
    """Models already normalised into CircuitComponent classes still dispatch.

    ``sax_component`` records the function it wrapped, so detection works
    whether it runs before or after the compiler has normalised the mapping.
    """
    net, _ = mzi
    wrapped = {
        "coupler": sax_component(coupler, name="coupler"),
        "straight": sax_component(straight, name="straight"),
    }
    assert wrapped["straight"]._sax_model_fn is straight
    assert check_sax_dispatch(net, wrapped)


def test_circuit_exposes_dispatch_verdict(mzi, simple_lrc_netlist) -> None:
    net, models = mzi
    assert compile_circuit(net, models, is_complex=True).sax_dispatch

    lrc_net, lrc_models = simple_lrc_netlist
    assert not compile_circuit(lrc_net, lrc_models).sax_dispatch


# --- Equivalence -----------------------------------------------------------


def _dense(s_dict, port_order):
    return jnp.array([[s_dict[(po, pi)] for pi in port_order] for po in port_order])


def test_sax_path_matches_sax_directly(mzi) -> None:
    """The dispatched result must be what a SAX user would have got unaided."""
    net, models = mzi
    circuit = compile_circuit(net, models, is_complex=True)

    reference, _ = sax.circuit(netlist=net, models=models)
    expected = reference(wl=1.55)
    got = circuit.sdict(wl=1.55)

    ports = ("in0", "in1", "out0", "out1")
    assert jnp.allclose(_dense(got, ports), _dense(expected, ports), atol=1e-12)


@pytest.mark.parametrize("wl", [1.5, 1.55, 1.6])
def test_sax_and_nodal_paths_agree(mzi, wl: float) -> None:
    """The whole premise of dispatch: both backends give the same S-matrix.

    If this ever fails, dispatch is not a shortcut but a behaviour change.
    """
    net, models = mzi
    circuit = compile_circuit(net, models, is_complex=True)

    ports = ("in0", "in1", "out0", "out1")
    via_sax = _dense(circuit.sdict(wl=wl, backend="sax"), ports)
    via_nodal = _dense(circuit.sdict(wl=wl, backend="nodal"), ports)

    assert jnp.allclose(via_sax, via_nodal, atol=1e-8)


def test_auto_backend_picks_sax(mzi) -> None:
    net, models = mzi
    circuit = compile_circuit(net, models, is_complex=True)

    ports = ("in0", "in1", "out0", "out1")
    auto = _dense(circuit.sdict(wl=1.55), ports)
    forced = _dense(circuit.sdict(wl=1.55, backend="sax"), ports)
    assert jnp.array_equal(auto, forced)


def test_parameter_override_reaches_the_sax_path(mzi) -> None:
    """Changing a swept parameter must change the answer, not be silently dropped."""
    net, models = mzi
    circuit = compile_circuit(net, models, is_complex=True)

    at_1550 = circuit.sdict(wl=1.55)[("out0", "in0")]
    at_1600 = circuit.sdict(wl=1.60)[("out0", "in0")]
    assert not jnp.allclose(at_1550, at_1600)


def test_sax_path_broadcasts_over_arrays(mzi) -> None:
    """SAX sweeps a wavelength array in one call; the nodal path cannot."""
    net, models = mzi
    circuit = compile_circuit(net, models, is_complex=True)

    wls = jnp.linspace(1.5, 1.6, 7)
    swept = circuit.sdict(wl=wls)[("out0", "in0")]
    assert swept.shape == (7,)

    pointwise = jnp.array([circuit.sdict(wl=float(w))[("out0", "in0")] for w in wls])
    assert jnp.allclose(swept, pointwise, atol=1e-12)


def test_smatrix_returns_dense_matrix_and_port_order(mzi) -> None:
    net, models = mzi
    circuit = compile_circuit(net, models, is_complex=True)

    s_matrix, port_order = circuit.smatrix(wl=1.55)
    assert set(port_order) == {"in0", "in1", "out0", "out1"}
    assert s_matrix.shape == (4, 4)

    s_dict = circuit.sdict(wl=1.55)
    i, j = port_order.index("out0"), port_order.index("in0")
    assert jnp.allclose(s_matrix[i, j], s_dict[("out0", "in0")])


def test_sdict_is_differentiable(mzi) -> None:
    """Dispatch must not cost the differentiability circulax exists for."""
    net, models = mzi
    circuit = compile_circuit(net, models, is_complex=True)

    def transmission(wl):
        return jnp.abs(circuit.sdict(wl=wl)[("out0", "in0")]) ** 2

    grad = jax.grad(transmission)(1.55)
    assert jnp.isfinite(grad)
    assert grad != 0.0


# --- API -------------------------------------------------------------------


def test_forcing_sax_on_unsupported_circuit_raises(simple_lrc_netlist) -> None:
    net, models = simple_lrc_netlist
    circuit = compile_circuit(net, models)

    with pytest.raises(ValueError, match="not SAX-dispatchable"):
        circuit.sdict(backend="sax")


def test_unknown_backend_raises(mzi) -> None:
    net, models = mzi
    circuit = compile_circuit(net, models, is_complex=True)

    with pytest.raises(ValueError, match="backend must be"):
        circuit.sdict(backend="fastest")


def test_build_sax_circuit_rejects_unsupported_netlist(simple_lrc_netlist) -> None:
    net, models = simple_lrc_netlist
    with pytest.raises(ValueError, match="not SAX-dispatchable"):
        build_sax_circuit(net, models)
