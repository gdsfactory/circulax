"""Tests for :func:`circulax.s_transforms.sax_component`.

These tests specifically cover the signature-threading path — the decorator
must forward SAX parameter defaults through the wrapper into the resulting
``CircuitComponent`` subclass, which was previously broken because the
physics wrapper's ``__signature__`` did not include the reserved ``signals``
and ``s`` arguments that ``base_component._build_component`` requires.
"""

import jax
import jax.numpy as jnp
import pytest
from sax.models import straight

from circulax.components.base_component import CircuitComponent
from circulax.s_transforms import s_to_y, sax_component

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


@pytest.fixture(scope="module")
def sax_straight_cls() -> type[CircuitComponent]:
    """Wrap ``sax.models.straight`` once per module."""
    return sax_component(straight)


def test_sax_component_returns_circuit_component(sax_straight_cls: type[CircuitComponent]) -> None:
    assert issubclass(sax_straight_cls, CircuitComponent)
    assert sax_straight_cls.__name__ == "straight"


def test_sax_component_ports_match_sax(sax_straight_cls: type[CircuitComponent]) -> None:
    # sax.models.straight returns an S-dict over ports ('in0', 'out0').
    assert set(sax_straight_cls.ports) == {"in0", "out0"}


def test_sax_component_defaults_are_threaded(sax_straight_cls: type[CircuitComponent]) -> None:
    """Every SAX parameter must surface on the class with its SAX default."""
    expected = {
        "wl": 1.55,
        "wl0": 1.55,
        "neff": 2.34,
        "ng": 3.4,
        "length": 10.0,
        "loss_dB_cm": 0.0,
    }
    inst = sax_straight_cls()
    for name, default in expected.items():
        assert hasattr(inst, name), f"missing attribute {name!r}"
        assert jnp.allclose(getattr(inst, name), default), f"{name} default mismatch"


def test_sax_component_override_param(sax_straight_cls: type[CircuitComponent]) -> None:
    inst = sax_straight_cls(length=42.0, neff=2.5)
    assert jnp.allclose(inst.length, 42.0)
    assert jnp.allclose(inst.neff, 2.5)
    # Untouched params keep their defaults.
    assert jnp.allclose(inst.ng, 3.4)


def test_sax_component_signature_is_well_formed(sax_straight_cls: type[CircuitComponent]) -> None:
    """The class must expose every SAX parameter as an attribute with a default.

    That is the contract base_component._build_component relies on when it
    inspects the wrapper's signature during decoration.
    """
    for name in ("wl", "wl0", "neff", "ng", "length", "loss_dB_cm"):
        assert hasattr(sax_straight_cls, name), f"class missing default for {name!r}"


def test_sax_component_physics_matches_s_to_y(sax_straight_cls: type[CircuitComponent]) -> None:
    """The component's port currents must equal ``Y @ V`` with ``Y`` derived from the underlying SAX model."""
    from sax import sdense

    params = {"wl": 1.55, "wl0": 1.55, "neff": 2.34, "ng": 3.4, "length": 25.0, "loss_dB_cm": 0.5}
    inst = sax_straight_cls(**params)

    s_dict = straight(**params)
    s_matrix, port_order = sdense(s_dict)
    y_ref = s_to_y(s_matrix)

    v_by_port = {"in0": 1.0 + 0.2j, "out0": -0.3 + 0.1j}
    v_vec = jnp.array([v_by_port[p] for p in port_order], dtype=jnp.complex128)
    i_ref = y_ref @ v_vec
    expected = {p: i_ref[k] for k, p in enumerate(port_order)}

    f_dict, q_dict = inst(**v_by_port)
    assert q_dict == {}
    for p in port_order:
        assert jnp.allclose(f_dict[p], expected[p], atol=1e-10)


def test_sax_component_handles_param_without_default() -> None:
    """Parameters missing a SAX default must be filled with 1.0 by the decorator."""

    def custom_coupler(coupling: float, loss: float = 0.0) -> dict:  # no default for ``coupling``
        t = jnp.sqrt(jnp.maximum(1.0 - coupling - loss, 0.0))
        c = 1j * jnp.sqrt(coupling)
        return {("p1", "p1"): 0.0 + 0j, ("p1", "p2"): t + c, ("p2", "p1"): t + c, ("p2", "p2"): 0.0 + 0j}

    cls = sax_component(custom_coupler)
    inst = cls()
    # Missing default is filled by the decorator with 1.0.
    assert jnp.allclose(inst.coupling, 1.0)
    assert jnp.allclose(inst.loss, 0.0)
