"""Tests for :func:`circulax.s_transforms.sax_component`.

These tests specifically cover the signature-threading path — the decorator
must forward SAX parameter defaults through the wrapper into the resulting
``CircuitComponent`` subclass, which was previously broken because the
physics wrapper's ``__signature__`` did not include the reserved ``signals``
and ``s`` arguments that ``base_component._build_component`` requires.
"""

import functools

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


def test_sax_component_accepts_functools_partial() -> None:
    """SAX PDKs bind fab-specific defaults via ``functools.partial``; the
    decorator must unwrap these so ``__name__`` lookup does not crash.

    Regression: previously raised ``AttributeError: 'functools.partial'
    object has no attribute '__name__'``.
    """
    wg = functools.partial(straight, neff=2.5, ng=3.6)
    cls = sax_component(wg)
    # Class name falls back to the unwrapped base function.
    assert cls.__name__ == "straight"
    inst = cls()
    # Partial-bound args become the component's defaults.
    assert jnp.allclose(inst.neff, 2.5)
    assert jnp.allclose(inst.ng, 3.6)
    # Non-bound params keep their SAX defaults.
    assert jnp.allclose(inst.length, 10.0)


def test_sax_component_partial_with_name_override() -> None:
    """The ``name=`` kwarg sets the class name — needed for PDK dicts where
    many partials share the same underlying ``straight``."""
    wg = functools.partial(straight, length=5.0)
    cls = sax_component(wg, name="wg_short")
    assert cls.__name__ == "wg_short"
    assert jnp.allclose(cls().length, 5.0)


def test_sax_component_nested_partials() -> None:
    """Nested partials are unwrapped to the innermost callable."""
    outer = functools.partial(functools.partial(straight, neff=2.5), length=50.0)
    cls = sax_component(outer)
    assert cls.__name__ == "straight"
    inst = cls()
    assert jnp.allclose(inst.neff, 2.5)
    assert jnp.allclose(inst.length, 50.0)


def test_sax_component_numeric_port_names_are_sanitized() -> None:
    """Some SAX PDKs label ports numerically ('1', '2'); namedtuple field
    names must be valid identifiers, so the decorator prefixes digits with
    ``'p'``.

    Regression: previously raised ``ValueError: Type names and field names
    must be valid identifiers: '1'``.
    """

    def numeric_port_wg(length: float = 10.0, neff: float = 2.4) -> dict:
        phase = jnp.exp(1j * 2 * jnp.pi * neff * length / 1.55)
        return {
            ("1", "1"): 0.0 + 0j,
            ("1", "2"): phase,
            ("2", "1"): phase,
            ("2", "2"): 0.0 + 0j,
        }

    cls = sax_component(numeric_port_wg)
    assert set(cls.ports) == {"p1", "p2"}

    inst = cls(length=5.0)
    f_dict, q_dict = inst(p1=1.0 + 0j, p2=0.0 + 0j)
    # Returned current dict keys are sanitized port names.
    assert set(f_dict) == {"p1", "p2"}
    assert q_dict == {}


def test_sax_component_pdk_dict_pattern() -> None:
    """Exercise the dict-comprehension pattern the user flagged in the bug report:

    ``{k: sax_component(v, name=k) for k, v in pdk.items()}``
    """
    pdk = {
        "wg_short": functools.partial(straight, length=5.0),
        "wg_long": functools.partial(straight, length=100.0),
    }
    circulax_models = {k: sax_component(v, name=k) for k, v in pdk.items()}

    assert set(circulax_models) == {"wg_short", "wg_long"}
    assert circulax_models["wg_short"].__name__ == "wg_short"
    assert circulax_models["wg_long"].__name__ == "wg_long"
    assert jnp.allclose(circulax_models["wg_short"]().length, 5.0)
    assert jnp.allclose(circulax_models["wg_long"]().length, 100.0)


# ---------------------------------------------------------------------------
# Auto-detection (compile_netlist normalizer)
# ---------------------------------------------------------------------------


def test_is_sax_model_accepts_all_three_s_types() -> None:
    """SAX PDKs ship SDict, SDense, and SCoo models — all three must auto-detect."""
    import sax  # noqa: PLC0415

    from circulax.s_transforms import _is_sax_model  # noqa: PLC0415

    def m_sdict(*, wl: float = 1.55) -> sax.SDict:
        return {("in0", "out0"): 1.0 + 0j}

    def m_sdense(*, wl: float = 1.55) -> sax.SDense:
        return jnp.eye(2, dtype=complex), {"in0": 0, "out0": 1}

    def m_scoo(*, wl: float = 1.55) -> sax.SCoo:
        return jnp.array([0]), jnp.array([1]), jnp.array([1.0 + 0j]), {"in0": 0, "out0": 1}

    def m_stype(*, wl: float = 1.55) -> sax.SType:
        return {("in0", "out0"): 1.0 + 0j}

    assert _is_sax_model(m_sdict)
    assert _is_sax_model(m_sdense)
    assert _is_sax_model(m_scoo)
    assert _is_sax_model(m_stype)
    # Real SAX PDK model — the whole point of auto-detect.
    assert _is_sax_model(straight)
    # functools.partial wrapping a SAX model.
    assert _is_sax_model(functools.partial(straight, length=5.0))


def test_is_sax_model_rejects_non_sax() -> None:
    """Rejection cases: no-default arg, wrong return annotation, classes."""
    import sax  # noqa: PLC0415

    from circulax.components.electronic import Resistor  # noqa: PLC0415
    from circulax.s_transforms import _is_sax_model  # noqa: PLC0415

    def bad_no_default(wl: float) -> sax.SDict:  # positional no-default
        return {("in0", "out0"): 1.0 + 0j}

    def bad_wrong_return(*, wl: float = 1.55) -> dict:  # wrong return annotation
        return {("in0", "out0"): 1.0 + 0j}

    def bad_no_return_ann(*, wl: float = 1.55):  # missing return annotation
        return {("in0", "out0"): 1.0 + 0j}

    assert not _is_sax_model(bad_no_default)
    assert not _is_sax_model(bad_wrong_return)
    assert not _is_sax_model(bad_no_return_ann)
    # CircuitComponent classes must not be mistaken for SAX models.
    assert not _is_sax_model(Resistor)


def test_normalize_model_passes_through_circuit_component() -> None:
    """A CircuitComponent subclass must be returned unchanged."""
    from circulax.components.electronic import Resistor  # noqa: PLC0415
    from circulax.s_transforms import _normalize_model  # noqa: PLC0415

    assert _normalize_model(Resistor, name="R") is Resistor


def test_normalize_model_wraps_sax_function() -> None:
    """A raw SAX function must be wrapped into a CircuitComponent subclass."""
    from circulax.components.base_component import CircuitComponent  # noqa: PLC0415
    from circulax.s_transforms import _normalize_model  # noqa: PLC0415

    cls = _normalize_model(straight, name="wg")
    assert issubclass(cls, CircuitComponent)
    assert set(cls.ports) == {"in0", "out0"}


def test_normalize_model_rejects_invalid() -> None:
    """Anything that is neither a CircuitComponent nor a SAX model must raise TypeError."""
    from circulax.s_transforms import _normalize_model  # noqa: PLC0415

    def not_sax(x):  # no default, no return annotation
        return x

    with pytest.raises(TypeError, match="CircuitComponent subclass or a SAX model"):
        _normalize_model(not_sax, name="bad")


def test_sax_component_respects_sdense_port_order() -> None:
    """Regression: 4-port SDict whose dict-insertion order differs from sorted order.

    ``get_ports(sdict)`` returns ports sorted (``o1, o2, o3, o4``), while
    ``sdense(sdict)`` orders matrix rows by dict-insertion order (here
    ``o1, o3, o4, o2``). The wrapper used to discard ``sdense``'s port_map and
    use the sorted order to index signals, silently scrambling rows/columns of
    the Y-matrix. For a 2×2 MMI-style directional coupler this made the bar/
    cross coupling appear where the self-reflection should be; reciprocal
    devices still passed energy-conservation spot checks so the bug hid.

    Constructing an MMI2x2 S-matrix with unsorted insertion order and comparing
    against ``y_matrix @ v_vec`` computed in ``sdense``'s own port order.
    """
    import sax  # noqa: PLC0415

    from circulax.s_transforms import s_to_y, sax_component  # noqa: PLC0415

    def mmi_2x2(*, tau: float = 0.7071067811865476) -> sax.SDict:
        # Ideal 50/50 2x2 MMI: bar coupling = tau, cross coupling = j*sqrt(1-tau^2).
        # `tau` (not `t`) because `t` is reserved for `@source` time-dependent components.
        # KEY ORDER IS DELIBERATELY NON-SORTED (o1, o3, o4, o2) to trigger the bug.
        c = jnp.sqrt(1.0 - tau * tau)
        return {
            ("o1", "o3"): tau + 0j,
            ("o3", "o1"): tau + 0j,
            ("o1", "o4"): 1j * c,
            ("o4", "o1"): 1j * c,
            ("o2", "o3"): 1j * c,
            ("o3", "o2"): 1j * c,
            ("o2", "o4"): tau + 0j,
            ("o4", "o2"): tau + 0j,
        }

    # Reference computed directly from the S-matrix using sdense's port order.
    sdict = mmi_2x2()
    s_matrix, port_map = sax.sdense(sdict)
    y_ref = s_to_y(s_matrix)
    v_by_port = {"o1": 1.0 + 0j, "o2": 0.2 + 0.1j, "o3": -0.3 + 0j, "o4": 0.5 - 0.2j}
    matrix_order = sorted(port_map, key=port_map.get)
    v_vec = jnp.array([v_by_port[p] for p in matrix_order], dtype=jnp.complex128)
    i_ref = y_ref @ v_vec
    expected = {p: i_ref[port_map[p]] for p in v_by_port}

    # Component-level output must match port-for-port.
    cls = sax_component(mmi_2x2)
    inst = cls()
    f, q = inst(**v_by_port)
    assert q == {}
    for p in v_by_port:
        assert jnp.allclose(f[p], expected[p], atol=1e-10), (
            f"port {p!r} scrambled: got {complex(f[p])}, expected {complex(expected[p])}"
        )


def test_compile_netlist_auto_wraps_sax() -> None:
    """End-to-end: a netlist with a raw SAX model in models_map compiles without manual wrapping."""
    from circulax.compiler import compile_netlist  # noqa: PLC0415

    # Two waveguides in a loop — minimal topology that connects every port.
    netlist = {
        "instances": {
            "wg1": {"component": "straight", "settings": {"length": 10.0}},
            "wg2": {"component": "straight", "settings": {"length": 20.0}},
        },
        "connections": {
            "wg1,out0": "wg2,in0",
            "wg2,out0": "wg1,in0",
        },
    }
    # `straight` is a plain SAX PjitFunction — passed directly, no sax_component() call.
    groups, _sys_size, _port_map = compile_netlist(netlist, {"straight": straight})
    assert "straight" in groups
