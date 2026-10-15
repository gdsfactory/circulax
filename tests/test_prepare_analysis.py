"""Backend setup selects registrations before entering simulation execution."""

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from circulax import Circuit, compile_circuit
from circulax.components.electronic import Resistor


def _forbid_solve(*_args: object, **_kwargs: object) -> None:
    """Preparation must never perform a simulation solve."""
    pytest.fail("prepare_analysis must not solve")


def _native_circuit(*, num_states: int = 0, complex_mode: bool = False, holomorphic: bool = True) -> Circuit:
    """Represent a native registration factory without provisioning artifacts."""
    indices = jnp.array([[0, 1]])
    group = SimpleNamespace(var_indices=indices, jac_rows=indices, jac_cols=indices, num_states=num_states, holomorphic=holomorphic)
    solver = SimpleNamespace(is_complex=complex_mode, solve_dc=_forbid_solve)
    return Circuit(solver, {"native": group}, 2, {"out": 1})


@pytest.mark.parametrize(
    ("analysis", "expected"),
    [
        ("op", "dc"),
        ("dc", "dc"),
        ("sp_wavelength", "dc"),
        ("ac_response", "ac"),
        ("sp_frequency", "ac"),
        ("tran", "tran"),
        ("hb", "tran"),
    ],
)
def test_preparation_selects_and_caches_native_registration_without_solving(analysis: str, expected: str) -> None:
    """Supported analysis names select immutable native registration modes."""
    circuit = _native_circuit()
    calls = []

    def factory(mode: str) -> Circuit:
        """Record requested native modes and return a topology-identical variant."""
        calls.append(mode)
        return _native_circuit()

    circuit._analysis_factory = factory  # noqa: SLF001 -- exercise compiler-installed registration factory
    prepared = circuit.prepare_analysis(analysis)
    assert calls[-1] == expected
    assert isinstance(prepared, Circuit)
    prior = list(calls)
    assert circuit.prepare_analysis(analysis) is prepared
    assert calls == prior


def test_builtin_preparation_does_not_solve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Builtin preparation passes without any operating-point calculation."""
    circuit = compile_circuit(
        {
            "instances": {"R": {"component": "resistor"}, "GND": {"component": "ground"}},
            "connections": {"R,p2": "GND,p1"},
            "ports": {"out": "R,p1"},
        },
        {"resistor": Resistor},
    )
    monkeypatch.setattr(Circuit, "dc", _forbid_solve)
    assert circuit.prepare_analysis("ac_response") is circuit
    assert circuit.prepare_analysis("tran") is circuit


def test_native_hb_state_slots_reject_before_registration_or_solve() -> None:
    """Unsupported native state history is a preparation-time capability reason."""
    circuit = _native_circuit(num_states=1)
    circuit._analysis_factory = _forbid_solve  # noqa: SLF001 -- reject before unnecessary registration
    with pytest.raises(NotImplementedError, match="ABI state slots"):
        circuit.prepare_analysis("hb")


def test_nonholomorphic_optical_driven_ac_rejects_during_preparation() -> None:
    """Optical quadrature limitations are discovered before running a solve."""
    with pytest.raises(ValueError, match="quadrature/sideband"):
        _native_circuit(complex_mode=True, holomorphic=False).prepare_analysis("ac_response")


def test_unsupported_analysis_rejects_before_setup() -> None:
    """Unknown backend analyses cannot silently select a registration mode."""
    with pytest.raises(ValueError, match="analysis"):
        _native_circuit().prepare_analysis("noise")


def _counting_compile(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], list[str]]:
    """Count wavelength-bench compilations and native registration-factory calls."""
    import circulax.circuit as circuit_module

    compiles: list[str] = []
    factory_modes: list[str] = []
    real = circuit_module.compile_circuit

    def counting(*args: object, **kwargs: object) -> Circuit:
        compiles.append("compile")
        compiled = real(*args, **kwargs)

        def factory(mode: str) -> Circuit:
            factory_modes.append(mode)
            return real(*args, **kwargs)

        compiled._analysis_factory = factory  # noqa: SLF001 -- stand in for a native registration factory
        return compiled

    monkeypatch.setattr(circuit_module, "compile_circuit", counting)
    return compiles, factory_modes


def _optical_two_port() -> Circuit:
    from circulax.components.base_component import PhysicsReturn, Signals, component

    @component(ports=("p1", "p2"), holomorphic=True)
    def conductance(signals: Signals, g: float = 1.0) -> PhysicsReturn:
        current = g * (signals.p1 - signals.p2)
        return {"p1": current, "p2": -current}, {}

    return compile_circuit(
        {
            "instances": {"X": {"component": "m"}, "GND": {"component": "ground"}},
            "connections": {},
            "ports": {"a": "X,p1", "b": "X,p2"},
        },
        {"m": conductance},
        is_complex=True,
        g_leak=0,
    )


def test_sp_wavelength_preparation_builds_the_bench_once_without_solving(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prepared bench and its native registrations are reused by every later sp_wavelength call."""
    circuit = _optical_two_port()
    compiles, factory_modes = _counting_compile(monkeypatch)
    kwargs = {"ports": ["a", "b"], "wavelengths": jnp.array([1.55]), "amplitude_unit": "sqrt(W)"}
    with monkeypatch.context() as patch:
        patch.setattr(Circuit, "dc", _forbid_solve)
        patch.setattr(Circuit, "dc_checked", _forbid_solve)
        circuit.prepare_analysis("sp_wavelength", ports=["a", "b"], z0=1.0)
    assert len(compiles) == 1
    assert factory_modes == ["dc"]
    first = circuit.sp_wavelength(**kwargs)
    second = circuit.sp_wavelength(**kwargs)
    assert jnp.allclose(first["s"], second["s"])
    assert len(compiles) == 1
    assert factory_modes == ["dc"]


def test_sp_wavelength_bench_is_keyed_by_ports_and_z0_not_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """Parameters are runtime inputs; a different port selection or z0 builds a new bench."""
    circuit = _optical_two_port()
    compiles, _ = _counting_compile(monkeypatch)
    kwargs = {"wavelengths": jnp.array([1.55]), "amplitude_unit": "sqrt(W)"}
    both = circuit.sp_wavelength(ports=["a", "b"], **kwargs)
    shifted = circuit.sp_wavelength(ports=["a", "b"], params={"X.g": 2.0}, **kwargs)
    assert len(compiles) == 1
    assert not jnp.allclose(both["s"], shifted["s"])
    circuit.sp_wavelength(ports=["a"], **kwargs)
    assert len(compiles) == 2
    circuit.sp_wavelength(ports=["a", "b"], z0=2.0, **kwargs)
    assert len(compiles) == 3
    circuit.sp_wavelength(ports=["a", "b"], z0=2.0, **kwargs)
    assert len(compiles) == 3


def test_sp_wavelength_bench_does_not_leak_across_rebuilt_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    """A newly compiled circuit owns its own bench cache."""
    compiles, _ = _counting_compile(monkeypatch)
    kwargs = {"ports": ["a", "b"], "wavelengths": jnp.array([1.55]), "amplitude_unit": "sqrt(W)"}
    _optical_two_port().sp_wavelength(**kwargs)
    _optical_two_port().sp_wavelength(**kwargs)
    assert len(compiles) == 2


def test_prepared_native_variant_reuses_the_bench_built_by_preparation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The selected DC variant, not just the original circuit, runs on the prepared bench."""
    parent = _optical_two_port()
    parent._analysis_factory = lambda _mode: _optical_two_port()  # noqa: SLF001 -- stand in for native registrations
    compiles, _ = _counting_compile(monkeypatch)
    kwargs = {"ports": ["a", "b"], "wavelengths": jnp.array([1.55]), "amplitude_unit": "sqrt(W)"}
    prepared = parent.prepare_analysis("sp_wavelength", ports=["a", "b"])
    assert prepared is not parent
    assert len(compiles) == 1
    prepared.sp_wavelength(**kwargs)
    prepared.sp_wavelength(**kwargs)
    parent.sp_wavelength(**kwargs)
    assert len(compiles) == 1
