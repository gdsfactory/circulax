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
