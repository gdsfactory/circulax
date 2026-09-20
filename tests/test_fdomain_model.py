"""Tests for :func:`circulax.s_transforms.y_to_s` and :func:`circulax.s_transforms.fdomain_model`."""

import jax.numpy as jnp
import numpy as np
import pytest

from circulax import compile_circuit, fdomain_component, fdomain_model
from circulax.components.electronic import Capacitor, Resistor
from circulax.s_transforms import s_to_y, sax_component, y_to_s


@pytest.mark.parametrize("seed", range(5))
def test_y_to_s_is_exact_inverse_of_s_to_y(seed):
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    S = jnp.asarray(m / (3 * np.abs(m).max()))
    S_back = y_to_s(s_to_y(S))
    assert jnp.max(jnp.abs(S - S_back)) < 1e-9


def _skin_effect_Y(f, R0=1.0, a=0.1):
    Z = R0 + a * jnp.sqrt(jnp.abs(f) + 1e-30)
    Y = 1.0 / Z
    return jnp.array([[Y, -Y], [-Y, Y]], dtype=jnp.complex128)


def _skin_effect_R(f, R0=1.0, a=0.1):
    return R0 + a * jnp.sqrt(jnp.abs(f) + 1e-30)


def _rc_netlist(resistor_component: str, resistor_settings: dict) -> dict:
    return {
        "instances": {
            "GND": {"component": "ground"},
            "R1": {"component": resistor_component, "settings": resistor_settings},
            "C1": {"component": "capacitor", "settings": {"C": 1e-9}},
        },
        "connections": {
            "R1,p2": "C1,p1",
            "C1,p2": "GND,p1",
        },
        "ports": {"in": "R1,p1"},
    }


def test_fdomain_model_matches_frozen_constant_resistor():
    """A y_to_s-wrapped skin-effect resistor, embedded in an RC circuit, must behave
    exactly like a constant resistor evaluated once at its own 'f' parameter — its
    value is frozen at compile/param-update time, not re-evaluated per AC-sweep point
    (unlike @fdomain_component, which does track the sweep).
    """
    sweep_freqs = jnp.array([1e6, 1e8, 1e9, 1e10])
    freq0 = float(sweep_freqs[2])
    r_frozen = float(_skin_effect_R(freq0))

    skin_sax = sax_component(fdomain_model(_skin_effect_Y, ports=("p1", "p2")))
    net_b = _rc_netlist("skinres", {"R0": 1.0, "a": 0.1, "f": freq0})
    circuit_b = compile_circuit(net_b, {"skinres": skin_sax, "capacitor": Capacitor, "ground": lambda: 0}, is_complex=True)
    S_b = circuit_b.sp(ports=["in"], freqs=sweep_freqs, z0=1.0)[:, 0, 0]

    net_ref = _rc_netlist("resistor", {"R": r_frozen})
    circuit_ref = compile_circuit(
        net_ref, {"resistor": Resistor, "capacitor": Capacitor, "ground": lambda: 0}, is_complex=True
    )
    S_ref = circuit_ref.sp(ports=["in"], freqs=sweep_freqs, z0=1.0)[:, 0, 0]

    assert jnp.allclose(S_b, S_ref, atol=1e-9)


def test_fdomain_model_does_not_track_ac_sweep_unlike_native_fdomain_component():
    """Documents the key semantic difference: @fdomain_component's Y(f) tracks the AC
    sweep frequency; a y_to_s-wrapped sax_component freezes at its own parameter value.
    """
    sweep_freqs = jnp.array([1e6, 1e8, 1e9, 1e10])
    freq0 = float(sweep_freqs[2])

    skin_native = fdomain_component(ports=("p1", "p2"))(_skin_effect_Y)
    net_a = _rc_netlist("skinres", {"R0": 1.0, "a": 0.1})
    circuit_a = compile_circuit(
        net_a, {"skinres": skin_native, "capacitor": Capacitor, "ground": lambda: 0}, is_complex=True
    )
    S_a = circuit_a.sp(ports=["in"], freqs=sweep_freqs, z0=1.0)[:, 0, 0]

    r_frozen = float(_skin_effect_R(freq0))
    net_ref = _rc_netlist("resistor", {"R": r_frozen})
    circuit_ref = compile_circuit(
        net_ref, {"resistor": Resistor, "capacitor": Capacitor, "ground": lambda: 0}, is_complex=True
    )
    S_ref = circuit_ref.sp(ports=["in"], freqs=sweep_freqs, z0=1.0)[:, 0, 0]

    # Native fdomain tracks the sweep: it matches the frozen reference only at freq0...
    assert jnp.allclose(S_a[2], S_ref[2], atol=1e-9)
    # ...and diverges from it elsewhere, since R(f) genuinely varies with the sweep there.
    assert not jnp.allclose(S_a[0], S_ref[0], atol=1e-9)
    assert not jnp.allclose(S_a[1], S_ref[1], atol=1e-9)
    assert not jnp.allclose(S_a[3], S_ref[3], atol=1e-9)


def test_fdomain_model_rejects_missing_f_argument():
    def bad_model(R0=1.0):
        return jnp.eye(2, dtype=jnp.complex128)

    with pytest.raises(TypeError, match="must have 'f' as its first argument"):
        fdomain_model(bad_model, ports=("p1", "p2"))


def test_fdomain_model_rejects_param_without_default():
    def bad_model(f, R0):
        return jnp.eye(2, dtype=jnp.complex128)

    with pytest.raises(TypeError, match="must have a default value"):
        fdomain_model(bad_model, ports=("p1", "p2"))
