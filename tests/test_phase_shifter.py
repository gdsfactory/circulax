# Copyright 2026 Circulax contributors

"""Tests for the electro-optic phase-shifter factory, junction models and data helpers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from circulax import Circuit, compile_circuit
from circulax.components import (
    JunctionModel,
    charge_from_capacitance_table,
    delta_neff_from_vpil,
    depletion_charge,
    diode_current,
    junction_model,
    phase_shifter_factory,
    pn_diode,
    tabulated,
)
from circulax.components.electronic import WAVE_DC, Resistor, VoltageSource, WaveformVoltageSource
from circulax.components.photonic import OpticalSource, OpticalSourcePulse

jax.config.update("jax_enable_x64", val=True)

C_UM_PER_S = 2.99792458e14
LENGTH_UM, N_GROUP, NEFF, WAVELENGTH_NM = 300.0, 4.0, 2.4, 1550.0


def quadratic_dn(v: ArrayLike, a1: float = -2e-4, a2: float = -3e-5) -> ArrayLike:
    return a1 * v + a2 * v**2


def linear_loss(v: ArrayLike, loss0: float = 2.0, loss1: float = -0.5) -> ArrayLike:
    return loss0 + loss1 * v


def transmission(v: ArrayLike, a1: float = -2e-4, a2: float = -3e-5, length_um: float = LENGTH_UM) -> np.ndarray:
    neff = NEFF + quadratic_dn(v, a1, a2)
    loss_db = linear_loss(v) * length_um * 1e-4
    return 10 ** (-loss_db / 20) * np.exp(-2j * np.pi * neff * length_um * 1e3 / WAVELENGTH_NM)


DIODE = pn_diode()


def link(
    *,
    source: str = "laser",
    source_settings: dict | None = None,
    drive_settings: dict | None = None,
    models: dict | None = None,
    reverse: bool = False,
    junction: JunctionModel | None = DIODE,
) -> Circuit:
    """Optical source -> phase shifter -> matched 1-ohm load; anode driven by V."""
    near, far = ("o2", "o1") if reverse else ("o1", "o2")
    ps = phase_shifter_factory(delta_neff=quadratic_dn, loss_dB_cm=linear_loss, junction=junction)
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "LASER": {"component": source, "settings": source_settings or {"power": 1.0}},
            "PS": {
                "component": "ps",
                "settings": {"length_um": LENGTH_UM, "n_group": N_GROUP, "neff": NEFF, "wavelength_nm": WAVELENGTH_NM},
            },
            "V": {"component": "V", "settings": drive_settings or {"V": 0.0}},
            "LOAD": {"component": "R", "settings": {"R": 1.0}},
        },
        "connections": {
            "LASER,p1": f"PS,{near}",
            f"PS,{far}": "LOAD,p1",
            "V,p1": "PS,anode",
            "GND,p1": ("LASER,p2", "LOAD,p2", "V,p2", "PS,cathode"),
        },
        "ports": {"out": "LOAD,p1"},
    }
    return compile_circuit(
        net, {"ps": ps, "R": Resistor, "V": VoltageSource, **(models or {"laser": OpticalSource})}, is_complex=True
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_dc_transmission_matches_analytic(*, reverse: bool) -> None:
    circuit = link(reverse=reverse)
    voltages = jnp.array([-2.0, -1.0, 0.0, 0.4])
    out = np.asarray(circuit.port(circuit.dc(params={"V.V": voltages}), "out"))
    np.testing.assert_allclose(out, transmission(np.asarray(voltages)), rtol=1e-7, atol=1e-12)


def test_function_keywords_are_sweepable_and_differentiable() -> None:
    circuit = link(drive_settings={"V": -1.0})
    a1 = jnp.array([-2e-4, -4e-4])
    out = np.asarray(circuit.port(circuit.dc(params={"PS.a1": a1}), "out"))
    np.testing.assert_allclose(out, [transmission(-1.0, a1=float(a)) for a in a1], rtol=1e-7)

    def phase(a: jax.Array) -> jax.Array:
        return jnp.angle(circuit.port(circuit.dc(params={"PS.a1": a}), "out"))

    grad = jax.grad(phase)(-2e-4)
    expected = -2 * np.pi * LENGTH_UM * 1e3 / WAVELENGTH_NM * (-1.0)  # d(-phi)/d a1 at v=-1
    np.testing.assert_allclose(grad, expected, rtol=1e-6)


def test_small_signal_preserves_field_and_power_modulation_delay() -> None:
    bias = -1.0
    circuit = link(drive_settings={"V": bias})
    freqs = jnp.array([0.0, 1e9, 100e9])
    response = circuit.small_signal()(freqs=freqs, sources={"V": 1.0})
    half_transit = LENGTH_UM * N_GROUP / (2 * C_UM_PER_S)
    delay = np.exp(-2j * np.pi * np.asarray(freqs) * half_transit)
    loss_slope = -np.log(10) / 20 * (-0.5) * LENGTH_UM * 1e-4
    phase_slope = -2 * np.pi * LENGTH_UM * 1e3 / WAVELENGTH_NM * (-2e-4 - 6e-5 * bias)
    field_slope = transmission(bias) * (loss_slope + 1j * phase_slope)
    dx, dy = response.quadratures("out")
    np.testing.assert_allclose(dx, field_slope.real * delay, rtol=1e-7, atol=1e-12)
    np.testing.assert_allclose(dy, field_slope.imag * delay, rtol=1e-7, atol=1e-12)
    power_slope = 2 * loss_slope * abs(transmission(bias)) ** 2
    np.testing.assert_allclose(response.power("out", field_unit="sqrt(W)"), power_slope * delay, rtol=1e-7, atol=1e-12)


def test_optical_and_modulation_delays() -> None:
    """Light crosses in L*ng/c; a voltage step reaches the output half a transit later."""
    transit = LENGTH_UM * N_GROUP / C_UM_PER_S
    t_light, t_step, rise = 0.5e-12, 4e-12, 0.05e-12
    circuit = link(
        source="pulse",
        source_settings={"power": 1.0, "delay": t_light, "rise": rise},
        drive_settings={"V": -1.0, "delay": t_step},
        models={"pulse": OpticalSourcePulse},
    )
    dt = transit / 200
    times = jnp.arange(0.0, t_step + transit, dt)
    solution = circuit.transient(t0=0.0, t1=float(times[-1]), dt0=dt, saveat=times, max_steps=len(times) + 20, throw=True)
    out = np.asarray(circuit.port(solution.ys, "out"))
    t = np.asarray(times)

    source = np.asarray(jax.nn.sigmoid((t - transit - t_light) / rise))
    before = t < t_step + transit / 2
    np.testing.assert_allclose(out[before], source[before] * transmission(0.0), atol=2e-3)
    after = t > t_step + transit / 2 + 4 * dt
    np.testing.assert_allclose(out[after], transmission(-1.0), atol=2e-3)


def test_vpil_helper_slope_sign_and_reference() -> None:
    v, vpil = np.array([-2.5, -1.5, -0.5]), np.array([1.6, 1.5, 1.4])
    dn = delta_neff_from_vpil(v, vpil, wavelength_nm=1310.0, reference_voltage=-0.5)
    assert float(dn(-0.5)) == pytest.approx(0.0, abs=1e-15)
    assert float(dn(-2.5)) > 0  # reverse bias depletes carriers and raises n_eff
    slope = jax.vmap(jax.grad(dn))(jnp.asarray(v))
    np.testing.assert_allclose(slope, -1310e-9 / (2 * vpil * 1e-2), rtol=1e-12)


def test_capacitance_table_charge_is_continuous_and_exact() -> None:
    v, cap = np.array([-3.0, -1.0, 0.0, 0.5]), np.array([1.0, 2.0, 3.0, 5.0]) * 1e-10
    charge = charge_from_capacitance_table(v, cap)
    assert float(charge(0.0)) == pytest.approx(0.0, abs=1e-25)
    probe = jnp.array([-4.0, -3.0, -2.0, -1.0, -1e-9, 1e-9, 0.25, 0.5, 1.0])
    np.testing.assert_allclose(jax.vmap(jax.grad(charge))(probe), np.interp(probe, v, cap), rtol=1e-6)


def test_capacitance_quadrature_matches_depletion_charge() -> None:
    def depletion_capacitance(v: ArrayLike, cj0_F_per_m: float = 3.0e-10, vbi: float = 0.9, grading: float = 0.5) -> ArrayLike:
        return cj0_F_per_m / (1 - v / vbi) ** grading

    v = jnp.array([-3.0, -1.0, 0.2])
    from_capacitance = junction_model(capacitance=depletion_capacitance)
    np.testing.assert_allclose(from_capacitance.charge(v), pn_diode().charge(v), rtol=1e-8)


def test_junction_model_evaluates_standalone() -> None:
    diode = pn_diode(cj0_F_per_m=3e-10, vbi=0.9, grading=0.5, dark_current_A_per_m=1e-5, dark_bias_V=-2.0)
    v = jnp.array([-3.0, -1.0, 0.0, 0.3])
    np.testing.assert_allclose(diode.capacitance(v), 3e-10 / np.sqrt(1 - np.asarray(v) / 0.9), rtol=1e-12)
    np.testing.assert_allclose(diode.capacitance(-1.0, cj0_F_per_m=6e-10), 2 * diode.capacitance(-1.0), rtol=1e-12)
    saturation = 1e-8 * (1 - np.exp(-2.0 / 0.025852))
    np.testing.assert_allclose(diode.current(-2.0), -(1e-5 + saturation), rtol=1e-12)
    assert diode.current(v).shape == v.shape
    assert diode.defaults["dark_bias_V"] == -2.0


def junction_bench(source_settings: dict | None, *, length_um: float = 100.0) -> tuple[JunctionModel, Circuit]:
    diode = pn_diode(dark_current_A_per_m=1e-5)
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "V": {"component": "V", "settings": source_settings},
            "R": {"component": "R", "settings": {"R": 1e3}},
            "D": {"component": "D", "settings": {"length_um": length_um}},
        },
        "connections": {"V,p1": "R,p1", "R,p2": "D,anode", "GND,p1": ("V,p2", "D,cathode")},
        "ports": {"anode": "D,anode"},
    }
    models = {"V": WaveformVoltageSource, "R": Resistor, "D": diode.component("Diode")}
    return diode, compile_circuit(net, models, is_complex=False)


def test_junction_component_dc_matches_root_solve() -> None:
    diode, circuit = junction_bench({"kind": WAVE_DC, "dc": 0.8})
    v_anode = float(circuit.port(circuit.dc(), "anode"))
    expected = brentq(lambda v: (0.8 - v) / 1e3 - 100e-6 * float(diode.current(v)), 0.0, 0.8, xtol=1e-14)
    np.testing.assert_allclose(v_anode, expected, rtol=1e-6)


def test_junction_component_ac_uses_bias_dependent_capacitance() -> None:
    diode, circuit = junction_bench({"kind": WAVE_DC, "dc": -1.0})
    y_dc = circuit.dc()
    v_dc = float(circuit.port(y_dc, "anode"))
    freqs = jnp.array([1e6, 1e9, 1e10])
    ac = circuit.ac_response(freqs=freqs, sources={"V": 1.0}, y_dc=y_dc)
    length_m = 100e-6
    conductance = length_m * float(jax.grad(diode.current)(v_dc))
    admittance = conductance + 2j * np.pi * np.asarray(freqs) * length_m * float(diode.capacitance(v_dc))
    np.testing.assert_allclose(np.asarray(circuit.port(ac["solution"], "anode")), 1 / (1 + 1e3 * admittance), rtol=1e-6)


def test_unloaded_phase_shifter_has_no_electrical_stamp() -> None:
    ps = phase_shifter_factory(delta_neff=quadratic_dn, loss_dB_cm=linear_loss)()
    signals = {"anode": -1.0, "cathode": 0.0, "o1": 1.0, "o2": 0.0}
    states = dict.fromkeys(("a_forward", "a_reverse", "m_forward", "m_reverse"), 0.0)
    f, q = ps(**signals, **states)
    assert "anode" not in f
    assert not q
    loaded, unloaded = link(drive_settings={"V": -1.0}), link(drive_settings={"V": -1.0}, junction=None)
    np.testing.assert_allclose(unloaded.port(unloaded.dc(), "out"), loaded.port(loaded.dc(), "out"), rtol=1e-7)


def test_depletion_and_diode_are_smooth_through_forward_bias() -> None:
    v = jnp.linspace(-3.0, 2.0, 501)
    cap = jax.vmap(jax.grad(depletion_charge))(v)
    assert np.all(np.isfinite(cap))
    assert np.all(cap > 0)
    assert np.max(np.abs(np.diff(cap))) < 0.05 * float(cap.max())
    di = jax.vmap(jax.grad(diode_current))(v)
    assert np.all(np.isfinite(di))
    assert np.all(di > 0)


def test_tabulated_holds_end_values() -> None:
    table = tabulated([1.0, 0.0], [3.0, 1.0])
    np.testing.assert_allclose(table(jnp.array([-1.0, 0.5, 2.0])), [1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"delta_neff": lambda v, a=1.0: a * v, "loss_dB_cm": lambda v, a=2.0: a * v}, ValueError),
        ({"delta_neff": lambda v, vbi=0.5: v + 0 * vbi, "junction": pn_diode()}, ValueError),
        ({"delta_neff": lambda v, length_um=1.0: v + 0 * length_um}, ValueError),
        ({"delta_neff": lambda v, a: a * v}, TypeError),
        ({"delta_neff": lambda v, **_kw: v}, TypeError),
        ({"delta_neff": 0.0, "junction": depletion_charge}, TypeError),
    ],
)
def test_invalid_phase_shifter_inputs_raise(kwargs: dict, error: type[Exception]) -> None:
    with pytest.raises(error):
        phase_shifter_factory(**kwargs)


def test_invalid_junction_inputs_raise() -> None:
    with pytest.raises(ValueError, match="either charge or capacitance"):
        junction_model(charge=depletion_charge, capacitance=1e-10)
    with pytest.raises(ValueError, match="unknown parameters"):
        junction_model(charge=depletion_charge, defaults={"cj0": 1.0})
    with pytest.raises(ValueError, match="negative"):
        pn_diode(dark_bias_V=1.0)
    with pytest.raises(TypeError, match="Unknown junction parameters"):
        pn_diode().current(0.0, cj=1.0)
