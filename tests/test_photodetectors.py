# Copyright (c) 2026 Circulax contributors
"""Promoted detector regression and independent optical/electrical limits."""

# ruff: noqa: ANN001, ANN201, ANN202, FBT002, FBT003
import ast
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax import attach_testbench, compile_circuit
from circulax.components.electronic import Resistor, VoltageSource
from circulax.components.photodetectors import (
    CarrierPhotodiode,
    IdealPhotodetector,
    OpticalCW,
    OpticalGaussianPulse,
    Photodiode,
    junction_charge,
    make_photodetector_subcircuit,
    optical_conversion,
)

jax.config.update("jax_enable_x64", True)


def fixture(kind="diode", package=False, pulse=False):
    model = {"ideal": IdealPhotodetector, "diode": Photodiode, "carrier": CarrierPhotodiode}[kind]
    model = make_photodetector_subcircuit(kind) if package else model
    ports = ("optical_in", "anode", "cathode", "pad_ground") if package else ("optical_in", "anode", "cathode")
    device = {"instances": {"PD": {"component": "detector"}}, "ports": {p: f"PD,{p}" for p in ports}}
    net = attach_testbench(
        device,
        sources={
            "optical_in": {"name": "LIGHT", "component": "light", "settings": {"power": 1e-6}},
            "cathode": {"name": "BIAS", "component": "bias", "settings": {"V": 3.0}},
        },
        loads={"anode": {"name": "LOAD", "component": "load", "settings": {"R": 50.0}}},
        gnd=["pad_ground"] if package else [],
    )
    return compile_circuit(
        net,
        {
            "detector": model,
            "light": OpticalGaussianPulse if pulse else OpticalCW,
            "bias": VoltageSource,
            "load": Resistor,
            "ground": lambda: 0,
        },
        is_complex=True,
        backend="dense",
    )


@pytest.mark.parametrize("kind", ["ideal", "diode", "carrier"])
def test_loaded_responsivity_reflection_phase_and_real_states(kind):
    c = fixture(kind)
    params = {"power": 1e-6, "screening": 0.0, "tau_rec": 1e6}
    dark = c.dc(params={**params, "power": 0.0})
    for phase in (0.0, 0.7, 2.2):
        y = c.dc(params={**params, "phase": phase})
        gain = np.real(c.port(y - dark, "PD,anode")) / 50 / 1e-6
        np.testing.assert_allclose(gain, 0.8 * 0.96, rtol=2e-5)
        incident = np.sqrt(1e-6) * np.exp(1j * phase)
        reflected = c.port(y, "PD,optical_in") - incident
        np.testing.assert_allclose(reflected / incident, 0.2, atol=1e-10)
        assert abs(np.imag(c.port(y, "PD,anode"))) < 1e-12
        if kind == "carrier":
            assert abs(np.imag(c.port(y, "PD,x_e"))) < 1e-12


def test_charge_is_integrated_capacitance_and_c1_join():
    for v in (-4.0, -0.2, 0.3, 0.4, 0.6):
        cap = jax.grad(junction_charge)(v)
        expected = 20e-15 / np.sqrt(1 - v / 0.8) if v <= 0.4 else 20e-15 / np.sqrt(0.5) + (v - 0.4) * 20e-15 / (2 * 0.8 * 0.5**1.5)
        np.testing.assert_allclose(cap, expected, rtol=1e-12)


def test_optical_absorption_not_total_field_squared():
    for phase in (0.0, 0.8, 2.0):
        r = 0.2 * np.exp(1j * phase)
        field = (1 + r) * np.sqrt(0.001)
        i, iph = optical_conversion(field, reflection_phase_rad=phase)
        np.testing.assert_allclose(iph, 0.8 * 0.001 * 0.96, rtol=1e-12)
        np.testing.assert_allclose(np.real(field * np.conj(i)), 0.001 * 0.96, rtol=1e-12)


def test_package_topology_and_dc_division():
    c = fixture("ideal", package=True)
    assert c.port_map["PD~CORE,anode"] != c.port_map["LOAD,p1"]
    y = c.dc(rtol=1e-11, atol=1e-14)
    output = c.port(y, "LOAD,p1")
    np.testing.assert_allclose(output, 0.8 * 0.96 * 1e-6 / (1 / 50 + 1 / 1e6), rtol=1e-7)
    np.testing.assert_allclose(c.port(y, "PD~CORE,anode") - output, 20 * output * (1 / 50 + 1 / 1e6), rtol=1e-7)


def test_real_block_jit_vmap_forward_reverse_and_parameter_update():
    def evaluate(x, responsivity):
        y = x[:5] + 1j * x[5:]
        f, q = CarrierPhotodiode.solver_call(0.0, y, CarrierPhotodiode(responsivity=responsivity))
        return jnp.concatenate((jnp.real(f), jnp.imag(f), jnp.real(q), jnp.imag(q)))

    x = jnp.array([0.01, 0.0, 3.0, 0.01, 0.02, 0.003, 0.0, 0.0, 0.01, -0.01])
    np.testing.assert_allclose(jax.jacfwd(evaluate)(x, 0.8), jax.jacrev(evaluate)(x, 0.8), atol=1e-12)
    batch = jax.jit(jax.vmap(evaluate, in_axes=(0, None)))
    assert np.isfinite(batch(jnp.stack([x, x]), 0.8)).all()
    assert not np.array_equal(batch(jnp.stack([x]), 0.8), batch(jnp.stack([x]), 0.9))
    f, _ = CarrierPhotodiode()(optical_in=0.01, cathode=3, x_e=0.01 + 0.02j, x_h=0.02 - 0.03j)
    assert np.imag(f["x_e"]) == pytest.approx(0.02)
    assert np.imag(f["x_h"]) == pytest.approx(-0.03)


def test_dark_iv_loaded_clamp():
    from circulax.components.photodetectors import dark_iv

    v = jnp.array([-4.0, -0.1, 0.1, 0.5])
    expected = 1e-12 * np.expm1(np.asarray(v) / (1.5 * 0.02585)) + 1e-9 * np.tanh(np.asarray(v) / 0.2) + np.asarray(v) / 1e12
    np.testing.assert_allclose(jax.jit(dark_iv)(v), expected, rtol=1e-12)


@pytest.mark.parametrize("kind", ["diode", "carrier"])
def test_small_signal_rc_and_transit_against_independent_transfer(kind):
    c = fixture(kind)
    params = {"screening": 0.0, "tau_rec": 1e6}
    y = c.dc(params=params)
    freqs = jnp.array([0.0, 1e9, 10e9, 50e9])
    s = c.sp(ports=["LIGHT,p1", "PD,anode"], freqs=freqs, z0=1.0, y_dc=y, params=params, holomorphic=False)
    identity = jnp.eye(2)
    z = jnp.linalg.solve((identity - s).swapaxes(-1, -2), (identity + s).swapaxes(-1, -2)).swapaxes(-1, -2)
    gain = z[:, 1, 0] / (50 * np.sqrt(1e-6))
    v = float(np.real(c.port(y, "PD,anode") - c.port(y, "PD,cathode")))
    cap = 20e-15 / np.sqrt(1 - v / 0.8)
    conductance = 1e-12 / (1.5 * 0.02585) * np.exp(v / (1.5 * 0.02585)) + 1 / 1e12
    omega = 2 * np.pi * np.asarray(freqs)
    expected = 0.8 * 0.96 / (1 + 50 * conductance + 1j * omega * 50 * cap)
    if kind == "carrier":
        te, th = np.array([8e-12, 40e-12]) * 3.8 / (0.8 - v)
        transport = 0.5 / (1 + 1j * omega * te) + 0.5 / (1 + 1j * omega * th)
        # The load modulates junction bias and hence both extraction times.
        # Linearizing tau proportional to 1/(Vbi-V) gives this feedback term.
        feedback = 50 * (0.8 * 0.96 * 1e-6) / (0.8 - v) * (1 - transport)
        expected = 0.8 * 0.96 * transport / (1 + 50 * conductance + 1j * omega * 50 * cap + feedback)
    np.testing.assert_allclose(gain, expected, rtol=2e-6, atol=1e-10)


@pytest.mark.parametrize("kind", ["diode", "carrier"])
def test_gaussian_pulse_loaded_charge_and_timestep_convergence(kind):
    c = fixture(kind, pulse=True)
    times = jnp.linspace(0, 400e-12, 401)
    params = {"power": 1e-6, "screening": 0.0, "tau_rec": 1e6}
    y0 = c.dc(params={**params, "power": 0.0})
    charges = []
    for dt in (1e-12, 0.5e-12):
        sol = c.transient(
            t0=0.0, t1=400e-12, dt0=dt, y0=y0, params=params, saveat=diffrax.SaveAt(ts=times), max_steps=2000, throw=True
        )
        current = np.real(c.port(sol.ys - y0, "PD,anode")) / 50
        charges.append(np.trapezoid(current, np.asarray(times)))
    expected = 0.8 * 0.96 * 1e-6 * np.sqrt(2 * np.pi) * 12e-12
    np.testing.assert_allclose(charges, expected, rtol=0.003)
    np.testing.assert_allclose(charges[0], charges[1], rtol=0.001)


def test_notebook_imports_models_instead_of_redefining():
    import json

    nb = json.loads(Path("examples/photonic/photodetectors.ipynb").read_text())
    defs = {
        n.name
        for c in nb["cells"]
        if c["cell_type"] == "code"
        for n in ast.parse("".join(c["source"])).body
        if isinstance(n, ast.FunctionDef)
    }
    assert not defs.intersection({"Photodiode", "CarrierPhotodiode", "IdealPhotodetector", "make_photodetector_subcircuit"})


def test_frozen_original_notebook_constitutive_regression():
    # Captured from original notebook cells 3/5/7 at the same real electrical
    # and carrier states; complex optical input exercises both quadratures.
    common = [0.00900300081211926 + 0.001990629773818126j]
    cases = [
        (IdealPhotodetector, [*common, -9.485909692413835e-05, 9.485909692413835e-05], [0, 0, 0]),
        (
            Photodiode,
            [*common, -9.486009990413332e-05, 9.486009990413332e-05],
            [0, -2.7652326023383193e-14, 2.7652326023383193e-14],
        ),
        (
            CarrierPhotodiode,
            [*common, -1.1492362738818494e-05, 1.1492362738818494e-05, -0.10046992629628491, -0.10764226891754396],
            [0, -2.7652326023383193e-14, 2.7652326023383193e-14, 2e-13, 6e-13],
        ),
    ]
    for cls, expected_f, expected_q in cases:
        model = cls(wavelength_nm=1500, reflection_phase_rad=0.7)
        y = jnp.array([0.013 + 0.004j, 0.02, 2.0, 0.01, 0.03])[: len(model.ports + model.states)]
        f, q = model.solver_call(0.0, y, model)
        np.testing.assert_allclose(f, expected_f, rtol=1e-12, atol=1e-16)
        np.testing.assert_allclose(q, expected_q, rtol=1e-12, atol=1e-27)


def test_carrier_small_signal_hb_vs_public_sp_and_harmonic_convergence():
    c = fixture("carrier")
    params = {"power": 1e-6, "frequency": 10e9, "screening": 0.0, "tau_rec": 1e6}
    op = c.dc(params=params)
    s = c.sp(ports=["LIGHT,p1", "PD,anode"], freqs=jnp.array([10e9]), z0=1.0, y_dc=op, params=params, holomorphic=False)
    eye = jnp.eye(2)
    z = jnp.linalg.solve((eye - s).swapaxes(-1, -2), (eye + s).swapaxes(-1, -2)).swapaxes(-1, -2)
    gain = z[0, 1, 0] / (50 * np.sqrt(1e-6))
    fundamentals = []
    for harmonics in (3, 5):
        _, spectrum = c.hb(freq=10e9, harmonics=harmonics, y0=op, params={**params, "modulation": 1e-3}, max_steps=100)
        fundamental = c.port(spectrum, "PD,anode")[1] / 50
        fundamentals.append(fundamental)
        np.testing.assert_allclose(fundamental / (1e-6 * 1e-3 / 2), gain, rtol=0.002, atol=1e-6)
    np.testing.assert_allclose(fundamentals[0], fundamentals[1], rtol=1e-4, atol=1e-13)
