# Copyright 2026 Circulax contributors
"""Detector acceptance through public circuits and independent physical laws."""

import importlib
from collections.abc import Callable
from typing import NoReturn

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax import Circuit, QuadratureDrive, compile_circuit
from circulax.components import carrier_photodiode, ideal_photodetector, optical_wave_source, photodiode
from circulax.components.detectors import QE, dark_iv, junction_charge
from circulax.components.electronic import VoltageSource

FACTORIES = (ideal_photodetector, photodiode, carrier_photodiode)
A = 0.003 + 0.004j
Q0 = QE * 100e-18 * 1e21


@pytest.fixture(autouse=True)
def forbid_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> NoReturn:
        msg = "Detector realization called s_to_y"
        raise AssertionError(msg)

    for name in ("circulax.s_transforms", "circulax.components.photonic"):
        monkeypatch.setattr(importlib.import_module(name), "s_to_y", forbidden)


def bench(factory: Callable = carrier_photodiode, r0: float = 1.0, waveform: Callable | None = None) -> Circuit:
    return compile_circuit(
        {
            "instances": {
                "PD": {"component": "pd"},
                "S": {"component": "s", "settings": {"amplitude": A}},
                "V": {"component": "v", "settings": {"V": -3.0}},
                "GND": {"component": "ground"},
            },
            "connections": {"S,p": "PD,optical_in", "V,p1": "PD,anode", "GND,p1": ("V,p2", "PD,cathode")},
        },
        {"pd": factory(r0=r0), "s": optical_wave_source(r0=r0, waveform=waveform), "v": VoltageSource},
        is_complex=True,
        backend="dense",
        g_leak=0.0,
    )


def dark(v: float | np.ndarray) -> float | np.ndarray:
    return 1e-12 * np.expm1(v / (1.5 * 0.02585)) + 1e-9 * np.tanh(v / 0.2) + v / 1e12


def capacitance(v: float) -> float:
    return 20e-15 / np.sqrt(1 - v / 0.8) if v <= 0.4 else 20e-15 / np.sqrt(0.5) + 20e-15 / (1.6 * 0.5**1.5) * (v - 0.4)


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
@pytest.mark.parametrize("phase", [0.0, np.pi / 2, np.pi])
def test_dc_power_reflective_limits(factory: Callable, r0: float, phase: float) -> None:
    c = bench(factory, r0)
    for reflectivity in (0.0, 0.36, 1.0):
        params = {"reflectivity": reflectivity, "reflection_phase_rad": phase, "screening": 0.0, "tau_rec": 1e6}
        # Only carrier model accepts the last two settings.
        if factory is not carrier_photodiode:
            params.pop("screening")
            params.pop("tau_rec")
        y, ok = c.dc_checked(params=params, rtol=1e-11, atol=1e-13)
        assert ok
        w = c.port_waves(y, "PD,optical_in", params=params)
        r = np.sqrt(reflectivity) * np.exp(1j * phase)
        np.testing.assert_allclose([w.incident, w.outgoing], [A, r * A], atol=2e-12)
        np.testing.assert_allclose(c.port(y, "PD,optical_in"), np.sqrt(r0) * (1 + r) * A, atol=2e-12)
        np.testing.assert_allclose(
            [w.incident_power, w.outgoing_power, w.net_power],
            abs(A) ** 2 * np.array([1, reflectivity, 1 - reflectivity]),
            atol=1e-14,
        )
        np.testing.assert_allclose(c.port_waves(y, "S,p").net_power + w.net_power, 0, atol=1e-14)
        generation = 0.8 * (1 - reflectivity) * abs(A) ** 2
        expected = generation - (0 if factory is ideal_photodetector else dark(-3.0))
        np.testing.assert_allclose(c.port(y, "V,i_src"), expected, atol=2e-13)
        if factory is carrier_photodiode:
            np.testing.assert_allclose(
                [c.port(y, "PD,x_e"), c.port(y, "PD,x_h")], generation / Q0 * np.array([8e-12, 40e-12]), atol=2e-10
            )


@pytest.mark.parametrize("factory", FACTORIES)
def test_observable_and_analysis_rejections(factory: Callable) -> None:
    c = bench(factory)
    y = c.dc()
    for terminal in ("anode", "cathode"):
        with pytest.raises(ValueError, match="optical terminal"):
            c.port_waves(y, f"PD,{terminal}")
    if factory is carrier_photodiode:
        for state in ("x_e", "x_h", "wave_optical_in"):
            with pytest.raises(KeyError):
                c.port_waves(y, f"PD,{state}")
    for analysis in ("ac", "hb", "hb_coefficients"):
        with pytest.raises(ValueError, match="not AC"):
            c.port_waves(y, "PD,optical_in", analysis=analysis)
    with pytest.raises(ValueError, match="quadrature"):
        c.ac_response(freqs=jnp.array([1e9]), sources={"S": 1.0})
    response = c.small_signal(y_dc=y)(freqs=jnp.array([1e9]), sources={"S": 1.0})
    with pytest.raises(ValueError, match="terminal potential"):
        response.power("PD,optical_in", field_unit="sqrt(W)")
    if factory is carrier_photodiode:
        for state in ("x_e", "x_h"):
            with pytest.raises(ValueError, match="non-wave state"):
                response.power(f"PD,{state}", field_unit="sqrt(W)")


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
def test_small_signal_intensity_quadratures_and_carrier_poles(factory: Callable, r0: float) -> None:
    c = bench(factory, r0)
    params = {} if factory is not carrier_photodiode else {"screening": 0.0, "tau_rec": 1e6, "electron_weight": 0.3}
    y = c.dc(params=params)
    f = jnp.array([0.0, 1e9, 10e9, 100e9])
    drive = QuadratureDrive(x=0.7 + 0.2j, y=-0.4j)
    response = jax.jit(c.small_signal(y_dc=y, params=params))(freqs=f, sources={"S": drive})
    gain = 0.8 * 0.96
    expected = 2 * gain * (A.real * drive.x + A.imag * drive.y)
    if factory is carrier_photodiode:
        expected *= 0.3 / (1 + 2j * np.pi * f * 8e-12) + 0.7 / (1 + 2j * np.pi * f * 40e-12)
    np.testing.assert_allclose(response.voltage("V,i_src"), expected, rtol=1e-8, atol=1e-12)
    # An optical phase perturbation is orthogonal to intensity for every phase.
    phase = c.small_signal(y_dc=y, params=params)(freqs=f, sources={"S": QuadratureDrive(x=-A.imag, y=A.real)})
    np.testing.assert_allclose(phase.voltage("V,i_src"), 0, atol=1e-15)


@pytest.mark.parametrize("v", [-3.0, -0.2, 0.2, 0.6])
def test_dark_iv_conductance_capacitance_and_complex_charge(v: float) -> None:
    c = bench(photodiode)
    params = {"V.V": v, "S.amplitude": 0.0}
    y = c.dc(params=params)
    np.testing.assert_allclose(-c.port(y, "V,i_src"), dark(v), rtol=1e-9, atol=1e-13)
    f = jnp.array([0.0, 1e9, 30e9])
    response = c.small_signal(y_dc=y, params=params)(freqs=f, sources={"V": 1.0})
    g = 1e-12 / (1.5 * 0.02585) * np.exp(v / (1.5 * 0.02585)) + 1e-9 / 0.2 / np.cosh(v / 0.2) ** 2 + 1e-12
    np.testing.assert_allclose(response.voltage("V,i_src"), -(g + 2j * np.pi * f * capacitance(v)), rtol=1e-9, atol=1e-13)
    # The original electrical imaginary coordinate is analytically continued,
    # not projected to real as the phase-shifter junction is.
    dv = 1e-7
    np.testing.assert_allclose(jnp.imag(junction_charge(v + 1j * dv)) / dv, capacitance(v), rtol=1e-7)
    np.testing.assert_allclose(jnp.imag(dark_iv(v + 1j * dv)) / dv, g, rtol=1e-7)


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
def test_dc_spectral_runtime_jit_vmap_and_parameter_gradients(r0: float) -> None:
    c = bench(ideal_photodetector, r0)

    def measured(p: jax.Array) -> jax.Array:
        y = c.dc(params={"wavelength_nm": p[0], "reflectivity": p[1], "responsivity": p[2], "reflection_phase_rad": p[3]})
        return jnp.real(c.port(y, "V,i_src"))

    p = jnp.array([1500.0, 0.3, 0.7, 1.2])

    def reference(p: jax.Array) -> jax.Array:
        detuning = p[0] - 1550
        return (
            p[2]
            * (p[0] / 1550)
            * jnp.exp(-0.5 * (detuning / 250) ** 2)
            * (1 - p[1] * jnp.exp(-0.5 * (detuning / 180) ** 2))
            * abs(A) ** 2
        )

    values = jnp.stack([p, p + jnp.array([100, 0.1, 0.1, 0.8])])
    np.testing.assert_allclose(jax.jit(jax.vmap(measured))(values), jax.vmap(reference)(values), rtol=1e-9)
    np.testing.assert_allclose(jax.grad(measured)(p), jax.grad(reference)(p), rtol=1e-8, atol=1e-14)


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("bad", [0.0, -1.0, 1j, np.inf, np.nan])
def test_static_reference_validation(factory: Callable, bad: complex) -> None:
    with pytest.raises(ValueError, match="finite, positive, and real"):
        factory(r0=bad)


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
def test_complete_real_block_state_and_parameter_derivatives(factory: Callable, r0: float) -> None:
    cls = factory(r0=r0)
    # Exercise nonzero imaginary voltage AND both complex carrier coordinates.
    state = jnp.array(
        [0.007 + 0.002j, -0.3 + 0.02j, 0.1 - 0.01j] + ([0.2 + 0.03j, 0.4 - 0.02j] if factory is carrier_photodiode else []) + [A]
    )
    n = len(state)
    x = jnp.concatenate((state.real, state.imag))
    names = ["responsivity", "wavelength_nm", "reflectivity", "reflection_phase_rad", "reflection_phase_slope_rad_nm"]
    p = [0.7, 1500.0, 0.36, 1.2, 0.012]
    if factory is not ideal_photodetector:
        names += ["c0", "i_s", "v_bi", "fc"]
        p += [25e-15, 2e-12, 0.9, 0.4]
    if factory is carrier_photodiode:
        names += ["tau_e0", "tau_h0", "screening", "tau_rec", "electron_weight", "tau_ref", "volume_m3", "density_scale_m3"]
        p += [9e-12, 35e-12, 1.2, 300e-12, 0.3, 18e-12, 110e-18, 0.9e21]
    scale = jnp.array(p)

    def residual(x: jax.Array, normalized: jax.Array) -> jax.Array:
        f, q = cls.solver_call(0.0, x[:n] + 1j * x[n:], cls(**dict(zip(names, normalized * scale, strict=True))))
        return jnp.concatenate((f.real, f.imag, q.real * 1e12, q.imag * 1e12))

    for arg, point in ((0, x), (1, jnp.ones(len(p)))):
        actual = jax.jit(jax.jacfwd(residual, argnums=arg))(x, jnp.ones(len(p)))

        def evaluate(value: jax.Array, arg: int = arg) -> jax.Array:
            return residual(value, jnp.ones(len(p))) if arg == 0 else residual(x, value)

        eps = 1e-6
        finite = np.stack(
            [(evaluate(point + eps * e) - evaluate(point - eps * e)) / (2 * eps) for e in jnp.eye(len(point))], axis=1
        )
        np.testing.assert_allclose(actual, finite, rtol=2e-5, atol=1e-8)
    f, q = cls.solver_call(0.0, state, cls())
    np.testing.assert_array_equal([f[1] + f[2], q[1] + q[2]], 0)
    np.testing.assert_array_equal(q[jnp.array([0, n - 1])], 0)


def pulse(t: jax.Array, width: float = 80e-12) -> jax.Array:
    u = (t - 40e-12) / width
    return jnp.where((u > 0) & (u < 1), jnp.sin(jnp.pi * u), 0.0)


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
@pytest.mark.parametrize(("screening", "tau_rec"), [(0.0, 1e6), (0.0, 60e-12), (2.0, 60e-12)])
def test_pulse_generation_collection_and_stored_charge_balance(r0: float, screening: float, tau_rec: float) -> None:
    c = bench(carrier_photodiode, r0, pulse)
    params = {"screening": screening, "tau_rec": tau_rec, "electron_weight": 0.3}
    ts = jnp.linspace(0, 800e-12, 1601)
    sol = c.transient(t0=0, t1=float(ts[-1]), dt0=0.5e-12, params=params, saveat=diffrax.SaveAt(ts=ts), max_steps=2000)
    w = c.port_waves(sol.ys, "PD,optical_in", analysis="tran", t=sol.ts, params=params)
    np.testing.assert_allclose(w.incident, A * pulse(ts), atol=2e-12)
    np.testing.assert_allclose(w.outgoing, 0.2 * A * pulse(ts), atol=2e-12)
    t = np.asarray(ts)
    generated = 0.8 * np.asarray(w.net_power)
    xe, xh = [np.real(c.port(sol.ys, f"PD,{state}")) for state in ("x_e", "x_h")]
    collected = np.real(c.port(sol.ys, "V,i_src")) + dark(-3.0)
    tau_e = 8e-12 * (1 + screening * (xe + xh) / 2)
    tau_h = 40e-12 * (1 + screening * (xe + xh) / 2)
    recombined = Q0 * xe * xh / tau_rec
    expected_collected = Q0 * (0.3 * xe / tau_e + 0.7 * xh / tau_h)
    np.testing.assert_allclose(collected, expected_collected, atol=2e-13)
    # Pair loss enters each population once; collection weighting sums to one.
    stored = Q0 * (0.3 * xe + 0.7 * xh)
    balance = np.trapezoid(generated - collected - recombined, t)
    np.testing.assert_allclose(balance, stored[-1] - stored[0], atol=3e-5 * np.trapezoid(generated, t))
    np.testing.assert_allclose(np.trapezoid(generated, t), 0.8 * 0.96 * abs(A) ** 2 * 80e-12 / 2, rtol=1e-10)
    assert max(xe) > 0
    assert max(xh) > max(xe)
    if tau_rec == 1e6:
        np.testing.assert_allclose(np.trapezoid(collected, t), np.trapezoid(generated, t), rtol=2e-5)
        # Independent analytic convolution of the known pulse with each lifetime.
        from scipy.integrate import quad

        for state, tau in ((xe, 8e-12), (xh, 40e-12)):
            for k in (120, 240, 400, 800):
                now = t[k] / 1e-12
                value = (
                    quad(
                        lambda u, now=now, tau=tau: np.sin(np.pi * (u - 40) / 80) ** 2 * np.exp(-(now - u) / (tau / 1e-12)),
                        40,
                        min(120, now),
                        epsabs=1e-11,
                    )[0]
                    * 1e-12
                    * 0.8
                    * 0.96
                    * abs(A) ** 2
                    / Q0
                )
                np.testing.assert_allclose(state[k], value, rtol=2e-4, atol=1e-9)
    else:
        assert np.trapezoid(recombined, t) > 0
        assert np.trapezoid(collected, t) < np.trapezoid(generated, t)


def test_pulse_timestep_refinement_and_screening_control() -> None:
    c = bench(carrier_photodiode, waveform=pulse)
    ts = jnp.linspace(0, 500e-12, 1001)

    def solve(dt: float, screening: float) -> np.ndarray:
        sol = c.transient(
            t0=0,
            t1=float(ts[-1]),
            dt0=dt,
            params={"screening": screening, "tau_rec": 1e6, "S.amplitude": 0.05},
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=3000,
        )
        return np.real(c.port(sol.ys, "V,i_src")) + dark(-3.0)

    coarse, fine = solve(1e-12, 2.0), solve(0.5e-12, 2.0)
    assert np.max(abs(coarse - fine)) / max(fine) < 0.003
    unscreened = solve(0.5e-12, 0.0)
    assert max(fine) < max(unscreened)
    assert fine[500] > unscreened[500]


@pytest.mark.parametrize("factory", FACTORIES)
@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
def test_hb_intensity_carrier_transfer_and_harmonic_convergence(factory: Callable, r0: float) -> None:
    freq, depth = 5e9, 0.3

    def modulation(t: jax.Array) -> jax.Array:
        return jnp.sqrt(1 + depth * jnp.cos(2 * jnp.pi * freq * t))

    c = bench(factory, r0, modulation)
    params = {} if factory is not carrier_photodiode else {"screening": 0.0, "tau_rec": 1e6, "electron_weight": 0.3}
    op = c.dc(params=params)
    solutions = [c.hb(freq=freq, harmonics=h, y0=op, params=params, max_steps=100) for h in (5, 9)]
    spectra = [c.port(spectrum, "V,i_src") for _, spectrum in solutions]
    g = 0.8 * 0.96 * abs(A) ** 2
    expected = g * depth / 2
    if factory is carrier_photodiode:
        expected *= 0.3 / (1 + 2j * np.pi * freq * 8e-12) + 0.7 / (1 + 2j * np.pi * freq * 40e-12)
    for spectrum in spectra:
        np.testing.assert_allclose(spectrum[0], g - (0 if factory is ideal_photodetector else dark(-3.0)), atol=1e-11)
        np.testing.assert_allclose(spectrum[1], expected, rtol=2e-5, atol=1e-12)
        np.testing.assert_allclose(spectrum[2:4], 0, atol=1e-11)
    np.testing.assert_allclose(spectra[0][:4], spectra[1][:4], atol=1e-11)
    time, _ = solutions[-1]
    ts = jnp.arange(len(time)) / (len(time) * freq)
    waves = c.port_waves(time, "PD,optical_in", analysis="hb_time", t=ts, params=params)
    np.testing.assert_allclose(waves.net_power, 0.96 * abs(A) ** 2 * (1 + depth * np.cos(2 * np.pi * freq * ts)), atol=2e-11)


def test_source_runtime_waveform_settings_and_gradient() -> None:
    def envelope(t: jax.Array, width: float = 80e-12) -> jax.Array:
        return pulse(t + 80e-12, width)

    c = bench(ideal_photodetector, waveform=envelope)

    def value(width: float) -> jax.Array:
        y = c.dc(params={"width": width})
        return jnp.real(c.port(y, "V,i_src"))

    widths = jnp.array([70e-12, 90e-12, 110e-12])
    np.testing.assert_allclose(
        jax.jit(jax.vmap(value))(widths), 0.8 * 0.96 * abs(A) ** 2 * jnp.sin(jnp.pi * 40e-12 / widths) ** 2, rtol=1e-9
    )
    d = jax.grad(value)(90e-12)
    finite = (value(90.001e-12) - value(89.999e-12)) / 0.002e-12
    np.testing.assert_allclose(d, finite, rtol=1e-7)


@pytest.mark.parametrize(
    "waveform", [lambda: 1.0, lambda _t, amplitude=1.0: amplitude, lambda _t, missing: missing, lambda t, **_kw: t]
)
def test_source_rejects_ambiguous_waveform_settings(waveform: Callable) -> None:
    with pytest.raises(TypeError, match="waveform"):
        optical_wave_source(waveform=waveform)


@pytest.mark.parametrize("r0", [0.01, 1.0, 50.0])
def test_carrier_equilibrium_recombination_runtime_gradients(r0: float) -> None:
    c = bench(carrier_photodiode, r0)
    scales = jnp.array([8e-12, 40e-12, 60e-12, 100e-18, 1e21])
    names = ("tau_e0", "tau_h0", "tau_rec", "volume_m3", "density_scale_m3")

    def measured(p: jax.Array) -> jax.Array:
        params = {**dict(zip(names, p * scales, strict=True)), "screening": 0.0}
        y = c.dc(params=params)
        return jnp.real(c.port(y, "V,i_src")) + dark(-3.0)

    def analytic(p: jax.Array) -> jax.Array:
        te, th, tr, volume, density = p * scales
        charge = QE * volume * density
        generation = 0.8 * 0.96 * abs(A) ** 2 / charge
        loss = te * th / tr
        collection_rate = 2 * generation / (1 + jnp.sqrt(1 + 4 * loss * generation))
        return charge * collection_rate

    p = jnp.ones(5)
    np.testing.assert_allclose(measured(p), analytic(p), rtol=1e-9)
    np.testing.assert_allclose(jax.jit(jax.grad(measured))(p), jax.grad(analytic)(p), rtol=1e-6, atol=1e-13)
    points = jnp.stack((p, p * jnp.array([1.2, 0.8, 1.5, 0.9, 1.3])))
    np.testing.assert_allclose(jax.jit(jax.vmap(measured))(points), jax.vmap(analytic)(points), rtol=1e-9)
