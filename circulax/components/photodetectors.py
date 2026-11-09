# Copyright (c) 2026 Circulax contributors
"""Direct detectors promoted from ``examples/photonic/photodetectors.ipynb``.

Optical pseudo-voltage is sqrt(W) only for a matched wave at z0=1.
The absorbing one-port below reconstructs the incident wave under reflection.
All times are seconds, electrical quantities SI, wavelengths nanometres.
The carrier model is illustrative phenomenology, not a calibrated material model.
"""

# ruff: noqa: ANN001, ANN201, PLR0917, FBT002
import equinox.internal as eqxi
import jax.numpy as jnp

from circulax.components.base_component import PhysicsReturn, Signals, component, source
from circulax.disciplines import electrical, normalized_carrier, optical

QE = 1.602176634e-19
PORTS = ("optical_in", "anode", "cathode")


def spectral_response(
    responsivity=0.8,
    wavelength_nm=1550.0,
    center_wavelength_nm=1550.0,
    responsivity_bandwidth_nm=250.0,
    reflectivity=0.04,
    reflection_bandwidth_nm=180.0,
    reflection_phase_rad=0.0,
    reflection_phase_slope_rad_nm=0.01,
):
    """Return absorbed-power responsivity (A/W) and complex field reflection."""
    reflectivity = jnp.asarray(reflectivity)
    reflectivity = eqxi.error_if(
        reflectivity,
        ~jnp.isfinite(reflectivity) | (reflectivity < 0) | (reflectivity >= 1),
        "Detector reflectivity must satisfy 0 <= reflectivity < 1; use OpticalInterface for ideal reflection.",
    )
    detuning_nm = wavelength_nm - center_wavelength_nm
    r_abs = responsivity * (wavelength_nm / center_wavelength_nm)
    r_abs *= jnp.exp(-0.5 * (detuning_nm / responsivity_bandwidth_nm) ** 2)
    reflection = jnp.sqrt(reflectivity) * jnp.exp(-0.25 * (detuning_nm / reflection_bandwidth_nm) ** 2)
    reflection = reflection * jnp.exp(1j * (reflection_phase_rad + reflection_phase_slope_rad_nm * detuning_nm))
    return r_abs, reflection


def optical_conversion(
    field,
    responsivity=0.8,
    wavelength_nm=1550.0,
    center_wavelength_nm=1550.0,
    responsivity_bandwidth_nm=250.0,
    reflectivity=0.04,
    reflection_bandwidth_nm=180.0,
    reflection_phase_rad=0.0,
    reflection_phase_slope_rad_nm=0.01,
):
    """Return inward optical current and absorbed-power photocurrent at z0=1."""
    r_abs, reflection = spectral_response(
        responsivity,
        wavelength_nm,
        center_wavelength_nm,
        responsivity_bandwidth_nm,
        reflectivity,
        reflection_bandwidth_nm,
        reflection_phase_rad,
        reflection_phase_slope_rad_nm,
    )
    optical_current = (1 - reflection) / (1 + reflection) * field
    incident = field / (1 + reflection)
    absorbed_power = (1 - jnp.abs(reflection) ** 2) * jnp.abs(incident) ** 2
    return optical_current, r_abs * absorbed_power


@component(port_disciplines={"optical_in": optical, "anode": electrical, "cathode": electrical}, ports=PORTS, holomorphic=False)
def IdealPhotodetector(
    signals: Signals,
    responsivity: float = 0.8,
    wavelength_nm: float = 1550.0,
    center_wavelength_nm: float = 1550.0,
    responsivity_bandwidth_nm: float = 250.0,
    reflectivity: float = 0.04,
    reflection_bandwidth_nm: float = 180.0,
    reflection_phase_rad: float = 0.0,
    reflection_phase_slope_rad_nm: float = 0.01,
) -> PhysicsReturn:
    """Instantaneous absorbed optical power to current.

    Device group:
        Photonic / Direct detection

    Device reference:
        photodetectors.md

    Examples:
        ```python
        import numpy as np

        from circulax.components.photodetectors import IdealPhotodetector

        # At zero reflection and z0=1, sqrt(1 mW) is the incident field.
        f, q = IdealPhotodetector(responsivity=0.8, reflectivity=0.0)(optical_in=np.sqrt(1e-3), anode=0.0, cathode=0.0)
        np.testing.assert_allclose(-f["anode"], 0.8e-3)
        print("Photocurrent leaving the anode:", -f["anode"], "A")
        ```

    """
    optical_current, i_photo = optical_conversion(
        signals.optical_in,
        responsivity,
        wavelength_nm,
        center_wavelength_nm,
        responsivity_bandwidth_nm,
        reflectivity,
        reflection_bandwidth_nm,
        reflection_phase_rad,
        reflection_phase_slope_rad_nm,
    )
    return {"optical_in": optical_current, "anode": -i_photo, "cathode": i_photo}, {}


def dark_iv(v, i_s=1e-12, ideality=1.5, v_t=0.02585, i_leak=1e-9, v_leak=0.2, r_sh=1e12):
    """Return Shockley, tanh leakage and shunt currents, in amperes."""
    return i_s * jnp.expm1(v / (ideality * v_t)) + i_leak * jnp.tanh(v / v_leak) + v / r_sh


def junction_charge(v, c0=20e-15, v_bi=0.8, fc=0.5):
    """Integrated depletion charge with C1 capacitance continuation."""
    v_join = fc * v_bi
    # A complex V is retained for the electrical imaginary-block equations.
    v_depletion = jnp.where(jnp.real(v) <= v_join, v, v_join)
    q_depletion = 2 * c0 * v_bi * (1 - jnp.sqrt(1 - v_depletion / v_bi))
    dv = jnp.where(jnp.real(v) > v_join, v - v_join, 0.0)
    c_join = c0 / jnp.sqrt(1 - fc)
    dc_join = c0 / (2 * v_bi * (1 - fc) ** 1.5)
    return q_depletion + c_join * dv + 0.5 * dc_join * dv**2


@component(port_disciplines={"optical_in": optical, "anode": electrical, "cathode": electrical}, ports=PORTS, holomorphic=False)
def Photodiode(
    signals: Signals,
    responsivity: float = 0.8,
    wavelength_nm: float = 1550.0,
    center_wavelength_nm: float = 1550.0,
    responsivity_bandwidth_nm: float = 250.0,
    reflectivity: float = 0.04,
    reflection_bandwidth_nm: float = 180.0,
    reflection_phase_rad: float = 0.0,
    reflection_phase_slope_rad_nm: float = 0.01,
    i_s: float = 1e-12,
    ideality: float = 1.5,
    v_t: float = 0.02585,
    i_leak: float = 1e-9,
    v_leak: float = 0.2,
    r_sh: float = 1e12,
    c0: float = 20e-15,
    v_bi: float = 0.8,
    fc: float = 0.5,
) -> PhysicsReturn:
    """Adds dark conduction and integrated junction depletion charge.

    Device group:
        Photonic / Direct detection

    Device reference:
        photodetectors.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.photodetectors import Photodiode

        jax.config.update("jax_enable_x64", True)
        model = Photodiode()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.complex128)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(Photodiode.__name__, "ports:", model.ports)
        ```

    """
    v = signals.anode - signals.cathode
    current = dark_iv(v, i_s, ideality, v_t, i_leak, v_leak, r_sh)
    optical_current, i_photo = optical_conversion(
        signals.optical_in,
        responsivity,
        wavelength_nm,
        center_wavelength_nm,
        responsivity_bandwidth_nm,
        reflectivity,
        reflection_bandwidth_nm,
        reflection_phase_rad,
        reflection_phase_slope_rad_nm,
    )
    current -= i_photo
    charge = junction_charge(v, c0, v_bi, fc)
    return {"optical_in": optical_current, "anode": current, "cathode": -current}, {"anode": charge, "cathode": -charge}


@source(discipline=optical, ports=("p1", "p2"), holomorphic=True)
def OpticalCW(
    signals: Signals,
    t: float,
    power: float = 1e-6,
    phase: float = 0.0,
    modulation: float = 0.0,
    frequency: float = 1e9,
) -> PhysicsReturn:
    # Sinusoidal incident POWER, not sinusoidal field amplitude.
    """A prescribed continuous optical carrier, optionally modulated in power.

    Device group:
        Photonic / Optical sources and propagation

    Device reference:
        photodetectors.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.photodetectors import OpticalCW

        jax.config.update("jax_enable_x64", True)
        model = OpticalCW()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.complex128)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(OpticalCW.__name__, "ports:", model.ports)
        ```

    """
    incident_power = power * (1 + modulation * jnp.cos(2 * jnp.pi * frequency * t))
    incident = jnp.sqrt(incident_power) * jnp.exp(1j * phase)
    current = signals.p1 - signals.p2 - 2 * incident
    return {"p1": current, "p2": -current}, {}


def internal_field(x_e, x_h, v, v_bi=0.8, screening=1.0, x_screen=1.0, depletion_width_m=1e-6):
    """Return unscreened and population-screened internal fields in V/m."""
    population = (jnp.maximum(jnp.real(x_e), 0) + jnp.maximum(jnp.real(x_h), 0)) / 2
    reverse_bias = jnp.maximum(-jnp.real(v), 0)
    unscreened = (v_bi + reverse_bias) / depletion_width_m
    effective = unscreened / (1 + screening * population / x_screen)
    return unscreened, effective


def carrier_rates(x_e, x_h, v, tau_e0=8e-12, tau_h0=40e-12, v_bi=0.8, v_ref=3.0, screening=1.0, x_screen=1.0, tau_rec=500e-12):
    """Return electron/hole extraction times and scaled recombination rate."""
    _, effective_field = internal_field(x_e, x_h, v, v_bi, screening, x_screen)
    reference_field = (v_bi + v_ref) / 1e-6
    tau_e, tau_h = tau_e0 * reference_field / effective_field, tau_h0 * reference_field / effective_field
    recombination = x_e * x_h / tau_rec
    return tau_e, tau_h, recombination


@component(
    port_disciplines={"optical_in": optical, "anode": electrical, "cathode": electrical},
    state_natures={"x_e": normalized_carrier, "x_h": normalized_carrier},
    equation_natures={"x_e": normalized_carrier, "x_h": normalized_carrier},
    ports=PORTS,
    states=("x_e", "x_h"),
    holomorphic=False,
)
def CarrierPhotodiode(
    signals: Signals,
    responsivity: float = 0.8,
    wavelength_nm: float = 1550.0,
    center_wavelength_nm: float = 1550.0,
    responsivity_bandwidth_nm: float = 250.0,
    reflectivity: float = 0.04,
    reflection_bandwidth_nm: float = 180.0,
    reflection_phase_rad: float = 0.0,
    reflection_phase_slope_rad_nm: float = 0.01,
    i_s: float = 1e-12,
    ideality: float = 1.5,
    v_t: float = 0.02585,
    i_leak: float = 1e-9,
    v_leak: float = 0.2,
    r_sh: float = 1e12,
    c0: float = 20e-15,
    v_bi: float = 0.8,
    fc: float = 0.5,
    volume_m3: float = 100e-18,
    density_scale_m3: float = 1e21,
    tau_e0: float = 8e-12,
    tau_h0: float = 40e-12,
    v_ref: float = 3.0,
    screening: float = 1.0,
    x_screen: float = 1.0,
    tau_rec: float = 500e-12,
    electron_weight: float = 0.5,
    tau_ref: float = 20e-12,
) -> PhysicsReturn:
    """Adds finite electron/hole extraction, recombination and screening.

    Device group:
        Photonic / Direct detection

    Device reference:
        photodetectors.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.photodetectors import CarrierPhotodiode

        jax.config.update("jax_enable_x64", True)
        model = CarrierPhotodiode()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.complex128)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(CarrierPhotodiode.__name__, "ports:", model.ports)
        ```

    """
    v = signals.anode - signals.cathode
    x_e, x_h = jnp.real(signals.x_e), jnp.real(signals.x_h)
    charge_scale = QE * volume_m3 * density_scale_m3
    optical_current, generation_current = optical_conversion(
        signals.optical_in,
        responsivity,
        wavelength_nm,
        center_wavelength_nm,
        responsivity_bandwidth_nm,
        reflectivity,
        reflection_bandwidth_nm,
        reflection_phase_rad,
        reflection_phase_slope_rad_nm,
    )
    generation = generation_current / charge_scale
    tau_e, tau_h, recombination = carrier_rates(x_e, x_h, v, tau_e0, tau_h0, v_bi, v_ref, screening, x_screen, tau_rec)
    i_photo = charge_scale * (electron_weight * x_e / tau_e + (1 - electron_weight) * x_h / tau_h)
    current = dark_iv(v, i_s, ideality, v_t, i_leak, v_leak, r_sh) - i_photo
    charge = junction_charge(v, c0, v_bi, fc)
    return {
        "optical_in": optical_current,
        "anode": current,
        "cathode": -current,
        "x_e": tau_ref * (x_e / tau_e + recombination - generation) + 1j * jnp.imag(signals.x_e),
        "x_h": tau_ref * (x_h / tau_h + recombination - generation) + 1j * jnp.imag(signals.x_h),
    }, {"anode": charge, "cathode": -charge, "x_e": tau_ref * signals.x_e, "x_h": tau_ref * signals.x_h}


@source(discipline=optical, ports=("p1", "p2"), holomorphic=True)
def OpticalGaussianPulse(
    signals: Signals, t: float, power: float = 1e-3, center: float = 120e-12, sigma: float = 12e-12, phase: float = 0.0
) -> PhysicsReturn:
    """A prescribed Gaussian incident-power pulse for arrival-time and detector-charge studies.

    Device group:
        Photonic / Optical sources and propagation

    Device reference:
        photodetectors.md

    Examples:
        ```python
        import jax
        import jax.numpy as jnp
        import numpy as np

        from circulax.components.photodetectors import OpticalGaussianPulse

        jax.config.update("jax_enable_x64", True)
        model = OpticalGaussianPulse()
        # Constitutive F/Q evaluation; physical circuit boundaries are shown in the model reference.
        y = jnp.zeros(len(model.ports) + len(model.states), dtype=jnp.complex128)
        f, q = model.solver_call(0.0, y, model)
        assert np.isfinite(f).all()
        assert np.isfinite(q).all()
        print(OpticalGaussianPulse.__name__, "ports:", model.ports)
        ```

    """
    incident = jnp.sqrt(power) * jnp.exp(-0.25 * ((t - center) / sigma) ** 2 + 1j * phase)
    current = signals.p1 - signals.p2 - 2 * incident
    return {"p1": current, "p2": -current}, {}


def make_photodetector_subcircuit(
    model_kind="carrier", series_resistance=20.0, pad_resistance=1e6, pad_capacitance=100e-15, with_parasitics=True
):
    """Reusable PD package: intrinsic core, true series R, and pad R || C.

    Physical ports: optical_in, anode (external), cathode, and pad_ground.
    The core's junction anode is an internal node on the other side of R_series.

    Wraps an intrinsic detector with series resistance and pad R/C.

    Device group:
        Photonic / Direct detection

    Device reference:
        photodetectors.md

    Examples:
        ```python
        from circulax.components.photodetectors import make_photodetector_subcircuit

        device = make_photodetector_subcircuit()
        assert device.sys_size > 0
        print("External ports:", tuple(device.ports))
        ```

    """
    from circulax import compile_circuit
    from circulax.components.electronic import Capacitor, Resistor

    models = {"ideal": IdealPhotodetector, "diode": Photodiode, "carrier": CarrierPhotodiode, "load": Resistor}
    if series_resistance <= 0 or pad_resistance <= 0 or pad_capacitance < 0:
        msg = "package resistances must be positive and pad capacitance nonnegative"
        raise ValueError(msg)
    if model_kind not in ("ideal", "diode", "carrier"):
        msg = "model_kind must be ideal, diode, or carrier"
        raise ValueError(msg)
    if not with_parasitics:
        return compile_circuit(
            {
                "instances": {"CORE": {"component": model_kind}},
                "connections": {},
                "ports": {port: f"CORE,{port}" for port in PORTS},
            },
            models,
            is_complex=True,
            backend="dense",
            rtol=1e-8,
            atol=1e-10,
        )
    package_netlist = {
        "instances": {
            "CORE": {"component": model_kind},
            "R_series": {"component": "load", "settings": {"R": series_resistance}},
            "R_pad": {"component": "load", "settings": {"R": pad_resistance}},
        },
        "connections": {},
        "nets": [
            {"p1": "CORE,anode", "p2": "R_series,p1"},
            {"p1": "R_series,p2", "p2": "R_pad,p1"},
        ],
        "ports": {
            "optical_in": "CORE,optical_in",
            "anode": "R_series,p2",
            "cathode": "CORE,cathode",
            "pad_ground": "R_pad,p2",
        },
    }
    if pad_capacitance > 0:
        package_netlist["instances"]["C_pad"] = {"component": "pad_capacitor", "settings": {"C": pad_capacitance}}
        package_netlist["nets"].extend(
            [
                {"p1": "R_series,p2", "p2": "C_pad,p1"},
                {"p1": "R_pad,p2", "p2": "C_pad,p2"},
            ]
        )
    return compile_circuit(
        package_netlist, {**models, "pad_capacitor": Capacitor}, is_complex=True, backend="dense", rtol=1e-8, atol=1e-10
    )
