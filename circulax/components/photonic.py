"""Photonic components for optical circuit simulation.

All wavelength parameters are in nanometres and power in watts.
"""

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp

from circulax.components.base_component import CircuitComponent, PhysicsReturn, Signals, component, source
from circulax.power_waves import power_wave_constraints, validate_reference
from circulax.s_transforms import s_to_y, sax_component  # noqa: F401 -- preserve conversion utility import compatibility

# ===========================================================================
# Passive Optical Components (S-Matrix based)
# ===========================================================================


def _bounded_parameter(value: float, name: str, *, lower: float = 0.0, upper: float = float("inf")) -> jnp.ndarray:
    """Validate a real passive parameter without clipping its physical law."""
    value = jnp.asarray(value)
    invalid = ~jnp.isfinite(value) | (value < lower) | (value > upper)
    return eqx.error_if(value, invalid, f"{name} must be finite and in [{lower}, {upper}]")


def _waveguide_scattering(
    length_um: float = 100.0,
    loss_dB_cm: float = 1.0,
    neff: float = 2.4,
    n_group: float = 4.0,
    center_wavelength_nm: float = 1310.0,
    wavelength_nm: float = 1310.0,
) -> dict:
    """Reciprocal matched waveguide with linearised index dispersion.

    Length is in um, loss in dB/cm, wavelengths in nm. Transmission is
    10**(-loss_dB_cm*length_um/10000/20) * exp(-1j*phi), with
    phi=2*pi*n_eff_disp*length_um*1000/wavelength_nm and
    n_eff_disp=neff+(neff-n_group)*(wavelength_nm-center_wavelength_nm)/center_wavelength_nm.
    This instantaneous carrier-envelope law adds no envelope flight time.
    Passive operation requires nonnegative length and propagation loss,
    and positive wavelengths. Both incident quadratures are retained per port.
    """
    length_um = _bounded_parameter(length_um, "length_um")
    loss_dB_cm = _bounded_parameter(loss_dB_cm, "loss_dB_cm")
    wavelength_nm = _bounded_parameter(wavelength_nm, "wavelength_nm", lower=jnp.finfo(float).tiny)
    center_wavelength_nm = _bounded_parameter(center_wavelength_nm, "center_wavelength_nm", lower=jnp.finfo(float).tiny)
    d_lam = wavelength_nm - center_wavelength_nm
    n_eff_disp = neff + (neff - n_group) / center_wavelength_nm * d_lam
    phi = 2.0 * jnp.pi * n_eff_disp * (length_um / wavelength_nm) * 1000.0
    transmission = 10.0 ** (-loss_dB_cm * length_um / 10000.0 / 20.0) * jnp.exp(-1j * phi)
    return {("p1", "p2"): transmission, ("p2", "p1"): transmission}


def optical_waveguide(*, r0: float = 1.0) -> type[CircuitComponent]:
    """Build OpticalWaveguide with a static finite positive real reference."""
    return sax_component(_waveguide_scattering, name="OpticalWaveguide", r0=r0)


OpticalWaveguide = optical_waveguide()


def optical_delay_line(*, r0: float = 1.0) -> type[CircuitComponent]:
    """Build a reciprocal carrier-envelope delay with a static real reference.

    Both incident waves are retained. Outgoing at each port is the opposite
    incident wave at t-tau, multiplied by physical attenuation and exp(-j*phi).
    Length is um, wavelength nm, loss dB/cm, tau=length_um*n_group/c.
    DC uses constant fields; transient holds y0 before t0 and interpolates
    accepted history/current Newton trials. HB uses a periodic Fourier shift.
    See docs/bidirectional_optical_delays.md for analysis and history limits.
    """
    r0 = validate_reference(r0)

    @component(ports=("p1", "p2"), states=("wave_p1", "wave_p2"), holomorphic=True)
    def OpticalDelayLine(  # noqa: PLR0917 -- public component parameters
        signals: Signals,
        length_um: float = 100.0,
        loss_dB_cm: float = 1.0,
        neff: float = 2.4,
        n_group: float = 4.0,
        wavelength_nm: float = 1310.0,
    ) -> PhysicsReturn:
        """Reciprocal propagation b1=T*a2(t-tau), b2=T*a1(t-tau)."""
        length_um = _bounded_parameter(length_um, "length_um")
        loss_dB_cm = _bounded_parameter(loss_dB_cm, "loss_dB_cm")
        n_group = _bounded_parameter(n_group, "n_group")
        wavelength_nm = _bounded_parameter(wavelength_nm, "wavelength_nm", lower=jnp.finfo(float).tiny)
        phi = 2.0 * jnp.pi * neff * (length_um / wavelength_nm) * 1000.0
        transmission = 10.0 ** (-loss_dB_cm * length_um / 10000.0 / 20.0) * jnp.exp(-1j * phi)
        delayed = signals.at_delay(length_um * n_group / 2.99792458e14)
        outgoing = transmission * jnp.array([delayed.wave_p2, delayed.wave_p1])
        return power_wave_constraints(signals, ("p1", "p2"), outgoing, r0=r0, algebraic_flow=True)

    OpticalDelayLine._wave_r0 = r0  # noqa: SLF001 -- compiler reference metadata
    OpticalDelayLine._wave_algebraic_flow = True  # noqa: SLF001 -- terminal F is history independent
    return OpticalDelayLine


OpticalDelayLine = optical_delay_line()


def nonlinear_optical_waveguide(*, r0: float = 1.0) -> type[CircuitComponent]:
    """Build a bidirectional delayed waveguide with independent directional TPA.

    Preserves the law from photonics_transient.ipynb: each retarded incident
    envelope u produces T*u/sqrt(1+tpa_coeff*|u|**2*effective_length).
    T includes linear field loss and negative carrier phase; effective length
    is in cm and tpa_coeff in 1/(W cm). There is no cross-direction absorption,
    Kerr phase, carrier population or dispersion. References are static,
    finite, positive and real; envelopes are in sqrt(W), independently of r0.

    DC, transient, full real-block small_signal and periodic HB are supported.
    Legacy complex-linear ac_response is rejected for this intensity law.
    History, interpolation and initialization follow OpticalDelayLine.
    See docs/nonlinear_optical_power_waves.md for the observation-plane audit.
    """
    r0 = validate_reference(r0)

    @component(ports=("p1", "p2"), states=("wave_p1", "wave_p2"), holomorphic=False)
    def NonlinearOpticalWaveguide(  # noqa: PLR0917 -- public component parameters
        signals: Signals,
        length_um: float = 100.0,
        loss_dB_cm: float = 1.0,
        tpa_coeff: float = 0.0,
        neff: float = 2.4,
        n_group: float = 4.0,
        wavelength_nm: float = 1310.0,
    ) -> PhysicsReturn:
        """Propagate each delayed incident intensity without mixing directions."""
        length_um = _bounded_parameter(length_um, "length_um")
        loss_dB_cm = _bounded_parameter(loss_dB_cm, "loss_dB_cm")
        tpa_coeff = _bounded_parameter(tpa_coeff, "tpa_coeff")
        n_group = _bounded_parameter(n_group, "n_group")
        wavelength_nm = _bounded_parameter(wavelength_nm, "wavelength_nm", lower=jnp.finfo(float).tiny)
        length_cm = length_um / 10000.0
        alpha_cm = loss_dB_cm * jnp.log(10.0) / 10.0
        # Keep the notebook's small-alpha branch, but avoid 0/0 in inactive AD.
        denominator = jnp.where(alpha_cm > 1e-12, alpha_cm, 1.0)
        effective_length = jnp.where(alpha_cm > 1e-12, -jnp.expm1(-alpha_cm * length_cm) / denominator, length_cm)
        phase = 2.0 * jnp.pi * neff * length_um * 1000.0 / wavelength_nm
        transmission = 10.0 ** (-loss_dB_cm * length_cm / 20.0) * jnp.exp(-1j * phase)
        past = signals.at_delay(length_um * n_group / 2.99792458e14)
        incident = jnp.array([past.wave_p2, past.wave_p1])
        outgoing = transmission * incident / jnp.sqrt(1.0 + tpa_coeff * jnp.abs(incident) ** 2 * effective_length)
        return power_wave_constraints(signals, ("p1", "p2"), outgoing, r0=r0, algebraic_flow=True)

    NonlinearOpticalWaveguide._wave_r0 = r0  # noqa: SLF001 -- compiler reference metadata
    NonlinearOpticalWaveguide._wave_algebraic_flow = True  # noqa: SLF001 -- complete instantaneous terminal F
    return NonlinearOpticalWaveguide


NonlinearOpticalWaveguide = nonlinear_optical_waveguide()


def _delay_scattering(
    f: float,
    length_um: float = 100.0,
    loss_dB_cm: float = 1.0,
    n_group: float = 4.0,
) -> dict:
    """Modulation-only reciprocal delay; deliberately excludes carrier phase."""
    length_um = _bounded_parameter(length_um, "length_um")
    loss_dB_cm = _bounded_parameter(loss_dB_cm, "loss_dB_cm")
    n_group = _bounded_parameter(n_group, "n_group")
    tau = length_um * n_group / 2.99792458e14
    transmission = 10.0 ** (-loss_dB_cm * length_um / 10000.0 / 20.0) * jnp.exp(-2j * jnp.pi * f * tau)
    return {("p1", "p2"): transmission, ("p2", "p1"): transmission}


def optical_delay_line_fdomain(*, r0: float = 1.0) -> type[CircuitComponent]:
    """Build a retained-wave DC/AC/HB delay operator at a static real reference.

    Transmission is attenuation*exp(-j*2*pi*f*tau), with f the modulation
    frequency in Hz. No optical carrier phase is included. Transient and
    time-domain port-flow reconstruction remain explicitly unsupported.
    """
    return sax_component(_delay_scattering, frequency_param="f", name="delay_line_fdomain", r0=r0)


delay_line_fdomain = optical_delay_line_fdomain()


def _grating_scattering(
    center_wavelength_nm: float = 1310.0,
    peak_loss_dB: float = 0.0,
    bandwidth_1dB: float = 20.0,
    wavelength_nm: float = 1310.0,
) -> dict:
    """Reciprocal matched grating with Gaussian wavelength insertion loss.

    Wavelengths and full 1 dB bandwidth are in nm. Loss is peak_loss_dB plus
    ((wavelength_nm-center_wavelength_nm)/(bandwidth_1dB/2))**2 in dB.
    Passive operation requires nonnegative peak loss and positive bandwidth.
    Zero insertion loss gives exactly unit transmission, without a cap.
    Port order is (grating, waveguide).
    """
    peak_loss_dB = _bounded_parameter(peak_loss_dB, "peak_loss_dB")
    bandwidth_1dB = _bounded_parameter(bandwidth_1dB, "bandwidth_1dB", lower=jnp.finfo(float).tiny)
    delta = wavelength_nm - center_wavelength_nm
    loss = peak_loss_dB + (delta / (0.5 * bandwidth_1dB)) ** 2
    transmission = 10.0 ** (-loss / 20.0)
    return {("grating", "waveguide"): transmission, ("waveguide", "grating"): transmission}


def grating(*, r0: float = 1.0) -> type[CircuitComponent]:
    """Build Grating with a static finite positive real reference."""
    return sax_component(_grating_scattering, name="Grating", r0=r0)


Grating = grating()


def _splitter_scattering(split_ratio: float = 0.5) -> dict:
    """Reciprocal passive three-port splitter with an omitted loss channel.

    For 0 <= split_ratio <= 1, p1 sends that power fraction to p2 and the
    remainder to p3 with +j phase. All physical ports are matched and there
    is no p2/p3 coupling. This matrix is NOT unitary: its singular values
    are (1, 1, 0). For arbitrary incident a, missing power is
    abs(1j*sqrt(1-split_ratio)*a_p2 + sqrt(split_ratio)*a_p3)**2.
    That orthogonal reverse-input mode exits an omitted matched loss channel.
    Forward-only input is lossless. Port order is (p1, p2, p3).
    """
    split_ratio = _bounded_parameter(split_ratio, "split_ratio", upper=1.0)
    r = jnp.sqrt(split_ratio)
    tc = 1j * jnp.sqrt(1.0 - split_ratio)
    return {("p1", "p2"): r, ("p2", "p1"): r, ("p1", "p3"): tc, ("p3", "p1"): tc}


def splitter(*, r0: float = 1.0) -> type[CircuitComponent]:
    """Build Splitter with a static finite positive real reference."""
    return sax_component(_splitter_scattering, name="Splitter", r0=r0)


Splitter = splitter()


def _coupler_scattering(t: jnp.ndarray, k: jnp.ndarray) -> dict:
    """Reciprocal four-port scattering with bar t and cross j*k."""
    return {
        ("p1", "p3"): t,
        ("p3", "p1"): t,
        ("p2", "p4"): t,
        ("p4", "p2"): t,
        ("p1", "p4"): 1j * k,
        ("p4", "p1"): 1j * k,
        ("p2", "p3"): 1j * k,
        ("p3", "p2"): 1j * k,
    }


def _directional_coupler_scattering(coupling: float = 0.5) -> dict:
    """Lossless reciprocal directional coupler, ordered (p1, p2, p3, p4).

    For 0 <= coupling <= 1, bar amplitude sqrt(1-coupling) connects p1/p3
    and p2/p4; cross amplitude j*sqrt(coupling) connects p1/p4 and p2/p3.
    Exact endpoints are supported. Square-root parameter derivatives need
    not be finite at endpoints; use TunableBeamSplitter for a smooth angle.
    """
    coupling = _bounded_parameter(coupling, "coupling", upper=1.0)
    return _coupler_scattering(jnp.sqrt(1.0 - coupling), jnp.sqrt(coupling))


def directional_coupler(*, r0: float = 1.0) -> type[CircuitComponent]:
    """Build DirectionalCoupler with a static finite positive real reference."""
    return sax_component(_directional_coupler_scattering, name="DirectionalCoupler", r0=r0)


DirectionalCoupler = directional_coupler()


# ===========================================================================
# Optical Sources
# ===========================================================================


@component(
    ports=("p1", "p2"),
    states=("i_src",),
    holomorphic=True,
    ac_source_stamp={"i_src": 1.0},
    ideal_voltage_constraint_ports=("p1", "p2"),
)
def OpticalSource(signals: Signals, power: float = 1.0, phase: float = 0.0) -> PhysicsReturn:
    """Ideal CW optical source for DC and small-signal AC analysis.

    Enforces a fixed complex field amplitude ``sqrt(power) * exp(j * phase)``
    across its ports, analogous to an ideal voltage source in electrical circuits.
    ``power`` sets the squared potential difference, not a prescribed launched
    traveling-wave power under reflection. For that contract use
    :class:`circulax.components.OpticalWaveSource`.

    Args:
        signals: Port fields and source-current state ``i_src``.
        power: Squared potential-difference amplitude. Defaults to ``1.0``.
        phase: Output field phase in radians. Defaults to ``0.0``.

    """
    v_val = jnp.sqrt(power) * jnp.exp(1j * phase)
    constraint = (signals.p1 - signals.p2) - v_val

    return {"p1": signals.i_src, "p2": -signals.i_src, "i_src": constraint}, {}


@source(
    ports=("p1", "p2"),
    states=("i_src",),
    holomorphic=True,
    ac_source_stamp={"i_src": 1.0},
    ideal_voltage_constraint_ports=("p1", "p2"),
)
def OpticalSourcePulse(
    signals: Signals,
    t: float,
    power: float = 1.0,
    phase: float = 0.0,
    delay: float = 0.2e-9,
    rise: float = 0.05e-9,
) -> PhysicsReturn:
    """Time-dependent optical source with a sigmoid turn-on profile.

    Field amplitude ramps smoothly from zero to ``sqrt(power)`` around
    ``delay``, with the steepness of the transition controlled by ``rise``.
    Suitable for transient simulations of optical pulse propagation.
    This remains a potential-difference constraint; ``power`` does not prescribe
    launched traveling-wave power under reflection. Use
    :func:`circulax.components.optical_wave_source` for a prescribed wave envelope.

    Args:
        signals: Port fields and source-current state ``i_src``.
        t: Current simulation time in seconds.
        power: Peak squared potential-difference amplitude. Defaults to ``1.0``.
        phase: Output field phase in radians. Defaults to ``0.0``.
        delay: Turn-on delay in seconds. Defaults to ``0.2e-9``.
        rise: Sigmoid rise time constant in seconds. Defaults to ``0.05e-9``.

    """
    val = jnp.sqrt(power) * jnn.sigmoid((t - delay) / rise)
    v_val = val * jnp.exp(1j * phase)

    constraint = (signals.p1 - signals.p2) - v_val

    return {"p1": signals.i_src, "p2": -signals.i_src, "i_src": constraint}, {}


# ===========================================================================
# Tunable Active Components
# ===========================================================================


def _tunable_beam_splitter_scattering(theta: float = jnp.pi / 4) -> dict:
    """Lossless reciprocal four-port coupler with bar cos(theta), cross j*sin(theta).

    Port order is (p1, p2, p3, p4). Forward transmission retains the former
    convention; reverse illumination and reflecting loads now obey the same
    reciprocal scattering law. Four incident-wave states replace i_top/i_bot.
    The angle parameter is smooth, including bar/cross endpoints.
    """
    return _coupler_scattering(jnp.cos(theta), jnp.sin(theta))


def tunable_beam_splitter(*, r0: float = 1.0) -> type[CircuitComponent]:
    """Build TunableBeamSplitter with a static finite positive real reference."""
    return sax_component(_tunable_beam_splitter_scattering, name="TunableBeamSplitter", r0=r0)


TunableBeamSplitter = tunable_beam_splitter()
