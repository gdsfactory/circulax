"""Photonic components for optical circuit simulation.

All wavelength parameters are in nanometres and power in watts.
"""

import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp

from circulax.components.base_component import CircuitComponent, PhysicsReturn, Signals, component, source
from circulax.s_transforms import fdomain_component, s_to_y, sax_component

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


@component(ports=("p1", "p2"), states=("i_p2",))
def OpticalDelayLine(
    signals: Signals,
    length_um: float = 100.0,
    loss_dB_cm: float = 1.0,
    neff: float = 2.4,
    n_group: float = 4.0,
    wavelength_nm: float = 1310.0,
) -> PhysicsReturn:
    """Waveguide modelled as an explicit time-of-flight delay line (real group delay).

    Unlike :func:`OpticalWaveguide` -- an instantaneous carrier-envelope stamp
    without envelope flight time -- this component enforces
    the actual transient envelope delay: the output field at ``p2`` is the
    input field at ``p1`` from ``tau = length_um * n_group / c`` seconds ago,
    attenuated and phase-shifted, via a legacy VCVS-style constraint. ``p1``
    carries no self-stamp and must be driven by a connected source/component,
    using a legacy directional convention.

    See ``circulax.solvers.assembly`` for how the delayed local state is
    computed -- a fixed-size accepted-step history buffer read via
    ``jnp.interp``, vmapped per-instance since ``tau`` may vary across
    instances of different length. Adaptive and fixed step-size controllers
    may take steps longer than ``tau``; interpolation against the current
    Newton trial supplies the required delay Jacobian in that regime.

    Args:
        signals: Current and delayed component-local fields and branch state.
        length_um: Waveguide length in micrometres. Defaults to ``100.0``.
        loss_dB_cm: Propagation loss in dB/cm. Defaults to ``1.0``.
        neff: Effective refractive index, used for the carrier phase shift
            over the delay. Defaults to ``2.4``.
        n_group: Group refractive index; sets the propagation delay via
            ``tau = length_um * n_group / c``. Defaults to ``4.0``.
        wavelength_nm: Operating wavelength in nm. Defaults to ``1310.0``.

    """
    phi = 2.0 * jnp.pi * neff * (length_um / wavelength_nm) * 1000.0
    loss_val = loss_dB_cm * (length_um / 10000.0)
    T_mag = 10.0 ** (-loss_val / 20.0)
    T = T_mag * jnp.exp(-1j * phi)

    c_um_per_s = 2.99792458e14
    delay = (length_um * n_group) / c_um_per_s
    constraint = signals.p2 - T * signals.at_delay(delay).p1

    return {"p1": 0.0, "p2": signals.i_p2, "i_p2": constraint}, {}


@fdomain_component(ports=("p1", "p2"))
def delay_line_fdomain(
    f: float,
    length_um: float = 100.0,
    loss_dB_cm: float = 1.0,
    n_group: float = 4.0,
) -> jnp.ndarray:
    """Frequency-domain delay line for AC and harmonic-balance analysis.

    Models a pure transmission-line group delay: ``S21 = T_mag * exp(-j 2pi f tau)``
    where ``tau = length_um * n_group / c`` and ``T_mag`` accounts for
    propagation loss.  The S-matrix is converted to a Y-matrix via
    :func:`s_to_y`.

    This is a frequency-domain-only component (``f`` = modulation / signal
    frequency, e.g. 1 GHz for an AC sweep).  Carrier-phase effects
    (``neff``, ``wavelength_nm``) belong in the wavelength-domain
    :func:`OpticalWaveguide` and are intentionally excluded here.

    This component is AC/HB-only; it cannot be used in transient simulation
    (fdomain components raise at transient setup time).

    Args:
        f: Modulation frequency in Hz (supplied by the AC/HB solver).
        length_um: Waveguide length in micrometres. Defaults to ``100.0``.
        loss_dB_cm: Propagation loss in dB/cm. Defaults to ``1.0``.
        n_group: Group refractive index; sets the propagation delay via
            ``tau = length_um * n_group / c``. Defaults to ``4.0``.

    """
    c_um_per_s = 2.99792458e14
    loss_val = loss_dB_cm * (length_um / 10000.0)
    T_mag = 10.0 ** (-loss_val / 20.0)
    tau = (length_um * n_group) / c_um_per_s
    T = T_mag * jnp.exp(-1j * 2.0 * jnp.pi * f * tau)
    S = jnp.array([[0.0, T], [T, 0.0]], dtype=jnp.complex128)
    return s_to_y(S)


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
