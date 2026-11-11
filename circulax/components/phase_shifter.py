# Copyright 2026 Circulax contributors

"""Electro-optic phase shifters built from designer-supplied physics.

:func:`phase_shifter_factory` turns functions of junction voltage into a
four-port component with electrical ports ``anode``/``cathode`` and optical
ports ``o1``/``o2``. The optical path has a native, bidirectional group delay,
so the same component serves DC operating points and transient simulation,
including traveling-wave modulators built from repeated sections. Compile
circuits that contain it with ``is_complex=True``. The junction voltage is
taken as a real part, so the component is not holomorphic. Use
``Circuit.small_signal()`` for its complete field quadratures;
the legacy ``ac_response()`` interface excludes this mixed optical model.

The electrical junction is a separate :class:`JunctionModel`, built with
:func:`pn_diode` or :func:`junction_model`, which can be evaluated on its own
(I-V, C-V) before it is passed to the phase shifter. The helpers in this
module convert typical extracted data (VπL and C(V) tables, depletion and
diode parameters) into functions of voltage. Any keyword arguments of a
supplied function become runtime component parameters that can be swept with
``circuit.dc(params=...)`` or differentiated with JAX.

Units follow :mod:`circulax.components.photonic`: wavelengths in nm, device
length in µm and optical loss in dB/cm. Electrical quantities are per metre
of device length and carry their unit in the argument name.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from circulax.components.base_component import CircuitComponent, PhysicsReturn, Signals, component
from circulax.power_waves import power_wave_constraints, validate_reference

__all__ = [
    "JunctionModel",
    "charge_from_capacitance_table",
    "delta_neff_from_vpil",
    "depletion_charge",
    "diode_current",
    "junction_model",
    "phase_shifter_factory",
    "pn_diode",
    "tabulated",
]

C_UM_PER_S = 2.99792458e14
_RUNTIME_DEFAULTS = {"length_um": 100.0, "neff": 2.4, "n_group": 4.0, "wavelength_nm": 1310.0}
_RESERVED = {"signals", "s", "t", "init", *_RUNTIME_DEFAULTS}
_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(16)

VoltageFunction = Callable[..., Any]


# ---------------------------------------------------------------------------
# Table and physics helpers
# ---------------------------------------------------------------------------
def _table(voltages: ArrayLike, values: ArrayLike, label: str) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(voltages, dtype=float)
    y = np.asarray(values, dtype=float)
    if v.ndim != 1 or y.shape != v.shape or len(v) < 2:
        msg = f"{label}: voltages and values must be 1-D arrays of equal length >= 2"
        raise ValueError(msg)
    order = np.argsort(v)
    v, y = v[order], y[order]
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(y))) or np.any(np.diff(v) <= 0):
        msg = f"{label}: values must be finite and voltages distinct"
        raise ValueError(msg)
    return v, y


def tabulated(voltages: ArrayLike, values: ArrayLike) -> VoltageFunction:
    """Return a piecewise-linear function of voltage from a sampled table.

    Use it for ``delta_neff``, ``loss_dB_cm`` or a junction ``current`` when
    data is tabulated. Values are held constant outside the table. For a measured
    capacitance table use :func:`charge_from_capacitance_table` instead, which
    integrates exactly and keeps the capacitance continuous.
    """
    v, y = _table(voltages, values, "tabulated")
    v_j, y_j = jnp.asarray(v), jnp.asarray(y)

    def table(voltage: Any) -> Any:
        return jnp.interp(voltage, v_j, y_j)

    return table


def _integrated(voltages: np.ndarray, slopes: np.ndarray, reference_voltage: float) -> VoltageFunction:
    """Exact integral of a piecewise-linear slope, extrapolated with the end slopes."""
    x, s = jnp.asarray(voltages), jnp.asarray(slopes)
    nodes = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (s[1:] + s[:-1]) * jnp.diff(x))])
    n = len(voltages)

    def integral(v: Any) -> Any:
        # Clamp only to the table ends: an interior clip would halve the
        # gradient at knots. The end terms below restore the slope outside.
        v_in = jnp.clip(v, x[0], x[-1])
        k = jnp.clip(jnp.searchsorted(x, v_in, side="right") - 1, 0, n - 2)
        h = x[k + 1] - x[k]
        d = v_in - x[k]
        inside = nodes[k] + s[k] * d + 0.5 * (s[k + 1] - s[k]) / h * d**2
        return inside + jnp.minimum(v - x[0], 0.0) * s[0] + jnp.maximum(v - x[-1], 0.0) * s[-1]

    offset = integral(jnp.asarray(float(reference_voltage)))

    def function(v: Any) -> Any:
        return integral(v) - offset

    return function


def delta_neff_from_vpil(
    voltages: ArrayLike,
    vpil_V_cm: ArrayLike,
    *,
    wavelength_nm: float,
    reference_voltage: float = 0.0,
) -> VoltageFunction:
    """Integrate a small-signal VπL table into ``delta_neff(v)``.

    ``vpil_V_cm`` is the small-signal VπL in V·cm at each junction voltage
    ``v = V_anode - V_cathode`` (negative for reverse bias), measured at
    ``wavelength_nm``. The slope magnitude is ``|dn_eff/dV| = λ / (2·VπL)``.
    Its sign follows plasma dispersion in a depletion-mode junction: reverse
    bias removes carriers, so ``n_eff`` *increases* as ``v`` decreases.

    The slope is interpolated linearly between samples and integrated exactly,
    so ``delta_neff`` has a continuous derivative inside the table and
    extrapolates with the end slopes. ``delta_neff(reference_voltage) = 0``;
    the factory's ``neff`` is therefore the effective index at that voltage.
    """
    v, vpil = _table(voltages, vpil_V_cm, "delta_neff_from_vpil")
    if np.any(vpil <= 0):
        msg = "delta_neff_from_vpil: VπL values must be positive"
        raise ValueError(msg)
    slope = -(wavelength_nm * 1e-9) / (2.0 * vpil * 1e-2)
    return _integrated(v, slope, reference_voltage)


def charge_from_capacitance_table(
    voltages: ArrayLike,
    capacitance_F_per_m: ArrayLike,
    *,
    reference_voltage: float = 0.0,
) -> VoltageFunction:
    """Integrate a C(V) table, in F/m, into a junction charge function in C/m.

    The capacitance is interpolated linearly and integrated exactly, so
    ``d charge / dv`` reproduces the table and stays continuous; outside the
    table the end capacitances are held. Pass the result as
    ``charge`` to :func:`junction_model`.
    """
    v, cap = _table(voltages, capacitance_F_per_m, "charge_from_capacitance_table")
    if np.any(cap < 0):
        msg = "charge_from_capacitance_table: capacitance must be nonnegative"
        raise ValueError(msg)
    return _integrated(v, cap, reference_voltage)


def depletion_charge(
    v: Any,
    cj0_F_per_m: float = 3.0e-10,
    vbi: float = 0.9,
    grading: float = 0.5,
    fc: float = 0.5,
) -> Any:
    """Depletion charge per metre with SPICE forward-bias linearisation.

    ``C(v) = cj0 / (1 - v/vbi)**grading`` for ``v < fc*vbi``. Above that the
    capacitance continues linearly (the standard SPICE ``FC`` treatment), so
    the charge and its derivative stay finite if the junction is driven
    forward during Newton iterations. ``charge(0) = 0``.
    """
    v_fc = fc * vbi
    one_m = 1.0 - grading
    v_safe = jnp.minimum(v, v_fc)
    q_rev = cj0_F_per_m * vbi / one_m * (1.0 - (1.0 - v_safe / vbi) ** one_m)
    f1 = (1.0 - fc) ** (-1.0 - grading)
    dv = jnp.maximum(v - v_fc, 0.0)
    q_fwd = cj0_F_per_m * f1 * ((1.0 - fc * (1.0 + grading)) * dv + grading / (2.0 * vbi) * ((v_fc + dv) ** 2 - v_fc**2))
    return q_rev + q_fwd


def diode_current(
    v: Any,
    *,
    is_A_per_m: float = 1.0e-8,
    ideality: float = 1.0,
    vt: float = 0.025852,
    v_limit: float = 0.9,
    dark_current_A_per_m: float = 0.0,
    dark_bias_V: float = -1.0,
) -> Any:
    """Diode current per metre: Shockley equation plus a dark-current shunt.

    ``I = is*(exp(v/(ideality*vt)) - 1) - dark_current * v / dark_bias_V``.
    The second term is an ohmic leakage sized so that it carries
    ``dark_current_A_per_m`` at the reverse bias ``dark_bias_V`` (negative),
    in addition to the saturation current. The exponential is replaced by its
    tangent above ``v_limit`` so the current stays finite during Newton
    iterations.
    """
    n_vt = ideality * vt
    e = jnp.exp(jnp.minimum(v, v_limit) / n_vt)
    shockley = is_A_per_m * (e * (1.0 + jnp.maximum(v - v_limit, 0.0) / n_vt) - 1.0)
    return shockley - dark_current_A_per_m * v / dark_bias_V


# ---------------------------------------------------------------------------
# Function inspection
# ---------------------------------------------------------------------------
def _as_function(value: Any, label: str) -> tuple[VoltageFunction | None, list[inspect.Parameter]]:
    """Normalise a callable/scalar input and return its keyword parameters."""
    if value is None:
        return None, []
    if not callable(value):
        constant = float(value)
        return (lambda v: constant * jnp.ones_like(v)), []
    try:
        parameters = list(inspect.signature(value).parameters.values())
    except (TypeError, ValueError) as exc:
        msg = f"{label}: cannot inspect the signature of {value!r}"
        raise TypeError(msg) from exc
    if not parameters or parameters[0].kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
        msg = f"{label}: the first argument must be the junction voltage"
        raise TypeError(msg)
    extra = []
    for p in parameters[1:]:
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            msg = f"{label}: *args/**kwargs are not supported; name each parameter with a default"
            raise TypeError(msg)
        if p.default is inspect.Parameter.empty:
            msg = f"{label}: parameter {p.name!r} needs a default value"
            raise TypeError(msg)
        default = p.default
        if isinstance(default, bool) or not isinstance(default, (int, float)):
            msg = f"{label}: parameter {p.name!r} must have a numeric default"
            raise TypeError(msg)
        if p.name in _RESERVED:
            msg = f"{label}: parameter name {p.name!r} is reserved"
            raise ValueError(msg)
        extra.append(inspect.Parameter(p.name, inspect.Parameter.KEYWORD_ONLY, default=float(default)))
    return value, extra


def _merge(*groups: list[inspect.Parameter] | tuple[inspect.Parameter, ...]) -> dict[str, inspect.Parameter]:
    """Merge keyword parameters by name; a shared name must share its default."""
    merged: dict[str, inspect.Parameter] = {}
    for p in (p for group in groups for p in group):
        if p.name in merged and merged[p.name].default != p.default:
            msg = f"Parameter {p.name!r} is declared with different defaults ({merged[p.name].default}, {p.default})"
            raise ValueError(msg)
        merged[p.name] = p
    return merged


def _keywords(params: tuple[inspect.Parameter, ...] | list[inspect.Parameter], kw: dict[str, Any]) -> dict[str, Any]:
    return {p.name: kw[p.name] for p in params}


# ---------------------------------------------------------------------------
# Junction model
# ---------------------------------------------------------------------------
class JunctionModel:
    """Electrical junction model per metre of length, usable on its own.

    Create one with :func:`junction_model` or :func:`pn_diode`. Evaluate it
    directly with :meth:`current`, :meth:`charge` and :meth:`capacitance`
    (arrays and parameter overrides are accepted), build a two-terminal
    circuit component with :meth:`component`, or pass it to
    :func:`phase_shifter_factory` as ``junction``.
    """

    def __init__(
        self,
        current_fn: VoltageFunction | None,
        current_params: list[inspect.Parameter],
        charge_fn: VoltageFunction | None,
        charge_params: list[inspect.Parameter],
    ) -> None:
        """Store the per-metre current and charge functions and their parameters."""
        self._current_fn, self._current_params = current_fn, tuple(current_params)
        self._charge_fn, self._charge_params = charge_fn, tuple(charge_params)
        self.parameters = tuple(_merge(current_params, charge_params).values())

    @property
    def defaults(self) -> dict[str, float]:
        """Default value of every runtime parameter."""
        return {p.name: p.default for p in self.parameters}

    def _resolve(self, overrides: dict[str, Any]) -> dict[str, Any]:
        unknown = set(overrides) - set(self.defaults)
        if unknown:
            msg = f"Unknown junction parameters {sorted(unknown)}; available: {sorted(self.defaults)}"
            raise TypeError(msg)
        return {**self.defaults, **overrides}

    def current_per_m(self, v: Any, kw: dict[str, Any]) -> Any:
        """Junction current in A/m for already-resolved parameters (traced use)."""
        if self._current_fn is None:
            return jnp.zeros_like(v)
        return self._current_fn(v, **_keywords(self._current_params, kw))

    def charge_per_m(self, v: Any, kw: dict[str, Any]) -> Any:
        """Junction charge in C/m for already-resolved parameters (traced use)."""
        if self._charge_fn is None:
            return jnp.zeros_like(v)
        return self._charge_fn(v, **_keywords(self._charge_params, kw))

    def current(self, v: ArrayLike, **overrides: Any) -> Any:
        """DC current per metre, anode to cathode, in A/m."""
        kw = self._resolve(overrides)
        return jnp.vectorize(lambda x: self.current_per_m(x, kw))(jnp.asarray(v, dtype=float))

    def charge(self, v: ArrayLike, **overrides: Any) -> Any:
        """Stored charge per metre in C/m, zero at 0 V for the built-in models."""
        kw = self._resolve(overrides)
        return jnp.vectorize(lambda x: self.charge_per_m(x, kw))(jnp.asarray(v, dtype=float))

    def capacitance(self, v: ArrayLike, **overrides: Any) -> Any:
        """Small-signal capacitance per metre, ``d charge / dv``, in F/m."""
        kw = self._resolve(overrides)
        return jnp.vectorize(jax.grad(lambda x: self.charge_per_m(x, kw)))(jnp.asarray(v, dtype=float))

    def component(self, name: str = "Junction") -> type[CircuitComponent]:
        """Return a two-terminal component with ports ``anode`` and ``cathode``.

        Its runtime parameters are ``length_um`` plus :attr:`defaults`; the
        component carries ``length * current(v)`` and stores ``length * charge(v)``.
        """
        junction = self

        def physics(signals: Signals, **kw: Any) -> PhysicsReturn:
            v = jnp.real(signals.anode - signals.cathode)
            length_m = kw["length_um"] * 1e-6
            current = length_m * junction.current_per_m(v, kw)
            charge = length_m * junction.charge_per_m(v, kw)
            return {"anode": current, "cathode": -current}, {"anode": charge, "cathode": -charge}

        _set_signature(physics, name, {"length_um": _RUNTIME_DEFAULTS["length_um"]}, self.parameters)
        return component(ports=("anode", "cathode"))(physics)


def junction_model(
    *,
    current: VoltageFunction | float | None = None,
    charge: VoltageFunction | None = None,
    capacitance: VoltageFunction | float | None = None,
    defaults: dict[str, float] | None = None,
) -> JunctionModel:
    """Build a junction model from functions of voltage, per metre of length.

    Args:
        current: DC current in A/m, anode to cathode, e.g. :func:`diode_current`.
        charge: Stored charge in C/m, e.g. :func:`depletion_charge` or
            :func:`charge_from_capacitance_table`.
        capacitance: Alternative to ``charge``: capacitance in F/m, integrated
            from 0 V with 16-point Gauss-Legendre quadrature. Must be smooth
            over the voltages used.
        defaults: Replacement defaults for keyword parameters of the functions.

    Functions take the junction voltage ``v = V_anode - V_cathode`` first and
    may declare keyword parameters with numeric defaults; those become
    runtime parameters wherever the model is used.

    """
    if charge is not None and capacitance is not None:
        msg = "Supply either charge or capacitance, not both"
        raise ValueError(msg)
    current_fn, current_params = _as_function(current, "current")
    if capacitance is not None:
        cap_fn, charge_params = _as_function(capacitance, "capacitance")
        gauss_x, gauss_w = jnp.asarray(_GAUSS_X), jnp.asarray(_GAUSS_W)

        def charge_fn(v: Any, **kw: Any) -> Any:
            half = 0.5 * v
            return half * jnp.sum(gauss_w * cap_fn(half * (gauss_x + 1.0), **kw))

    else:
        charge_fn, charge_params = _as_function(charge, "charge")

    if defaults:
        known = {p.name for p in (*current_params, *charge_params)}
        unknown = set(defaults) - known
        if unknown:
            msg = f"defaults: unknown parameters {sorted(unknown)}; available: {sorted(known)}"
            raise ValueError(msg)
        current_params = [p.replace(default=float(defaults.get(p.name, p.default))) for p in current_params]
        charge_params = [p.replace(default=float(defaults.get(p.name, p.default))) for p in charge_params]
    return JunctionModel(current_fn, current_params, charge_fn, charge_params)


def pn_diode(**defaults: float) -> JunctionModel:
    """PN diode per metre: :func:`diode_current` plus :func:`depletion_charge`.

    Keyword arguments replace parameter defaults, for example
    ``pn_diode(cj0_F_per_m=4e-10, vbi=0.8, dark_current_A_per_m=1e-5)``.
    Parameters: ``is_A_per_m``, ``ideality``, ``vt``, ``v_limit``,
    ``dark_current_A_per_m``, ``dark_bias_V`` (negative), ``cj0_F_per_m``,
    ``vbi``, ``grading`` and ``fc``.
    """
    if defaults.get("dark_bias_V", -1.0) >= 0:
        msg = "dark_bias_V must be a negative (reverse) bias"
        raise ValueError(msg)
    return junction_model(current=diode_current, charge=depletion_charge, defaults=defaults)


def _set_signature(
    physics: Callable[..., PhysicsReturn],
    name: str,
    runtime: dict[str, float],
    parameters: tuple[inspect.Parameter, ...] | list[inspect.Parameter],
) -> None:
    kind = inspect.Parameter.KEYWORD_ONLY
    physics.__signature__ = inspect.Signature(
        [
            inspect.Parameter("signals", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            *(inspect.Parameter(k, kind, default=v) for k, v in runtime.items()),
            *parameters,
        ]
    )
    physics.__name__ = physics.__qualname__ = name
    physics.__doc__ = f"{name} generated by circulax (parameters: {', '.join([*runtime, *(p.name for p in parameters)])})."


# ---------------------------------------------------------------------------
# Phase shifter
# ---------------------------------------------------------------------------
def phase_shifter_factory(
    *,
    delta_neff: VoltageFunction | float,
    loss_dB_cm: VoltageFunction | float = 0.0,
    junction: JunctionModel | None = None,
    name: str = "PhaseShifter",
    r0: float = 1.0,
) -> type[CircuitComponent]:
    """Build an electro-optic phase shifter component from functions of voltage.

    The optical inputs are functions of the voltage ``v = Re(V_anode -
    V_cathode)`` (or constants). Functions must be JAX-traceable and may
    declare extra keyword parameters with numeric defaults, for example
    ``def delta_neff(v, a1=1e-4, a2=1e-5): ...``. Those parameters, and the
    junction's, become runtime parameters of the returned component, shared
    by name, so they can be set in netlist settings, swept or fitted.

    Args:
        delta_neff: Change in optical effective index, dimensionless. By
            convention it increases under reverse bias (``v < 0``) in a
            depletion junction, where carriers are removed. Use
            :func:`delta_neff_from_vpil` for VπL data, :func:`tabulated` for
            a Δn_eff(V) table (for example from a 2D carrier/mode solve), or
            any function of ``v``.
        loss_dB_cm: Optical propagation loss in dB/cm.
        junction: Electrical model from :func:`pn_diode` or
            :func:`junction_model`. ``None`` leaves the electrodes unloaded,
            as for a Pockels (e.g. thin-film lithium niobate) modulator with
            no junction, whose electrode capacitance belongs in the RF line.
        name: Class name of the generated component.
        r0: Static finite positive real optical reference (default 1). Does
            not rescale the electrical voltage, current or charge.

    Returns:
        A component class with ports ``("anode", "cathode", "o1", "o2")`` and
        runtime parameters ``length_um``, ``neff`` (index where
        ``delta_neff`` is zero), ``n_group``, ``wavelength_nm`` plus the
        keyword parameters of the supplied functions.

    Optical model: the section is a matched, reciprocal power-wave guide.
    Optical potential is sqrt(r0)*(a+b), flow into the section is
    (a-b)/sqrt(r0), with incident power |a|² and outgoing power |b|². Light
    takes ``length_um * n_group / c`` to cross it in either direction. The
    voltage-dependent transmission
    ``T(v) = 10**(-loss_dB_cm(v)*L_cm/20) * exp(-2j*pi*(neff + delta_neff(v))*L/λ)``
    is applied at the section midpoint using the junction voltage half a
    transit earlier, which is when the light passed it. Model a long
    traveling-wave electrode as several short sections so the RF voltage is
    approximately uniform along each. Transient analysis reads each delay by
    interpolating accepted history; when the half-transit of a section is
    shorter than the time step, that error accumulates along a chain of
    sections, so check convergence by reducing the step.

    Electrical model: the anode receives ``length * junction.current(v)`` and
    the junction stores ``length * junction.charge(v)``. Series resistance
    and electrode loading are external; connect them in the netlist.

    Example, a depletion PN phase shifter from VπL and loss tables::

        diode = pn_diode(cj0_F_per_m=3e-10, dark_current_A_per_m=1e-5)
        diode.capacitance([-2.0, -1.0, 0.0])  # evaluate the junction alone
        PN = phase_shifter_factory(
            delta_neff=delta_neff_from_vpil(v_data, vpil_data, wavelength_nm=1310.0),
            loss_dB_cm=tabulated(v_data, loss_data),
            junction=diode,
        )
        # The diode parameters (cj0_F_per_m, vbi, ...) are settings of every "PN" instance.

    """
    r0 = validate_reference(r0)
    if junction is not None and not isinstance(junction, JunctionModel):
        msg = "junction must be a JunctionModel from pn_diode() or junction_model()"
        raise TypeError(msg)
    dn_fn, dn_params = _as_function(delta_neff, "delta_neff")
    loss_fn, loss_params = _as_function(loss_dB_cm, "loss_dB_cm")
    merged = _merge(dn_params, loss_params, junction.parameters if junction is not None else ())

    def physics(signals: Signals, **kw: Any) -> PhysicsReturn:
        length_um, wavelength_nm = kw["length_um"], kw["wavelength_nm"]
        past = signals.at_delay(length_um * kw["n_group"] / (2.0 * C_UM_PER_S))
        v_mid = jnp.real(past.anode - past.cathode)
        neff = kw["neff"] + dn_fn(v_mid, **_keywords(dn_params, kw))
        loss_db = loss_fn(v_mid, **_keywords(loss_params, kw)) * length_um * 1e-4
        transmission = 10.0 ** (-loss_db / 20.0) * jnp.exp(-2j * jnp.pi * neff * length_um * 1e3 / wavelength_nm)
        b1 = transmission * past.m_reverse
        b2 = transmission * past.m_forward
        f, q = power_wave_constraints(
            signals,
            ("o1", "o2"),
            jnp.array([b1, b2]),
            r0=r0,
            algebraic_flow=True,
            incident_states=("a_forward", "a_reverse"),
        )
        # One delayed snapshot implements both half-transits. These midpoint
        # states carry unmodulated sqrt(W) envelopes, in both directions.
        f["m_forward"] = signals.m_forward - past.a_forward
        f["m_reverse"] = signals.m_reverse - past.a_reverse
        if junction is not None:
            v_now = jnp.real(signals.anode - signals.cathode)
            length_m = length_um * 1e-6
            current = length_m * junction.current_per_m(v_now, kw)
            charge = length_m * junction.charge_per_m(v_now, kw)
            f["anode"], f["cathode"] = current, -current
            q["anode"], q["cathode"] = charge, -charge
        return f, q

    _set_signature(physics, name, _RUNTIME_DEFAULTS, tuple(merged.values()))
    cls = component(
        ports=("anode", "cathode", "o1", "o2"),
        states=("a_forward", "a_reverse", "m_forward", "m_reverse"),
        holomorphic=False,
    )(physics)
    # Only optical terminals declare complete history-independent F and zero Q.
    cls._wave_r0 = r0
    cls._wave_ports = ("o1", "o2")
    cls._wave_algebraic_flow = True
    return cls
