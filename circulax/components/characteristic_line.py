"""Uniform transmission lines realized by their characteristic admittance and exact delay.

The line obeys ``Yc*V1 - I1 = H*(Yc*V2 + I2)`` and the mirrored relation, with
``H = exp(-s*delay) * propagation(s)`` (method of characteristics). Removing the
delay keeps ``Yc`` and ``propagation`` smooth, so a long line needs a few poles
rather than enough poles to follow its phase rotation.

Each pole is a separate small component, keeping the Jacobian sparse. Internal
wave nodes are in volts (currents times ``zref``):

- ``sum_k = zref * Yc * V_k``;
- ``r_k = zref * (Yc*V_k - I_k)``, the wave arriving at port ``k``;
- ``u_k = zref * (Yc*V_k + I_k) = 2*sum_k - r_k``, the wave launched from port ``k``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from circulax.components.base_component import PhysicsReturn, Signals, component

if TYPE_CHECKING:
    from collections.abc import Callable

    from circulax.circuit import Circuit
    from circulax.fitting.rlgc import CharacteristicLineModel


def _pole_terms(signals: Signals, drive, pole_re, pole_im, residue_re, residue_im):  # noqa: ANN001, ANN202
    """Contribution ``residue/(s - pole)`` (plus its conjugate when ``pole_im != 0``).

    States are scaled by ``|pole|`` so that they have the units of ``drive``.
    """
    scale = (pole_re**2 + pole_im**2) ** 0.5
    xr, xi = signals.xr, signals.xi
    output = 2 * (residue_re * xr - residue_im * xi) / scale
    f = {"xr": -((pole_re * xr - pole_im * xi) / scale + drive), "xi": -(pole_im * xr + pole_re * xi) / scale}
    return output, f, {"xr": xr / scale, "xi": xi / scale}


def _real_pole_terms(signals: Signals, drive, pole, residue):  # noqa: ANN001, ANN202
    scale = -pole
    return residue * signals.x / scale, {"x": -(pole * signals.x / scale + drive)}, {"x": signals.x / scale}


@component(ports=("v", "sum"), states=("x",), holomorphic=True)
def AdmittancePole(signals: Signals, pole: float = -1e9, residue: float = 0.0, zref: float = 50.0) -> PhysicsReturn:
    """Real pole of ``Yc``: shunt current at ``v`` and its ``zref``-scaled copy into ``sum``."""
    current, f, q = _real_pole_terms(signals, signals.v, pole, residue)
    return {"v": current, "sum": -zref * current, **f}, q


@component(ports=("v", "sum"), states=("xr", "xi"), holomorphic=True)
def AdmittancePolePair(
    signals: Signals,
    pole_re: float = -1e9,
    pole_im: float = 1e9,
    residue_re: float = 0.0,
    residue_im: float = 0.0,
    zref: float = 50.0,
) -> PhysicsReturn:
    """Complex-conjugate pole pair of ``Yc``."""
    current, f, q = _pole_terms(signals, signals.v, pole_re, pole_im, residue_re, residue_im)
    return {"v": current, "sum": -zref * current, **f}, q


@component(ports=("u", "r"), states=("x",), holomorphic=True)
def PropagationPole(signals: Signals, pole: float = -1e9, residue: float = 0.0, delay: float = 0.0) -> PhysicsReturn:
    """Real pole of the delay-free propagation, driven by the launched wave at ``t - delay``."""
    output, f, q = _real_pole_terms(signals, signals.at_delay(delay).u, pole, residue)
    return {"r": -output, **f}, q


@component(ports=("u", "r"), states=("xr", "xi"), holomorphic=True)
def PropagationPolePair(
    signals: Signals,
    pole_re: float = -1e9,
    pole_im: float = 1e9,
    residue_re: float = 0.0,
    residue_im: float = 0.0,
    delay: float = 0.0,
) -> PhysicsReturn:
    """Complex-conjugate pole pair of the delay-free propagation."""
    output, f, q = _pole_terms(signals, signals.at_delay(delay).u, pole_re, pole_im, residue_re, residue_im)
    return {"r": -output, **f}, q


@component(ports=("p1", "p2", "sum1", "sum2", "u1", "u2", "r1", "r2"), holomorphic=True)
def CharacteristicLineCore(
    signals: Signals, admittance: float = 0.02, propagation: float = 0.0, delay: float = 0.0, zref: float = 50.0
) -> PhysicsReturn:
    """Constant terms, port currents and wave bookkeeping; the wave nodes are unit-resistance loaded."""
    past = signals.at_delay(delay)
    return {
        "p1": admittance * signals.p1 - signals.r1 / zref,
        "p2": admittance * signals.p2 - signals.r2 / zref,
        "sum1": signals.sum1 - zref * admittance * signals.p1,
        "sum2": signals.sum2 - zref * admittance * signals.p2,
        "u1": signals.u1 - (2 * signals.sum1 - signals.r1),
        "u2": signals.u2 - (2 * signals.sum2 - signals.r2),
        "r1": signals.r1 - propagation * past.u2,
        "r2": signals.r2 - propagation * past.u1,
    }, {}


_MODELS = {
    "characteristic_core": CharacteristicLineCore,
    "characteristic_admittance_pole": AdmittancePole,
    "characteristic_admittance_pair": AdmittancePolePair,
    "characteristic_propagation_pole": PropagationPole,
    "characteristic_propagation_pair": PropagationPolePair,
}


def _pole_instances(poles: np.ndarray, residues: np.ndarray, kind: str, extra: dict) -> list[tuple[str, dict]]:
    """One instance per real pole or conjugate pair (upper half-plane member)."""
    out = []
    for pole, residue in zip(poles, residues, strict=True):
        if pole.imag < 0:
            continue
        if pole.imag == 0:
            out.append((f"characteristic_{kind}_pole", {"pole": float(pole.real), "residue": float(residue.real), **extra}))
        else:
            settings = {"pole_re": float(pole.real), "pole_im": float(pole.imag)}
            settings.update(residue_re=float(residue.real), residue_im=float(residue.imag), **extra)
            out.append((f"characteristic_{kind}_pair", settings))
    return out


def characteristic_line_circuit(model: CharacteristicLineModel) -> Circuit:
    """Realize a :class:`~circulax.fitting.rlgc.CharacteristicLineModel` as a two-port ``Circuit``.

    Ports ``p1`` and ``p2`` are single-ended to ground. Works in real and
    complex circuits, in DC, AC, harmonic balance and transient analysis.
    """
    from circulax import compile_circuit

    # Scale the wave nodes to volts with the high-frequency characteristic impedance.
    zref = 1 / model.admittance.constant if model.admittance.constant > 0 else 50.0
    delay = float(model.delay)
    core = {"admittance": model.admittance.constant, "propagation": model.propagation.constant, "delay": delay, "zref": zref}
    instances = {"core": {"component": "characteristic_core", "settings": core}}
    connections = {}
    for port in (1, 2):
        admittance = _pole_instances(model.admittance.poles, model.admittance.residues, "admittance", {"zref": zref})
        for i, (component_name, settings) in enumerate(admittance):
            instances[f"y{port}_{i}"] = {"component": component_name, "settings": settings}
            connections[f"y{port}_{i},v"] = f"core,p{port}"
            connections[f"y{port}_{i},sum"] = f"core,sum{port}"
        propagation = _pole_instances(model.propagation.poles, model.propagation.residues, "propagation", {"delay": delay})
        for i, (component_name, settings) in enumerate(propagation):
            # Waves launched at this port arrive at the other one.
            instances[f"h{port}_{i}"] = {"component": component_name, "settings": settings}
            connections[f"h{port}_{i},u"] = f"core,u{port}"
            connections[f"h{port}_{i},r"] = f"core,r{3 - port}"
    net = {"instances": instances, "connections": connections, "ports": {"p1": "core,p1", "p2": "core,p2"}}
    return compile_circuit(net, _MODELS, g_leak=0.0)


def characteristic_rlgc_transmission_line(
    frequencies: ArrayLike,
    *,
    R: ArrayLike | Callable[[np.ndarray], ArrayLike],
    L: ArrayLike | Callable[[np.ndarray], ArrayLike],
    G: ArrayLike | Callable[[np.ndarray], ArrayLike],
    C: ArrayLike | Callable[[np.ndarray], ArrayLike],
    length: float,
    tolerance: float = 1e-3,
    max_order: int = 40,
    delay_fraction: float = 0.98,
    z0: float = 50.0,
) -> Circuit:
    """Create a fast uniform line from RLGC versus Hz, with its delay de-embedded.

    Use it instead of :func:`~circulax.components.reduced_rlgc_transmission_line`
    when the line is electrically long. That model fits the S-parameters and
    needs many poles to follow the propagation phase. This one removes the delay
    exactly and fits only the smooth characteristic admittance and the
    remaining propagation. ``frequencies`` must start at DC, and the lowest
    nonzero frequency must be low enough for the DC floors described in
    :func:`~circulax.fitting.fit_rlgc_characteristic`. That function also
    documents the admission checks; a fit failing them raises ``ValueError``.
    ``z0`` is only the reference impedance for the S error check.

    Ports ``p1`` and ``p2`` are single-ended to ground, with no internal taps.
    Fitting depends on length, so refit for a different length. Use
    ``fit_rlgc_characteristic`` to inspect the fit and ``model.circuit()`` to
    realize it.
    """
    from circulax.fitting import fit_rlgc_characteristic

    model = fit_rlgc_characteristic(
        frequencies,
        R=R,
        L=L,
        G=G,
        C=C,
        length=length,
        tolerance=tolerance,
        max_order=max_order,
        delay_fraction=delay_fraction,
        z0=z0,
    )
    return model.circuit()
