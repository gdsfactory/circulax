"""Passive, causal RLC transmission-line factories from sampled RLGC data."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import nnls

from circulax.components.electronic import Capacitor, Inductor, Resistor

if TYPE_CHECKING:
    from circulax.circuit import Circuit


@dataclass(frozen=True)
class FosterFit:
    """Positive-real response ``constant + s*linear + sum(weights*s/(s+poles))``.

    ``poles`` are positive rates in rad/s, not signed Laplace poles. The actual
    branch poles are ``-poles``. Coefficients are per metre for a line fit.
    """

    constant: float
    linear: float
    weights: np.ndarray
    poles: np.ndarray
    normalized_rmse: float
    maximum_relative_error: float

    def evaluate(self, frequencies: ArrayLike) -> np.ndarray:
        """Evaluate the fitted impedance or admittance per metre, in Hz."""
        s = 2j * np.pi * np.asarray(frequencies, dtype=float)
        return self.constant + s * self.linear + np.sum(self.weights * s[..., None] / (s[..., None] + self.poles), axis=-1)


def _fit_response(response: np.ndarray, frequencies: np.ndarray, poles: np.ndarray, tolerance: float, label: str) -> FosterFit:
    scale = 2 * np.pi * frequencies[-1]
    s = 2j * np.pi * frequencies
    basis = np.column_stack([np.ones_like(s), s / scale, s[:, None] / (s[:, None] + poles)])
    magnitude = float(np.max(np.abs(response)))
    if magnitude == 0:
        return FosterFit(0.0, 0.0, np.array([]), np.array([]), 0.0, 0.0)
    # Relative weighting, capped near zeros so DC does not dominate the fit.
    denominator = np.maximum(np.abs(response), 0.01 * magnitude)
    matrix = np.concatenate([(basis / denominator[:, None]).real, (basis / denominator[:, None]).imag])
    target = np.concatenate([(response / denominator).real, (response / denominator).imag])
    column_scale = np.linalg.norm(matrix, axis=0)
    coefficients, _ = nnls(matrix / column_scale, target, maxiter=100 * matrix.shape[1])
    coefficients /= column_scale
    # Remove numerical dust before constructing elements. Include this in the reported error.
    coefficients[coefficients < 1e-12 * np.max(coefficients)] = 0
    error = np.abs(basis @ coefficients - response) / denominator
    nrmse = float(np.linalg.norm(error) / np.linalg.norm(response / denominator))
    peak = float(np.max(error))
    if nrmse > tolerance:
        msg = (
            f"{label} passive RLC fit NRMSE {nrmse:.3%} exceeds {tolerance:.3%}. "
            "Try more/different poles or a narrower band. This Foster topology cannot represent every passive response."
        )
        raise ValueError(msg)
    active = coefficients[2:] > 0
    weights, rates = coefficients[2:][active].copy(), poles[active].copy()
    weights.setflags(write=False)
    rates.setflags(write=False)
    return FosterFit(float(coefficients[0]), float(coefficients[1] / scale), weights, rates, nrmse, peak)


@dataclass(frozen=True)
class RLGCTransmissionLine:
    """Reusable fitted line; call with a length in metres to create a Circuit.

    Both fits use nonnegative RLC values, ensuring a passive, causal finite
    ladder. This does not certify that the supplied sampled data is causal,
    nor that the approximation is accurate outside ``frequency_range``.
    """

    series: FosterFit
    shunt: FosterFit
    frequency_range: tuple[float, float]

    def netlist(self, *, length: float, n_sections: int = 1) -> dict:  # noqa: C901, PLR0915 -- local RLC branch builders
        """Build an unterminated symmetric-T ladder from R, L and C primitives.

        Ports ``p1`` and ``p2`` are single-ended relative to the internal common
        ground. ``tap_0`` ... ``tap_{n_sections-1}`` expose section midpoints for
        junction loading. No source, termination, optical model or delay is added.
        Increasing ``n_sections`` reduces the spatial discretization error.
        """
        if not np.isfinite(length) or length <= 0:
            msg = "length must be finite and positive, in metres"
            raise ValueError(msg)
        if isinstance(n_sections, bool) or not isinstance(n_sections, Integral) or n_sections < 1:
            msg = "n_sections must be a positive integer"
            raise ValueError(msg)
        instances = {"GND": {"component": "ground"}}
        connections: dict[str, str] = {}
        ports = {}
        serial = 0

        def put(kind: str, value: float, start: str | None, end: str | None = None) -> tuple[str, str]:
            nonlocal serial
            serial += 1
            name = f"{kind}{serial}"
            instances[name] = {"component": kind, "settings": {kind: float(value)}}
            p1, p2 = f"{name},p1", f"{name},p2"
            if start is not None:
                connections[p1] = start
            if end is not None:
                connections[p2] = end
            return p1, p2

        def series(start: str | None, distance: float) -> str:
            tip = start
            first = None
            if self.series.constant > 0:
                first, tip = put("R", self.series.constant * distance, tip)
            if self.series.linear > 0:
                begin, tip = put("L", self.series.linear * distance, tip)
                first = first or begin
            for w, p in zip(self.series.weights, self.series.poles, strict=True):
                begin, end = put("R", w * distance, tip)
                put("L", w * distance / p, begin, end)
                first, tip = first or begin, end
            if tip is None:
                msg = "The line must have nonzero series impedance"
                raise ValueError(msg)
            if "p1" not in ports:
                ports["p1"] = first
            return tip

        def shunt(node: str, distance: float) -> None:
            if self.shunt.constant > 0:
                put("R", 1 / (self.shunt.constant * distance), node, "GND,p1")
            if self.shunt.linear > 0:
                put("C", self.shunt.linear * distance, node, "GND,p1")
            for w, p in zip(self.shunt.weights, self.shunt.poles, strict=True):
                _, tip = put("R", 1 / (w * distance), node)
                put("C", w * distance / p, tip, "GND,p1")

        last = None
        d = length / n_sections
        for k in range(n_sections):
            midpoint = series(last, d / 2)
            shunt(midpoint, d)
            ports[f"tap_{k}"] = midpoint
            last = series(midpoint, d / 2)
        ports["p2"] = last
        return {"instances": instances, "connections": connections, "ports": ports}

    def __call__(self, *, length: float, n_sections: int = 1) -> Circuit:
        """Create a Circulax Circuit, usable directly or as a parent subcircuit."""
        from circulax.circuit import compile_circuit

        return compile_circuit(
            self.netlist(length=length, n_sections=n_sections),
            {"R": Resistor, "L": Inductor, "C": Capacitor},
            is_complex=False,
            g_leak=0.0,
        )


def rlgc_transmission_line(  # noqa: C901 -- validate physical arrays before fitting
    frequencies: ArrayLike,
    *,
    R: ArrayLike,
    L: ArrayLike,
    G: ArrayLike,
    C: ArrayLike,
    n_poles: int = 12,
    poles: ArrayLike | None = None,
    tolerance: float = 0.01,
) -> RLGCTransmissionLine:
    """Fit frequency-dependent RLGC per metre and return a line factory.

    ``frequencies`` must be increasing, nonnegative Hz with a positive maximum.
    ``R``, ``L``, ``G``, ``C`` are real nonnegative scalars or arrays on that grid,
    in ohm/m, H/m, S/m and F/m. The series impedance ``R+sL`` and shunt
    admittance ``G+sC`` are fitted independently; constant characteristic
    impedance is not assumed. Skin-effect responses can be approximated over
    a finite band, but not every passive response belongs to this branch family.

    ``poles`` optionally supplies positive branch rates in rad/s. Otherwise
    ``n_poles`` logarithmically spaced rates are chosen. ``tolerance`` is the
    maximum weighted relative RMS fitting error for *each* response. The
    relative weighting has a floor of 1% of the maximum response magnitude.
    Excess error raises ValueError rather than returning a poor fit silently.
    Inspect ``series`` and ``shunt`` for coefficients and errors. Spatial
    discretization is separate: verify convergence as ``n_sections`` increases.

    Example::

        line = rlgc_transmission_line(f, R=r, L=l, G=g, C=c)
        section = line(length=200e-6, n_sections=2)
        # Use section as models_map["line"] in a parent circuit.

    Fitting runs once in NumPy/SciPy. Generated Circulax component values are
    ordinary real RLC parameters; no S-to-state-space conversion is used.
    """
    f = np.asarray(frequencies)
    if np.iscomplexobj(f) or f.ndim != 1 or len(f) < 2:
        msg = "frequencies must be a real 1-D array with at least two samples"
        raise ValueError(msg)
    f = f.astype(float)
    if not np.all(np.isfinite(f)) or f[0] < 0 or np.any(np.diff(f) <= 0):
        msg = "frequencies must be finite, nonnegative and strictly increasing"
        raise ValueError(msg)
    if not np.isfinite(tolerance) or tolerance <= 0:
        msg = "tolerance must be finite and positive"
        raise ValueError(msg)
    values = []
    for label, value in zip(("R", "L", "G", "C"), (R, L, G, C), strict=True):
        array = np.asarray(value)
        if np.iscomplexobj(array) or array.shape not in ((), f.shape):
            msg = f"{label} must be a real scalar or an array matching frequencies"
            raise ValueError(msg)
        array = np.broadcast_to(array.astype(float), f.shape)
        if not np.all(np.isfinite(array)) or np.any(array < 0):
            msg = f"{label} must be finite and nonnegative"
            raise ValueError(msg)
        values.append(array)
    r, l, g, cap = values
    if not np.any(r) and not np.any(l):
        msg = "The line must have nonzero series impedance"
        raise ValueError(msg)
    if poles is None:
        if isinstance(n_poles, bool) or not isinstance(n_poles, Integral) or n_poles < 1:
            msg = "n_poles must be a positive integer"
            raise ValueError(msg)
        f_min = max(f[f > 0][0] / 10, f[-1] / 1e6)
        rates = 2 * np.pi * np.geomspace(f_min, 10 * f[-1], n_poles)
    else:
        rates = np.asarray(poles)
        if np.iscomplexobj(rates) or rates.ndim != 1 or not len(rates):
            msg = "poles must be a nonempty real 1-D array of positive rates"
            raise ValueError(msg)
        rates = rates.astype(float)
        if not np.all(np.isfinite(rates)) or np.any(rates <= 0) or np.any(np.diff(rates) <= 0):
            msg = "poles must be finite, positive and strictly increasing"
            raise ValueError(msg)
    s = 2j * np.pi * f
    return RLGCTransmissionLine(
        series=_fit_response(r + s * l, f, rates, tolerance, "Series impedance"),
        shunt=_fit_response(g + s * cap, f, rates, tolerance, "Shunt admittance"),
        frequency_range=(float(f[0]), float(f[-1])),
    )
