# Copyright 2026 Circulax contributors

"""Compact causal two-port models of uniform, frequency-dependent RLGC lines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike

from .api import ModelCoefficients, ModelFitOptions, fit_model, validate_model


def rlgc_sparameters(
    frequencies: ArrayLike,
    *,
    R: ArrayLike | Callable[[np.ndarray], ArrayLike],
    L: ArrayLike | Callable[[np.ndarray], ArrayLike],
    G: ArrayLike | Callable[[np.ndarray], ArrayLike],
    C: ArrayLike | Callable[[np.ndarray], ArrayLike],
    length: float,
    z0: float = 50.0,
) -> np.ndarray:
    """Evaluate an exact uniform line on the supplied frequency grid.

    RLGC are nonnegative per-metre scalars, arrays or functions of Hz, in
    ohm/m, H/m, S/m and F/m. This frequency-domain calculation does not make
    arbitrary input curves causal. No junction loading or termination is added.
    """
    from circulax.components.transmission_line import _sample_rlgc

    f, r, l, g, c = _sample_rlgc(frequencies, R, L, G, C)
    if np.ndim(length) != 0 or not np.isreal(length) or not np.isfinite(length) or length <= 0:
        raise ValueError("length must be a finite positive number of metres")
    if np.ndim(z0) != 0 or not np.isreal(z0) or not np.isfinite(z0) or z0 <= 0:
        raise ValueError("z0 must be a finite positive real impedance")
    s = 2j * np.pi * f
    z, y = r + s * l, g + s * c
    gamma = np.sqrt(z * y)
    propagation = gamma * length
    # Divide the ABCD-to-S expressions by cosh(gamma*length) to avoid overflow
    # on long/lossy lines. tanh(gamma*length)/gamma tends to length at DC.
    factor = np.divide(np.tanh(propagation), gamma, out=np.full_like(gamma, length), where=gamma != 0)
    exponential = np.exp(-propagation)
    sech = 2 * exponential / (1 + exponential**2)
    denominator = 2 + (z / z0 + y * z0) * factor
    reflection = (z / z0 - y * z0) * factor / denominator
    transmission = 2 * sech / denominator
    result = np.empty((len(f), 2, 2), complex)
    result[:, 0, 0] = result[:, 1, 1] = reflection
    result[:, 0, 1] = result[:, 1, 0] = transmission
    return result


def fit_rlgc(
    frequencies: ArrayLike,
    *,
    R: ArrayLike | Callable[[np.ndarray], ArrayLike],
    L: ArrayLike | Callable[[np.ndarray], ArrayLike],
    G: ArrayLike | Callable[[np.ndarray], ArrayLike],
    C: ArrayLike | Callable[[np.ndarray], ArrayLike],
    length: float,
    z0: float = 50.0,
    options: ModelFitOptions | None = None,
) -> ModelCoefficients:
    """Fit a reduced uniform-line model, checked before admission to simulation.

    Fit the exact distributed two-port, reserving every fifth interior sample
    for validation. At least six frequency samples are required. Defaults use
    reciprocal vector fitting with sampled passivity enforcement. The proper
    stable realization is causal; this does not certify arbitrary input data or
    extrapolation. Passivity is enforced/tested on a grid, not globally certified.

    Supplied options can choose order, tolerances and optional native delays
    (``delay_mode='supplied'`` or ``'infer'``). Reciprocity is required. Disabling
    enforcement still requires the resulting fit to pass passivity validation.
    Returned coefficients can be saved or passed to ``circuit_from_coefficients``.
    """
    response = rlgc_sparameters(frequencies, R=R, L=L, G=G, C=C, length=length, z0=z0)
    f = np.asarray(frequencies, dtype=float)
    if len(f) < 6:
        raise ValueError("reduced RLGC fitting requires at least six frequency samples")
    options = options or ModelFitOptions(enforce_passivity=True)
    if not options.reciprocal:
        raise ValueError("uniform RLGC lines require reciprocal fitting")
    # Include DC and a wider band in passivity screening, independently of the
    # held-out accuracy samples. This is deliberately still a sampled check.
    check_f = np.unique(np.r_[0, f, np.geomspace(f[f > 0][0] / 10, f[-1] * 10, max(401, 4 * len(f)))])
    if options.enforcement_freqs is not None:
        check_f = np.unique(np.r_[check_f, options.enforcement_freqs])
    options = replace(options, enforcement_freqs=tuple(check_f))
    heldout = np.zeros(len(f), dtype=bool)
    heldout[3:-1:5] = True
    fitted = fit_model(response[~heldout], f[~heldout], z0=z0, options=options)
    report = validate_model(
        fitted,
        measured_S=response[~heldout],
        freqs=f[~heldout],
        validation_S=response[heldout],
        validation_freqs=f[heldout],
        normalized_rmse=options.normalized_rmse,
        max_absolute_error=options.max_absolute_error,
    )
    report.raise_for_simulation()
    maximum = float(np.max(np.linalg.svd(fitted.evaluate(check_f), compute_uv=False)))
    if maximum > 1 + 1e-9:
        raise ValueError("reduced RLGC model fails wider-grid sampled passivity")
    fitted.metadata.update(
        source="uniform-rlgc",
        length_m=float(length),
        validation={**asdict(report), "findings": list(report.findings)},
        passivity_grid_maximum_singular_value=maximum,
        global_passivity_certified=False,
    )
    return fitted


@dataclass(frozen=True)
class RationalResponse:
    """Scalar ``constant + sum(residues / (s - poles))`` with poles in rad/s.

    Complex poles are stored with their conjugates, so the response is real
    for real ``s``.
    """

    poles: np.ndarray
    residues: np.ndarray
    constant: float

    def evaluate(self, frequencies: ArrayLike) -> np.ndarray:
        """Evaluate the response at frequencies in Hz."""
        s = 2j * np.pi * np.asarray(frequencies, dtype=float)
        return self.constant + np.sum(self.residues / (s[..., None] - self.poles), axis=-1)


@dataclass(frozen=True)
class CharacteristicLineModel:
    """Uniform line as characteristic admittance and delayed propagation.

    With port currents ``I1``, ``I2`` flowing into the line,

    ``Yc*V1 - I1 = H * (Yc*V2 + I2)`` and ``Yc*V2 - I2 = H * (Yc*V1 + I1)``,

    where ``H(s) = exp(-gamma*length) = exp(-s*delay) * propagation(s)``. The
    exact delay is removed before fitting, so ``propagation`` and
    ``admittance`` are smooth and need few poles even for long lines.
    """

    length: float
    delay: float
    resistance_floor: float
    conductance_floor: float
    admittance: RationalResponse
    propagation: RationalResponse
    metadata: dict

    def propagation_function(self, frequencies: ArrayLike) -> np.ndarray:
        """Fitted ``exp(-gamma*length)``, including the exact delay."""
        f = np.asarray(frequencies, dtype=float)
        return self.propagation.evaluate(f) * np.exp(-2j * np.pi * f * self.delay)

    def sparameters(self, frequencies: ArrayLike, z0: float = 50.0) -> np.ndarray:
        """S-parameters of the fitted model, shape ``(frequencies, 2, 2)``."""
        f = np.asarray(frequencies, dtype=float)
        yc = self.admittance.evaluate(f)
        h = self.propagation_function(f)
        # Combine the even/odd expressions before division so H = +/-1
        # (a lossless through line at DC or a half wavelength) stays finite.
        normalized_y = z0 * yc
        denominator = (1 + normalized_y) ** 2 - ((1 - normalized_y) * h) ** 2
        result = np.empty((len(f), 2, 2), complex)
        result[:, 0, 0] = result[:, 1, 1] = (1 - normalized_y**2) * (1 - h**2) / denominator
        result[:, 0, 1] = result[:, 1, 0] = 4 * normalized_y * h / denominator
        return result

    def circuit(self):  # noqa: ANN201
        """Realize the model as a transient/AC/HB-capable two-port ``Circuit``."""
        from circulax.components.characteristic_line import characteristic_line_circuit

        return characteristic_line_circuit(self)


def _fit_scalar(data: np.ndarray, s: np.ndarray, order: int, dc_value: float) -> RationalResponse:
    """Vector-fit one response with 1/sqrt(|data|) weighting, then pin its DC value."""
    from .driver import vfdriver
    from .types import FitOptions

    options = FitOptions(N=order, asymp=2, parametertype="S", weightparam=3)
    model, _, _, _ = vfdriver(jnp.asarray(data)[None, None, :], jnp.asarray(s), None, options, verbose=False)
    poles = np.asarray(model.poles)
    residues = np.asarray(model.residues)[0, 0]
    constant = float(np.real(np.asarray(model.D)[0, 0]))
    constant += float(np.real(dc_value - (constant + np.sum(residues / -poles))))
    return RationalResponse(poles, residues, constant)


def fit_rlgc_characteristic(
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
) -> CharacteristicLineModel:
    """Fit a uniform line by its characteristic admittance and de-embedded delay.

    The propagation delay ``delay_fraction`` times the phase delay at the
    highest frequency is removed exactly, rather than approximated by poles.
    The characteristic admittance ``Yc = sqrt(Y/Z)`` and the remaining
    propagation ``exp(-gamma*length + s*delay)`` are fitted separately with
    the same order, increased from 4 until every check passes:

    - maximum absolute S error (reference ``z0``) on all samples, including
      held-out ones, against the supplied, unfloored RLGC is within ``tolerance``;
    - stable poles;
    - sampled passivity (largest singular value of S at most 1), ``Re(Yc) >= 0``
      and ``|propagation| <= 1`` on a grid from DC to ten times the band.

    A DC sample is required and the fitted DC values are pinned exactly, so the
    floored model is exact at DC. With ``G = 0`` (or ``R = 0``) the DC resistance is
    the ratio of two vanishing quantities. Conductance and resistance are
    therefore floored at ``2*pi*f_low*C`` and ``2*pi*f_low*L``, where
    ``f_low`` is the lowest nonzero frequency, putting the floors' corner
    inside the grid. Their effect is included in the S check. Lower ``f_low``
    if it fails. Sampled passivity is not a global certificate, and input
    causality is not certified.
    """
    from circulax.components.transmission_line import _sample_rlgc

    f, r, l, g, c = _sample_rlgc(frequencies, R, L, G, C)
    if np.ndim(length) != 0 or not np.isreal(length) or not np.isfinite(length) or length <= 0:
        raise ValueError("length must be a finite positive number of metres")
    if np.ndim(tolerance) != 0 or not np.isreal(tolerance) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if isinstance(max_order, bool) or not isinstance(max_order, (int, np.integer)) or max_order < 4:
        raise ValueError("max_order must be an integer of at least 4")
    if np.ndim(z0) != 0 or not np.isreal(z0) or not np.isfinite(z0) or z0 <= 0:
        raise ValueError("z0 must be a finite positive real impedance")
    if not (np.any(r) or np.any(l)) or not (np.any(g) or np.any(c)):
        raise ValueError("characteristic lines require nonzero series impedance and shunt admittance")
    if f[0] != 0 or len(f) < 7:
        raise ValueError("characteristic line fitting requires a DC sample and at least six nonzero frequencies")
    if not 0 < delay_fraction <= 1:
        raise ValueError("delay_fraction must be in (0, 1]")
    resistance_floor = 2 * np.pi * f[1] * l[1]
    conductance_floor = 2 * np.pi * f[1] * c[1]
    # The floors only matter where they exceed the supplied low-frequency values.
    floor_error = max(max(resistance_floor - r[0], 0.0) * length / z0, max(conductance_floor - g[0], 0.0) * length * z0)
    if floor_error > tolerance:
        raise ValueError(
            f"lowest nonzero frequency {f[1]:.3g} Hz is too high: the DC floors alone change S by about "
            f"{floor_error:.1e}; include frequencies down to about {f[1] * tolerance / floor_error:.2g} Hz"
        )
    s = 2j * np.pi * f
    z = np.maximum(r, resistance_floor) + s * l
    y = np.maximum(g, conductance_floor) + s * c
    gamma = np.sqrt(z * y)
    admittance = gamma / z
    delay = delay_fraction * float(gamma[-1].imag) * length / (2 * np.pi * f[-1])
    propagation = np.exp(-gamma * length + s * delay)

    reference = rlgc_sparameters(f, R=r, L=l, G=g, C=c, length=length, z0=z0)
    check_f = np.unique(np.r_[0.0, f, np.geomspace(f[1] / 10, f[-1] * 10, max(801, 4 * len(f)))])
    fit = np.ones(len(f), dtype=bool)
    fit[3:-1:5] = False
    best = None
    for order in range(4, max_order + 1, 2):
        yc = _fit_scalar(admittance[fit], s[fit], order, float(admittance[0].real))
        k = _fit_scalar(1 - propagation[fit], s[fit], order, float(1 - propagation[0].real))
        h = RationalResponse(k.poles, -k.residues, 1 - k.constant)
        model = CharacteristicLineModel(float(length), delay, resistance_floor, conductance_floor, yc, h, {})
        error = float(np.max(np.abs(model.sparameters(f, z0) - reference)))
        singular = float(np.max(np.linalg.svd(model.sparameters(check_f, z0), compute_uv=False)))
        diagnostics = {
            "order": order,
            "maximum_s_error": error,
            "passivity_grid_maximum_singular_value": singular,
            "minimum_admittance_real": float(np.min(yc.evaluate(check_f).real)),
            "maximum_propagation_magnitude": float(np.max(np.abs(h.evaluate(check_f)))),
            "stable": bool(np.all(yc.poles.real < 0) and np.all(h.poles.real < 0)),
        }
        passed = (
            diagnostics["stable"]
            and error <= tolerance
            and singular <= 1
            and diagnostics["minimum_admittance_real"] >= 0
            and diagnostics["maximum_propagation_magnitude"] <= 1
        )
        if passed:
            return replace(model, metadata={**diagnostics, "source": "uniform-rlgc-characteristic", "z0": z0})
        if best is None or error < best["maximum_s_error"]:
            best = diagnostics
    raise ValueError(f"no characteristic line fit up to order {max_order} passed all checks; best attempt: {best}")
