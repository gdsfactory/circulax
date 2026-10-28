"""Compact causal two-port models of uniform, frequency-dependent RLGC lines."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, replace

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
