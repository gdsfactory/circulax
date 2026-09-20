"""Pre-simulation validation for rational S-parameter surface fits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np

from .surface import (
    RationalSurface,
    evaluate_surface,
    surface_asymptotic_passivity_margins,
    surface_passivity_margins,
)

ValidationStatus = Literal["pass", "warn", "fail"]


class FitValidationError(RuntimeError):
    """Raised when a fit has not passed its simulation validation gate."""


@dataclass(frozen=True)
class FitValidationThresholds:
    """Configurable acceptance thresholds for :func:`validate_surface_fit`."""

    normalized_rmse: float = 2e-2
    max_absolute_error: float = 5e-2
    reciprocity_error: float = 1e-6
    passivity_tolerance: float = 1e-9
    slope_tolerance: float = 1e-12
    stability_tolerance: float = 0.0
    delay_tolerance_seconds: float = 1e-15


@dataclass(frozen=True)
class ValidationFinding:
    """One actionable validation result."""

    severity: Literal["warn", "fail"]
    code: str
    message: str


@dataclass(frozen=True)
class FitValidationReport:
    """Accuracy and physical-validity evidence for a fitted surface model."""

    status: ValidationStatus
    normalized_rmse: float
    rms_error: float
    max_absolute_error: float
    heldout_normalized_rmse: float | None
    heldout_max_absolute_error: float | None
    worst_error_corner: int
    worst_error_frequency_hz: float
    worst_error_port_pair: tuple[int, int]
    maximum_reciprocity_error: float
    maximum_pole_real_part: float
    minimum_delay_seconds: float
    minimum_passivity_margin: float
    minimum_D_margin: float
    minimum_E_margin: float
    validated_frequency_range_hz: tuple[float, float]
    training_corner_count: int
    validation_corner_count: int
    validation_frequency_count: int
    pole_count: int
    expected_passive: bool
    expected_reciprocal: bool
    findings: tuple[ValidationFinding, ...]

    @property
    def simulation_ready(self) -> bool:
        """Whether the report passed without warnings or failures."""
        return self.status == "pass"

    def raise_for_simulation(self, *, allow_warnings: bool = False) -> None:
        """Raise unless the model is ready for simulation.

        Warning-only reports may be explicitly admitted with
        ``allow_warnings=True``. Failures always raise.
        """
        blocked = self.status == "fail" or (self.status == "warn" and not allow_warnings)
        if blocked:
            details = "; ".join(f"{item.code}: {item.message}" for item in self.findings)
            raise FitValidationError(f"fit validation {self.status}: {details}")

    def summary(self) -> str:
        """Return a concise human-readable report."""
        holdout = "not supplied" if self.heldout_normalized_rmse is None else f"{self.heldout_normalized_rmse:.3e}"
        lines = [
            f"Fit validation: {self.status.upper()}",
            f"Expected physics: passive={self.expected_passive}, reciprocal={self.expected_reciprocal}",
            f"Normalized complex RMSE: {self.normalized_rmse:.3e}",
            f"Maximum |dS|: {self.max_absolute_error:.3e}",
            f"Held-out normalized RMSE: {holdout}",
            f"Minimum Y passivity margin: {self.minimum_passivity_margin:.3e}",
            f"Minimum D/E margins: {self.minimum_D_margin:.3e} / {self.minimum_E_margin:.3e}",
            f"Maximum pole real part: {self.maximum_pole_real_part:.3e} rad/s",
            f"Minimum delay: {self.minimum_delay_seconds:.3e} s",
            "Validated frequency range: "
            f"{self.validated_frequency_range_hz[0]:.6g}--{self.validated_frequency_range_hz[1]:.6g} Hz",
        ]
        lines.extend(f"{item.severity.upper()} [{item.code}]: {item.message}" for item in self.findings)
        return "\n".join(lines)


def _error_metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    error = np.abs(prediction - target)
    rms = float(np.sqrt(np.mean(error**2)))
    reference_rms = float(np.sqrt(np.mean(np.abs(target) ** 2)))
    return rms / max(reference_rms, 1e-30), rms, float(np.max(error))


def _status(findings: list[ValidationFinding]) -> ValidationStatus:
    if any(item.severity == "fail" for item in findings):
        return "fail"
    if findings:
        return "warn"
    return "pass"


def validate_surface_fit(
    model: RationalSurface,
    measured_S: np.ndarray,
    features: np.ndarray,
    freqs: np.ndarray,
    *,
    validation_S: np.ndarray | None = None,
    validation_features: np.ndarray | None = None,
    validation_freqs: np.ndarray | None = None,
    passivity_features: np.ndarray | None = None,
    passivity_freqs: np.ndarray | None = None,
    simulation_frequency_range: tuple[float, float] | None = None,
    expected_reciprocal: bool = True,
    expected_passive: bool = True,
    thresholds: FitValidationThresholds | None = None,
) -> FitValidationReport:
    """Validate accuracy and physical suitability before circuit simulation.

    Held-out validation requires all of ``validation_S``,
    ``validation_features``, and ``validation_freqs``. Passivity may be checked
    on an independent dense grid using ``passivity_features`` and
    ``passivity_freqs``. Set ``expected_passive=False`` for an intentionally
    active model; passivity margins remain available as diagnostics but do not
    gate simulation readiness.
    """
    limits = FitValidationThresholds() if thresholds is None else thresholds
    measured_S = np.asarray(measured_S)
    features = np.asarray(features)
    freqs = np.asarray(freqs, dtype=float)
    if measured_S.ndim != 4 or measured_S.shape[:2] != (len(features), len(freqs)):
        msg = "measured_S must have shape (len(features), len(freqs), ports, ports)"
        raise ValueError(msg)
    if measured_S.shape[-1] != measured_S.shape[-2]:
        msg = "measured_S port matrices must be square"
        raise ValueError(msg)
    heldout_values = (validation_S, validation_features, validation_freqs)
    if any(value is None for value in heldout_values) and not all(value is None for value in heldout_values):
        msg = "validation_S, validation_features, and validation_freqs must be supplied together"
        raise ValueError(msg)

    prediction = np.asarray(evaluate_surface(model, jnp.asarray(features), jnp.asarray(freqs)))
    normalized_rmse, rms_error, max_error = _error_metrics(prediction, measured_S)
    worst_index = np.unravel_index(int(np.argmax(np.abs(prediction - measured_S))), measured_S.shape)
    reciprocity_error = float(np.max(np.abs(prediction - np.swapaxes(prediction, -1, -2))))

    heldout_normalized_rmse = None
    heldout_max_error = None
    heldout_count = 0
    if validation_S is not None and validation_features is not None and validation_freqs is not None:
        validation_S = np.asarray(validation_S)
        validation_features = np.asarray(validation_features)
        validation_freqs = np.asarray(validation_freqs, dtype=float)
        expected_shape = (len(validation_features), len(validation_freqs), measured_S.shape[-2], measured_S.shape[-1])
        if validation_S.shape != expected_shape:
            msg = f"validation_S must have shape {expected_shape}; got {validation_S.shape}"
            raise ValueError(msg)
        heldout_prediction = np.asarray(evaluate_surface(model, validation_features, validation_freqs))
        heldout_normalized_rmse, _, heldout_max_error = _error_metrics(heldout_prediction, validation_S)
        heldout_count = len(validation_features)

    physical_features = np.asarray(passivity_features) if passivity_features is not None else features
    physical_freqs = np.asarray(passivity_freqs, dtype=float) if passivity_freqs is not None else freqs
    if physical_features.ndim != 2 or physical_freqs.ndim != 1 or len(physical_freqs) == 0:
        msg = "passivity_features must be 2-D and passivity_freqs must be a nonempty 1-D array"
        raise ValueError(msg)
    passivity_margin = float(jnp.min(surface_passivity_margins(model, physical_features, physical_freqs)))
    D_margins, E_margins = surface_asymptotic_passivity_margins(model, physical_features)
    D_margin = float(jnp.min(D_margins))
    E_margin = float(jnp.min(E_margins))
    delay_seconds = np.asarray(physical_features @ np.asarray(model.tau_coeffs) / float(model.omega_scale))
    minimum_delay = float(np.min(delay_seconds))
    maximum_pole_real = float(np.max(np.real(np.asarray(model.poles) * float(model.omega_scale))))
    validated_range = (float(np.min(physical_freqs)), float(np.max(physical_freqs)))

    findings: list[ValidationFinding] = []
    all_scalars = np.array(
        [
            normalized_rmse,
            rms_error,
            max_error,
            reciprocity_error,
            passivity_margin,
            D_margin,
            E_margin,
            minimum_delay,
            maximum_pole_real,
            *(value for value in (heldout_normalized_rmse, heldout_max_error) if value is not None),
        ]
    )
    if not np.all(np.isfinite(all_scalars)) or not np.all(np.isfinite(prediction)):
        findings.append(ValidationFinding("fail", "nonfinite", "model evaluation or validation metrics contain NaN/Inf"))
    if normalized_rmse > limits.normalized_rmse:
        findings.append(
            ValidationFinding(
                "fail",
                "fit_rmse",
                f"normalized RMSE {normalized_rmse:.3e} exceeds {limits.normalized_rmse:.3e}",
            )
        )
    if max_error > limits.max_absolute_error:
        findings.append(
            ValidationFinding(
                "fail",
                "fit_max_error",
                f"maximum |dS| {max_error:.3e} exceeds {limits.max_absolute_error:.3e}",
            )
        )
    if heldout_normalized_rmse is None:
        findings.append(ValidationFinding("warn", "no_holdout", "no independent held-out data were supplied"))
    elif heldout_normalized_rmse > limits.normalized_rmse:
        findings.append(
            ValidationFinding(
                "fail",
                "holdout_rmse",
                f"held-out normalized RMSE {heldout_normalized_rmse:.3e} exceeds {limits.normalized_rmse:.3e}",
            )
        )
    if heldout_max_error is not None and heldout_max_error > limits.max_absolute_error:
        findings.append(
            ValidationFinding(
                "fail",
                "holdout_max_error",
                f"held-out maximum |dS| {heldout_max_error:.3e} exceeds {limits.max_absolute_error:.3e}",
            )
        )
    if expected_passive and (passivity_features is None or passivity_freqs is None):
        findings.append(
            ValidationFinding(
                "warn",
                "no_passivity_grid",
                "passivity was not checked on an explicitly supplied independent dense grid",
            )
        )
    if expected_reciprocal and reciprocity_error > limits.reciprocity_error:
        findings.append(
            ValidationFinding(
                "fail",
                "reciprocity",
                f"maximum reciprocity error {reciprocity_error:.3e} exceeds {limits.reciprocity_error:.3e}",
            )
        )
    if maximum_pole_real >= -limits.stability_tolerance:
        findings.append(
            ValidationFinding("fail", "unstable", f"maximum pole real part is {maximum_pole_real:.3e} rad/s")
        )
    if minimum_delay < -limits.delay_tolerance_seconds:
        findings.append(ValidationFinding("fail", "negative_delay", f"minimum delay is {minimum_delay:.3e} s"))

    z0_scale = max(float(np.real(np.asarray(model.z0))), 1.0)
    if expected_passive and z0_scale * passivity_margin < -limits.passivity_tolerance:
        findings.append(
            ValidationFinding("fail", "passivity", f"minimum normalized Y margin is {z0_scale * passivity_margin:.3e}")
        )
    if expected_passive and z0_scale * D_margin < -limits.passivity_tolerance:
        findings.append(ValidationFinding("fail", "D_passivity", f"minimum normalized D margin is {z0_scale * D_margin:.3e}"))
    if expected_passive and E_margin < -limits.slope_tolerance:
        findings.append(ValidationFinding("fail", "E_passivity", f"minimum E margin is {E_margin:.3e}"))
    if simulation_frequency_range is not None:
        requested_low, requested_high = simulation_frequency_range
        if requested_low < validated_range[0] or requested_high > validated_range[1]:
            findings.append(
                ValidationFinding(
                    "fail",
                    "simulation_band",
                    f"requested range {requested_low:.6g}--{requested_high:.6g} Hz exceeds validated range "
                    f"{validated_range[0]:.6g}--{validated_range[1]:.6g} Hz",
                )
            )

    return FitValidationReport(
        status=_status(findings),
        normalized_rmse=normalized_rmse,
        rms_error=rms_error,
        max_absolute_error=max_error,
        heldout_normalized_rmse=heldout_normalized_rmse,
        heldout_max_absolute_error=heldout_max_error,
        worst_error_corner=int(worst_index[0]),
        worst_error_frequency_hz=float(freqs[worst_index[1]]),
        worst_error_port_pair=(int(worst_index[2]), int(worst_index[3])),
        maximum_reciprocity_error=reciprocity_error,
        maximum_pole_real_part=maximum_pole_real,
        minimum_delay_seconds=minimum_delay,
        minimum_passivity_margin=passivity_margin,
        minimum_D_margin=D_margin,
        minimum_E_margin=E_margin,
        validated_frequency_range_hz=validated_range,
        training_corner_count=len(features),
        validation_corner_count=heldout_count,
        validation_frequency_count=len(physical_freqs),
        pole_count=len(model.poles),
        expected_passive=expected_passive,
        expected_reciprocal=expected_reciprocal,
        findings=tuple(findings),
    )
