"""Stable S-parameter fitting API for Circulax."""

from .api import (
    DelayInferenceOptions,
    DelayInferenceWarning,
    ModelCoefficients,
    ModelFitOptions,
    ModelValidationReport,
    circuit_from_coefficients,
    fit_model,
    validate_model,
)
from .rlgc import fit_rlgc, rlgc_sparameters

__all__ = [
    "DelayInferenceOptions",
    "DelayInferenceWarning",
    "ModelCoefficients",
    "ModelFitOptions",
    "ModelValidationReport",
    "circuit_from_coefficients",
    "fit_model",
    "fit_rlgc",
    "rlgc_sparameters",
    "validate_model",
]
