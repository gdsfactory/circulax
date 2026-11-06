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
from .rlgc import CharacteristicLineModel, RationalResponse, fit_rlgc, fit_rlgc_characteristic, rlgc_sparameters

__all__ = [
    "CharacteristicLineModel",
    "DelayInferenceOptions",
    "DelayInferenceWarning",
    "ModelCoefficients",
    "ModelFitOptions",
    "ModelValidationReport",
    "RationalResponse",
    "circuit_from_coefficients",
    "fit_model",
    "fit_rlgc",
    "fit_rlgc_characteristic",
    "rlgc_sparameters",
    "validate_model",
]
