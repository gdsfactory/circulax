"""Circulax rational fitting.

Recommended S-model workflow:
    fit_model(S, freqs, options=ModelFitOptions(...)) -> ModelCoefficients
    component_from_coefficients(coefficients_or_path) -> CircuitComponent class

Coefficients can be saved/loaded independently of component generation.
The routines below remain available as advanced, backward-compatible APIs.

Public API:
    vfdriver           : two-phase iterative fitting (VFdriver.m equivalent)
    VFModel            : pole-residue model dataclass
    SSModel            : state-space model dataclass
    FitOptions         : fitting hyperparameters
    init_poles_logcmplx: logarithmically-spaced initial poles
    init_poles_lincmplx: linearly-spaced initial poles
"""

import jax

# Enable 64-bit precision by default — required for numerical stability.
jax.config.update("jax_enable_x64", True)

from .aaa import aaa_driver, aaa_scalar, aaa_scalar_numpy
from .aaa_jax import aaa_scalar_jax
from .api import ModelCoefficients, ModelFitOptions, component_from_coefficients, fit_model
from .conditioning_numpy import ConditioningReport, condition_sparameters, project_s_passive, project_s_reciprocal
from .driver import vfdriver
from .enforcement_numpy import enforce_s_passivity_numpy
from .passivity import enforce_passivity
from .pole_sweep import PoleCountSweep, contribution_masks, vmap_pole_count_sweep
from .reduction_numpy import fit_s_numpy
from .sparam import (
    CausalityError,
    CausalityWarning,
    deembed_delay,
    embed_delay,
    evaluate_sparameter_model,
    extract_group_delay,
    fit_with_delay,
    s_to_y,
    scattering_state_space_to_admittance,
    scattering_state_space_to_admittance_jax,
    vmap_scattering_state_space_to_admittance,
    y_to_s,
)
from .surface import (
    PassivityShifts,
    RationalSurface,
    evaluate_surface,
    evaluate_surface_y,
    initialize_surface,
    project_surface_passive,
    refine_surface,
    surface_asymptotic_passivity_margins,
    surface_from_fit,
    surface_loss,
    surface_passivity_loss,
    surface_passivity_margins,
)
from .types import FitOptions, SSModel, VFModel, eval_model
from .utils import (
    init_poles_lincmplx,
    init_poles_linlogcmplx,
    init_poles_logcmplx,
)
from .validation import (
    FitValidationError,
    FitValidationReport,
    FitValidationThresholds,
    ValidationFinding,
    validate_surface_fit,
)

__all__ = [
    "CausalityError",
    "CausalityWarning",
    "ConditioningReport",
    "FitOptions",
    "FitValidationError",
    "FitValidationReport",
    "FitValidationThresholds",
    "ModelCoefficients",
    "ModelFitOptions",
    "PassivityShifts",
    "PoleCountSweep",
    "RationalSurface",
    "SSModel",
    "VFModel",
    "ValidationFinding",
    "aaa_driver",
    "aaa_scalar",
    "aaa_scalar_jax",
    "aaa_scalar_numpy",
    "component_from_coefficients",
    "condition_sparameters",
    "contribution_masks",
    "deembed_delay",
    "embed_delay",
    "enforce_passivity",
    "enforce_s_passivity_numpy",
    "eval_model",
    "evaluate_sparameter_model",
    "evaluate_surface",
    "evaluate_surface_y",
    "extract_group_delay",
    "fit_model",
    "fit_s_numpy",
    "fit_with_delay",
    "init_poles_lincmplx",
    "init_poles_linlogcmplx",
    "init_poles_logcmplx",
    "initialize_surface",
    "project_s_passive",
    "project_s_reciprocal",
    "project_surface_passive",
    "refine_surface",
    "s_to_y",
    "scattering_state_space_to_admittance",
    "scattering_state_space_to_admittance_jax",
    "surface_asymptotic_passivity_margins",
    "surface_from_fit",
    "surface_loss",
    "surface_passivity_loss",
    "surface_passivity_margins",
    "validate_surface_fit",
    "vfdriver",
    "vmap_pole_count_sweep",
    "vmap_scattering_state_space_to_admittance",
    "y_to_s",
]
