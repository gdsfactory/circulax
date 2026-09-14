"""Circulax rational fitting — JAX port of SINTEF Vector Fitting (MFT-NNLS toolbox).

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

from .aaa import aaa_driver
from .driver import vfdriver
from .passivity import enforce_passivity
from .pole_sweep import PoleCountSweep, contribution_masks, vmap_pole_count_sweep
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
    "FitOptions",
    "FitValidationError",
    "FitValidationReport",
    "FitValidationThresholds",
    "PassivityShifts",
    "PoleCountSweep",
    "RationalSurface",
    "SSModel",
    "VFModel",
    "ValidationFinding",
    "aaa_driver",
    "contribution_masks",
    "deembed_delay",
    "embed_delay",
    "enforce_passivity",
    "eval_model",
    "evaluate_sparameter_model",
    "evaluate_surface",
    "evaluate_surface_y",
    "extract_group_delay",
    "fit_with_delay",
    "init_poles_lincmplx",
    "init_poles_linlogcmplx",
    "init_poles_logcmplx",
    "initialize_surface",
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
