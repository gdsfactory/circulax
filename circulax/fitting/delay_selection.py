"""Conservative propagation-delay inference for low-reflection two-ports."""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from .reduction_numpy import pole_groups

if TYPE_CHECKING:
    from .api import ModelCoefficients


def _errors(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    difference = prediction - target
    normalized = float(np.linalg.norm(difference) / max(np.linalg.norm(target), 1e-30))
    return normalized, float(np.max(np.abs(difference)))


def _reservation(count: int, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if count < 10:
        raise ValueError("delay inference requires at least 10 frequency samples for internal validation")
    stride = max(2, round(1 / fraction))
    validation = np.arange(stride - 1, count, stride)
    training = np.setdiff1d(np.arange(count), validation, assume_unique=True)
    if len(validation) < 2 or len(training) < 4:
        raise ValueError("delay inference could not reserve enough training and validation samples")
    return training, validation


def _condition_reciprocity(S: np.ndarray, tolerance: float) -> tuple[np.ndarray, float]:
    projected = 0.5 * (S + S.swapaxes(1, 2))
    asymmetry = float(np.max(np.abs(S - S.swapaxes(1, 2))))
    if asymmetry > tolerance:
        raise ValueError(f"reciprocity error {asymmetry:.6g} exceeds {tolerance:.6g}")
    return projected, asymmetry


def _proposal(S: np.ndarray, freqs: np.ndarray, options) -> tuple[float | None, str | None]:  # noqa: PLR0911
    controls = options.delay_inference
    if options.max_delay is None:
        return None, "supply max_delay; phase unwrapping cannot exclude aliasing"
    if options.max_delay * np.max(np.diff(freqs)) >= 0.5:
        return None, "sampling cannot resolve the asserted delay bound"
    if np.max(np.linalg.svd(S, compute_uv=False)) > 1 + controls.passivity_tolerance:
        return None, "active data require supplied delays"
    if np.max(np.abs(S[:, [0, 1], [0, 1]])) > controls.reflection_threshold:
        return None, "reflections exceed the supported inference limit; supply per-port delays"

    estimates = []
    span = freqs[-1] - freqs[0]
    x = (freqs - freqs.mean()) / span
    for i, j in ((0, 1), (1, 0)):
        response = S[:, i, j]
        if np.min(np.abs(response)) < controls.min_transmission:
            return None, "transmission null or insufficient transmission magnitude"
        phase = np.unwrap(np.angle(response))
        slope, intercept = np.polyfit(x, phase, 1)
        delay = -slope / (2 * np.pi * span)
        if np.max(np.abs(phase - (slope * x + intercept))) > controls.phase_residual:
            return None, "phase is not sufficiently linear"
        if not 0 < delay <= options.max_delay:
            return None, "phase slope is nonpositive or exceeds the asserted delay bound"
        estimates.append(delay)
    if abs(estimates[0] - estimates[1]) > controls.direction_tolerance * max(estimates):
        return None, "transmission directions disagree"
    return float(np.mean(estimates)), None


def fit_auto_delay(S, freqs, z0, options) -> ModelCoefficients:
    """Infer a delay using an internal frequency reservation, then refit all data."""
    from .api import DelayInferenceWarning, fit_model

    if S.shape[1:] != (2, 2) or not options.reciprocal:
        raise ValueError("delay inference requires a reciprocal two-port")

    controls = options.delay_inference
    conditioned, asymmetry = _condition_reciprocity(S, controls.reciprocity_tolerance)
    training_indices, validation_indices = _reservation(len(freqs), controls.validation_fraction)
    training_S, training_freqs = conditioned[training_indices], freqs[training_indices]
    validation_S, validation_freqs = S[validation_indices], freqs[validation_indices]

    records: list[dict] = []
    candidates: list[tuple[str, tuple[float, ...] | None, ModelCoefficients, float, float]] = []

    try:
        baseline = fit_model(
            training_S,
            training_freqs,
            z0=z0,
            options=replace(options, delay_mode="none", port_delays=None, max_delay=None),
        )
        baseline_nrmse, baseline_max = _errors(baseline.evaluate(validation_freqs), validation_S)
        if baseline_nrmse > options.normalized_rmse or baseline_max > options.max_absolute_error:
            message = f"reserved validation exceeds limits: NRMSE={baseline_nrmse:.6g}, max error={baseline_max:.6g}"
            raise ValueError(message)  # noqa: TRY301
        candidates.append(("baseline", None, baseline, baseline_nrmse, baseline_max))
        records.append(
            {
                "kind": "baseline",
                "status": "passed",
                "poles": len(baseline.poles),
                "pole_groups": len(pole_groups(baseline.poles)),
                "validation_nrmse": baseline_nrmse,
                "validation_max_error": baseline_max,
            }
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        records.append({"kind": "baseline", "status": "failed", "reason": str(exc)})

    proposal, declined_reason = _proposal(training_S, training_freqs, options)
    if proposal is not None:
        for fraction in dict.fromkeys(controls.delay_fractions):
            delays = (proposal * fraction / 2,) * 2
            record = {"kind": "delayed", "port_delays_seconds": list(delays), "allocation": "equal split"}
            try:
                candidate = fit_model(
                    training_S,
                    training_freqs,
                    z0=z0,
                    options=replace(options, delay_mode="supplied", port_delays=delays, max_delay=None),
                )
                peak = float(np.max(np.linalg.svd(candidate.evaluate_core(training_freqs), compute_uv=False)))
                if peak > 1 + controls.passivity_tolerance:
                    raise ValueError("core fails sampled passivity")  # noqa: TRY301
                validation_nrmse, validation_max = _errors(candidate.evaluate(validation_freqs), validation_S)
                if validation_nrmse > options.normalized_rmse or validation_max > options.max_absolute_error:
                    raise ValueError(  # noqa: TRY301
                        f"reserved validation exceeds limits: NRMSE={validation_nrmse:.6g}, max error={validation_max:.6g}"
                    )
                candidates.append(("delayed", delays, candidate, validation_nrmse, validation_max))
                record.update(
                    status="passed",
                    poles=len(candidate.poles),
                    pole_groups=len(pole_groups(candidate.poles)),
                    sampled_peak_singular_value=peak,
                    validation_nrmse=validation_nrmse,
                    validation_max_error=validation_max,
                )
            except (ValueError, np.linalg.LinAlgError) as exc:
                record.update(status="failed", reason=str(exc))
            records.append(record)

    baseline_entry = next((item for item in candidates if item[0] == "baseline"), None)
    selected = baseline_entry
    delayed_candidates = [item for item in candidates if item[0] == "delayed"]
    if delayed_candidates:
        delayed_candidates.sort(key=lambda item: (len(pole_groups(item[2].poles)), item[3], item[4]))
        delayed = delayed_candidates[0]
        if baseline_entry is None:
            selected = delayed
        else:
            baseline_groups = len(pole_groups(baseline_entry[2].poles))
            delayed_groups = len(pole_groups(delayed[2].poles))
            allowed_nrmse = baseline_entry[3] + controls.validation_degradation * options.normalized_rmse
            allowed_max = baseline_entry[4] + controls.validation_degradation * options.max_absolute_error
            if delayed_groups < baseline_groups and delayed[3] <= allowed_nrmse and delayed[4] <= allowed_max:
                selected = delayed

    if selected is None:
        raise ValueError(f"No candidate meets the reserved validation limits: {declined_reason}; {records}")

    selected_kind, selected_delays, _, _, _ = selected
    final_options = replace(
        options,
        delay_mode="none" if selected_delays is None else "supplied",
        port_delays=selected_delays,
        max_delay=None,
    )
    result = fit_model(conditioned, freqs, z0=z0, options=final_options)
    original_nrmse, original_max = _errors(result.evaluate(freqs), S)
    if original_nrmse > options.normalized_rmse or original_max > options.max_absolute_error:
        raise ValueError(
            f"Final inferred-delay fit exceeds original-data limits: NRMSE={original_nrmse:.6g}, max error={original_max:.6g}"
        )

    if selected_kind == "baseline":
        message = declined_reason or "inferred-delay candidates did not reduce complexity without validation degradation"
        warnings.warn(f"Delay inference returned the undelayed baseline: {message}", DelayInferenceWarning, stacklevel=2)

    result.metadata.update(
        training_nrmse=original_nrmse,
        training_max_error=original_max,
        delay_provenance="inferred" if selected_delays is not None else "none",
        delay_inference={
            "candidates": records,
            "declined_reason": declined_reason,
            "selected": selected_kind,
            "selection_metric": "complete pole groups with reserved-validation guard",
            "reservation": {
                "fraction": controls.validation_fraction,
                "training_samples": len(training_indices),
                "validation_samples": len(validation_indices),
            },
            "input_reciprocity_error": asymmetry,
            "reciprocity_projection_applied": bool(asymmetry),
            "sampling_assumption": "user-supplied maximum transmission delay; not a causality certificate",
        },
    )
    return result
