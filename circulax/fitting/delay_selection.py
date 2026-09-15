"""Conservative training-only proposals; phase slope is not a causality proof."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .api import ModelCoefficients

import numpy as np


def fit_auto_delay(S, freqs, z0, options) -> ModelCoefficients:
    """Select constructible candidates using training data and an asserted delay bound."""
    from .api import component_from_coefficients, fit_model

    records = []
    baseline = None
    try:
        baseline = fit_model(S, freqs, z0=z0, options=replace(options, delay_mode="none"))
        realized = component_from_coefficients(baseline)
        if isinstance(realized, type):
            from circulax import compile_circuit

            net = {
                "instances": {"core": {"component": "core"}},
                "connections": {},
                "ports": {f"p{i + 1}": f"core,p{i + 1}" for i in range(S.shape[1])},
            }
            realized = compile_circuit(net, {"core": realized}, g_leak=0)
        baseline.metadata.update(
            core_state_count=len(baseline.poles) * S.shape[1],
            circuit_unknowns=realized.sys_size,
            real_solver_unknowns=realized.sys_size * (2 if realized.solver.is_complex else 1),
            line_algebraic_unknowns=2 * sum(g.var_indices.shape[0] for g in realized.groups.values() if g.has_delay),
            history_cost="none for an ordinary delay-free core",
        )
        records.append(
            {
                "kind": "baseline",
                "status": "passed",
                "poles": len(baseline.poles),
                "training_nrmse": baseline.metadata["training_nrmse"],
                "training_max_error": baseline.metadata["training_max_error"],
                "circuit_unknowns": baseline.metadata["circuit_unknowns"],
            }
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        baseline = None
        records.append({"kind": "baseline", "status": "failed", "reason": str(exc)})
    reason = None
    proposal = None
    if S.shape[1] != 2 or not options.reciprocal:
        reason = "automatic inference requires a reciprocal two-port"
    elif options.auto_max_delay is None:
        reason = "supply auto_max_delay (maximum transmission delay); phase unwrapping cannot exclude aliasing"
    elif len(freqs) < 4 or options.auto_max_delay * np.max(np.diff(freqs)) >= 0.5:
        reason = "sampling cannot resolve the asserted delay bound"
    elif np.max(np.linalg.svd(S, compute_uv=False)) > 1 + 1e-8:
        reason = "active data require supplied delays"
    elif np.max(np.abs(S[:, [0, 1], [0, 1]])) > options.auto_reflection_threshold:
        reason = "reflections prevent a justified equal split; supply per-port delays"
    else:
        estimates = []
        for i, j in ((0, 1), (1, 0)):
            response = S[:, i, j]
            if np.min(np.abs(response)) < options.auto_min_transmission:
                reason = "transmission null or insufficient transmission magnitude"
                break
            phase = np.unwrap(np.angle(response))
            x = (freqs - freqs.mean()) / (freqs[-1] - freqs[0])
            slope, intercept = np.polyfit(x, phase, 1)
            delay = -slope / (2 * np.pi * (freqs[-1] - freqs[0]))
            if np.max(np.abs(phase - (slope * x + intercept))) > options.auto_phase_residual:
                reason = "phase is not sufficiently linear"
            elif not 0 < delay <= options.auto_max_delay:
                reason = "phase slope is nonpositive or exceeds the asserted delay bound"
            estimates.append(delay)
        if reason is None:
            if abs(estimates[0] - estimates[1]) > options.auto_direction_tolerance * max(estimates):
                reason = "transmission directions disagree"
            else:
                proposal = float(np.mean(estimates))
    best = baseline
    if proposal is not None:
        for fraction in dict.fromkeys(options.auto_delay_fractions):
            delays = (proposal * fraction / 2,) * 2
            record = {"kind": "delayed", "port_delays_seconds": list(delays), "allocation": "equal split"}
            try:
                candidate = fit_model(S, freqs, z0=z0, options=replace(options, delay_mode="supplied", port_delays=delays))
                peak = float(np.max(np.linalg.svd(candidate.evaluate_core(freqs), compute_uv=False)))
                if peak > 1 + 1e-8:
                    raise ValueError("core fails sampled passivity")  # noqa: TRY301
                record.update(
                    status="passed",
                    poles=len(candidate.poles),
                    sampled_peak_singular_value=peak,
                    training_nrmse=candidate.metadata["training_nrmse"],
                    training_max_error=candidate.metadata["training_max_error"],
                    circuit_unknowns=candidate.metadata.get("circuit_unknowns"),
                )
                if best is None or len(candidate.poles) < len(best.poles):
                    best = candidate
            except (ValueError, np.linalg.LinAlgError) as exc:
                record.update(status="failed", reason=str(exc))
            records.append(record)
    if best is None:
        raise ValueError(f"No constructible candidate meets accuracy limits: {reason}; {records}")
    best.metadata.update(
        delay_provenance="estimated" if np.any(best.port_delays) else "none",
        auto_selection={
            "candidates": records,
            "declined_reason": reason,
            "selected": "delayed" if np.any(best.port_delays) else "baseline",
            "selection_metric": "rational pole count; ties prefer baseline",
            "sampling_assumption": "user-supplied maximum transmission delay; not a causality certificate",
            "baseline_pole_reduction": None if baseline is None else len(baseline.poles) - len(best.poles),
        },
    )
    return best
