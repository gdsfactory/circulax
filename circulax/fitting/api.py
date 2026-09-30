"""Stable S-parameter fitting, validation, and circuit-construction API."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from .enforcement_numpy import enforce_s_passivity_numpy
from .reduction_numpy import candidate_subsets, errors_numpy, evaluate_numpy, fit_s_numpy, pole_groups, refine_numpy, refit_numpy
from .sparam import scattering_state_space_to_admittance
from .types import SSModel, VFModel, vfmodel_to_ss

if TYPE_CHECKING:
    from circulax.circuit import Circuit
    from circulax.components.base_component import CircuitComponent


@dataclass(frozen=True)
class DelayInferenceOptions:
    """Conservative controls for propagation-delay inference.

    Inference is limited to passive, approximately reciprocal, low-reflection
    two-ports. By default, 20% of the frequencies are reserved for candidate
    selection, and the selected configuration is refitted on all samples.
    """

    min_transmission: float = 0.05
    phase_residual: float = 0.05
    direction_tolerance: float = 0.05
    reflection_threshold: float = 0.05
    delay_fractions: tuple[float, ...] = (1.0, 0.5)
    reciprocity_tolerance: float = 0.02
    passivity_tolerance: float = 0.01
    validation_fraction: float = 0.2
    validation_degradation: float = 0.1

    def __post_init__(self) -> None:
        positive = (
            self.min_transmission,
            self.phase_residual,
            self.direction_tolerance,
            self.reflection_threshold,
            self.reciprocity_tolerance,
            self.passivity_tolerance,
        )
        if not np.all(np.isfinite(positive)) or min(positive) <= 0:
            raise ValueError("delay-inference tolerances must be finite and positive")
        if not self.delay_fractions or any(not np.isfinite(f) or not 0 < f <= 1 for f in self.delay_fractions):
            raise ValueError("delay_fractions must lie in (0, 1]")
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must lie in (0, 0.5)")
        if not np.isfinite(self.validation_degradation) or self.validation_degradation < 0:
            raise ValueError("validation_degradation must be finite and nonnegative")


class DelayInferenceWarning(RuntimeWarning):
    """Emitted when requested delay inference returns the undelayed baseline."""


@dataclass(frozen=True)
class ModelFitOptions:
    """Controls for proper S fitting with optional one-way port delays.

    Conventional scikit-rf vector fitting is the default. AAA is experimental.
    JAX selects AAA discovery only; refinement and enforcement use NumPy/SciPy.
    Supplied initial poles bypass either discovery backend and reduction, using
    the NumPy residue/refinement solver. They are relocated unless
    iterations=0. Error limits apply AFTER optional enforcement as well.
    """

    delay_mode: Literal["none", "supplied", "infer"] = "none"
    port_delays: tuple[float, ...] | None = None
    max_delay: float | None = None
    delay_inference: DelayInferenceOptions = field(default_factory=DelayInferenceOptions)
    method: Literal["vector_fitting", "aaa"] = "vector_fitting"
    vector_fit_order: tuple[int, int] | None = None
    aaa_backend: Literal["numpy", "jax"] = "numpy"
    tol: float = 1e-8
    mmax: int = 12
    iterations: int = 6
    reciprocal: bool = True
    screening: Literal["compact", "masked"] = "compact"
    reduction_stage: Literal["initial", "refined"] = "refined"
    normalized_rmse: float = 0.02
    max_absolute_error: float = 0.05
    enforce_passivity: bool = False
    passivity_limit: float = 0.999999
    enforcement_iterations: int = 300
    enforcement_freqs: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.delay_mode not in {"none", "supplied", "infer"}:
            raise ValueError("invalid delay_mode")
        if (self.port_delays is not None) != (self.delay_mode == "supplied"):
            raise ValueError("port_delays are required only for supplied delay mode")
        if self.port_delays is not None:
            delays = np.asarray(self.port_delays, float)
            if delays.ndim != 1 or not len(delays) or not np.all(np.isfinite(delays)) or np.any(delays < 0):
                raise ValueError("port_delays must be finite nonnegative one-way seconds")
            object.__setattr__(self, "port_delays", tuple(float(d) for d in delays))
        if self.max_delay is not None and (not np.isfinite(self.max_delay) or self.max_delay <= 0):
            raise ValueError("max_delay must be finite positive seconds")
        if self.delay_mode == "infer" and self.max_delay is None:
            raise ValueError("max_delay is required for inferred delay")
        if self.delay_mode != "infer" and self.max_delay is not None:
            raise ValueError("max_delay applies only to inferred delay")
        if not isinstance(self.delay_inference, DelayInferenceOptions):
            raise TypeError("delay_inference must be DelayInferenceOptions")
        if self.method not in {"vector_fitting", "aaa"}:
            raise ValueError("method must be 'vector_fitting' or 'aaa'")
        if self.vector_fit_order is not None:
            order = self.vector_fit_order
            if len(order) != 2 or any(not isinstance(n, int) or n < 0 for n in order) or sum(order) == 0:
                raise ValueError("vector_fit_order must be (real poles, complex pairs), nonnegative and nonzero")
            if self.method != "vector_fitting":
                raise ValueError("vector_fit_order applies only to vector_fitting")
        if self.aaa_backend not in {"numpy", "jax"} or self.screening not in {"compact", "masked"}:
            raise ValueError("invalid AAA backend or screening method")
        if self.reduction_stage not in {"initial", "refined"}:
            raise ValueError("invalid reduction stage")
        values = (self.tol, self.normalized_rmse, self.max_absolute_error, self.passivity_limit)
        if not np.all(np.isfinite(values)) or min(values) <= 0 or self.passivity_limit >= 1:
            raise ValueError("require finite positive tolerances and passivity_limit < 1")
        for value, minimum in ((self.mmax, 1), (self.iterations, 0), (self.enforcement_iterations, 1)):
            if not isinstance(value, int) or value < minimum:
                raise ValueError("iteration counts and capacity must be valid integers")
        if self.enforce_passivity and not self.reciprocal:
            raise ValueError("rational enforcement currently requires reciprocity")
        if self.enforcement_freqs is not None:
            grid = np.asarray(self.enforcement_freqs, float)
            if grid.ndim != 1 or not len(grid) or not np.all(np.isfinite(grid)) or np.any(grid < 0):
                raise ValueError("enforcement_freqs must be finite nonnegative frequencies")
            object.__setattr__(self, "enforcement_freqs", tuple(float(f) for f in grid))


@dataclass
class ModelCoefficients:
    """Portable S(s)=D+sum(R_k/(s-p_k)) coefficients; no hidden conversion.

    Arrays use full conjugate-pair storage and poles in rad/s. Frequencies are
    Hz, z0 is a common real impedance in ohms. port_delays are one-way seconds.
    evaluate returns P S_core P; there is no proportional term.
    Diagnostics are informational, never a trusted passivity certificate.
    """

    poles: np.ndarray
    residues: np.ndarray
    D: np.ndarray
    z0: float = 50.0
    frequency_range: tuple[float, float] = (0.0, 0.0)
    metadata: dict = field(default_factory=dict)
    port_delays: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.poles = np.array(self.poles, complex, copy=True)
        self.residues = np.array(self.residues, complex, copy=True)
        self.D = np.array(self.D, complex, copy=True)
        if self.poles.ndim != 1 or self.D.ndim != 2 or self.D.shape[0] != self.D.shape[1] or not len(self.D):
            raise ValueError("expected vector poles and a nonempty square D")
        self.port_delays = np.zeros(len(self.D)) if self.port_delays is None else np.array(self.port_delays, float, copy=True)
        if self.port_delays.shape != (len(self.D),) or not np.all(np.isfinite(self.port_delays)) or np.any(self.port_delays < 0):
            raise ValueError("port_delays must contain one finite nonnegative one-way duration per port")
        if self.residues.shape != (*self.D.shape, len(self.poles)):
            raise ValueError("residues must have shape (ports, ports, poles)")
        if not all(np.all(np.isfinite(a)) for a in (self.poles, self.residues, self.D)):
            raise ValueError("coefficients must be finite")
        if np.ndim(self.z0) != 0 or not np.isreal(self.z0) or not np.isfinite(self.z0) or np.real(self.z0) <= 0:
            raise ValueError("z0 must be a real positive scalar")
        self.z0 = float(np.real(self.z0))
        if not np.allclose(self.D.imag, 0, atol=1e-12):
            raise ValueError("D must be real")
        for group in pole_groups(self.poles):
            a = self.residues[..., group[0]]
            b = a.conj() if len(group) == 1 else self.residues[..., group[1]].conj()
            if not np.allclose(a, b, rtol=1e-8, atol=1e-12):
                raise ValueError("residues must have real/conjugate symmetry")
        if (
            len(self.frequency_range) != 2
            or not np.all(np.isfinite(self.frequency_range))
            or not 0 <= self.frequency_range[0] <= self.frequency_range[1]
        ):
            raise ValueError("invalid frequency range")

    def _model(self):
        return VFModel(self.poles, self.residues, self.D.real, np.zeros_like(self.D.real))

    def evaluate(self, freqs: np.ndarray) -> np.ndarray:
        """Evaluate complex S-parameters at frequencies in Hz."""
        phase = np.exp(-2j * np.pi * np.asarray(freqs)[:, None] * self.port_delays)
        return self.evaluate_core(freqs) * phase[:, :, None] * phase[:, None, :]

    def evaluate_core(self, freqs: np.ndarray) -> np.ndarray:
        """Evaluate the rational core, before one-way port delays (seconds)."""
        return evaluate_numpy(self._model(), np.asarray(freqs, float))

    def save(self, path: str | Path) -> None:
        """Write a versioned NPZ archive (no pickle); preserve the exact path."""
        header = json.dumps(
            {
                "version": 2,
                "delay_convention": "one-way-seconds-per-port",
                "domain": "s",
                "z0": float(self.z0),
                "frequency_range": self.frequency_range,
                "metadata": self.metadata,
            }
        )
        with Path(path).open("wb") as stream:
            np.savez_compressed(
                stream, poles=self.poles, residues=self.residues, D=self.D, port_delays=self.port_delays, header=header
            )

    @classmethod
    def load(cls, path: str | Path) -> ModelCoefficients:
        """Read and validate an archive without executing pickled objects."""
        with np.load(path, allow_pickle=False) as archive:
            header = json.loads(str(archive["header"].item()))
            if header["version"] not in (1, 2) or header["domain"] != "s":
                raise ValueError("unsupported coefficient schema or domain")
            if header["version"] == 2 and header.get("delay_convention") != "one-way-seconds-per-port":
                raise ValueError("unsupported delay convention")
            return cls(
                archive["poles"],
                archive["residues"],
                archive["D"],
                header["z0"],
                tuple(header["frequency_range"]),
                header["metadata"],
                archive["port_delays"] if header["version"] == 2 else None,
            )


@dataclass(frozen=True)
class ModelValidationReport:
    """Validation evidence for admitting fitted coefficients to simulation."""

    status: Literal["pass", "warn", "fail"]
    training_nrmse: float | None
    training_max_error: float | None
    validation_nrmse: float | None
    validation_max_error: float | None
    maximum_singular_value: float
    maximum_s_pole_real_part: float
    maximum_y_pole_real_part: float
    findings: tuple[str, ...]

    @property
    def simulation_ready(self) -> bool:
        """Whether validation passed without findings."""
        return self.status == "pass"

    def raise_for_simulation(self, *, allow_warnings: bool = False) -> None:
        """Raise when the report does not admit the model to simulation."""
        if self.status == "fail" or (self.status == "warn" and not allow_warnings):
            raise ValueError(f"model validation {self.status}: {'; '.join(self.findings)}")


def _validation_errors(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    difference = prediction - target
    return (
        float(np.linalg.norm(difference) / max(np.linalg.norm(target), 1e-30)),
        float(np.max(np.abs(difference))),
    )


def _validation_dataset(S: np.ndarray, freqs: np.ndarray, ports: int, label: str) -> tuple[np.ndarray, np.ndarray]:
    S = np.asarray(S, complex)
    freqs = np.asarray(freqs, float)
    if freqs.ndim != 1 or not len(freqs) or not np.all(np.isfinite(freqs)) or np.any(freqs < 0):
        raise ValueError(f"{label} frequencies must be a finite nonempty vector of nonnegative Hz values")
    if S.shape != (len(freqs), ports, ports) or not np.all(np.isfinite(S)):
        raise ValueError(f"{label} S data must be finite with shape (frequencies, {ports}, {ports})")
    return S, freqs


def validate_model(
    coefficients: ModelCoefficients | str | Path,
    *,
    measured_S: np.ndarray | None = None,
    freqs: np.ndarray | None = None,
    validation_S: np.ndarray | None = None,
    validation_freqs: np.ndarray | None = None,
    simulation_frequency_range: tuple[float, float] | None = None,
    expected_passive: bool = True,
    expected_reciprocal: bool = True,
    normalized_rmse: float = 0.02,
    max_absolute_error: float = 0.05,
    passivity_tolerance: float = 1e-9,
) -> ModelValidationReport:
    """Validate fitted coefficients without compiling a circuit.

    Accuracy checks are performed when measured data are supplied. Independent
    validation requires both ``validation_S`` and ``validation_freqs``.
    Passivity is required by default; active models must opt out explicitly.
    """
    source = ModelCoefficients.load(coefficients) if isinstance(coefficients, (str, Path)) else coefficients
    source = ModelCoefficients(
        source.poles, source.residues, source.D, source.z0, source.frequency_range, source.metadata, source.port_delays
    )
    if (measured_S is None) != (freqs is None):
        raise ValueError("measured_S and freqs must be supplied together")
    if (validation_S is None) != (validation_freqs is None):
        raise ValueError("validation_S and validation_freqs must be supplied together")
    limits = (normalized_rmse, max_absolute_error, passivity_tolerance)
    if not np.all(np.isfinite(limits)) or normalized_rmse <= 0 or max_absolute_error <= 0 or passivity_tolerance < 0:
        raise ValueError("validation limits must be finite; error limits must be positive and passivity tolerance nonnegative")

    ports = len(source.D)
    training_data = None
    validation_data = None
    if measured_S is not None:
        training_data = _validation_dataset(measured_S, freqs, ports, "training")
    if validation_S is not None:
        validation_data = _validation_dataset(validation_S, validation_freqs, ports, "held-out")

    findings: list[str] = []
    physical_grids = [data[1] for data in (training_data, validation_data) if data is not None]
    evaluation_freqs = np.unique(np.concatenate(physical_grids)) if physical_grids else np.asarray(source.frequency_range, float)
    response = source.evaluate(evaluation_freqs)
    training_nrmse = training_max = validation_nrmse = validation_max = None
    if training_data is not None:
        training_S, training_freqs = training_data
        training_nrmse, training_max = _validation_errors(source.evaluate(training_freqs), training_S)
        if training_nrmse > normalized_rmse or training_max > max_absolute_error:
            findings.append("training accuracy exceeds the requested limits")
    else:
        findings.append("no measured training data supplied")
    if validation_data is not None:
        heldout_S, heldout_freqs = validation_data
        validation_nrmse, validation_max = _validation_errors(source.evaluate(heldout_freqs), heldout_S)
        if validation_nrmse > normalized_rmse or validation_max > max_absolute_error:
            findings.append("held-out accuracy exceeds the requested limits")
    else:
        findings.append("no independent validation data supplied")

    if expected_reciprocal and np.max(np.abs(response - response.swapaxes(1, 2))) > 1e-8:
        findings.append("model is not reciprocal")
    maximum_singular = float(np.max(np.linalg.svd(response, compute_uv=False)))
    if expected_passive and maximum_singular > 1 + passivity_tolerance:
        findings.append("model fails sampled passivity")
    maximum_s_pole = float(np.max(source.poles.real)) if len(source.poles) else -np.inf
    if maximum_s_pole >= 0:
        findings.append("model has unstable S poles")

    n = len(source.D)
    through = not len(source.poles) and n == 2 and np.array_equal(source.D, [[0, 1], [1, 0]])
    maximum_y_pole = -np.inf
    if not through:
        if np.linalg.matrix_rank(np.eye(n) + source.D) < n:
            findings.append("model has an unsupported singular I+D core")
        elif len(source.poles):
            model = VFModel(source.poles, source.residues, source.D.real, np.zeros((n, n)))
            try:
                ss, _ = scattering_state_space_to_admittance(vfmodel_to_ss(model, n), z0=source.z0)
                maximum_y_pole = float(np.max(np.asarray(ss.A).real)) if len(ss.A) else -np.inf
                if maximum_y_pole >= 0:
                    findings.append("model has unstable Y poles")
            except np.linalg.LinAlgError:
                findings.append("S-to-Y realization failed")

    if simulation_frequency_range is not None:
        if len(simulation_frequency_range) != 2:
            raise ValueError("simulation_frequency_range must contain (minimum, maximum) Hz")
        low, high = simulation_frequency_range
        if not np.all(np.isfinite((low, high))) or not 0 <= low <= high:
            raise ValueError("simulation_frequency_range must be finite, nonnegative, and ordered")
        if low < source.frequency_range[0] or high > source.frequency_range[1]:
            findings.append("requested simulation range exceeds the fitted frequency range")

    failures = tuple(item for item in findings if not item.startswith("no measured") and not item.startswith("no independent"))
    status: Literal["pass", "warn", "fail"] = "fail" if failures else ("warn" if findings else "pass")
    return ModelValidationReport(
        status,
        training_nrmse,
        training_max,
        validation_nrmse,
        validation_max,
        maximum_singular,
        maximum_s_pole,
        maximum_y_pole,
        tuple(findings),
    )


def fit_model(
    S: np.ndarray,
    freqs: np.ndarray,
    *,
    options: ModelFitOptions | None = None,
    z0: float = 50.0,
    initial_poles: np.ndarray | None = None,
) -> ModelCoefficients:
    """Fit S data and return coefficients only, without making a component.

    Automatic order selection is the default. Optional enforcement is sampled,
    not globally certified; independent rational testing remains necessary.
    """
    options = options or ModelFitOptions()
    S, freqs = np.asarray(S, complex), np.asarray(freqs, float)
    if freqs.ndim != 1 or not len(freqs) or not np.all(np.isfinite(freqs)) or freqs[0] < 0 or np.any(np.diff(freqs) <= 0):
        raise ValueError("frequencies must be finite, nonnegative and increasing")
    if S.ndim != 3 or S.shape[0] != len(freqs) or S.shape[1] != S.shape[2] or not S.shape[1] or not np.all(np.isfinite(S)):
        raise ValueError("S must be finite with shape (frequencies, ports, ports)")
    # Validate impedance before doing any potentially expensive fitting.
    ModelCoefficients(np.array([], complex), np.zeros((*S.shape[1:], 0)), np.zeros(S.shape[1:]), z0)
    if initial_poles is not None and options.vector_fit_order is not None:
        raise ValueError("Choose initial_poles or vector_fit_order, not both")
    if options.delay_mode == "infer":
        if initial_poles is not None:
            raise ValueError("delay inference cannot be combined with supplied poles")
        from .delay_selection import fit_auto_delay

        return fit_auto_delay(S, freqs, z0, options)
    if options.reciprocal and not np.allclose(S, S.swapaxes(1, 2), atol=1e-8, rtol=0):
        raise ValueError("reciprocal fitting requires symmetric data")
    fit_started = time.perf_counter()
    original = S
    delays = np.zeros(S.shape[1]) if options.port_delays is None else np.asarray(options.port_delays)
    if delays.shape != (S.shape[1],):
        raise ValueError("port_delays must match the number of ports")
    phase = np.exp(2j * np.pi * freqs[:, None] * delays)
    S = S * phase[:, :, None] * phase[:, None, :]
    static = VFModel(np.empty(0, complex), np.empty((*S.shape[1:], 0), complex), S.real.mean(axis=0), np.zeros(S.shape[1:]))
    if S.shape[1] == 2 and np.max(np.abs(S - np.array([[0, 1], [1, 0]]))) <= min(
        1e-14, options.normalized_rmse / 2, options.max_absolute_error / 2
    ):
        static = VFModel(static.poles, static.residues, np.array([[0.0, 1.0], [1.0, 0.0]]), static.E)
    static_error, static_max = errors_numpy(static, S, freqs)
    if (
        options.delay_mode == "supplied"
        and initial_poles is None
        and static_error <= min(options.normalized_rmse, 1e-12)
        and static_max <= min(options.max_absolute_error, 1e-12)
    ):
        model, diagnostics = static, {"fitter": "static", "pole_count": 0}
    elif initial_poles is None and options.method == "vector_fitting":
        model, diagnostics = _fit_vector_fitting(S, freqs, z0, options)
    elif initial_poles is None:
        model, diagnostics = fit_s_numpy(
            S,
            freqs,
            tol=options.tol,
            mmax=options.mmax,
            iterations=options.iterations,
            reciprocal=options.reciprocal,
            screening=options.screening,
            reduction_stage=options.reduction_stage,
            aaa_backend=options.aaa_backend,
            normalized_rmse=options.normalized_rmse,
            max_absolute_error=options.max_absolute_error,
        )
    else:
        poles = np.asarray(initial_poles, complex)
        if poles.ndim != 1 or not np.all(np.isfinite(poles)) or np.any(poles.real >= 0):
            raise ValueError("initial poles must be finite and strictly stable")
        pole_groups(poles)
        model = refine_numpy(S, freqs, poles, iterations=options.iterations, reciprocal=options.reciprocal)
        diagnostics = {
            "fitter": "numpy-supplied-poles",
            "order_selection": "supplied",
            "initial_poles": [[float(p.real), float(p.imag)] for p in poles],
        }
    if options.delay_mode == "supplied" and initial_poles is None and options.vector_fit_order is None and len(model.poles):
        diagnostics["unreduced_core_poles"] = len(model.poles)
        for subset in candidate_subsets(model, freqs):
            candidate = refit_numpy(S, 2j * np.pi * freqs, model.poles[subset])
            candidate_error, candidate_max = errors_numpy(candidate, S, freqs)
            if candidate_error <= options.normalized_rmse and candidate_max <= options.max_absolute_error:
                model = candidate
                break
    fitting_seconds = time.perf_counter() - fit_started
    enforcement_started = time.perf_counter()
    if options.enforce_passivity:
        model, report = enforce_s_passivity_numpy(
            model,
            freqs,
            enforcement_freqs=options.enforcement_freqs,
            limit=options.passivity_limit,
            max_iterations=options.enforcement_iterations,
        )
        diagnostics["enforcement"] = report
        if not report["converged"]:
            raise ValueError(f"Passivity enforcement failed: {report['optimizer_message']}")
    enforcement_seconds = time.perf_counter() - enforcement_started if options.enforce_passivity else 0.0
    error, maximum = errors_numpy(model, S, freqs)
    if error > options.normalized_rmse or maximum > options.max_absolute_error:
        raise ValueError(f"Final fit exceeds accuracy limits: NRMSE={error:.6g}, max error={maximum:.6g}")
    diagnostics.update(
        training_nrmse=error,
        training_max_error=maximum,
        options=json.loads(json.dumps(asdict(options))),
        global_passivity_certified=False,
    )
    result = ModelCoefficients(model.poles, model.residues, model.D, z0, (float(freqs[0]), float(freqs[-1])), diagnostics, delays)
    difference = result.evaluate(freqs) - original
    error = float(np.linalg.norm(difference) / max(np.linalg.norm(original), 1e-30))
    maximum = float(np.max(np.abs(difference)))
    if error > options.normalized_rmse or maximum > options.max_absolute_error:
        raise ValueError("Final fit exceeds original-domain accuracy limits")
    diagnostics.update(
        training_nrmse=error,
        training_max_error=maximum,
        core_pole_count=len(model.poles),
        delay_provenance=options.delay_mode,
        port_delays_seconds=delays.tolist(),
    )
    diagnostics.update(fitting_seconds=fitting_seconds, enforcement_seconds=enforcement_seconds, pole_count=len(model.poles))
    return result


def _fit_vector_fitting(S, freqs, z0, options):
    # Handle exact real static data without asking a dynamic fitter to identify
    # poles of a constant (an underdetermined problem).
    if np.all(S[0] == S) and np.all(S.imag == 0):
        n = S.shape[1]
        return VFModel(np.empty(0, complex), np.empty((n, n, 0), complex), S[0].real.copy(), np.zeros((n, n))), {
            "fitter": "static",
            "pole_count": 0,
        }
    try:
        import skrf
        from skrf.vectorFitting import VectorFitting
    except ImportError as exc:
        raise ImportError("Default vector fitting requires scikit-rf. Install scikit-rf>=1.8,<2 or select method='aaa'.") from exc
    network = skrf.Network(frequency=skrf.Frequency.from_f(freqs, unit="Hz"), s=S, z0=z0)
    fitter = VectorFitting(network)
    if options.vector_fit_order is None:
        fitter.auto_fit(parameter_type="s")
    else:
        real, pairs = options.vector_fit_order
        fitter.vector_fit(n_poles_real=real, n_poles_cmplx=pairs, parameter_type="s", fit_constant=True, fit_proportional=False)
    poles, residues = [], []
    n = S.shape[1]
    for k, pole in enumerate(fitter.poles):
        residue = fitter.residues[:, k].reshape(n, n)
        poles.append(pole)
        residues.append(residue)
        if pole.imag != 0:
            poles.append(pole.conjugate())
            residues.append(residue.conjugate())
    residue_array = np.stack(residues, axis=-1) if residues else np.empty((n, n, 0), complex)
    model = VFModel(np.asarray(poles, complex), residue_array, fitter.constant_coeff.reshape(n, n), np.zeros((n, n)))
    return model, {
        "fitter": "scikit-rf",
        "fitter_version": skrf.__version__,
        "pole_count": len(poles),
        "order_selection": "automatic" if options.vector_fit_order is None else "prescribed",
    }


def _component_from_coefficients(
    coefficients: ModelCoefficients | str | Path, *, name: str = "FittedModel", holomorphic: bool = True
) -> type[CircuitComponent] | Circuit:
    """Return a leaf class or a Circuit composed with exact delay lines.

    Integrate either result through compile_circuit(models_map=...).

    No fitting or enforcement occurs here. Unstable S or converted Y poles are
    rejected. Stability checks do not certify passivity or extrapolation accuracy.
    The component supports AC/HB/transient using the existing complex-state API.
    """
    from circulax.components.rational import rational_component

    source = ModelCoefficients.load(coefficients) if isinstance(coefficients, (str, Path)) else coefficients
    # Revalidate even an in-memory object: callers may have modified its arrays.
    source = ModelCoefficients(
        source.poles, source.residues, source.D, source.z0, source.frequency_range, source.metadata, source.port_delays
    )
    if np.any(source.poles.real >= 0):
        raise ValueError("Cannot create component: unstable S poles")
    n = len(source.D)
    through = not len(source.poles) and n == 2 and np.array_equal(source.D, [[0, 1], [1, 0]])
    if np.any(source.port_delays) or through:
        from uuid import uuid4

        from circulax import compile_circuit
        from circulax.components.electronic import TransmissionLine

        key = "fitted_" + uuid4().hex
        if through:
            net = {
                "instances": {"core": {"component": key, "settings": {"tau": float(sum(source.port_delays)), "z0": source.z0}}},
                "connections": {},
                "ports": {"p1": "core,p1", "p2": "core,p2"},
            }
            return compile_circuit(net, {key: TransmissionLine}, g_leak=0)
        core = _component_from_coefficients(replace(source, port_delays=np.zeros(n)), name=name, holomorphic=holomorphic)
        models = {key: core, key + "_line": TransmissionLine}
        net = {"instances": {"core": {"component": key}}, "connections": {}, "ports": {}}
        for i, delay in enumerate(source.port_delays, 1):
            port = f"p{i}"
            if delay:
                line = f"line{i}"
                net["instances"][line] = {"component": key + "_line", "settings": {"tau": float(delay), "z0": source.z0}}
                net["connections"][f"{line},p2"] = f"core,{port}"
                net["ports"][port] = f"{line},p1"
            else:
                net["ports"][port] = f"core,{port}"
        return compile_circuit(net, models, g_leak=0)
    if np.linalg.matrix_rank(np.eye(n) + source.D) < n:
        raise ValueError("Cannot realize singular I+D core; only the ideal two-port through is supported")
    model = VFModel(source.poles, source.residues, source.D.real, np.zeros((n, n)))
    if len(source.poles):
        ss, _ = scattering_state_space_to_admittance(vfmodel_to_ss(model, n), z0=source.z0)
    else:
        # A static network has no eigenvectors to diagonalize or condition.
        direct = np.linalg.solve((np.eye(n) + source.D).T, (np.eye(n) - source.D).T).T / source.z0
        ss = SSModel(np.empty(0, complex), np.empty((0, n)), np.empty((n, 0)), direct, np.zeros((n, n)))
    if not all(np.all(np.isfinite(a)) for a in (ss.A, ss.B, ss.C, ss.D, ss.E)) or np.any(np.asarray(ss.A).real >= 0):
        raise ValueError("Cannot create component: nonfinite realization or unstable Y poles; reassess fitting/enforcement")
    return rational_component(ss, name=name, z0=source.z0, holomorphic=holomorphic)


def circuit_from_coefficients(
    coefficients: ModelCoefficients | str | Path, *, name: str = "FittedModel", holomorphic: bool = True
) -> Circuit:
    """Build a simulation-ready :class:`circulax.Circuit` from coefficients.

    The return type is the same for rational cores, delayed models, and ideal
    through lines. Fitting and passivity enforcement never occur here.
    """
    from circulax import Circuit, compile_circuit

    realized = _component_from_coefficients(coefficients, name=name, holomorphic=holomorphic)
    if isinstance(realized, Circuit):
        return realized
    ports = {port: f"model,{port}" for port in realized.ports}
    net = {"instances": {"model": {"component": "model"}}, "connections": {}, "ports": ports}
    return compile_circuit(net, {"model": realized}, g_leak=0)
