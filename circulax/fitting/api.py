"""Recommended two-stage S fitting API: coefficients first, component second."""

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
class ModelFitOptions:
    """Controls for proper S fitting with optional one-way port delays.

    Conventional scikit-rf vector fitting is the default. AAA is experimental.
    JAX selects AAA discovery only; refinement and enforcement use NumPy/SciPy.
    Supplied initial poles bypass either discovery backend and reduction, using
    the NumPy residue/refinement solver. They are relocated unless
    iterations=0. Error limits apply AFTER optional enforcement as well.
    """

    delay_mode: Literal["none", "supplied", "auto"] = "none"
    port_delays: tuple[float, ...] | None = None
    auto_max_delay: float | None = None
    auto_min_transmission: float = 0.05
    auto_phase_residual: float = 0.05
    auto_direction_tolerance: float = 0.05
    auto_reflection_threshold: float = 1e-3
    auto_delay_fractions: tuple[float, ...] = (1.0, 0.5)
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
        if self.delay_mode not in {"none", "supplied", "auto"}:
            raise ValueError("invalid delay_mode")
        if (self.port_delays is not None) != (self.delay_mode == "supplied"):
            raise ValueError("port_delays are required only for supplied delay mode")
        if self.port_delays is not None:
            delays = np.asarray(self.port_delays, float)
            if delays.ndim != 1 or not len(delays) or not np.all(np.isfinite(delays)) or np.any(delays < 0):
                raise ValueError("port_delays must be finite nonnegative one-way seconds")
            object.__setattr__(self, "port_delays", tuple(float(d) for d in delays))
        if self.auto_max_delay is not None and (not np.isfinite(self.auto_max_delay) or self.auto_max_delay <= 0):
            raise ValueError("auto_max_delay must be finite positive seconds")
        safeguards = (
            self.auto_min_transmission,
            self.auto_phase_residual,
            self.auto_direction_tolerance,
            self.auto_reflection_threshold,
        )
        if not np.all(np.isfinite(safeguards)) or min(safeguards) <= 0:
            raise ValueError("automatic safeguards must be finite and positive")
        if not self.auto_delay_fractions or any(not np.isfinite(f) or not 0 < f <= 1 for f in self.auto_delay_fractions):
            raise ValueError("auto_delay_fractions must lie in (0, 1]")
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
    if options.reciprocal and not np.allclose(S, S.swapaxes(1, 2), atol=1e-8, rtol=0):
        raise ValueError("reciprocal fitting requires symmetric data")
    # Validate impedance before doing any potentially expensive fitting.
    ModelCoefficients(np.array([], complex), np.zeros((*S.shape[1:], 0)), np.zeros(S.shape[1:]), z0)
    if initial_poles is not None and options.vector_fit_order is not None:
        raise ValueError("Choose initial_poles or vector_fit_order, not both")
    if options.delay_mode == "auto":
        if initial_poles is not None:
            raise ValueError("automatic delay inference cannot be combined with supplied poles")
        from .delay_selection import fit_auto_delay

        return fit_auto_delay(S, freqs, z0, options)
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
    if options.delay_mode != "none":
        started = time.perf_counter()
        realized = component_from_coefficients(result)
        diagnostics["conversion_compilation_seconds"] = time.perf_counter() - started
        diagnostics["realization_validated"] = True
        diagnostics["sampled_core_peak_singular_value"] = float(
            np.max(np.linalg.svd(result.evaluate_core(freqs), compute_uv=False))
        )
        diagnostics["core_state_count"] = len(model.poles) * S.shape[1]
        diagnostics["line_algebraic_unknowns"] = 0
        if hasattr(realized, "sys_size"):
            diagnostics["line_algebraic_unknowns"] = 2 * sum(
                group.var_indices.shape[0] for group in realized.groups.values() if group.has_delay
            )
            diagnostics["circuit_unknowns"] = realized.sys_size
            diagnostics["real_solver_unknowns"] = realized.sys_size * (2 if realized.solver.is_complex else 1)
        diagnostics["history_cost"] = "full solver state per accepted step; depends on transient max_steps"
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


def component_from_coefficients(
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
        core = component_from_coefficients(replace(source, port_delays=np.zeros(n)), name=name, holomorphic=holomorphic)
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
