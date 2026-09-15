"""Recommended two-stage S fitting API: coefficients first, component second."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from .enforcement_numpy import enforce_s_passivity_numpy
from .reduction_numpy import errors_numpy, evaluate_numpy, fit_s_numpy, pole_groups, refine_numpy
from .sparam import scattering_state_space_to_admittance
from .types import SSModel, VFModel, vfmodel_to_ss

if TYPE_CHECKING:
    from circulax.components.base_component import CircuitComponent


@dataclass(frozen=True)
class ModelFitOptions:
    """All controls for the proper, delay-free S fitting workflow.

    JAX selects AAA discovery only; refinement and enforcement use NumPy/SciPy.
    Supplied initial poles bypass AAA and reduction. They are relocated unless
    iterations=0. Error limits apply AFTER optional enforcement as well.
    """

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
    Hz, z0 is a common real impedance in ohms. No delay or proportional term.
    Diagnostics are informational, never a trusted passivity certificate.
    """

    poles: np.ndarray
    residues: np.ndarray
    D: np.ndarray
    z0: float = 50.0
    frequency_range: tuple[float, float] = (0.0, 0.0)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.poles = np.array(self.poles, complex, copy=True)
        self.residues = np.array(self.residues, complex, copy=True)
        self.D = np.array(self.D, complex, copy=True)
        if self.poles.ndim != 1 or self.D.ndim != 2 or self.D.shape[0] != self.D.shape[1] or not len(self.D):
            raise ValueError("expected vector poles and a nonempty square D")
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
        return evaluate_numpy(self._model(), np.asarray(freqs, float))

    def save(self, path: str | Path) -> None:
        """Write a versioned NPZ archive (no pickle); preserve the exact path."""
        header = json.dumps(
            {"version": 1, "domain": "s", "z0": float(self.z0), "frequency_range": self.frequency_range, "metadata": self.metadata}
        )
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, poles=self.poles, residues=self.residues, D=self.D, header=header)

    @classmethod
    def load(cls, path: str | Path) -> ModelCoefficients:
        """Read and validate an archive without executing pickled objects."""
        with np.load(path, allow_pickle=False) as archive:
            header = json.loads(str(archive["header"].item()))
            if header["version"] != 1 or header["domain"] != "s":
                raise ValueError("unsupported coefficient schema or domain")
            return cls(
                archive["poles"],
                archive["residues"],
                archive["D"],
                header["z0"],
                tuple(header["frequency_range"]),
                header["metadata"],
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
    if initial_poles is None:
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
        diagnostics = {"order_selection": "supplied", "initial_poles": [[float(p.real), float(p.imag)] for p in poles]}
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
    error, maximum = errors_numpy(model, S, freqs)
    if error > options.normalized_rmse or maximum > options.max_absolute_error:
        raise ValueError(f"Final fit exceeds accuracy limits: NRMSE={error:.6g}, max error={maximum:.6g}")
    diagnostics.update(training_nrmse=error, training_max_error=maximum, options=asdict(options), global_passivity_certified=False)
    return ModelCoefficients(model.poles, model.residues, model.D, z0, (float(freqs[0]), float(freqs[-1])), diagnostics)


def component_from_coefficients(
    coefficients: ModelCoefficients | str | Path, *, name: str = "FittedModel", holomorphic: bool = True
) -> type[CircuitComponent]:
    """Read coefficients, convert S to Y, and return a CircuitComponent class.

    No fitting or enforcement occurs here. Unstable S or converted Y poles are
    rejected. Stability checks do not certify passivity or extrapolation accuracy.
    The component supports AC/HB/transient using the existing complex-state API.
    """
    from circulax.components.rational import rational_component

    source = ModelCoefficients.load(coefficients) if isinstance(coefficients, (str, Path)) else coefficients
    # Revalidate even an in-memory object: callers may have modified its arrays.
    source = ModelCoefficients(source.poles, source.residues, source.D, source.z0, source.frequency_range, source.metadata)
    if np.any(source.poles.real >= 0):
        raise ValueError("Cannot create component: unstable S poles")
    n = len(source.D)
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
