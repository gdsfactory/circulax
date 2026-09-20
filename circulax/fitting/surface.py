"""Differentiable fixed-pole rational surface fitting prototype.

AAA discovers a pole topology once. With that topology fixed, this module
fits coefficient surfaces and refines them against complex S-parameter error
using JAX autodiff and batching.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .residue_id import identify_residues
from .sparam import deembed_delay, extract_group_delay, s_to_y
from .types import FitOptions, SSModel
from .utils import stack_upper_triangle


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RationalSurface:
    """Fixed-pole rational model whose coefficients vary over features."""

    poles: jax.Array
    residue_coeffs: jax.Array
    D_coeffs: jax.Array
    E_coeffs: jax.Array
    tau_coeffs: jax.Array
    omega_scale: jax.Array
    z0: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PassivityShifts:
    """Common diagonal shifts applied by sampled passivity projection."""

    conductance: jax.Array
    slope: jax.Array


def surface_from_fit(
    ss: SSModel,
    tau_per_port: np.ndarray,
    omega_scale: float,
    *,
    z0: complex = 50.0,
) -> RationalSurface:
    """Wrap one fitted state-space model as a constant rational surface."""
    nc = int(ss.D.shape[0])
    tau_per_port = np.asarray(tau_per_port, dtype=float)
    if tau_per_port.shape != (nc,):
        msg = f"tau_per_port must have shape ({nc},); got {tau_per_port.shape}"
        raise ValueError(msg)
    return RationalSurface(
        poles=jnp.asarray(ss.A / omega_scale),
        residue_coeffs=jnp.einsum("ik,kj->ijk", ss.C, ss.B)[None, ...] / omega_scale,
        D_coeffs=jnp.real(ss.D)[None, ...],
        E_coeffs=jnp.real(ss.E)[None, ...] * omega_scale,
        tau_coeffs=jnp.asarray(tau_per_port[None, ...] * omega_scale),
        omega_scale=jnp.asarray(omega_scale),
        z0=jnp.asarray(z0, dtype=jnp.complex128),
    )


def _regress(features: np.ndarray, values: np.ndarray) -> jax.Array:
    """Least-squares surface coefficients with arbitrary trailing dimensions."""
    shape = values.shape[1:]
    coeffs = np.linalg.lstsq(features, values.reshape(len(features), -1), rcond=None)[0]
    return jnp.asarray(coeffs.reshape(features.shape[1], *shape))


def initialize_surface(
    S: np.ndarray,
    freqs: np.ndarray,
    features: np.ndarray,
    poles: np.ndarray,
    *,
    z0: complex = 50.0,
    opts: FitOptions | None = None,
    delay_scale: float = 1.0,
) -> RationalSurface:
    """Initialize a shared-pole surface using fixed-pole linear fits.

    ``S`` has shape ``(corners, frequencies, ports, ports)``. Delay and
    residue identification are initialization steps outside the differentiable
    refinement path. ``features`` includes the desired surface basis and its
    intercept column.
    """
    S = np.asarray(S, dtype=np.complex128)
    freqs = np.asarray(freqs, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    poles = np.asarray(poles, dtype=np.complex128)
    if S.ndim != 4:
        msg = f"S must have shape (corners, frequencies, ports, ports); got {S.shape}"
        raise ValueError(msg)
    if features.shape[0] != S.shape[0]:
        msg = "features and S must contain the same number of corners"
        raise ValueError(msg)
    if opts is None:
        opts = FitOptions(N=len(poles), asymp=2, weightparam=2)

    omega_scale = float(2.0 * np.pi * np.max(np.abs(freqs)))
    s = jnp.asarray(1j * 2.0 * np.pi * freqs)
    taus, residues, Ds, Es = [], [], [], []

    for S_corner in S:
        tau = extract_group_delay(S_corner, freqs, scale=delay_scale)
        S_deembedded = deembed_delay(S_corner, freqs, tau)
        Y = np.stack([s_to_y(Sk, z0) for Sk in S_deembedded])
        bigH = jnp.asarray(np.moveaxis(Y, 0, -1))
        f = stack_upper_triangle(bigH)
        weights = jnp.ones((1, len(freqs)))
        C_flat, D_flat, E_flat = identify_residues(f, s, poles, weights, opts)

        nc = S.shape[-1]
        idx = np.triu_indices(nc)
        R = np.zeros((nc, nc, len(poles)), dtype=np.complex128)
        D = np.zeros((nc, nc), dtype=np.float64)
        E = np.zeros((nc, nc), dtype=np.float64)
        for k, (row, col) in enumerate(zip(*idx, strict=True)):
            R[row, col] = np.asarray(C_flat[k])
            D[row, col] = float(D_flat[k])
            E[row, col] = float(E_flat[k])
            R[col, row] = R[row, col]
            D[col, row] = D[row, col]
            E[col, row] = E[row, col]
        taus.append(tau)
        residues.append(R / omega_scale)
        Ds.append(D)
        Es.append(E * omega_scale)

    return RationalSurface(
        poles=jnp.asarray(poles / omega_scale),
        residue_coeffs=_regress(features, np.stack(residues)),
        D_coeffs=_regress(features, np.stack(Ds)),
        E_coeffs=_regress(features, np.stack(Es)),
        tau_coeffs=_regress(features, np.stack(taus) * omega_scale),
        omega_scale=jnp.asarray(omega_scale),
        z0=jnp.asarray(z0, dtype=jnp.complex128),
    )


def evaluate_surface(model: RationalSurface, features: jax.Array, freqs: jax.Array) -> jax.Array:
    """Evaluate delay-embedded S parameters for every feature row."""
    features = jnp.asarray(features)
    freqs = jnp.asarray(freqs)
    s = 1j * 2.0 * jnp.pi * freqs / model.omega_scale
    omega = 2.0 * jnp.pi * freqs / model.omega_scale
    Y_all = evaluate_surface_y(model, features, freqs)
    tau = jnp.einsum("bp,pi->bi", features, model.tau_coeffs)

    def evaluate_corner(Y_corner: jax.Array, tk: jax.Array) -> jax.Array:
        def evaluate_frequency(Y: jax.Array, wk: jax.Array) -> jax.Array:
            eye = jnp.eye(Y.shape[-1], dtype=jnp.complex128)
            S_deembedded = jnp.linalg.solve(
                (eye + model.z0 * Y).T,
                (eye - jnp.conj(model.z0) * Y).T,
            ).T
            phase = jnp.exp(-0.5j * wk * tk)
            return phase[:, None] * S_deembedded * phase[None, :]

        return jax.vmap(evaluate_frequency)(Y_corner, omega)

    return jax.vmap(evaluate_corner)(Y_all, tau)


def evaluate_surface_y(model: RationalSurface, features: jax.Array, freqs: jax.Array) -> jax.Array:
    """Evaluate de-embedded Y parameters for every feature row."""
    features = jnp.asarray(features)
    freqs = jnp.asarray(freqs)
    s = 1j * 2.0 * jnp.pi * freqs / model.omega_scale
    residues = jnp.einsum("bp,pijk->bijk", features, model.residue_coeffs)
    D = jnp.einsum("bp,pij->bij", features, model.D_coeffs)
    E = jnp.einsum("bp,pij->bij", features, model.E_coeffs)

    def evaluate_corner(R: jax.Array, Dk: jax.Array, Ek: jax.Array) -> jax.Array:
        def evaluate_frequency(sk: jax.Array) -> jax.Array:
            return jnp.sum(R / (sk - model.poles)[None, None, :], axis=-1) + Dk + sk * Ek

        return jax.vmap(evaluate_frequency)(s)

    return jax.vmap(evaluate_corner)(residues, D, E)


def surface_passivity_margins(model: RationalSurface, features: jax.Array, freqs: jax.Array) -> jax.Array:
    """Return ``lambda_min(Re(Y))`` at every corner and frequency.

    Nonnegative values mean that the model is passive on the sampled grid.
    """
    Y = evaluate_surface_y(model, features, freqs)
    hermitian = 0.5 * (Y + jnp.swapaxes(jnp.conj(Y), -1, -2))
    return jnp.linalg.eigvalsh(hermitian)[..., 0]


def surface_asymptotic_passivity_margins(
    model: RationalSurface,
    features: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return the minimum Hermitian eigenvalues of D and E per corner."""
    features = jnp.asarray(features)
    D = jnp.einsum("bp,pij->bij", features, model.D_coeffs)
    E = jnp.einsum("bp,pij->bij", features, model.E_coeffs)
    D_hermitian = 0.5 * (D + jnp.swapaxes(jnp.conj(D), -1, -2))
    E_hermitian = 0.5 * (E + jnp.swapaxes(jnp.conj(E), -1, -2))
    return jnp.linalg.eigvalsh(D_hermitian)[..., 0], jnp.linalg.eigvalsh(E_hermitian)[..., 0]


def surface_passivity_loss(
    model: RationalSurface,
    features: jax.Array,
    freqs: jax.Array,
    *,
    minimum_conductance: float = 0.0,
    minimum_slope: float = 0.0,
) -> jax.Array:
    """Dimensionless squared penalty for sampled Y-passivity violations."""
    margins = surface_passivity_margins(model, features, freqs)
    D_margins, E_margins = surface_asymptotic_passivity_margins(model, features)
    scale = jnp.maximum(jnp.real(model.z0), 1.0)
    dynamic_violations = jax.nn.relu(minimum_conductance - margins) * scale
    D_violations = jax.nn.relu(minimum_conductance - D_margins) * scale
    E_violations = jax.nn.relu(minimum_slope - E_margins) * scale
    return jnp.mean(dynamic_violations**2) + jnp.mean(D_violations**2) + jnp.mean(E_violations**2)


def _project_passive(
    model: RationalSurface,
    features: jax.Array,
    freqs: jax.Array,
    minimum_conductance: jax.Array,
    minimum_slope: jax.Array,
) -> tuple[RationalSurface, PassivityShifts]:
    """Project onto sampled passivity with a common diagonal D shift."""
    eye = jnp.eye(model.D_coeffs.shape[-1], dtype=model.D_coeffs.dtype)
    _, E_margins = surface_asymptotic_passivity_margins(model, features)
    E_shift = jax.nn.relu(minimum_slope - jnp.min(E_margins))
    E_coeffs = model.E_coeffs.at[0].add(E_shift * eye)
    slope_projected = RationalSurface(
        model.poles,
        model.residue_coeffs,
        model.D_coeffs,
        E_coeffs,
        model.tau_coeffs,
        model.omega_scale,
        model.z0,
    )
    dynamic_margin = jnp.min(surface_passivity_margins(slope_projected, features, freqs))
    D_margins, _ = surface_asymptotic_passivity_margins(slope_projected, features)
    worst_conductance_margin = jnp.minimum(dynamic_margin, jnp.min(D_margins))
    D_shift = jax.nn.relu(minimum_conductance - worst_conductance_margin)
    D_coeffs = model.D_coeffs.at[0].add(D_shift * eye)
    projected = RationalSurface(
        model.poles,
        model.residue_coeffs,
        D_coeffs,
        E_coeffs,
        model.tau_coeffs,
        model.omega_scale,
        model.z0,
    )
    return projected, PassivityShifts(D_shift, E_shift)


def project_surface_passive(
    model: RationalSurface,
    features: jax.Array,
    freqs: jax.Array,
    *,
    minimum_conductance: float = 1e-12,
    minimum_slope: float = 1e-12,
) -> tuple[RationalSurface, PassivityShifts]:
    """Guarantee Y-passivity on a grid by minimally shifting its D intercept.

    The first feature must be an intercept equal to one. The returned metadata
    contains the common diagonal shifts applied to D and E.
    """
    features = jnp.asarray(features)
    if features.ndim != 2 or not np.allclose(np.asarray(features[:, 0]), 1.0):
        msg = "passivity projection requires features[:, 0] to be an intercept of ones"
        raise ValueError(msg)
    return _project_passive(
        model,
        features,
        jnp.asarray(freqs),
        jnp.asarray(minimum_conductance),
        jnp.asarray(minimum_slope),
    )


def surface_loss(
    model: RationalSurface,
    features: jax.Array,
    freqs: jax.Array,
    target_S: jax.Array,
) -> jax.Array:
    """Relative complex S-parameter MSE plus a nonnegative-delay penalty."""
    prediction = evaluate_surface(model, features, freqs)
    data_loss = jnp.mean(jnp.abs(prediction - target_S) ** 2) / jnp.maximum(jnp.mean(jnp.abs(target_S) ** 2), 1e-30)
    tau = features @ model.tau_coeffs
    return data_loss + 1e-4 * jnp.mean(jax.nn.relu(-tau) ** 2)


@partial(jax.jit, static_argnames=("steps", "learning_rate", "enforce_passive"))
def refine_surface(
    model: RationalSurface,
    features: jax.Array,
    freqs: jax.Array,
    target_S: jax.Array,
    *,
    steps: int = 200,
    learning_rate: float = 1e-4,
    enforce_passive: bool = False,
    minimum_conductance: float = 1e-12,
    minimum_slope: float = 1e-12,
    passivity_weight: float = 1.0,
    passivity_features: jax.Array | None = None,
    passivity_freqs: jax.Array | None = None,
) -> tuple[RationalSurface, jax.Array]:
    """Refine coefficients, optionally enforcing sampled Y-passivity.

    Passive refinement uses a differentiable eigenvalue penalty and projects
    back onto sampled Y-passivity after every optimizer step. The first feature
    is therefore expected to be a constant intercept of one. Separate, denser
    feature and frequency grids may be supplied for projected updates without
    requiring target data on those grids. The cheaper differentiable penalty
    is evaluated on the training grid.
    """
    features, freqs, target_S = map(jnp.asarray, (features, freqs, target_S))
    constraint_features = features if passivity_features is None else jnp.asarray(passivity_features)
    constraint_freqs = freqs if passivity_freqs is None else jnp.asarray(passivity_freqs)
    trainable = (model.residue_coeffs, model.D_coeffs, model.E_coeffs, model.tau_coeffs)
    first = jax.tree.map(jnp.zeros_like, trainable)
    second = jax.tree.map(jnp.zeros_like, trainable)

    def with_params(params: tuple[jax.Array, ...]) -> RationalSurface:
        return RationalSurface(model.poles, params[0], params[1], params[2], params[3], model.omega_scale, model.z0)

    def objective(params: tuple[jax.Array, ...]) -> jax.Array:
        candidate = with_params(params)
        loss = surface_loss(candidate, features, freqs, target_S)
        if enforce_passive:
            loss += passivity_weight * surface_passivity_loss(
                candidate,
                features,
                freqs,
                minimum_conductance=minimum_conductance,
                minimum_slope=minimum_slope,
            )
        return loss

    value_and_grad = jax.value_and_grad(objective)

    def step(carry: tuple, index: jax.Array) -> tuple[tuple, jax.Array]:
        params, m, v = carry
        loss, grads = value_and_grad(params)
        # JAX returns the conjugate-covector convention for a real loss with
        # complex parameters. Conjugating gives the steepest-descent direction.
        grads = jax.tree.map(jnp.conj, grads)
        m = jax.tree.map(lambda old, grad: 0.9 * old + 0.1 * grad, m, grads)
        v = jax.tree.map(lambda old, grad: 0.999 * old + 0.001 * jnp.abs(grad) ** 2, v, grads)
        correction1, correction2 = 1.0 - 0.9 ** (index + 1), 1.0 - 0.999 ** (index + 1)
        params = jax.tree.map(
            lambda param, mk, vk: param - learning_rate * (mk / correction1) / (jnp.sqrt(vk / correction2) + 1e-8),
            params,
            m,
            v,
        )
        if enforce_passive:
            projected, _ = _project_passive(
                with_params(params),
                constraint_features,
                constraint_freqs,
                jnp.asarray(minimum_conductance),
                jnp.asarray(minimum_slope),
            )
            params = (
                projected.residue_coeffs,
                projected.D_coeffs,
                projected.E_coeffs,
                projected.tau_coeffs,
            )
        return (params, m, v), loss

    (trainable, _, _), losses = jax.lax.scan(step, (trainable, first, second), jnp.arange(steps))
    fitted = with_params(trainable)
    if enforce_passive:
        fitted, _ = _project_passive(
            fitted,
            constraint_features,
            constraint_freqs,
            jnp.asarray(minimum_conductance),
            jnp.asarray(minimum_slope),
        )
    return fitted, losses
