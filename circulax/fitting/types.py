"""Core data structures for Circulax rational fitting."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class VFModel:
    """Pole-residue model.

    Represents: H(s) = sum_m residues[:, :, m] / (s - poles[m]) + D + s*E
    """

    poles: jnp.ndarray  # (N,) complex128 — shared poles, all in left half-plane
    residues: jnp.ndarray  # (Nc, Nc, N) complex128 — residue matrices per pole
    D: jnp.ndarray  # (Nc, Nc) float64 — constant term (zero if asymp==1)
    E: jnp.ndarray  # (Nc, Nc) float64 — linear term (zero if asymp<3)


@dataclass
class SSModel:
    """State-space model with diagonal A.

    Represents: H(s) = C * diag(1/(s - A)) * B + D + s*E
    A is stored as a 1-D vector of diagonal entries to avoid materialising the
    large sparse block-diagonal matrix.
    """

    A: jnp.ndarray  # (Nc*N,) complex128 — diagonal entries (poles repeated Nc times)
    B: jnp.ndarray  # (Nc*N, Nc) complex128 — block identity structure
    C: jnp.ndarray  # (Nc, Nc*N) complex128 — residues arranged by port
    D: jnp.ndarray  # (Nc, Nc) complex128
    E: jnp.ndarray  # (Nc, Nc) complex128


@dataclass(frozen=True)
class FitOptions:
    """Fitting hyperparameters.

    Frozen so the instance can be used as a JIT static argument (it is hashable).
    """

    N: int  # pole order
    asymp: int = 2  # 1: D=E=0, 2: D≠0 E=0, 3: D≠0 E≠0
    stable: bool = True  # flip unstable poles to left half-plane
    relax: bool = True  # use relaxed non-triviality constraint
    weightparam: int = 1  # 1=uniform, 2=1/|H|, 3=1/√|H|, 4=1/‖H‖F, 5=1/√‖H‖F
    Niter1: int = 4  # iterations fitting only diagonal elements (phase 1)
    Niter2: int = 4  # iterations fitting full upper triangle (phase 2)
    parametertype: str = "Y"  # "Y" (admittance) or "S" (scattering)
    TOLG: float = 1e-6  # passivity eigenvalue tolerance
    TOLD: float = 1e-3  # D-term passivity threshold
    TOLE: float = 1e-12  # E-term passivity threshold
    Niter_out: int = 20  # outer passivity enforcement iterations
    passive_DE: bool = False  # project D and E to PSD after fitting
    nu: float = 1e-3  # relative damping for initial pole generation


# Register VFModel and SSModel as JAX pytrees so they pass through jit/vmap.
jax.tree_util.register_dataclass(
    VFModel,
    data_fields=["poles", "residues", "D", "E"],
    meta_fields=[],
)
jax.tree_util.register_dataclass(
    SSModel,
    data_fields=["A", "B", "C", "D", "E"],
    meta_fields=[],
)


# ---------------------------------------------------------------------------
# Conversion between representations
# ---------------------------------------------------------------------------


def vfmodel_to_ss(model: VFModel, Nc: int) -> SSModel:
    """Convert VFModel (pole-residue) to SSModel (state-space).

    Equivalent to pr2ss.m.

    For Nc ports and N poles, the block-diagonal structure is:
        A (as vector): [poles, poles, ..., poles]  (repeated Nc times) → (Nc*N,)
        B[n*N:(n+1)*N, n] = 1, rest 0             (Nc*N, Nc)
        C[row, col*N:(col+1)*N] = residues[row, col, :]  (Nc, Nc*N)
    """
    N = model.poles.shape[0]
    A_diag = jnp.tile(model.poles, Nc)  # (Nc*N,)

    # B: kron(eye(Nc), ones(N,1)) gives the block identity structure
    B = jnp.kron(
        jnp.eye(Nc, dtype=jnp.complex128),
        jnp.ones((N, 1), dtype=jnp.complex128),
    )  # (Nc*N, Nc)

    # C: residues (Nc, Nc, N) → reshape so C[row, col*N+k] = residues[row, col, k]
    C = model.residues.astype(jnp.complex128).reshape(Nc, Nc * N)

    return SSModel(
        A=A_diag,
        B=B,
        C=C,
        D=model.D.astype(jnp.complex128),
        E=model.E.astype(jnp.complex128),
    )


def ss_to_vfmodel(ss: SSModel, N: int, Nc: int) -> VFModel:
    """Convert SSModel back to VFModel.

    Inverse of vfmodel_to_ss. Equivalent to ss2pr.
    """
    poles = ss.A[:N]
    residues = ss.C.reshape(Nc, Nc, N)
    return VFModel(
        poles=poles,
        residues=residues,
        D=jnp.real(ss.D),
        E=jnp.real(ss.E),
    )


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def _eval_at_one_freq(
    sk: complex, A_diag: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray, D: jnp.ndarray, E: jnp.ndarray
) -> jnp.ndarray:
    """Evaluate H(sk) = C * diag(1/(sk - A)) * B + D + sk*E.

    Returns: (Nc, Nc) complex.
    """
    resolvent = 1.0 / (sk - A_diag)  # (Nc*N,)
    return C @ (resolvent[:, None] * B) + D + sk * E  # (Nc, Nc)


def eval_model(s: jnp.ndarray, ss: SSModel) -> jnp.ndarray:
    """Evaluate model at all frequency points.

    Args:
        s: (Ns,) complex frequency points (= j*omega).
        ss: SSModel.

    Returns:
        H: (Ns, Nc, Nc) complex — model response at each frequency.

    """
    return jax.vmap(lambda sk: _eval_at_one_freq(sk, ss.A, ss.B, ss.C, ss.D, ss.E))(s)
