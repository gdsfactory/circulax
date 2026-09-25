"""Utility functions: pole initialisation, data reshaping, weights, error."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# ---------------------------------------------------------------------------
# cindex — pole classification for real-representation LS systems
# ---------------------------------------------------------------------------


def compute_cindex(poles: np.ndarray) -> np.ndarray:
    """Classify each pole for the real-valued LS representation.

    Returns cindex array where:
        0 = real pole
        1 = first pole of a complex conjugate pair
        2 = second pole of a complex conjugate pair (conjugate of the previous)

    Equivalent to the cindex computation in vectfit4.m.
    """
    N = len(poles)
    cindex = np.zeros(N, dtype=int)
    for m in range(N):
        if poles[m].imag != 0.0:
            if m == 0:
                cindex[m] = 1
            elif cindex[m - 1] in (0, 2):
                cindex[m] = 1
                if m + 1 < N:
                    cindex[m + 1] = 2
            else:
                cindex[m] = 2
    return cindex


# ---------------------------------------------------------------------------
# Basis functions
# ---------------------------------------------------------------------------


def build_dk(s: jnp.ndarray, poles: jnp.ndarray, cindex: np.ndarray, offs: int) -> jnp.ndarray:
    """Build the Dk basis matrix (Ns, N+offs) using the real representation.

    For real pole (cindex[m]==0):
        Dk[:, m] = 1/(s - poles[m])
    For first of complex pair (cindex[m]==1):
        Dk[:, m]   = 1/(s-a) + 1/(s-a*)   (= 2*Re(1/(s-a)))
        Dk[:, m+1] = i/(s-a) - i/(s-a*)   (= -2*Im(1/(s-a))*i... see below)
    For second of pair (cindex[m]==2):
        (column already set by m-1 step above)

    Constant and s terms (for asymp):
        Dk[:, N]   = 1       (if offs >= 1)
        Dk[:, N+1] = s       (if offs >= 2)

    Complex pairs: poles[m+1] = conj(poles[m]) so dk_conj[:,m] = dk_direct[:,m+1].
    """
    N = len(poles)
    Ns = s.shape[0]
    cindex = np.asarray(cindex)

    dk_direct = 1.0 / (s[:, None] - poles[None, :])  # (Ns, N)
    dk_conj = 1.0 / (s[:, None] - jnp.conj(poles)[None, :])  # (Ns, N)

    is_real = jnp.array(cindex == 0, dtype=bool)  # (N,)
    is_first = jnp.array(cindex == 1, dtype=bool)
    is_second = jnp.array(cindex == 2, dtype=bool)

    # Real case: dk_direct
    # First of pair: sum of conjugate fractions
    # Second of pair: imaginary-difference representation
    Dk_poles = jnp.where(
        is_real[None, :],
        dk_direct,
        jnp.where(
            is_first[None, :],
            dk_direct + dk_conj,
            1j * (dk_conj - dk_direct),  # for cindex==2: 1j*(1/(s-a) - 1/(s-a*))
        ),
    )  # (Ns, N)

    extra_cols = []
    if offs >= 1:
        extra_cols.append(jnp.ones((Ns, 1), dtype=jnp.complex128))
    if offs >= 2:
        extra_cols.append(s[:, None].astype(jnp.complex128))

    if extra_cols:
        Dk = jnp.concatenate([Dk_poles] + extra_cols, axis=1)
    else:
        Dk = Dk_poles

    return Dk  # (Ns, N+offs)


# ---------------------------------------------------------------------------
# Pole initialisation
# ---------------------------------------------------------------------------


def _make_complex_pairs(omega: np.ndarray, nu: float) -> np.ndarray:
    """Return complex pairs [-nu*w ± jw] for each frequency w in omega."""
    pairs = np.empty(2 * len(omega), dtype=np.complex128)
    for k, w in enumerate(omega):
        pairs[2 * k] = -nu * w - 1j * w
        pairs[2 * k + 1] = -nu * w + 1j * w
    return pairs


def init_poles_lincmplx(s: jnp.ndarray, N: int, nu: float = 1e-3) -> np.ndarray:
    """N/2 complex conjugate pairs, linearly spaced over the frequency range."""
    omega = np.imag(np.asarray(s))
    omega_min = max(abs(omega.min()), 1e-3)
    omega_max = abs(omega.max())
    omegas = np.linspace(omega_min, omega_max, N // 2)
    return _make_complex_pairs(omegas, nu)


def init_poles_logcmplx(s: jnp.ndarray, N: int, nu: float = 1e-3) -> np.ndarray:
    """N/2 complex conjugate pairs, logarithmically spaced over the frequency range."""
    omega = np.imag(np.asarray(s))
    omega_min = max(abs(omega.min()), 1e-3)
    omega_max = abs(omega.max())
    omegas = np.logspace(np.log10(omega_min), np.log10(omega_max), N // 2)
    return _make_complex_pairs(omegas, nu)


def init_poles_linlogcmplx(s: jnp.ndarray, N: int, nu: float = 1e-3) -> np.ndarray:
    """N/2 complex pairs: first N/4 linearly spaced, next N/4 logarithmically spaced.

    Matches the default pole generation in VFdriver.m.
    """
    omega = np.imag(np.asarray(s))
    omega_min = max(abs(omega.min()), 1e-3)
    omega_max = abs(omega.max())
    n1 = N // 4
    n2 = N // 2 - n1
    omegas_lin = np.linspace(omega_min, omega_max, n1)
    omegas_log = np.logspace(np.log10(omega_min), np.log10(omega_max), n2)
    omegas = np.sort(np.concatenate([omegas_lin, omegas_log]))
    return _make_complex_pairs(omegas, nu)


# ---------------------------------------------------------------------------
# Pole sorting (real-to-complex conversion and sort order)
# ---------------------------------------------------------------------------


def sort_poles(poles: np.ndarray) -> np.ndarray:
    """Sort poles: real poles first, then complex pairs sorted by magnitude.

    Within each conjugate pair the negative-imaginary pole is first (matching
    MATLAB's sort-by-magnitude + sort-by-phase-angle behaviour). This ordering
    is required by compute_cindex which assumes pairs appear as (a-ib, a+ib).

    Also applies conjugation cleanup: poles with negligible imaginary parts
    become exactly real (matching vectfit4 line 530: roetter - 2i*imag(roetter)).
    """
    poles = np.asarray(poles)
    # Zero out tiny imaginary parts (relative tolerance)
    tol = 1e-14 * np.abs(poles)
    nearly_real = np.abs(poles.imag) < tol
    poles = np.where(nearly_real, poles.real + 0j, poles)

    real_poles = poles[poles.imag == 0.0]
    cmplx_poles = poles[poles.imag != 0.0]

    # Sort by (magnitude, imaginary part) — keeps conjugate pairs adjacent
    # with negative-imaginary member first, exactly like MATLAB sort on complex.
    if len(cmplx_poles) > 0:
        mag = np.abs(cmplx_poles)
        idx = np.lexsort((cmplx_poles.imag, mag))
        cmplx_sorted = cmplx_poles[idx]
    else:
        cmplx_sorted = cmplx_poles

    return np.concatenate([real_poles, cmplx_sorted])


# ---------------------------------------------------------------------------
# Frequency-domain data reshaping (upper triangle ↔ matrix)
# ---------------------------------------------------------------------------


def _upper_triangle_indices(Nc: int) -> list[tuple[int, int]]:
    """Return (row, col) indices of the upper triangle in column-major order.

    Matches the stacking in VFdriver.m:
        for col=1:Nc; for row=col:Nc → (row, col)
    """
    return [(row, col) for col in range(Nc) for row in range(col, Nc)]


def stack_upper_triangle(H: jnp.ndarray) -> jnp.ndarray:
    """Stack upper-triangle elements of H into a (nnn, Ns) matrix.

    Args:
        H: (Nc, Nc, Ns) complex — frequency-domain matrix data.

    Returns:
        f: (nnn, Ns) complex  where nnn = Nc*(Nc+1)//2.

    """
    Nc = H.shape[0]
    idx = _upper_triangle_indices(Nc)
    return jnp.stack([H[r, c, :] for r, c in idx], axis=0)


def unstack_upper_triangle(f: jnp.ndarray, Nc: int) -> jnp.ndarray:
    """Reconstruct symmetric H from upper-triangle elements.

    Args:
        f: (nnn, Ns) complex.
        Nc: number of ports.

    Returns:
        H: (Nc, Nc, Ns) complex — symmetric.

    """
    Ns = f.shape[1]
    idx = _upper_triangle_indices(Nc)
    H = jnp.zeros((Nc, Nc, Ns), dtype=jnp.complex128)
    for k, (r, c) in enumerate(idx):
        H = H.at[r, c, :].set(f[k])
        if r != c:
            H = H.at[c, r, :].set(f[k])
    return H


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------


def compute_weights(H: jnp.ndarray, weightparam: int, *, reciprocal: bool = True) -> jnp.ndarray:
    """Compute fitting weights for the selected matrix elements.

    Args:
        H:           (Nc, Nc, Ns) complex — full matrix data.
        weightparam: 1=uniform, 2=1/|H_ij|, 3=1/√|H_ij|, 4=1/‖H‖F, 5=1/√‖H‖F.
        reciprocal: Use only one triangle when true, or every ordered matrix
            element when false.

    Returns:
        weight: (nresponses, Ns) or (1, Ns) real — per-element weights.

    """
    Nc, _, Ns = H.shape
    idx = (
        _upper_triangle_indices(Nc)
        if reciprocal
        else [(row, col) for row in range(Nc) for col in range(Nc)]
    )

    if weightparam == 1:
        return jnp.ones((1, Ns))

    if weightparam in (2, 3):
        w = jnp.stack([jnp.abs(H[r, c, :]) for r, c in idx], axis=0)  # (nnn, Ns)
        w = jnp.where(w == 0, 1.0, w)  # avoid division by zero
        if weightparam == 2:
            return 1.0 / w
        return 1.0 / jnp.sqrt(w)

    if weightparam in (4, 5):
        # Frobenius norm at each frequency — same weight for all elements
        H_flat = H.reshape(Nc * Nc, Ns)
        norms = jnp.linalg.norm(H_flat, axis=0)  # (Ns,)
        norms = jnp.where(norms == 0, 1.0, norms)
        if weightparam == 4:
            return (1.0 / norms)[None, :]
        return (1.0 / jnp.sqrt(norms))[None, :]

    raise ValueError(f"Unknown weightparam: {weightparam}")


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------


def compute_rmserr(H: jnp.ndarray, Hfit: jnp.ndarray) -> float:
    """Root-mean-square error: sqrt(sum|H - Hfit|² / (Nc² * Ns))."""
    diff = H - Hfit
    return float(jnp.sqrt(jnp.sum(jnp.abs(diff) ** 2) / diff.size))
