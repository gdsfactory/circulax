"""S-parameter preprocessing with delay de-embedding for vector fitting.

Provides a pipeline that extracts per-port group delay from S-parameter data,
de-embeds it via a reference-plane shift, fits the smooth remainder with AAA,
and enforces passivity — yielding a low-order rational model.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from .aaa import _aaa_poles, _collect_poles, aaa_scalar
from .pole_sweep import PoleCountSweep, prune_poles_by_contribution, vmap_pole_count_sweep
from .residue_id import identify_residues
from .types import FitOptions, SSModel, VFModel, eval_model, vfmodel_to_ss
from .utils import (
    _upper_triangle_indices,
    compute_rmserr,
    compute_weights,
    stack_upper_triangle,
)


class CausalityWarning(RuntimeWarning):
    """A fitted delay model has evidence of a non-causal intermediate fit."""


class CausalityError(ValueError):
    """Strict causality policy rejected a delay-model fit."""


def s_to_y(S: np.ndarray, z0: complex = 50.0) -> np.ndarray:
    """Convert S-parameter matrix to Y-parameter matrix (Kurokawa form).

    Y = (I - S) @ inv(z0*S + conj(z0)*I)

    Args:
        S: (..., N, N) complex S-parameter matrix.
        z0: Reference impedance. A small imaginary part regularises near-lossless
            components; pass ``z0 + 1e-12j`` externally if needed.

    Returns:
        Y: (..., N, N) complex admittance matrix.

    """
    n = S.shape[-1]
    I = np.eye(n, dtype=np.complex128)
    return (I - S) @ np.linalg.inv(z0 * S + np.conj(z0) * I)


def extract_group_delay(
    S: np.ndarray,
    freqs: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Extract per-port group delay from S-parameter transmission phase.

    Uses least-squares slope of the unwrapped S21 phase vs angular frequency.
    For a 2-port, uses S21; for N-port, uses each off-diagonal S[0,i] (i>0).

    Args:
        S: (Ns, Nc, Nc) complex S-parameter data.
        freqs: (Ns,) real frequencies in Hz.
        scale: Scale factor applied to the fitted delay (default 1.0).
            Use < 1.0 (e.g. 0.95) to deliberately under-estimate and avoid
            de-embedding more delay than physically present.

    Returns:
        tau: (Nc,) per-port group delay in seconds.

    Raises:
        ValueError: If adjacent phase differences exceed pi (undersampled data).

    """
    tau_per_port, _ = _extract_group_delay_diagnostics(S, freqs, scale=scale)
    return tau_per_port


def _extract_group_delay_diagnostics(
    S: np.ndarray,
    freqs: np.ndarray,
    scale: float = 1.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Extract delay and retain diagnostics needed for causality feedback."""
    _, Nc, _ = S.shape
    omega = 2.0 * np.pi * freqs

    tau_per_port = np.zeros(Nc, dtype=np.float64)
    raw_tau_per_port = np.zeros(Nc, dtype=np.float64)
    phase_slope_rmse = np.zeros(Nc, dtype=np.float64)

    for i in range(Nc):
        j = (i + 1) % Nc if Nc > 1 else i
        if i == j:
            continue
        sij = S[:, i, j]
        phase = np.unwrap(np.angle(sij))

        d_phase = np.diff(phase)
        if np.any(np.abs(d_phase) > np.pi):
            raise ValueError(
                f"Phase jump > pi between adjacent frequency samples in S[{i},{j}]. "
                f"Data is undersampled for the group delay present — increase "
                f"frequency resolution."
            )

        coeffs = np.polynomial.polynomial.polyfit(omega, phase, 1)
        tau_i = -coeffs[1] * scale
        raw_tau_per_port[i] = tau_i
        tau_per_port[i] = max(tau_i, 0.0)
        fitted_phase = np.polynomial.polynomial.polyval(omega, coeffs)
        phase_slope_rmse[i] = float(np.sqrt(np.mean((phase - fitted_phase) ** 2)))

    if Nc == 2:
        avg = 0.5 * (tau_per_port[0] + tau_per_port[1])
        tau_per_port[:] = avg

    return tau_per_port, {
        "raw_tau": raw_tau_per_port,
        "phase_slope_rmse": phase_slope_rmse,
        "negative_raw_delay_ports": np.flatnonzero(raw_tau_per_port < 0),
    }


def deembed_delay(
    S: np.ndarray,
    freqs: np.ndarray,
    tau_per_port: np.ndarray,
) -> np.ndarray:
    """De-embed group delay via reference-plane shift S' = P @ S @ P.

    P = diag(exp(+j * omega * tau_i / 2)) advances each port's reference
    plane by half the total delay, preserving S-matrix structure (reciprocity,
    passivity, unitarity for lossless).

    Args:
        S: (Ns, Nc, Nc) complex S-parameter data.
        freqs: (Ns,) real frequencies in Hz.
        tau_per_port: (Nc,) per-port group delay in seconds.

    Returns:
        S_deembedded: (Ns, Nc, Nc) complex.

    """
    Ns, Nc, _ = S.shape
    omega = 2.0 * np.pi * freqs  # (Ns,)

    S_out = np.empty_like(S)
    for k in range(Ns):
        P = np.diag(np.exp(1j * omega[k] * tau_per_port / 2.0))
        S_out[k] = P @ S[k] @ P

    return S_out


def embed_delay(
    S_deembedded: np.ndarray,
    freqs: np.ndarray,
    tau_per_port: np.ndarray,
) -> np.ndarray:
    """Re-embed group delay (inverse of deembed_delay).

    S = P_inv @ S' @ P_inv  where P_inv = diag(exp(-j * omega * tau_i / 2)).
    """
    Ns, Nc, _ = S_deembedded.shape
    omega = 2.0 * np.pi * freqs

    S_out = np.empty_like(S_deembedded)
    for k in range(Ns):
        P_inv = np.diag(np.exp(-1j * omega[k] * tau_per_port / 2.0))
        S_out[k] = P_inv @ S_deembedded[k] @ P_inv

    return S_out


def _aaa_all_elements(
    bigH: jnp.ndarray,
    s: jnp.ndarray,
    opts: FitOptions,
    tol: float = 1e-10,
    mmax: int = 100,
    dedup_rtol: float = 1e-2,
    reciprocal: bool = True,
    pole_selection: Literal["most_complex", "largest_response"] = "most_complex",
    causality: Literal["warn", "error", "ignore"] = "warn",
    verbose: bool = True,
    aaa_backend: Literal["numpy", "jax"] = "numpy",
) -> tuple[VFModel, SSModel, float, jnp.ndarray, int, float]:
    """AAA fitting that runs on the selected matrix elements.

    Runs AAA on each selected element to find candidate poles, then
    selects the best element's poles (most support points) and filters out
    spurious poles far outside the data bandwidth. Uses a single element's
    poles to avoid near-duplicate clusters from independent AAA runs, which
    cause ill-conditioned residue identification.

    Returns ``(model, ss, rmserr, bigHfit, n_rhp, max_raw_pole_real)``.
    The final two values describe RHP poles before `_collect_poles` reflects
    them into the LHP.
    """
    Nc = bigH.shape[0]
    Ns = s.shape[0]
    bigH_np = np.asarray(bigH, dtype=np.complex128)
    s_np = np.asarray(s, dtype=np.complex128)

    idx = (
        _upper_triangle_indices(Nc)
        if reciprocal
        else [(row, col) for row in range(Nc) for col in range(Nc)]
    )
    elem_poles = []
    elem_nsupport = []

    for r, c in idx:
        f_elem = bigH_np[r, c, :]
        w, zj, fj = aaa_scalar(f_elem, s_np, tol=tol, mmax=mmax, backend=aaa_backend)
        if len(zj) > 1:
            pols = _aaa_poles(w, zj)
        else:
            pols = np.array([], dtype=np.complex128)
        elem_poles.append(pols)
        elem_nsupport.append(len(zj))
        if verbose:
            print(f"  AAA [{r},{c}]: {len(zj)} support pts, {len(pols)} raw poles")

    if pole_selection == "most_complex":
        best_idx = int(np.argmax(elem_nsupport))
    elif pole_selection == "largest_response":
        response_norms = [np.linalg.norm(bigH_np[row, col, :]) for row, col in idx]
        best_idx = int(np.argmax(response_norms))
    else:
        msg = f"unknown pole_selection {pole_selection!r}"
        raise ValueError(msg)
    raw_poles = elem_poles[best_idx]

    if len(raw_poles) == 0:
        for pols in elem_poles:
            if len(pols) > 0:
                raw_poles = pols
                break

    if len(raw_poles) == 0:
        raise RuntimeError("AAA found no poles across the selected matrix responses.")

    n_rhp = int(np.sum(raw_poles.real > 0))
    max_raw_pole_real = float(np.max(raw_poles.real))
    if n_rhp and causality == "error":
        raise CausalityError(
            f"AAA found {n_rhp} right-half-plane pole(s) (maximum real part "
            f"{max_raw_pole_real:.3e} rad/s) before stabilization. "
            "The extracted delay may be over-estimated; reduce delay_scale or use causality='warn'."
        )
    poles_np = _collect_poles([raw_poles])

    s_max = float(np.max(np.abs(s_np)))
    bw_mask = np.abs(poles_np) < 5.0 * s_max
    if np.any(bw_mask):
        poles_np = poles_np[bw_mask]

    if len(poles_np) > 1:
        keep = np.ones(len(poles_np), dtype=bool)
        for i in range(len(poles_np)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(poles_np)):
                if not keep[j]:
                    continue
                if abs(poles_np[j] - poles_np[i]) / max(abs(poles_np[i]), 1.0) < dedup_rtol:
                    keep[j] = False
        poles_np = poles_np[keep]

    if verbose:
        print(f"  AAA total: {len(poles_np)} poles (from element {best_idx}, bandwidth-filtered)")

    if len(poles_np) == 0:
        raise RuntimeError("All AAA poles were outside the data bandwidth.")

    f_full = (
        stack_upper_triangle(bigH)
        if reciprocal
        else jnp.stack([bigH[row, col, :] for row, col in idx], axis=0)
    )
    w_full = compute_weights(bigH, opts.weightparam, reciprocal=reciprocal)
    C_flat, D_vec, E_vec = identify_residues(f_full, s, poles_np, w_full, opts)

    N = len(poles_np)
    residues = jnp.zeros((Nc, Nc, N), dtype=jnp.complex128)
    D_mat = jnp.zeros((Nc, Nc), dtype=jnp.float64)
    E_mat = jnp.zeros((Nc, Nc), dtype=jnp.float64)

    for k, (r, c) in enumerate(idx):
        residues = residues.at[r, c, :].set(C_flat[k])
        D_mat = D_mat.at[r, c].set(float(D_vec[k]))
        E_mat = E_mat.at[r, c].set(float(E_vec[k]))
        if reciprocal and r != c:
            residues = residues.at[c, r, :].set(C_flat[k])
            D_mat = D_mat.at[c, r].set(float(D_vec[k]))
            E_mat = E_mat.at[c, r].set(float(E_vec[k]))

    model = VFModel(poles=jnp.array(poles_np), residues=residues, D=D_mat, E=E_mat)
    ss = vfmodel_to_ss(model, Nc)
    bigHfit_stacked = eval_model(s, ss)
    bigHfit = jnp.moveaxis(bigHfit_stacked, 0, -1)
    H_data = jnp.moveaxis(bigH, -1, 0)
    rmserr = compute_rmserr(H_data, bigHfit_stacked)

    return model, ss, rmserr, bigHfit, n_rhp, max_raw_pole_real


def fit_with_delay(
    S: np.ndarray,
    freqs: np.ndarray,
    z0: complex = 50.0,
    opts: FitOptions | None = None,
    delay_scale: float = 1.0,
    tol: float = 1e-10,
    mmax: int = 100,
    enforce_passive: bool = True,
    reciprocal: bool = True,
    delay_mode: Literal["auto", "port", "none"] = "auto",
    causality: Literal["warn", "error", "ignore"] = "warn",
    fit_domain: Literal["y", "s"] = "y",
    s_refinement_iterations: int = 6,
    max_poles: int | None = None,
    pole_count_candidates: tuple[int, ...] | None = None,
    verbose: bool = True,
    aaa_backend: Literal["numpy", "jax"] = "numpy",
) -> tuple[SSModel, np.ndarray, dict]:
    """De-embed delay and fit a simulation-ready admittance realization.

    Args:
        S: (Ns, Nc, Nc) complex S-parameter data.
        freqs: (Ns,) real frequencies in Hz.
        z0: Reference impedance for S-to-Y conversion.
        opts: FitOptions (N is ignored for AAA; asymp=2 default).
        delay_scale: Scale factor for delay extraction (< 1.0 to under-estimate).
        tol: AAA convergence tolerance.
        mmax: Maximum AAA support points per selected matrix response.
        enforce_passive: Whether to enforce passivity after fitting.
        reciprocal: Whether the network is reciprocal. When false, fit every
            ordered matrix response independently instead of mirroring one
            triangle.
        delay_mode: ``"port"`` extracts a shared reference-plane delay,
            ``"none"`` disables delay extraction, and ``"auto"`` uses port
            delay only for reciprocal networks. A shared port delay is usually
            inappropriate for directional active devices.
        causality: How to handle evidence of a non-causal intermediate fit.
            ``"warn"`` (default) emits :class:`CausalityWarning`, ``"error"``
            rejects negative raw delay estimates and RHP AAA poles before they
            are stabilized, and ``"ignore"`` records diagnostics only.
        fit_domain: Fit admittance directly with ``"y"`` or discover poles
            with AAA and refine them against all complex scattering responses
            with ``"s"``. The S-domain realization is transformed exactly to
            an admittance realization for circuit simulation.
        s_refinement_iterations: Common-pole vector-fitting iterations after
            AAA initialization when ``fit_domain="s"``.
        max_poles: Optional pole budget. AAA pole pairs are ranked by their
            aggregate contribution over all measured responses before fitting
            their residues again. In the S domain, the retained poles are also
            refined by common-pole vector fitting.
        pole_count_candidates: Optional candidate budgets to refit and screen
            concurrently with a fixed-shape ``vmap`` before pole relocation.
            Results are returned as ``metadata["pole_sweep"]``.
        verbose: Print progress.
        aaa_backend: NumPy or JAX for AAA support discovery only. Pole
            extraction uses SciPy; residue fitting and VF still use JAX.

    Returns:
        ss: SSModel — fitted state-space model of the de-embedded response.
        tau_per_port: (Nc,) per-port group delay in seconds.
        metadata: dict with keys:
            - pole_count: int
            - rmserr_Y: float — RMS error in Y-parameters
            - rmserr_S: float — RMS error in S-parameters (after re-embedding)
            - passivity_margin: float — min eigenvalue of Re(Y) (None if not enforced)
            - pole_flips: int — number of RHP poles flipped by _collect_poles
            - reciprocal: bool — whether response matrices were mirrored
            - delay_mode: str — resolved delay strategy
            - causality: raw delay, phase-fit, and pre-stabilization pole diagnostics

    """
    if aaa_backend not in {"numpy", "jax"}:
        raise ValueError("AAA backend must be 'numpy' or 'jax'")
    S = np.asarray(S, dtype=np.complex128)
    freqs = np.asarray(freqs, dtype=np.float64)
    Ns, Nc, _ = S.shape

    if opts is None:
        weightparam = 1 if fit_domain == "s" else 2
        opts = FitOptions(N=0, asymp=2, weightparam=weightparam)

    if fit_domain not in {"y", "s"}:
        msg = f"unknown fit_domain {fit_domain!r}; expected 'y' or 's'"
        raise ValueError(msg)
    if s_refinement_iterations < 1:
        msg = "s_refinement_iterations must be at least one"
        raise ValueError(msg)
    if max_poles is not None and max_poles < 1:
        msg = "max_poles must be positive"
        raise ValueError(msg)
    if pole_count_candidates is not None and fit_domain != "s":
        msg = "pole_count_candidates is currently available only with fit_domain='s'"
        raise ValueError(msg)

    if delay_mode not in {"auto", "port", "none"}:
        msg = f"unknown delay_mode {delay_mode!r}; expected 'auto', 'port', or 'none'"
        raise ValueError(msg)
    if causality not in {"warn", "error", "ignore"}:
        msg = f"unknown causality policy {causality!r}; expected 'warn', 'error', or 'ignore'"
        raise ValueError(msg)
    resolved_delay_mode = "port" if delay_mode == "auto" and reciprocal else delay_mode
    if resolved_delay_mode == "auto":
        resolved_delay_mode = "none"
    if enforce_passive and not reciprocal:
        msg = "passivity enforcement currently requires reciprocal=True"
        raise ValueError(msg)
    if fit_domain == "s" and enforce_passive:
        msg = "S-domain passivity enforcement is not implemented; use enforce_passive=False"
        raise ValueError(msg)

    if resolved_delay_mode == "port":
        tau_per_port, delay_diagnostics = _extract_group_delay_diagnostics(S, freqs, scale=delay_scale)
    else:
        tau_per_port = np.zeros(Nc, dtype=np.float64)
        delay_diagnostics = {
            "raw_tau": tau_per_port.copy(),
            "phase_slope_rmse": np.zeros(Nc, dtype=np.float64),
            "negative_raw_delay_ports": np.array([], dtype=np.intp),
        }

    negative_delay_ports = delay_diagnostics["negative_raw_delay_ports"]
    if len(negative_delay_ports):
        message = (
            "Group-delay extraction produced negative raw delay(s) for port(s) "
            f"{negative_delay_ports.tolist()}; they were clamped to zero. "
            "The measured response may not admit a causal fixed-delay decomposition."
        )
        if causality == "error":
            raise CausalityError(message)
        if causality == "warn":
            warnings.warn(message, CausalityWarning, stacklevel=2)

    if verbose:
        for i, t in enumerate(tau_per_port):
            print(f"  Port {i} group delay: {t * 1e12:.3f} ps")

    S_deemb = deembed_delay(S, freqs, tau_per_port)

    z0_conv = complex(z0)
    s = jnp.array(1j * 2.0 * np.pi * freqs)

    if fit_domain == "y":
        Y_deemb = np.stack([s_to_y(S_deemb[k], z0_conv) for k in range(Ns)])
        bigH = jnp.array(np.moveaxis(Y_deemb, 0, -1))
        model, ss, rmserr_Y, _, pole_flips, max_raw_pole_real = _aaa_all_elements(
            bigH,
            s,
            opts,
            aaa_backend=aaa_backend,
            tol=tol,
            mmax=mmax,
            reciprocal=reciprocal,
            causality=causality,
            verbose=verbose,
        )
        if max_poles is not None and len(model.poles) > max_poles:
            reduced_poles = prune_poles_by_contribution(model, s, max_poles)
            f_full = stack_upper_triangle(bigH)
            weights = compute_weights(bigH, opts.weightparam, reciprocal=reciprocal)
            C_flat, D_vec, E_vec = identify_residues(f_full, s, reduced_poles, weights, opts)
            N = len(reduced_poles)
            residues = jnp.zeros((Nc, Nc, N), dtype=jnp.complex128)
            D_mat = jnp.zeros((Nc, Nc), dtype=jnp.float64)
            E_mat = jnp.zeros((Nc, Nc), dtype=jnp.float64)
            idx = (
                _upper_triangle_indices(Nc)
                if reciprocal
                else [(row, col) for row in range(Nc) for col in range(Nc)]
            )
            for k, (row, col) in enumerate(idx):
                residues = residues.at[row, col, :].set(C_flat[k])
                D_mat = D_mat.at[row, col].set(float(D_vec[k]))
                E_mat = E_mat.at[row, col].set(float(E_vec[k]))
                if reciprocal and row != col:
                    residues = residues.at[col, row, :].set(C_flat[k])
                    D_mat = D_mat.at[col, row].set(float(D_vec[k]))
                    E_mat = E_mat.at[col, row].set(float(E_vec[k]))
            model = VFModel(poles=jnp.array(reduced_poles), residues=residues, D=D_mat, E=E_mat)
            ss = vfmodel_to_ss(model, Nc)
            rmserr_Y = compute_rmserr(jnp.moveaxis(bigH, -1, 0), eval_model(s, ss))
        S_fit_deemb = np.stack([y_to_s(np.asarray(value), z0_conv) for value in eval_model(s, ss)])
        direct_S_order = None
        transform_condition = None
        pole_sweep: PoleCountSweep | None = None
    else:
        from .driver import vfdriver

        bigS = jnp.array(np.moveaxis(S_deemb, 0, -1))
        aaa_model, _, _, _, pole_flips, max_raw_pole_real = _aaa_all_elements(
            bigS,
            s,
            opts,
            aaa_backend=aaa_backend,
            tol=tol,
            mmax=mmax,
            reciprocal=reciprocal,
            pole_selection="largest_response",
            causality=causality,
            verbose=verbose,
        )
        aaa_pole_count = len(aaa_model.poles)
        pole_sweep = (
            None
            if pole_count_candidates is None
            else vmap_pole_count_sweep(
                aaa_model,
                freqs,
                S_deemb,
                pole_count_candidates,
                asymp=opts.asymp,
                z0=float(np.real(z0_conv)),
            )
        )
        initial_poles = prune_poles_by_contribution(
            aaa_model,
            s,
            aaa_pole_count if max_poles is None else max_poles,
        )
        vf_opts = replace(
            opts,
            N=len(initial_poles),
            Niter1=0,
            Niter2=s_refinement_iterations,
            passive_DE=False,
        )
        _, scattering_ss, _, _ = vfdriver(
            bigS,
            s,
            initial_poles,
            vf_opts,
            reciprocal=reciprocal,
            verbose=verbose,
        )
        S_fit_deemb = np.asarray(eval_model(s, scattering_ss))
        ss, transform_condition = scattering_state_space_to_admittance(scattering_ss, z0_conv)
        fitted_Y = np.asarray(eval_model(s, ss))
        measured_Y = np.stack([s_to_y(value, z0_conv) for value in S_deemb])
        rmserr_Y = float(np.sqrt(np.mean(np.abs(fitted_Y - measured_Y) ** 2)))
        model = None
        direct_S_order = len(initial_poles)

    if pole_flips > 0:
        # The current AAA cleanup reflects these poles into the LHP. Preserve
        # the event in metadata and warn regardless of ``verbose``.
        message = (
            f"AAA found {pole_flips} right-half-plane pole(s) before stabilization; "
            "they were reflected into the LHP. The de-embedded fit may be non-causal; "
            "when delay extraction is enabled, the delay may be over-estimated."
        )
        if causality == "warn":
            warnings.warn(message, CausalityWarning, stacklevel=2)
        if verbose:
            print(f"  WARNING: {message}")

    passivity_margin = None
    if enforce_passive:
        from .passivity import enforce_passivity as _enforce

        model, gmin = _enforce(model, s, opts, verbose=verbose)
        ss = vfmodel_to_ss(model, Nc)
        passivity_margin = float(np.min(gmin))

    if fit_domain == "y":
        S_fit_deemb = np.stack([y_to_s(np.asarray(value), z0_conv) for value in eval_model(s, ss)])

    S_fit = embed_delay(S_fit_deemb, freqs, tau_per_port)

    rmserr_S = float(np.sqrt(np.mean(np.abs(S - S_fit) ** 2)))

    metadata = {
        "pole_count": len(np.asarray(model.poles)) if model is not None else direct_S_order,
        "aaa_backend": aaa_backend,
        "state_count": len(np.asarray(ss.A)),
        "rmserr_Y": float(rmserr_Y),
        "rmserr_S": rmserr_S,
        "passivity_margin": passivity_margin,
        "pole_flips": pole_flips,
        "causality": {
            "policy": causality,
            "raw_tau": delay_diagnostics["raw_tau"],
            "tau": tau_per_port.copy(),
            "negative_raw_delay_ports": delay_diagnostics["negative_raw_delay_ports"],
            "phase_slope_rmse": delay_diagnostics["phase_slope_rmse"],
            "pole_flips": pole_flips,
            "max_raw_pole_real": max_raw_pole_real,
            "status": "warning" if pole_flips or len(negative_delay_ports) else "pass",
        },
        "reciprocal": reciprocal,
        "delay_mode": resolved_delay_mode,
        "fit_domain": fit_domain,
        "direct_S_order": direct_S_order,
        "transform_condition": transform_condition,
        "aaa_pole_count": None if fit_domain == "y" else aaa_pole_count,
        "requested_max_poles": max_poles,
        "pole_sweep": pole_sweep,
    }

    if verbose:
        print(f"  Fit: {metadata['pole_count']} poles, RMS(S) = {rmserr_S:.2e}, RMS(Y) = {float(rmserr_Y):.2e}")

    return ss, tau_per_port, metadata


@jax.jit
def scattering_state_space_to_admittance_jax(
    scattering_ss: SSModel,
    z0: jax.Array,
) -> tuple[SSModel, jax.Array]:
    """JAX kernel for exact S-to-Y state-space conversion.

    The transformation eliminates the incident waves through
    ``v=sqrt(z0)*(a+b)`` and ``i=(a-b)/sqrt(z0)``. Its feedback generally
    changes the poles, so the dense transformed state matrix is diagonalized
    with :func:`jax.numpy.linalg.eig` before constructing the diagonal
    :class:`SSModel` consumed by Circulax.

    This kernel assumes a positive real ``z0`` and a proper model (``E=0``).
    It deliberately performs no Python-side validation so it can be composed
    under :func:`jax.jit` and :func:`jax.vmap`; use
    :func:`scattering_state_space_to_admittance` for checked scalar calls.
    """
    A = jnp.diag(scattering_ss.A)
    B = scattering_ss.B
    C = scattering_ss.C
    D = scattering_ss.D
    nc = D.shape[0]
    eye = jnp.eye(nc, dtype=D.dtype)
    inverse_sum = jnp.linalg.solve(eye + D, eye)
    root_z0 = jnp.sqrt(z0)

    A_y = A - B @ inverse_sum @ C
    B_y = B @ inverse_sum / root_z0
    C_y = -2.0 * inverse_sum @ C / root_z0
    D_y = (eye - D) @ inverse_sum / z0

    eigenvalues, eigenvectors = jnp.linalg.eig(A_y)
    condition = jnp.linalg.cond(eigenvectors)
    B_diagonal = jnp.linalg.solve(eigenvectors, B_y)
    C_diagonal = C_y @ eigenvectors
    return (
        SSModel(
            A=eigenvalues,
            B=B_diagonal,
            C=C_diagonal,
            D=D_y,
            E=jnp.zeros_like(D_y),
        ),
        condition,
    )


@jax.jit
def vmap_scattering_state_space_to_admittance(
    scattering_ss: SSModel,
    z0: jax.Array,
) -> tuple[SSModel, jax.Array]:
    """Convert a leading batch of fixed-shape S realizations to Y in parallel."""
    return jax.vmap(scattering_state_space_to_admittance_jax, in_axes=(0, None))(
        scattering_ss,
        z0,
    )


def scattering_state_space_to_admittance(
    scattering_ss: SSModel,
    z0: complex = 50.0,
) -> tuple[SSModel, float]:
    """Checked scalar wrapper around the JAX S-to-Y conversion kernel."""
    if abs(complex(z0).imag) > 1e-14 or float(np.real(z0)) <= 0:
        msg = "S-to-Y state-space conversion currently requires a positive real z0"
        raise ValueError(msg)
    if not np.allclose(np.asarray(scattering_ss.E), 0.0):
        msg = "S-to-Y state-space conversion requires a proper S model with E=0"
        raise ValueError(msg)

    admittance_ss, condition_array = scattering_state_space_to_admittance_jax(
        scattering_ss,
        jnp.asarray(float(np.real(z0)), dtype=jnp.float64),
    )
    condition = float(condition_array)
    if not np.isfinite(condition) or condition > 1e12:
        msg = f"S-to-Y state-space diagonalization is ill-conditioned ({condition:.3e})"
        raise ValueError(msg)
    return admittance_ss, condition


def y_to_s(Y: np.ndarray, z0: complex = 50.0) -> np.ndarray:
    """Convert Y-parameter matrix to S-parameter matrix.

    S = (I - z0*Y) @ inv(I + z0*Y)
    """
    n = Y.shape[-1]
    I = np.eye(n, dtype=np.complex128)
    z0_conj = np.conj(z0)
    return (I - z0_conj * Y) @ np.linalg.inv(I + z0 * Y)


def evaluate_sparameter_model(
    ss: SSModel,
    freqs: np.ndarray,
    tau_per_port: np.ndarray,
    z0: complex = 50.0,
) -> np.ndarray:
    """Evaluate a fitted, delay-embedded state-space model as S parameters."""
    freqs = np.asarray(freqs, dtype=np.float64)
    s = jnp.asarray(1j * 2.0 * np.pi * freqs)
    Y = np.asarray(eval_model(s, ss))
    S_deembedded = np.stack([y_to_s(Yk, z0) for Yk in Y])
    return embed_delay(S_deembedded, freqs, np.asarray(tau_per_port))


# Backward compatibility for early fitting prototypes that imported the
# private spelling before ``y_to_s`` became part of the public API.
_y_to_s = y_to_s
