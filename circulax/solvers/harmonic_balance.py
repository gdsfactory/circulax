"""Harmonic Balance solver for periodic steady-state circuit analysis.

Harmonic Balance (HB) finds the periodic steady-state solution of a nonlinear
circuit driven by periodic sources, without time-stepping to steady state.

The circuit DAE F(y) + dQ/dt = 0 is solved by representing y(t) as K equally-
spaced time samples over one period. The HB residual at harmonic k is:

    R_k = FFT{F(y(t))}[k] + jkw0 * FFT{Q(y(t))}[k] = 0

JAX makes this clean: ``jax.vmap`` evaluates circuit physics at all K time points
in parallel, ``jnp.fft.rfft``/``irfft`` handle the frequency transforms, and
``jax.jacobian`` provides the exact Newton Jacobian with no manual derivation.

Example::

    groups, num_vars, _ = compile_netlist(netlist, models)
    linear_strat = analyze_circuit(groups, num_vars)
    y_dc = linear_strat.solve_dc(groups, jnp.zeros(num_vars))

    run_hb = setup_harmonic_balance(groups, num_vars, freq=1e6, num_harmonics=5)
    y_time, y_freq = run_hb(y_dc)

    # JIT-able for repeated calls with different initial conditions:
    y_time, y_freq = jax.jit(run_hb)(y_dc)

    # y_time: shape (K, num_vars) -- waveform at K equally-spaced time points
    # y_freq: shape (N+1, num_vars) complex -- normalised Fourier coefficients
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array

from circulax.solvers.assembly import assemble_residual_only_complex, assemble_residual_only_real
from circulax.solvers.linear import (
    DAMPING_EPS,
    DAMPING_FACTOR,
    GROUND_STIFFNESS,
    _build_index_arrays,
)

__all__ = ["setup_harmonic_balance"]


def _hb_residual(
    y_time: Array,
    component_groups: dict,
    t_points: Array,
    omega: float,
    ground_indices: Array,
    *,
    is_complex: bool = False,
) -> Array:
    """Evaluate the Harmonic Balance residual for a real or unrolled-complex circuit.

    Evaluates circuit physics at K time points via ``jax.vmap``, transforms
    to the frequency domain, applies the ``jkw`` scaling to reactive terms,
    and transforms back to a real time-domain residual for Newton iteration.

    Args:
        y_time: State waveform, shape ``(K, sys_size)``.
        component_groups: Compiled component groups from :func:`compile_netlist`.
        t_points: Time sample points, shape ``(K,)``, covering one period.
        omega: Fundamental angular frequency ``2*pi*f0``.
        ground_indices: Indices of ground nodes (enforced to zero voltage).
        is_complex: If ``True``, use the complex (unrolled) assembly function.
            The state vector is expected in ``[re | im]`` block format of shape
            ``(K, 2*num_vars)``.

    Returns:
        Real residual array of shape ``(K, sys_size)``.

    """
    K = y_time.shape[0]
    N_harm = (K - 1) // 2

    # Evaluate (f, q) at all K time points simultaneously.
    # The dt argument is unused for residual-only assembly.
    assemble_fn = assemble_residual_only_complex if is_complex else assemble_residual_only_real
    f_time, q_time = jax.vmap(lambda y_t, t: assemble_fn(y_t, component_groups, t, 1.0))(y_time, t_points)
    # f_time, q_time: shape (K, sys_size)

    # Transform to frequency domain.
    F_k = jnp.fft.rfft(f_time, axis=0)  # (N_harm+1, sys_size) complex
    Q_k = jnp.fft.rfft(q_time, axis=0)  # (N_harm+1, sys_size) complex

    # Apply jkw scaling to reactive terms: R_k = F_k + jkw * Q_k
    k = jnp.arange(N_harm + 1, dtype=jnp.float64)
    jkw = (1j * omega) * k[:, None]  # (N_harm+1, 1)
    R_k = F_k + jkw * Q_k

    # Add frequency-domain component contributions directly in the spectral domain.
    # For each f-domain group and each harmonic k, I_k = Y(k*f0) @ V_k where
    # V_k = rfft(y_time)[k] is the raw (unnormalized) Fourier coefficient — consistent
    # with the units of F_k above.
    fdomain_keys = [gk for gk in sorted(component_groups.keys()) if component_groups[gk].is_fdomain]
    if fdomain_keys:
        sys_size = y_time.shape[1]
        f0 = omega / (2.0 * jnp.pi)
        freqs = jnp.arange(N_harm + 1, dtype=jnp.float64) * f0  # (N_harm+1,)
        y_freq = jnp.fft.rfft(y_time, axis=0)  # (N_harm+1, sys_size) complex

        for gk in fdomain_keys:
            group = component_groups[gk]

            def _fdomain_contrib(v_k: jax.Array, f_k: float) -> jax.Array:
                """Compute f-domain current contribution at a single harmonic."""
                v_ports = v_k[group.var_indices]  # (N, n_ports) complex
                Y_mats = jax.vmap(lambda p: group.physics_func(f_k, p))(group.params)
                i_ports = jnp.einsum("nij,nj->ni", Y_mats, v_ports)  # (N, n_ports) complex
                contrib = jnp.zeros(sys_size, dtype=jnp.complex128)
                return contrib.at[group.eq_indices].add(i_ports)

            # Evaluate over all harmonics simultaneously.
            fdomain_R_k = jax.vmap(_fdomain_contrib)(y_freq, freqs)  # (N_harm+1, sys_size)
            R_k = R_k + fdomain_R_k

    # Transform back to time domain to yield a real residual.
    R_time = jnp.fft.irfft(R_k, n=K, axis=0)  # (K, sys_size)

    # Enforce V_ground = 0 at every time point via penalty.
    for idx in ground_indices:
        R_time = R_time.at[:, idx].add(GROUND_STIFFNESS * y_time[:, idx])

    return R_time


def setup_harmonic_balance(
    groups: dict[str, Any],
    num_vars: int,
    freq: float,
    num_harmonics: int = 5,
    *,
    is_complex: bool = False,
    g_leak: float = 1e-9,
) -> Callable[[Array], tuple[Array, Array]]:
    """Configure and return a callable for Harmonic Balance analysis.

    This is the primary entry point for HB analysis, mirroring the pattern of
    :func:`~circulax.solvers.setup_transient`: circuit-specific data (groups,
    frequency, number of harmonics) are captured at setup time; the returned
    callable takes only the DC operating point and solver options.

    Args:
        groups: Compiled component groups returned by
            :func:`~circulax.compiler.compile_netlist`.
        num_vars: Total number of state variables (second return value of
            :func:`~circulax.compiler.compile_netlist`).
        freq: Fundamental frequency in Hz of the periodic drive.
        num_harmonics: Number of harmonics to include. The solver uses
            K = 2*num_harmonics + 1 time points. More harmonics improve
            accuracy for strongly nonlinear circuits at the cost of a larger
            Jacobian (K*sys_size x K*sys_size).
        is_complex: Set ``True`` for photonic (complex-valued) circuits.
            The state vector is stored in unrolled ``[re | im]`` block format
            of length ``2 * num_vars``.
        g_leak: Small leakage conductance added to the Jacobian diagonal for
            regularisation. Prevents singular matrices when floating nodes
            have no DC path to ground.

    Returns:
        A callable ``run_hb(y_dc, *, max_iter=50, tol=1e-6) -> (y_time, y_freq)``
        that finds the periodic steady state starting from the DC operating
        point ``y_dc``. Compatible with ``jax.jit``.

    """
    _, _, ground_idxs, sys_size = _build_index_arrays(groups, num_vars, is_complex=is_complex)
    ground_indices = jnp.array(ground_idxs)

    K = 2 * num_harmonics + 1
    omega = 2.0 * jnp.pi * freq
    t_points = jnp.linspace(0.0, 1.0 / freq, K, endpoint=False)

    def run_hb(
        y_dc: Array,
        *,
        y_flat_init: Array | None = None,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> tuple[Array, Array]:
        """Find the periodic steady state via Newton–Raphson on the HB residual.

        Args:
            y_dc: DC operating point, shape ``(sys_size,)``. Used as the
                initial guess (zero AC amplitude). Obtain from
                :meth:`~circulax.solvers.CircuitLinearSolver.solve_dc`.
            y_flat_init: Optional flat initial waveform, shape ``(K * sys_size,)``.
                If provided, used as the starting point instead of tiling ``y_dc``.
                Useful for autonomous oscillators where the zero state is a trivial
                fixed point — pass a sinusoidal initial guess to escape it.
            max_iter: Maximum number of Newton iterations.
            tol: Convergence tolerance on the infinity norm of the residual.

        Returns:
            A two-tuple ``(y_time, y_freq)`` where:

            - **y_time** -- periodic waveform samples, shape ``(K, sys_size)``.
              The k-th row is the state at time ``t_k = k*T/K``.
            - **y_freq** -- normalised Fourier coefficients, shape
              ``(N_harm+1, sys_size)`` complex. ``y_freq[0]`` is the DC
              component, ``y_freq[1]`` is the fundamental, and so on.
              Two-sided amplitude at harmonic k>=1 is ``2 * |y_freq[k]|``.

        """
        # Initial guess: use provided flat waveform or tile the DC operating point.
        y_flat = y_flat_init if y_flat_init is not None else jnp.tile(y_dc, K)  # shape (K * sys_size,)

        def residual_fn(y_flat: Array) -> Array:
            y_time = y_flat.reshape(K, sys_size)
            return _hb_residual(y_time, groups, t_points, omega, ground_indices, is_complex=is_complex).flatten()

        def newton_step(y_flat: Array, _: Any) -> Array:
            r = residual_fn(y_flat)
            J = jax.jacobian(residual_fn)(y_flat)
            # Regularise: prevents singular Jacobian when floating nodes have
            # no DC path to ground (mirrors the DC solver's g_leak).
            J = J + g_leak * jnp.eye(J.shape[0], dtype=J.dtype)
            delta = jnp.linalg.solve(J, -r)
            # Voltage damping mirrors the DC solver: limits the maximum step to
            # avoid crashing exponential nonlinearities (diodes, transistors).
            max_change = jnp.max(jnp.abs(delta))
            damping = jnp.minimum(1.0, DAMPING_FACTOR / (max_change + DAMPING_EPS))
            return y_flat + delta * damping

        # optx.FixedPointIteration uses jax.lax.while_loop internally —
        # this makes run_hb compatible with jax.jit.
        hb_solver = optx.FixedPointIteration(rtol=tol, atol=tol)
        sol = optx.fixed_point(newton_step, hb_solver, y_flat, max_steps=max_iter, throw=False)
        y_flat = sol.value

        y_time = y_flat.reshape(K, sys_size)
        # Normalise by K so that y_freq[k] is the true complex amplitude.
        y_freq = jnp.fft.rfft(y_time, axis=0) / K
        return y_time, y_freq

    return run_hb
