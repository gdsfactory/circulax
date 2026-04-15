"""Circuit linear solver strategies.

Separates physics assembly (Jacobian values) from the linear solve (matrix inversion).
All solvers implement the ``lineax`` abstract interface and are JAX-transformable.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import klujax
import lineax as lx
import numpy as np
import optimistix as optx

from circulax.solvers.assembly import assemble_system_complex, assemble_system_real

# ---------------------------------------------------------------------------
# Solver constants
# ---------------------------------------------------------------------------
GROUND_STIFFNESS: float = 1e9
"""Penalty added to ground-node diagonal entries to enforce V=0."""

DC_DT: float = 1e18
"""Effective timestep used for DC analysis; makes capacitor stamps vanish (C/dt → 0)."""

DAMPING_FACTOR: float = 0.5
"""Newton-step damping coefficient: limits each step to at most ``DAMPING_FACTOR / |δy|_max``."""

DAMPING_EPS: float = 1e-9
"""Small additive epsilon that prevents division by zero in the damping formula."""

# Check if split solver available — KLUHandleManager was added in a later version of klujax.
split_solver_available = True
try:
    from klujax import KLUHandleManager

    try:
        import klujax_rs as klurs
        from klujax_rs import KLUHandleManager as KLURSHandleManager
    except ImportError:
        # Silently falling back to klujax until package is ready
        klurs = klujax
        from klujax import KLUHandleManager as KLURSHandleManager
except ImportError:
    split_solver_available = False
    # Provide dummy sentinels so the eqx.field annotations below don't cause NameErrors
    KLUHandleManager = object  # type: ignore[assignment,misc]
    KLURSHandleManager = object  # type: ignore[assignment,misc]

split_refactor_available: bool = split_solver_available and hasattr(klujax, "refactor")
split_rs_refactor_available: bool = split_solver_available and hasattr(klurs, "refactor")
split_rs_fused_available: bool = split_solver_available and hasattr(klurs, "refactor_and_solve")


# ---------------------------------------------------------------------------
# Index-building helpers shared across all solver factory classmethods
# ---------------------------------------------------------------------------


def _build_index_arrays(component_groups: dict, num_vars: int, is_complex: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Extract COO row/col index arrays from component groups and expand for complex systems.

    Returns:
        (static_rows, static_cols, ground_idxs, sys_size) as numpy arrays.

    """
    all_rows, all_cols = [], []
    for k in sorted(component_groups.keys()):
        g = component_groups[k]
        all_rows.append(np.array(g.jac_rows).reshape(-1))
        all_cols.append(np.array(g.jac_cols).reshape(-1))

    static_rows = np.concatenate(all_rows)
    static_cols = np.concatenate(all_cols)
    sys_size = num_vars
    ground_idxs = np.array([0], dtype=np.int32)

    if is_complex:
        # Expand to 2N x 2N block structure:  [ RR  RI ]
        #                                      [ IR  II ]
        N = num_vars
        static_rows = np.concatenate([static_rows, static_rows, static_rows + N, static_rows + N])
        static_cols = np.concatenate([static_cols, static_cols + N, static_cols, static_cols + N])
        sys_size = N * 2
        ground_idxs = np.array([0, num_vars], dtype=np.int32)

    return static_rows, static_cols, ground_idxs, sys_size


def _klu_deduplicate(
    static_rows: np.ndarray,
    static_cols: np.ndarray,
    ground_idxs: np.ndarray,
    sys_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build unique COO index arrays for KLU, coalescing circuit + ground + leakage entries.

    Returns:
        (u_rows, u_cols, map_idx, n_unique)

    """
    leak_diag = np.arange(sys_size, dtype=np.int32)
    full_rows = np.concatenate([static_rows, ground_idxs, leak_diag])
    full_cols = np.concatenate([static_cols, ground_idxs, leak_diag])
    rc_hashes = full_rows.astype(np.int64) * sys_size + full_cols.astype(np.int64)
    unique_hashes, map_indices = np.unique(rc_hashes, return_inverse=True)
    u_rows = (unique_hashes // sys_size).astype(np.int32)
    u_cols = (unique_hashes % sys_size).astype(np.int32)
    return u_rows, u_cols, map_indices, len(unique_hashes)


class CircuitLinearSolver(lx.AbstractLinearSolver):
    """Abstract base for all circuit linear solvers.

    Attributes:
        ground_indices (jax.Array): Indices of nodes connected to ground (forced to 0V).
        is_complex (bool): If True, the system is 2N×2N (real/imag unrolled); otherwise N×N.

    """

    ground_indices: jax.Array
    is_complex: bool = eqx.field(static=True)

    def init(self, operator: Any, options: Any) -> Any:  # noqa: ARG002
        """Initialize the solver state (no-op for stateless solvers)."""
        return None

    def compute(self, state: Any, vector: jax.Array, options: Any) -> lx.Solution:
        """Satisfies the lineax API; call ``_solve_impl`` directly for internal use."""
        msg = "Directly call _solve_impl for internal use."
        raise NotImplementedError(msg)

    def transpose(self, state: Any, options: Any) -> Any:  # noqa: D102
        raise NotImplementedError

    def conj(self, state: Any, options: Any) -> "CircuitLinearSolver":  # noqa: ARG002, D102
        return self

    def allow_dependent_columns(self, operator: Any) -> bool:  # noqa: ARG002, D102
        return False

    def allow_dependent_rows(self, operator: Any) -> bool:  # noqa: ARG002, D102
        return False

    def _solve_impl(self, all_vals: jax.Array, residual: jax.Array) -> lx.Solution:
        """Internal implementation of the linear solve: Ax = b.

        Args:
            all_vals (jax.Array): Flattened array of non-zero Jacobian values.
            residual (jax.Array): The Right-Hand Side (RHS) vector 'b'.

        Returns:
            lineax.Solution: Wrapper containing the solution vector 'x'.

        """
        raise NotImplementedError

    def assume_full_rank(self) -> bool:
        """Indicate if the solver assumes the operator is full rank."""
        return False

    def _run_newton(
        self,
        component_groups: dict[str, Any],
        y_guess: jax.Array,
        source_scale: float = 1.0,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        max_steps: int = 100,
    ) -> tuple[jax.Array, jax.Array]:
        """Inner Newton-Raphson loop, optionally with source scaling.

        Assembles the system at ``source_scale`` (1.0 for a normal solve, < 1.0
        during source-stepping homotopy), applies ground constraints, and runs
        ``optx.fixed_point`` to find the DC operating point.  This method is
        fully JAX-traceable and may be called inside ``jax.lax.scan``.

        Args:
            component_groups: The circuit components and their parameters.
            y_guess: Initial guess vector (shape ``[N]`` or ``[2N]``).
            source_scale: Multiplicative scale applied to source amplitudes
                (``amplitude_param`` leaf).  ``1.0`` is a plain solve.
            rtol: Relative tolerance for ``optx.fixed_point``.
            atol: Absolute tolerance for ``optx.fixed_point``.
            max_steps: Maximum Newton iterations.

        Returns:
            ``(y, converged)`` where ``converged`` is a boolean JAX scalar that
            is ``True`` when ``optx.fixed_point`` reported success.  Both values
            are JAX arrays and can be used inside ``jax.lax.cond`` / ``scan``.

        """
        assemble_fn = assemble_system_complex if self.is_complex else assemble_system_real

        def dc_step(y: jax.Array, _: Any) -> jax.Array:
            total_f, _, all_vals = assemble_fn(y, component_groups, t1=0.0, dt=DC_DT, source_scale=source_scale)

            total_f_grounded = total_f
            for idx in self.ground_indices:
                total_f_grounded = total_f_grounded.at[idx].add(GROUND_STIFFNESS * y[idx])

            sol = self._solve_impl(all_vals, -total_f_grounded)
            delta = sol.value

            max_change = jnp.max(jnp.abs(delta))
            damping = jnp.minimum(1.0, DAMPING_FACTOR / (max_change + DAMPING_EPS))

            return y + delta * damping

        solver = optx.FixedPointIteration(rtol=rtol, atol=atol)
        sol = optx.fixed_point(dc_step, solver, y_guess, max_steps=max_steps, throw=False)
        return sol.value, sol.result == optx.RESULTS.successful

    def solve_dc(
        self,
        component_groups: dict[str, Any],
        y_guess: jax.Array,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        max_steps: int = 100,
    ) -> jax.Array:
        """DC operating point via damped Newton-Raphson.

        Args:
            component_groups: Compiled circuit components.
            y_guess: Initial guess vector (shape ``[N]`` or ``[2N]``).

        Returns:
            Converged solution vector.

        """
        y, _ = self._run_newton(component_groups, y_guess, rtol=rtol, atol=atol, max_steps=max_steps)
        return y

    def solve_dc_checked(
        self,
        component_groups: dict[str, Any],
        y_guess: jax.Array,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        max_steps: int = 100,
    ) -> tuple[jax.Array, jax.Array]:
        """DC Operating Point with convergence status.

        Identical to :meth:`solve_dc` but additionally returns a boolean JAX
        scalar indicating whether the Newton-Raphson fixed-point iteration
        reported success.  Because the flag is a JAX array (not a Python bool)
        it can be consumed inside compiled programs:

        .. code-block:: python

            y, converged = solver.solve_dc_checked(groups, y0)
            # Outside JIT — inspect in Python:
            if not converged:
                y = solver.solve_dc_gmin(groups, y0)
            # Inside JIT — branch without Python-level control flow:
            y = jax.lax.cond(converged, lambda: y, lambda: solver.solve_dc_gmin(groups, y0))

        Args:
            component_groups: Compiled circuit components.
            y_guess: Initial guess vector (shape ``[N]`` or ``[2N]``).
            rtol: Relative tolerance for ``optx.fixed_point``.
            atol: Absolute tolerance for ``optx.fixed_point``.
            max_steps: Maximum Newton iterations.

        Returns:
            ``(y, converged)`` — solution vector and boolean success flag.

        """
        return self._run_newton(component_groups, y_guess, rtol=rtol, atol=atol, max_steps=max_steps)

    def solve_dc_gmin(
        self,
        component_groups: dict[str, Any],
        y_guess: jax.Array,
        g_start: float = 1e-2,
        n_steps: int = 10,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        max_steps: int = 100,
    ) -> jax.Array:
        """DC Operating Point via GMIN stepping (homotopy rescue).

        Steps the diagonal regularisation conductance ``g_leak`` logarithmically
        from ``g_start`` down to ``self.g_leak``, using each converged solution
        as the warm start for the next step.  The large initial ``g_leak``
        linearises highly nonlinear components (diodes above threshold, lasers)
        that would otherwise cause Newton to diverge from a flat 0V start.

        Implemented with ``jax.lax.scan`` — fully JIT/grad/vmap-compatible.

        Args:
            component_groups: Compiled circuit components.
            y_guess: Initial guess (typically ``jnp.zeros(sys_size)``).
            g_start: Starting leakage conductance (large value, e.g. ``1e-2``).
            n_steps: Number of log-uniform steps from ``g_start`` to
                ``self.g_leak``.
            rtol: Relative tolerance for each inner Newton solve.
            atol: Absolute tolerance for each inner Newton solve.
            max_steps: Max Newton iterations per step.

        Returns:
            Converged solution vector after the full GMIN schedule.

        """
        g_values = jnp.logspace(jnp.log10(g_start), jnp.log10(self.g_leak), n_steps)

        def step(y: jax.Array, g_leak_val: jax.Array) -> tuple[jax.Array, None]:
            stepped_solver = eqx.tree_at(lambda s: s.g_leak, self, g_leak_val)
            y_new, _ = stepped_solver._run_newton(  # noqa: SLF001
                component_groups, y, rtol=rtol, atol=atol, max_steps=max_steps
            )
            return y_new, None

        y_final, _ = jax.lax.scan(step, y_guess, g_values)
        return y_final

    def solve_dc_source(
        self,
        component_groups: dict[str, Any],
        y_guess: jax.Array,
        n_steps: int = 10,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        max_steps: int = 100,
    ) -> jax.Array:
        """DC Operating Point via source stepping (homotopy rescue).

        Ramps all source amplitudes (components tagged with ``amplitude_param``)
        from 10 % to 100 % of their netlist values, using each converged
        solution as the warm start for the next step.  This guides Newton
        through the nonlinear region without the large initial step from 0V to
        full excitation.

        Implemented with ``jax.lax.scan`` — fully JIT/grad/vmap-compatible.

        Args:
            component_groups: Compiled circuit components.
            y_guess: Initial guess (typically ``jnp.zeros(sys_size)``).
            n_steps: Number of uniformly-spaced steps from 0.1 to 1.0.
            rtol: Relative tolerance for each inner Newton solve.
            atol: Absolute tolerance for each inner Newton solve.
            max_steps: Max Newton iterations per step.

        Returns:
            Converged solution vector at full source amplitude.

        """
        scales = jnp.linspace(0.1, 1.0, n_steps)

        def step(y: jax.Array, scale: jax.Array) -> tuple[jax.Array, None]:
            y_new, _ = self._run_newton(component_groups, y, source_scale=scale, rtol=rtol, atol=atol, max_steps=max_steps)
            return y_new, None

        y_final, _ = jax.lax.scan(step, y_guess, scales)
        return y_final

    def solve_dc_auto(
        self,
        component_groups: dict[str, Any],
        y_guess: jax.Array,
        g_start: float = 1e-2,
        n_gmin: int = 10,
        n_source: int = 10,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        max_steps: int = 100,
    ) -> jax.Array:
        """DC Operating Point with automatic homotopy fallback.

        Attempts a direct Newton solve first.  If it fails to converge, falls
        back to GMIN stepping followed by source stepping — all inside a single
        JIT-compiled kernel via ``jax.lax.cond``.

        Strategy:
        1.  ``_run_newton`` — plain damped Newton from ``y_guess``.
        2.  On failure: ``solve_dc_gmin`` (GMIN stepping) starting from
            ``y_guess``, then ``solve_dc_source`` (source stepping) from the
            GMIN result.

        Because ``jax.lax.cond`` evaluates both branches at *trace* time but
        only one at *runtime*, this compiles to a single kernel with no Python-
        level branching.

        Args:
            component_groups: Compiled circuit components.
            y_guess: Initial guess (typically ``jnp.zeros(sys_size)``).
            g_start: Starting leakage for GMIN stepping (rescue branch).
            n_gmin: Number of GMIN steps in the rescue branch.
            n_source: Number of source steps in the rescue branch.
            rtol: Relative tolerance for each inner Newton solve.
            atol: Absolute tolerance for each inner Newton solve.
            max_steps: Max Newton iterations per step.

        Returns:
            Converged solution vector.

        """
        y_direct, converged = self._run_newton(component_groups, y_guess, rtol=rtol, atol=atol, max_steps=max_steps)

        def rescue(_: None) -> jax.Array:
            y_gmin = self.solve_dc_gmin(
                component_groups,
                y_guess,
                g_start=g_start,
                n_steps=n_gmin,
                rtol=rtol,
                atol=atol,
                max_steps=max_steps,
            )
            return self.solve_dc_source(
                component_groups,
                y_gmin,
                n_steps=n_source,
                rtol=rtol,
                atol=atol,
                max_steps=max_steps,
            )

        return jax.lax.cond(converged, lambda _: y_direct, rescue, None)


# --- DenseSolver ---


class DenseSolver(CircuitLinearSolver):
    """Solves the system using dense matrix factorization (LU).

    Best For:
        - Small to Medium circuits (N < 2000).
        - Wavelength sweeps (AC Analysis) on GPU.
        - Systems where VMAP parallelism is critical.

    Attributes:
        static_rows (jax.Array): Row indices for placing values into dense matrix.
        static_cols (jax.Array): Column indices.
        g_leak (float): Leakage conductance added to diagonal to prevent singularity.

    """

    static_rows: jax.Array
    static_cols: jax.Array
    sys_size: int = eqx.field(static=True)
    g_leak: float = 1e-9

    def _solve_impl(self, all_vals: jax.Array, residual: jax.Array) -> lx.Solution:
        J = jnp.zeros((self.sys_size, self.sys_size), dtype=residual.dtype)
        J = J.at[self.static_rows, self.static_cols].add(all_vals)

        diag_idx = jnp.arange(self.sys_size)
        J = J.at[diag_idx, diag_idx].add(self.g_leak)

        for idx in self.ground_indices:
            J = J.at[idx, idx].add(GROUND_STIFFNESS)

        x = jnp.linalg.solve(J, residual)
        return lx.Solution(value=x, result=lx.RESULTS.successful, state=None, stats={})

    @classmethod
    def from_component_groups(
        cls, component_groups: dict[str, Any], num_vars: int, *, is_complex: bool = False, g_leak: float = 1e-9
    ) -> "DenseSolver":
        """Factory method to pre-calculate indices for the dense matrix."""
        rows, cols, ground_idxs, sys_size = _build_index_arrays(component_groups, num_vars, is_complex)
        return cls(
            static_rows=jnp.array(rows),
            static_cols=jnp.array(cols),
            sys_size=sys_size,
            ground_indices=jnp.array(ground_idxs),
            is_complex=is_complex,
            g_leak=g_leak,
        )


# --- KLU Solvers ---


class KLUSplitSolver(CircuitLinearSolver):
    """Solves the system using the KLU sparse solver (via `klujax`) with split interface.

    This solver performs symbolic analysis ONCE during initialization and reuses
    the symbolic handle for subsequent solves, significantly speeding up non-linear
    simulations (Newton-Raphson iterations).

    Best For:
        - Large circuits (N > 5000) running on CPU.
        - DC Operating Points of massive meshes.

    Attributes:
        Bp, Bi: CSC format indices (fixed structure).
        csc_map_idx: Mapping from raw value indices to CSC value vector.
        symbolic_handle: Pointer to the pre-computed KLU symbolic analysis.

    """

    u_rows: jax.Array
    u_cols: jax.Array
    map_idx: jax.Array

    n_unique: int = eqx.field(static=True)
    sys_size: int = eqx.field(static=True)
    _handle_wrapper: KLUHandleManager = eqx.field(static=True)

    g_leak: float = 1e-9

    def cleanup(self) -> None:  # noqa: D102
        del self._handle_wrapper

    def _solve_impl(self, all_vals: jax.Array, residual: jax.Array) -> lx.Solution:
        g_vals = jnp.full(self.ground_indices.shape[0], GROUND_STIFFNESS, dtype=all_vals.dtype)
        l_vals = jnp.full(self.sys_size, self.g_leak, dtype=all_vals.dtype)
        raw_vals = jnp.concatenate([all_vals, g_vals, l_vals])
        coalesced_vals = jax.ops.segment_sum(raw_vals, self.map_idx, num_segments=self.n_unique)
        solution = klujax.solve_with_symbol(
            self.u_rows,
            self.u_cols,
            coalesced_vals,
            residual,
            self._handle_wrapper.handle,
        )
        return lx.Solution(value=solution, result=lx.RESULTS.successful, state=None, stats={})

    @classmethod
    def from_component_groups(
        cls, component_groups: dict[str, Any], num_vars: int, *, is_complex: bool = False, g_leak: float = 1e-9
    ) -> "KLUSplitSolver":
        """Factory method to pre-hash indices for sparse coalescence."""
        rows, cols, ground_idxs, sys_size = _build_index_arrays(component_groups, num_vars, is_complex)
        u_rows, u_cols, map_idx, n_unique = _klu_deduplicate(rows, cols, ground_idxs, sys_size)
        symbolic = klujax.analyze(u_rows, u_cols, sys_size)
        return cls(
            u_rows=jnp.array(u_rows),
            u_cols=jnp.array(u_cols),
            map_idx=jnp.array(map_idx),
            n_unique=n_unique,
            _handle_wrapper=symbolic,
            ground_indices=jnp.array(ground_idxs),
            sys_size=sys_size,
            is_complex=is_complex,
            g_leak=g_leak,
        )


class KlursSplitSolver(KLUSplitSolver):
    """Solves the system using the rust wrapped KLU sparse solver (via `klu-rs`) with split interface.

    This solver performs symbolic analysis ONCE during initialization and reuses
    the symbolic handle for subsequent solves, significantly speeding up non-linear
    simulations (Newton-Raphson iterations).

    Best For:
        - Large circuits (N > 5000) running on CPU.
        - DC Operating Points of massive meshes.

    Attributes:
        Bp, Bi: CSC format indices (fixed structure).
        csc_map_idx: Mapping from raw value indices to CSC value vector.
        symbolic_handle: Pointer to the pre-computed KLU symbolic analysis.


    """

    _handle_wrapper: KLURSHandleManager = eqx.field(static=True)

    def _solve_impl(self, all_vals: jax.Array, residual: jax.Array) -> lx.Solution:
        g_vals = jnp.full(self.ground_indices.shape[0], GROUND_STIFFNESS, dtype=all_vals.dtype)
        l_vals = jnp.full(self.sys_size, self.g_leak, dtype=all_vals.dtype)
        raw_vals = jnp.concatenate([all_vals, g_vals, l_vals])
        coalesced_vals = jax.ops.segment_sum(raw_vals, self.map_idx, num_segments=self.n_unique)
        solution = klurs.solve_with_symbol(
            self.u_rows,
            self.u_cols,
            coalesced_vals,
            residual,
            self._handle_wrapper.handle,
        )
        return lx.Solution(value=solution, result=lx.RESULTS.successful, state=None, stats={})

    @classmethod
    def from_component_groups(
        cls, component_groups: dict[str, Any], num_vars: int, *, is_complex: bool = False, g_leak: float = 1e-9
    ) -> "KlursSplitSolver":
        """Factory method to pre-hash indices for sparse coalescence."""
        rows, cols, ground_idxs, sys_size = _build_index_arrays(component_groups, num_vars, is_complex)
        u_rows, u_cols, map_idx, n_unique = _klu_deduplicate(rows, cols, ground_idxs, sys_size)
        symbol = klurs.analyze(u_rows, u_cols, sys_size)
        return cls(
            u_rows=jnp.array(u_rows),
            u_cols=jnp.array(u_cols),
            map_idx=jnp.array(map_idx),
            n_unique=n_unique,
            _handle_wrapper=symbol,
            ground_indices=jnp.array(ground_idxs),
            sys_size=sys_size,
            is_complex=is_complex,
            g_leak=g_leak,
        )


class KLUSplitLinear(KLUSplitSolver):
    """KLU split solver paired with Modified Newton (frozen-Jacobian) for linear convergence.

    Extends :class:`KLUSplitSolver` with an explicit numeric factorization step so the
    Jacobian can be **factored once per time step** and reused across all Newton iterations
    within that step (Modified Newton / frozen-Jacobian scheme).  Use together with
    :class:`~circulax.solvers.transient.FactorizedTransientSolver`.

    Best For:
        - Large circuits (N > 5000) running on CPU where the Jacobian changes slowly.
        - Transient simulations with many Newton iterations per step.

    """

    def cleanup(self) -> None:  # noqa: D102
        del self._handle_wrapper

    def factor_jacobian(self, all_vals: jax.Array) -> jax.Array:
        """Factor the Jacobian and return a numeric handle for repeated solves.

        Args:
            all_vals: Flattened non-zero Jacobian values (COO format).

        Returns:
            Opaque numeric handle (``int32`` JAX scalar) to pass to
            :meth:`solve_with_frozen_jacobian`.  Must be freed with
            ``klujax.free_numeric`` after use to avoid C++ memory leaks.

        """
        g_vals = jnp.full(self.ground_indices.shape[0], GROUND_STIFFNESS, dtype=all_vals.dtype)
        l_vals = jnp.full(self.sys_size, self.g_leak, dtype=all_vals.dtype)

        raw_vals = jnp.concatenate([all_vals, g_vals, l_vals])
        coalesced_vals = jax.ops.segment_sum(raw_vals, self.map_idx, num_segments=self.n_unique)

        return klujax.factor(self.u_rows, self.u_cols, coalesced_vals, self._handle_wrapper.handle)

    def solve_with_frozen_jacobian(self, residual: jax.Array, numeric: jax.Array) -> lx.Solution:
        """Solve using a pre-computed numeric factorization.

        Args:
            residual: The right-hand side vector ``-F(y)``.
            numeric: Handle returned by :meth:`factor_jacobian`.

        Returns:
            :class:`lineax.Solution` with the Newton step ``δy``.

        """
        solution = klujax.solve_with_numeric(numeric, residual, self._handle_wrapper.handle)
        return lx.Solution(
            value=solution.reshape(residual.shape),
            result=lx.RESULTS.successful,
            state=None,
            stats={},
        )

    @classmethod
    def from_component_groups(
        cls, component_groups: dict[str, Any], num_vars: int, *, is_complex: bool = False, g_leak: float = 1e-9
    ) -> "KLUSplitLinear":
        """Factory — delegates to :meth:`KLUSplitSolver.from_component_groups`."""
        return super().from_component_groups(  # type: ignore[return-value]
            component_groups, num_vars, is_complex=is_complex, g_leak=g_leak
        )


KLUSplitFactorSolver = KLUSplitLinear  # backward-compat alias


class KLUSplitQuadratic(KLUSplitLinear):
    """KLU split solver paired with full Newton for quadratic convergence via ``klu_refactor``.

    Extends :class:`KLUSplitLinear` with :meth:`refactor_jacobian`, which updates the numeric
    LU factorization in-place using ``klujax.refactor``.  The sparsity pattern is fixed for a
    given circuit topology, so KLU reuses the existing memory allocation and fill-reducing
    permutation — only the L/U values are recomputed.  This gives full Newton (quadratic)
    convergence at a fraction of the cost of re-calling ``klu_factor`` at every iteration.

    Use together with :class:`~circulax.solvers.transient.RefactoringTransientSolver`.

    Best For:
        - Large circuits on CPU with nonlinear devices where quadratic convergence is desired.
        - Transient simulations where the Jacobian changes significantly between Newton iterates.

    """

    def refactor_jacobian(self, all_vals: jax.Array, numeric: jax.Array) -> jax.Array:
        """Update the numeric factorization in-place with new Jacobian values.

        Reuses the existing memory allocation and fill-reducing permutation from the
        symbolic analysis; only the L/U values are recomputed.  Faster than calling
        :meth:`~KLUSplitLinear.factor_jacobian` from scratch each Newton iteration.

        Args:
            all_vals: Flattened non-zero Jacobian values (COO format).
            numeric: Existing handle returned by :meth:`~KLUSplitLinear.factor_jacobian`.

        Returns:
            Refreshed numeric handle (same underlying C++ object, now connected in the
            XLA computation graph so the refactor cannot be eliminated as dead code).

        """
        g_vals = jnp.full(self.ground_indices.shape[0], GROUND_STIFFNESS, dtype=all_vals.dtype)
        l_vals = jnp.full(self.sys_size, self.g_leak, dtype=all_vals.dtype)
        raw_vals = jnp.concatenate([all_vals, g_vals, l_vals])
        coalesced_vals = jax.ops.segment_sum(raw_vals, self.map_idx, num_segments=self.n_unique)
        return klujax.refactor(self.u_rows, self.u_cols, coalesced_vals, numeric, self._handle_wrapper.handle)

    @classmethod
    def from_component_groups(
        cls, component_groups: dict[str, Any], num_vars: int, *, is_complex: bool = False, g_leak: float = 1e-9
    ) -> "KLUSplitQuadratic":
        """Factory — delegates to :meth:`KLUSplitSolver.from_component_groups`."""
        return super().from_component_groups(  # type: ignore[return-value]
            component_groups, num_vars, is_complex=is_complex, g_leak=g_leak
        )


class KLURSplitLinear(KlursSplitSolver):
    """KLU-RS split solver paired with Modified Newton (frozen-Jacobian) for linear convergence.

    Extends :class:`KlursSplitSolver` with an explicit numeric factorization step so the
    Jacobian can be **factored once per time step** and reused across all Newton iterations
    within that step (Modified Newton / frozen-Jacobian scheme).  Use together with
    :class:`~circulax.solvers.transient.FactorizedTransientSolver`.

    Best For:
        - Large circuits (N > 5000) running on CPU where the Jacobian changes slowly.
        - Transient simulations with many Newton iterations per step.

    """

    def cleanup(self) -> None:  # noqa: D102
        del self._handle_wrapper

    def _solve_impl(self, all_vals: jax.Array, residual: jax.Array) -> lx.Solution:
        """Full factor + solve in one call (used by DC solver)."""
        g_vals = jnp.full(self.ground_indices.shape[0], GROUND_STIFFNESS, dtype=all_vals.dtype)
        l_vals = jnp.full(self.sys_size, self.g_leak, dtype=all_vals.dtype)

        raw_vals = jnp.concatenate([all_vals, g_vals, l_vals])
        coalesced_vals = jax.ops.segment_sum(raw_vals, self.map_idx, num_segments=self.n_unique)

        numeric = klurs.factor(self.u_rows, self.u_cols, coalesced_vals, self._handle_wrapper.handle)
        solution = klurs.solve_with_numeric(numeric, residual)
        klurs.free_numeric(numeric)
        return lx.Solution(
            value=solution.reshape(residual.shape),
            result=lx.RESULTS.successful,
            state=None,
            stats={},
        )

    def factor_jacobian(self, all_vals: jax.Array) -> jax.Array:
        """Factor the Jacobian and return a numeric handle for repeated solves.

        Args:
            all_vals: Flattened non-zero Jacobian values (COO format).

        Returns:
            Numeric handle to pass to :meth:`solve_with_frozen_jacobian`.
            Must be freed after use to avoid C++ memory leaks.

        """
        g_vals = jnp.full(self.ground_indices.shape[0], GROUND_STIFFNESS, dtype=all_vals.dtype)
        l_vals = jnp.full(self.sys_size, self.g_leak, dtype=all_vals.dtype)

        raw_vals = jnp.concatenate([all_vals, g_vals, l_vals])
        coalesced_vals = jax.ops.segment_sum(raw_vals, self.map_idx, num_segments=self.n_unique)

        return klurs.factor(self.u_rows, self.u_cols, coalesced_vals, self._handle_wrapper.handle)

    def solve_with_frozen_jacobian(self, residual: jax.Array, numeric: jax.Array) -> lx.Solution:
        """Solve using a pre-computed numeric factorization.

        Args:
            residual: The right-hand side vector ``-F(y)``.
            numeric: Handle returned by :meth:`factor_jacobian`.

        Returns:
            :class:`lineax.Solution` with the Newton step ``δy``.

        """
        solution = klurs.solve_with_numeric(numeric, residual)
        return lx.Solution(
            value=solution.reshape(residual.shape),
            result=lx.RESULTS.successful,
            state=None,
            stats={},
        )

    @classmethod
    def from_component_groups(
        cls, component_groups: dict[str, Any], num_vars: int, *, is_complex: bool = False, g_leak: float = 1e-9
    ) -> "KLURSplitLinear":
        """Factory — delegates to :meth:`KlursSplitSolver.from_component_groups`."""
        return super().from_component_groups(  # type: ignore[return-value]
            component_groups, num_vars, is_complex=is_complex, g_leak=g_leak
        )


class KLURSplitQuadratic(KLURSplitLinear):
    """KLURs split solver paired with full Newton for quadratic convergence via ``klu_refactor``.

    Extends :class:`KLUSplitLinear` with :meth:`refactor_jacobian`, which updates the numeric
    LU factorization in-place using ``klujax.refactor``.  The sparsity pattern is fixed for a
    given circuit topology, so KLU reuses the existing memory allocation and fill-reducing
    permutation — only the L/U values are recomputed.  This gives full Newton (quadratic)
    convergence at a fraction of the cost of re-calling ``klu_factor`` at every iteration.

    Use together with :class:`~circulax.solvers.transient.RefactoringTransientSolver`.

    Best For:
        - Large circuits on CPU with nonlinear devices where quadratic convergence is desired.
        - Transient simulations where the Jacobian changes significantly between Newton iterates.

    """

    def refactor_jacobian(self, all_vals: jax.Array, numeric: jax.Array) -> jax.Array:
        """Update the numeric factorization in-place with new Jacobian values.

        Reuses the existing memory allocation and fill-reducing permutation from the
        symbolic analysis; only the L/U values are recomputed.  Faster than calling
        :meth:`~KLUSplitLinear.factor_jacobian` from scratch each Newton iteration.

        Args:
            all_vals: Flattened non-zero Jacobian values (COO format).
            numeric: Existing handle returned by :meth:`~KLUSplitLinear.factor_jacobian`.

        Returns:
            Refreshed numeric handle (same underlying C++ object, now connected in the
            XLA computation graph so the refactor cannot be eliminated as dead code).

        """
        g_vals = jnp.full(self.ground_indices.shape[0], GROUND_STIFFNESS, dtype=all_vals.dtype)
        l_vals = jnp.full(self.sys_size, self.g_leak, dtype=all_vals.dtype)
        raw_vals = jnp.concatenate([all_vals, g_vals, l_vals])
        coalesced_vals = jax.ops.segment_sum(raw_vals, self.map_idx, num_segments=self.n_unique)
        return klurs.refactor(self.u_rows, self.u_cols, coalesced_vals, numeric, self._handle_wrapper.handle)

    if split_rs_fused_available:

        def refactor_and_solve_jacobian(
            self, all_vals: jax.Array, residual: jax.Array, numeric: jax.Array
        ) -> tuple["lx.Solution", jax.Array]:
            """Fused refactor + solve: one XLA custom call instead of two.

            Args:
                all_vals: Flattened non-zero Jacobian values (COO format).
                residual: The right-hand side vector ``-F(y)`` (pass ``-residual``).
                numeric: Handle returned by :meth:`~KLURSplitLinear.factor_jacobian`.

            Returns:
                ``(solution, refreshed_numeric)`` where ``refreshed_numeric`` is a
                non-owning alias of ``numeric`` — safe to discard with ``_`` since
                ``numeric_handle`` is closure-captured and the Newton loop does not
                thread the handle through its carry state.

            """
            g_vals = jnp.full(self.ground_indices.shape[0], GROUND_STIFFNESS, dtype=all_vals.dtype)
            l_vals = jnp.full(self.sys_size, self.g_leak, dtype=all_vals.dtype)
            raw_vals = jnp.concatenate([all_vals, g_vals, l_vals])
            coalesced_vals = jax.ops.segment_sum(raw_vals, self.map_idx, num_segments=self.n_unique)
            x, refreshed = _klurs_refactor_and_solve(
                self.u_rows, self.u_cols, coalesced_vals, residual, numeric, self._handle_wrapper.handle
            )
            return lx.Solution(value=x.reshape(residual.shape), result=lx.RESULTS.successful, state=None, stats={}), refreshed

    @classmethod
    def from_component_groups(
        cls, component_groups: dict[str, Any], num_vars: int, *, is_complex: bool = False, g_leak: float = 1e-9
    ) -> "KLURSplitQuadratic":
        """Factory — delegates to :meth:`KLUSplitSolver.from_component_groups`."""
        return super().from_component_groups(  # type: ignore[return-value]
            component_groups, num_vars, is_complex=is_complex, g_leak=g_leak
        )


class KLUSolver(CircuitLinearSolver):
    """Solves the system using the KLU sparse solver (via `klujax`).

    Best For:
        - Large circuits (N > 5000) running on CPU.
        - DC Operating Points of massive meshes.
        - Cases where DenseSolver runs out of memory (OOM).

    Note:
        Does NOT support `vmap` (batching) automatically.

    """

    u_rows: jax.Array
    u_cols: jax.Array
    map_idx: jax.Array
    n_unique: int = eqx.field(static=True)
    sys_size: int = eqx.field(static=True)

    g_leak: float = 1e-9

    def _solve_impl(self, all_vals: jax.Array, residual: jax.Array) -> lx.Solution:
        g_vals = jnp.full(self.ground_indices.shape[0], GROUND_STIFFNESS, dtype=all_vals.dtype)
        l_vals = jnp.full(self.sys_size, self.g_leak, dtype=all_vals.dtype)
        raw_vals = jnp.concatenate([all_vals, g_vals, l_vals])
        coalesced_vals = jax.ops.segment_sum(raw_vals, self.map_idx, num_segments=self.n_unique)
        solution = klujax.solve(self.u_rows, self.u_cols, coalesced_vals, residual)
        return lx.Solution(value=solution, result=lx.RESULTS.successful, state=None, stats={})

    @classmethod
    def from_component_groups(
        cls, component_groups: dict[str, Any], num_vars: int, *, is_complex: bool = False, g_leak: float = 1e-9
    ) -> "KLUSolver":
        """Factory method to pre-hash indices for sparse coalescence."""
        rows, cols, ground_idxs, sys_size = _build_index_arrays(component_groups, num_vars, is_complex)
        u_rows, u_cols, map_idx, n_unique = _klu_deduplicate(rows, cols, ground_idxs, sys_size)
        return cls(
            u_rows=jnp.array(u_rows),
            u_cols=jnp.array(u_cols),
            map_idx=jnp.array(map_idx),
            n_unique=n_unique,
            ground_indices=jnp.array(ground_idxs),
            sys_size=sys_size,
            is_complex=is_complex,
            g_leak=g_leak,
        )


# --- SparseSolver ---


class SparseSolver(CircuitLinearSolver):
    """Solves the system using JAX's Iterative BiCGStab solver.

    Best For:
        - Large Transient Simulations on GPU (uses previous step as warm start).
        - Systems where N is too large for Dense, but we need VMAP support.

    Attributes:
        diag_mask (jax.Array): Mask to extract diagonal elements for preconditioning.

    """

    static_rows: jax.Array
    static_cols: jax.Array
    diag_mask: jax.Array
    sys_size: int = eqx.field(static=True)

    g_leak: float = 1e-9

    def _solve_impl(self, all_vals: jax.Array, residual: jax.Array) -> lx.Solution:
        # Jacobi preconditioner: extract diagonal, add leakage and ground stiffness.
        diag_vals = jax.ops.segment_sum(all_vals * self.diag_mask, self.static_rows, num_segments=self.sys_size)
        diag_vals = diag_vals + self.g_leak
        for idx in self.ground_indices:
            diag_vals = diag_vals.at[idx].add(GROUND_STIFFNESS)
        inv_diag = jnp.where(jnp.abs(diag_vals) < 1e-12, 1.0, 1.0 / diag_vals)

        def matvec(x: jax.Array) -> jax.Array:
            Ax = jax.ops.segment_sum(all_vals * x[self.static_cols], self.static_rows, num_segments=self.sys_size)
            Ax = Ax + (x * self.g_leak)
            for idx in self.ground_indices:
                Ax = Ax.at[idx].add(GROUND_STIFFNESS * x[idx])
            return Ax

        delta_guess = residual * inv_diag
        x, _ = jax.scipy.sparse.linalg.bicgstab(
            matvec,
            residual,
            x0=delta_guess,
            M=lambda v: inv_diag * v,
            tol=1e-5,
            maxiter=200,
        )

        return lx.Solution(value=x, result=lx.RESULTS.successful, state=None, stats={})

    @classmethod
    def from_component_groups(
        cls, component_groups: dict[str, Any], num_vars: int, *, is_complex: bool = False, g_leak: float = 1e-9
    ) -> "SparseSolver":
        """Factory method to prepare indices and diagonal mask."""
        rows, cols, ground_idxs, sys_size = _build_index_arrays(component_groups, num_vars, is_complex)
        return cls(
            static_rows=jnp.array(rows),
            static_cols=jnp.array(cols),
            diag_mask=jnp.array(rows == cols),
            sys_size=sys_size,
            ground_indices=jnp.array(ground_idxs),
            is_complex=is_complex,
            g_leak=g_leak,
        )

    def cleanup(self):
        pass


backends: dict[str, type[CircuitLinearSolver]] = {
    "default": KLUSolver,
    "klu": KLUSolver,
    "dense": DenseSolver,
    "sparse": SparseSolver,
}

if split_solver_available:
    backends["klu_split_linear"] = KLUSplitLinear
    backends["klu_split"] = KLUSplitQuadratic if split_refactor_available else KLUSplitLinear
    backends["klu_rs_split"] = KLURSplitQuadratic if split_rs_refactor_available else KlursSplitSolver
    # Legacy aliases
    backends["klu_split_factor"] = KLUSplitLinear
    backends["klu_split_refactor"] = KLUSplitQuadratic if split_refactor_available else KLUSplitLinear
    # KLU-RS factor/refactor variants
    backends["klu_rs_split_factor"] = KLURSplitLinear
    backends["klu_rs_split_refactor"] = KLURSplitQuadratic if split_rs_refactor_available else KLURSplitLinear
    # Default uses factor (BDF2FactorizedTransientSolver): wins for linear and mildly nonlinear
    # circuits; use klu_split_refactor / klu_rs_split_refactor explicitly for strongly nonlinear.
    backends["default"] = backends["klu_split_factor"]
else:
    # Silently fall back to KLUSolver when KLUHandleManager is not available
    backends["klu_split"] = KLUSolver
    backends["klu_split_linear"] = KLUSolver
    backends["klu_split_factor"] = KLUSolver
    backends["klu_split_refactor"] = KLUSolver
    backends["klu_rs_split"] = KLUSolver
    backends["klu_rs_split_factor"] = KLUSolver
    backends["klu_rs_split_refactor"] = KLUSolver


def analyze_circuit(
    groups: list, num_vars: int, backend: str = "default", *, is_complex: bool = False, g_leak: float = 1e-9
) -> CircuitLinearSolver:
    """Initializes a linear solver strategy for circuit analysis.

    This function serves as a factory and wrapper to select and configure the
    appropriate numerical backend for solving the linear system of equations
    derived from a circuit's topology.

    The available backends are

    "default": KLUSolver,
    "klu": KLUSolver,
    "dense": DenseSolver,
    "sparse": SparseSolver,

    Args:
        groups (list): A list of component groups that define the circuit's
            structure and properties.
        num_vars (int): The total number of variables in the linear system.
        backend (str, optional): The name of the solver backend to use.
            Supported backends are 'klu', 'klu_split', 'dense', and 'sparse'.
            Defaults to 'default', which uses the 'klu' solver.
        is_complex (bool, optional): A flag indicating whether the circuit
            analysis involves complex numbers. Defaults to False.

    Returns:
        CircuitLinearSolver: An instance of a circuit linear solver strategy
        configured for the specified backend and circuit parameters.

    Raises:
        ValueError: If the specified backend is not supported.

    """
    solver_class = backends.get(backend)
    if solver_class is None:
        msg = f"Unknown backend: '{backend}'. Available backends are {list(backends.keys())}"
        raise ValueError(msg)

    linear_strategy = solver_class.from_component_groups(groups, num_vars, is_complex=is_complex, g_leak=g_leak)

    return linear_strategy
