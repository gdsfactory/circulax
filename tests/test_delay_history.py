"""Bounded-history regression tests against the original full-history lookup."""

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax.solvers.assembly import _interp_delayed
from circulax.solvers.circuit_diffeq import circuit_diffeqsolve
from circulax.solvers.delay_history import DelayTimes


def full_history_interp(hist_t, hist_y, tau, idx, t1, current=None):
    """Original #94 implementation, retained as an independent numerical oracle."""
    finite = jnp.isfinite(hist_t)
    last = jnp.maximum(jnp.sum(finite, dtype=jnp.int32) - 1, 0)
    previous_t = hist_t[last]
    times = jnp.where(finite, hist_t, previous_t + jnp.arange(hist_t.size) - last)
    if current is None:
        current = hist_y[last, idx]

    def read(delay, columns, trial):
        query = t1 - delay
        values = hist_y[:, columns]
        values = jnp.where(finite[:, None], values, values[last])
        accepted = jax.vmap(lambda col: jnp.interp(query, times, col), in_axes=1)(values)
        previous = hist_y[last, columns]
        span = t1 - previous_t
        weight = jnp.clip(jnp.where(span > 0, (query - previous_t) / jnp.where(span > 0, span, 1.0), 0.0), 0, 1)
        return jnp.where(query > previous_t, previous + weight * (trial - previous), accepted)

    return jax.vmap(read)(tau, idx, current)


@pytest.mark.parametrize("count", [1, 4, 9])
@pytest.mark.parametrize("start", [0, 3, 8])
def test_lookup_wrap_startup_hits_and_derivatives(count, start):
    capacity = 9
    times = jnp.arange(count, dtype=float) * 0.3
    values = jnp.sin(times[:, None] + jnp.arange(6)) * (1 + 0.2j)
    slots = (start + jnp.arange(count)) % capacity
    ring = DelayTimes(jnp.zeros(capacity).at[slots].set(times), jnp.int32(start), jnp.int32(count))
    ring_y = jnp.zeros((capacity, 6), dtype=complex).at[slots].set(values)
    oracle_t = jnp.full(capacity + 1, jnp.inf).at[:count].set(times)
    oracle_y = jnp.full((capacity + 1, 6), jnp.inf, dtype=complex).at[:count].set(values)
    idx = jnp.array([[0, 2], [3, 5]])
    now = times[-1] + 0.4
    # Before t0, exact accepted hits, accepted intervals, substep, zero delay.
    for query in [-0.7, 0.0, 0.3, 0.45, float(times[-1]), float(now - 0.13), float(now)]:
        tau = jnp.array([now - query, now - query + 0.07])
        trial = jnp.ones((2, 2), dtype=complex) * (2 + 3j)

        def evaluate(delay, current, data, bounded):
            return (
                _interp_delayed(ring, data, delay, idx, now, current)
                if bounded
                else full_history_interp(oracle_t, data, delay, idx, now, current)
            )

        np.testing.assert_allclose(evaluate(tau, trial, ring_y, True), evaluate(tau, trial, oracle_y, False), atol=1e-12)

        # Real scalar losses allow comparison of complex cotangents too.
        def loss(delay, current, data, bounded):
            return jnp.real(evaluate(delay, current, data, bounded)).sum()

        actual = jax.grad(loss, (0, 1))(tau, trial, ring_y, True)
        expected = jax.grad(loss, (0, 1))(tau, trial, oracle_y, False)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(a, b, atol=1e-12)


class FeedbackSolver(diffrax.AbstractSolver):
    """Linear DDE with feedback; history affects later history and its adjoint."""

    oracle: bool = eqx.field(static=True, default=False)
    term_structure = diffrax.AbstractTerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def order(self, terms):
        return 1

    def init(self, terms, t0, t1, y0, args):
        return None

    def func(self, terms, t0, y0, args):
        return jnp.zeros_like(y0)

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        tau, amplitude, times, values = args
        if self.oracle:
            # No wrap in the oracle run; reconstruct its original inf padding.
            times = times.times if isinstance(times, DelayTimes) else times
            read = full_history_interp
        else:
            read = _interp_delayed
        idx = jnp.array([[0], [1]])

        def delayed(trial):
            return read(times, values, tau, idx, t1, trial[:, None])[:, 0]

        offset = delayed(jnp.zeros(2))
        slope = jax.jacfwd(delayed)(jnp.zeros(2))
        forcing = amplitude * jnp.sin(t1 * jnp.array([1.7, 2.1]))
        y = jnp.linalg.solve(jnp.eye(2) - 0.3 * slope, forcing + 0.3 * offset)
        return y, (y - y0) * (t1 - t0), {"y0": y0, "y1": y}, None, diffrax.RESULTS.successful


def zero_term(t, y, args):
    return jnp.zeros_like(y)


TERM = diffrax.ODETerm(zero_term)


def solve(parameters, capacity, oracle=False, adaptive=False, checkpoints=None, stop=4.0):
    tau, amp, initial = parameters[:2], parameters[2], parameters[3:]
    controller = diffrax.PIDController(rtol=0.03, atol=0.01) if adaptive else diffrax.ConstantStepSize()
    return circuit_diffeqsolve(
        TERM,
        FeedbackSolver(oracle),
        0.0,
        stop,
        0.1 if not adaptive else 0.8,
        initial,
        args=(tau, amp),
        saveat=diffrax.SaveAt(ts=jnp.concatenate([jnp.array([0.0]), jnp.linspace(0.013, stop - 0.013, 79), jnp.array([stop])])),
        record_history=True,
        max_delay=jnp.max(tau),
        history_capacity=capacity,
        max_steps=512,
        stepsize_controller=controller,
        checkpoints=checkpoints,
    )


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("checkpoints", [2, 8])
def test_complete_trace_and_checkpointed_adjoint_match_full_history(adaptive, checkpoints):
    parameters = jnp.array([0.23, 0.07, 0.9, 0.12, -0.2])
    capacity = 12 if adaptive else 6
    actual = solve(parameters, capacity, adaptive=adaptive, checkpoints=checkpoints)
    expected = solve(parameters, 513, oracle=True, adaptive=adaptive, checkpoints=checkpoints)
    np.testing.assert_allclose(actual.ys, expected.ys, atol=2e-13, rtol=2e-13)
    # Entire trace Jacobian, including delay, forcing and startup values, not
    # just a final-state loss. This crosses the same ring slots many times.
    a = jax.jacrev(lambda p: solve(p, capacity, adaptive=adaptive, checkpoints=checkpoints).ys)(parameters)
    b = jax.jacrev(lambda p: solve(p, 513, oracle=True, adaptive=adaptive, checkpoints=checkpoints).ys)(parameters)
    np.testing.assert_allclose(a, b, atol=2e-11, rtol=2e-11)
    if adaptive:
        assert actual.stats["num_rejected_steps"] > 0
        assert actual.stats["num_accepted_steps"] > capacity


def test_capacity_exhaustion_is_explicit():
    with pytest.raises(eqx.EquinoxRuntimeError, match="Delay history capacity exhausted"):
        solve(jnp.array([0.31, 0.07, 0.9, 0.12, -0.2]), 3).ys.block_until_ready()


def test_capacity_does_not_grow_with_duration():
    p = jnp.array([0.23, 0.07, 0.9, 0.12, -0.2])
    for stop in [2.0, 8.0]:
        sol = solve(p, 6, stop=stop)
        assert sol.stats["history_capacity"] == 6
        assert sol.stats["history_bytes"] == 6 * 3 * 8


@pytest.mark.parametrize("padded", [False, True])
def test_legacy_array_final_sample_slope(padded):
    times = jnp.array([0.0, 0.4, 0.9] + ([jnp.inf] if padded else []))
    values = jnp.array([[1.0], [3.0], [2.0]] + ([[jnp.inf]] if padded else []))

    def read(tau, implementation):
        return implementation(times, values, tau[None], jnp.array([[0]]), 1.2)[0, 0]

    for tau in [0.3, 0.4, 1.2, 1.3]:
        np.testing.assert_allclose(read(jnp.array(tau), _interp_delayed), read(jnp.array(tau), full_history_interp))
        a = jax.grad(read)(jnp.array(tau), _interp_delayed)
        b = jax.grad(read)(jnp.array(tau), full_history_interp)
        np.testing.assert_allclose(a, b, atol=1e-12)


def test_batched_independent_runs():
    parameters = jnp.array([[0.23, 0.07, 0.9, 0.12, -0.2], [0.28, 0.04, 0.6, -0.1, 0.3]])
    actual = jax.jit(jax.vmap(lambda p: solve(p, 6).ys))(parameters)
    expected = jax.vmap(lambda p: solve(p, 513, oracle=True).ys)(parameters)
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)


def test_no_delay_has_no_history_allocation():
    sol = circuit_diffeqsolve(TERM, diffrax.Euler(), 0.0, 1.0, 0.1, jnp.ones(2), max_steps=12)
    assert sol.stats["history_capacity"] == 0
    assert sol.stats["history_bytes"] == 0


def test_checkpoint_recomputation_preserves_on_grid_output_gradients():
    term = diffrax.ODETerm(lambda t, y, amplitude: jnp.ones_like(y) * amplitude)
    times = jnp.linspace(0, 4, 81)

    def trace(amplitude):
        return circuit_diffeqsolve(
            term,
            diffrax.Euler(),
            0.0,
            4.0,
            0.1,
            jnp.zeros(1),
            args=amplitude,
            max_steps=512,
            checkpoints=2,
            saveat=diffrax.SaveAt(ts=times),
        ).ys[:, 0]

    np.testing.assert_allclose(jax.jacrev(trace)(jnp.array(1.0)), times, atol=2e-13)


def test_feedback_trace_gradients_match_finite_differences():
    parameters = jnp.array([0.23, 0.07, 0.9, 0.12, -0.2])
    actual = jax.jacrev(lambda p: solve(p, 6, checkpoints=2).ys)(parameters)
    epsilon = 1e-5
    expected = jnp.stack(
        [
            (solve(parameters.at[k].add(epsilon), 6).ys - solve(parameters.at[k].add(-epsilon), 6).ys) / (2 * epsilon)
            for k in range(parameters.size)
        ],
        axis=-1,
    )
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-8)
