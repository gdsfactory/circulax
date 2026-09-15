"""Fixed-capacity JAX AAA discovery, comparable to the NumPy scalar AAA.

Returns padded barycentric arrays and an active support count. Adaptation stays
inside one compiled loop; changing sample count or capacity still recompiles.
Pole extraction and stable circuit realization are separate operations.
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp


class AAAResult(NamedTuple):
    """Only entries before ``count`` are active in each barycentric array."""

    weights: jax.Array
    nodes: jax.Array
    values: jax.Array
    count: jax.Array


@partial(jax.jit, static_argnames=("mmax",))
def aaa_scalar_jax(f, z, tol=1e-10, mmax=100) -> AAAResult:
    """Adaptive AAA with a fixed SVD shape and no host-side iteration.

    Inactive columns are isolated in an orthogonal block with singular value
    two; the active Loewner block is normalized to norm at most one. Thus the
    smallest right singular vector belongs to the active approximation.
    Discrete support selection is not a differentiable topology search.
    """
    f = jnp.asarray(f, dtype=jnp.complex128)
    z = jnp.asarray(z, dtype=jnp.complex128)
    if f.ndim != 1 or z.shape != f.shape or not f.size or mmax < 1:
        raise ValueError("expected matching nonempty vectors and positive mmax")
    capacity = min(mmax, f.size)
    indices = jnp.arange(capacity)
    empty = jnp.zeros(capacity, dtype=f.dtype)
    norm = jnp.maximum(jnp.linalg.norm(f), jnp.finfo(jnp.float64).tiny)
    # count, support mask, nodes, values, weights, residual, relative error
    initial = (jnp.array(0), jnp.zeros(f.size, bool), empty, empty, empty, f, jnp.array(jnp.inf))

    def condition(state):
        return (state[0] < capacity) & (state[-1] >= tol)

    def step(state):
        count, support, nodes, values, _, residual, _ = state
        selected = jnp.argmax(jnp.where(support, -jnp.inf, jnp.abs(residual)))
        support = support.at[selected].set(True)
        nodes = nodes.at[count].set(z[selected])
        values = values.at[count].set(f[selected])
        count = count + 1
        active = indices < count
        delta = z[:, None] - nodes[None, :]
        valid = (~support[:, None]) & active[None, :]
        cauchy = jnp.where(valid, 1 / jnp.where(valid, delta, 1), 0)
        loewner = cauchy * (f[:, None] - values[None, :])
        scale = jnp.maximum(jnp.linalg.norm(loewner), jnp.finfo(jnp.float64).tiny)
        padded = jnp.concatenate([loewner / scale, jnp.diag(jnp.where(active, 0., 2.))])

        def solve(_):
            _, _, vh = jnp.linalg.svd(padded, full_matrices=False)
            return jnp.where(active, vh[-1].conj(), 0)

        weights = jax.lax.cond(count == 1, lambda _: empty.at[0].set(1), solve, operand=None)
        numerator = cauchy @ (weights * values)
        denominator = cauchy @ weights
        safe = jnp.abs(denominator) > 1e-300
        prediction = jnp.where(safe, numerator / jnp.where(safe, denominator, 1), f)
        residual = jnp.where(support, 0, f - prediction)
        error = jnp.linalg.norm(residual) / norm
        return count, support, nodes, values, weights, residual, error

    count, _, nodes, values, weights, _, _ = jax.lax.while_loop(condition, step, initial)
    return AAAResult(weights, nodes, values, count)
