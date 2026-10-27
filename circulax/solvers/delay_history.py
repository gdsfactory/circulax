"""Chronological indexing of bounded, overwritable accepted-step history."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


class DelayTimes(eqx.Module):
    """Ring timestamps with a logical ascending view, without rotating storage."""

    times: jax.Array
    start: jax.Array
    count: jax.Array

    def physical(self, logical: ArrayLike) -> jax.Array:
        """Map chronological indices to physical storage slots."""
        return (self.start + logical) % self.times.shape[0]

    def time(self, logical: ArrayLike) -> jax.Array:
        """Gather chronological timestamps without rotating storage."""
        return self.times[self.physical(logical)]

    def search(self, query: jax.Array) -> jax.Array:
        """Right insertion point in the valid chronological prefix; O(log capacity)."""

        def body(_: int, bounds: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            lo, hi = bounds
            mid = (lo + hi) // 2
            right = (mid < self.count) & (self.time(mid) <= query)
            return jnp.where(right, mid + 1, lo), jnp.where(right, hi, mid)

        lo, _ = jax.lax.fori_loop(
            0,
            self.times.shape[0].bit_length(),
            body,
            (jnp.zeros_like(query, dtype=jnp.int32), jnp.full_like(query, self.count, dtype=jnp.int32)),
        )
        return lo
