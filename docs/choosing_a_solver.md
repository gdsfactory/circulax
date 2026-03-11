# Choosing a Solver

Circulax separates the **linear algebra backend** from the simulation algorithm.
The backend is selected when calling `analyze_circuit`:

```python
linear_strat = analyze_circuit(groups, num_vars, backend="dense")
```

Four backends are available. All expose the same interface — you can swap them without changing the rest of your simulation code.

---

## Dense

```python
backend="dense"
```

Assembles the full $N \times N$ Jacobian as a dense array and solves with JAX's built-in LU factorisation (`jnp.linalg.solve`).

**When to use:**

- Small to medium circuits ($N \lesssim 2000$ nodes).
- Any workflow that uses `jax.vmap` or `jax.grad` through the linear solve — e.g., frequency sweeps with Harmonic Balance or inverse design.
- GPU execution: dense BLAS routines are highly optimised on GPU.

**Trade-offs:**

Memory scales as $O(N^2)$, so large circuits become expensive. The solve is fully differentiable and vmap-compatible.

---

## Sparse

```python
backend="sparse"
```

Solves the sparse system iteratively using JAX's BiCGStab implementation. The sparsity pattern is pre-computed at setup time; only non-zero values are stored.

**When to use:**

- Large transient simulations on GPU where $N$ is too large for Dense.
- Situations where vmap support is still required at large $N$.

**Trade-offs:**

Iterative solvers can fail to converge for ill-conditioned systems. Convergence depends on problem conditioning and the diagonal preconditioner used internally. Not recommended for DC operating-point solves of strongly nonlinear circuits.

---

## KLU

```python
backend="klu"
```

Calls the [KLU](https://github.com/flaport/klujax) direct sparse solver via `klujax`. KLU performs full symbolic and numeric factorisation on every Newton step.

**When to use:**

- Large circuits ($N \gtrsim 5000$) on CPU.
- DC operating-point solves of large meshes.
- Cases where Dense runs out of memory.

**Trade-offs:**

Does not support `jax.vmap` or `jax.grad` through the linear solve. KLU is a CPU-only external library — GPU execution falls back to the Dense path.

---

## KLU Split *(experimental)*

```python
backend="klu_split"
```

An extended KLU interface that separates symbolic analysis (sparsity pattern, done once at setup) from numeric factorisation (done each Newton step). This avoids repeating the costly symbolic phase on every iteration, which gives a measurable speedup for circuits with many Newton steps.

!!! warning "Experimental"
    `klu_split` requires a custom build of `klujax` that exposes the split symbolic/numeric API. It is not available in the standard `klujax` release. Falls back silently to `klu` if the split interface is not present.

**When to use:**

- Large circuits on CPU where `klu` is already the right choice but Newton convergence requires many iterations.
- Circuits with strongly nonlinear devices (diodes, transistors) where the symbolic sparsity pattern is fixed but numeric values change every step.

**Trade-offs:**

Same vmap/grad limitations as `klu`. Requires the extended `klujax` build.

---

## Summary

| Backend | Best for | vmap / grad | GPU | Requires |
|---------|----------|-------------|-----|---------|
| `dense` | $N < 2000$, sweeps, inverse design | Yes | Yes | — |
| `sparse` | Large $N$, GPU transient | Yes | Yes | — |
| `klu` | Large $N$, CPU DC/transient | No | No | `klujax` |
| `klu_split` | Large $N$, CPU, many Newton steps | No | No | custom `klujax` build |

If you are unsure, start with `dense`. Switch to `klu` only when memory or runtime becomes a bottleneck on CPU.
