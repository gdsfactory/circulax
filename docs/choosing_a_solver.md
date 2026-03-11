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
- GPU execution: dense BLAS routines are highly optimised on GPU.
- Frequency sweeps with Harmonic Balance where `jax.vmap` over the full solve is desired.

**Trade-offs:**

Memory scales as $O(N^2)$, so large circuits become expensive.

---

## Sparse

```python
backend="sparse"
```

Solves the sparse system iteratively using JAX's BiCGStab implementation. The sparsity pattern is pre-computed at setup time; only non-zero values are stored. Because BiCGStab is implemented entirely in JAX, it runs natively on GPU and TPU.

**When to use:**

- Large circuits ($N \gtrsim 2000$) on GPU or TPU where Dense runs out of memory.
- Large transient simulations where the sparsity of the circuit can be exploited for speed.

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

CPU-only external library. GPU execution falls back to the Dense path.

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

Same limitations as `klu`. Requires the extended `klujax` build.

---

## Summary

| Backend | Best for | GPU | Requires |
|---------|----------|-----|---------|
| `dense` | $N < 2000$, GPU, HB frequency sweeps | Yes | — |
| `sparse` | Large $N$, GPU transient | Yes | — |
| `klu` | Large $N$, CPU DC/transient | No | `klujax` |
| `klu_split` | Large $N$, CPU, many Newton steps | No | custom `klujax` build |

If you are unsure, start with `dense`. Switch to `klu` only when memory or runtime becomes a bottleneck on CPU.
