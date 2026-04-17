# Choosing a Solver

Circulax separates the **linear algebra backend** from the simulation algorithm.
The backend is selected when calling `compile_circuit`:

```python
circuit = compile_circuit(net_dict, models_map, backend="klu")
```

Four backends are available. All expose the same interface — you can swap them without changing the rest of your simulation code.

---

## KLU *(default)*

```python
circuit = compile_circuit(net_dict, models_map, backend="klu")
```

The same sparse direct solver (AMD ordering + LU) used in SPICE, Spectre, and HSPICE.

**When to use:** Any circuit on CPU. The right default for almost all simulations.

**Trade-offs:** CPU-only (`klujax`). GPU falls back to Dense.

---

## KLU Split *(experimental)*

```python
backend="klu_split"
```

Separates symbolic analysis (sparsity pattern, done once) from numeric factorisation (done each Newton step). Avoids repeating the symbolic phase, giving a speedup when Newton needs many iterations.

!!! warning "Experimental"
    Requires a custom `klujax` build exposing the split symbolic/numeric API (available in main branch). Falls back silently to `klu` if not present.

**When to use:** Same cases as `klu`, but with many Newton iterations per timestep (strongly nonlinear devices).

**Trade-offs:** Same as `klu`. Requires the extended `klujax` build.

---

## Dense

```python
backend="dense"
```

Assembles the full $N \times N$ Jacobian as a dense array and solves with JAX's built-in LU factorisation (`jnp.linalg.solve`).

**When to use:**

- Very small circuits ($N \lesssim 50$ nodes) where sparse overhead dominates.
- GPU execution: dense BLAS routines are highly optimised on GPU.
- Frequency sweeps with Harmonic Balance where `jax.vmap` over the full solve is desired.

**Trade-offs:**

Memory scales as $O(N^2)$ and runtime as $O(N^3)$, so it becomes impractical quickly. For any real circuit, `klu` will outperform `dense`.

---

## Sparse

```python
backend="sparse"
```

Iterative BiCGStab in pure JAX. Sparsity pattern is pre-computed at setup; only non-zero values are stored. Runs natively on GPU and TPU.

**When to use:** Large circuits ($N \gtrsim 2000$) on GPU/TPU where Dense runs out of memory.

**Trade-offs:** Can fail to converge on ill-conditioned systems. Not recommended for DC solves of strongly nonlinear circuits.

---

## Summary

| Backend | Best for | GPU | Requires |
|---------|----------|-----|---------|
| `klu` | All circuits on CPU — **default** | No | `klujax` |
| `klu_split` | CPU, many Newton steps per timestep | No | custom `klujax` build |
| `dense` | $N \lesssim 50$, GPU, HB frequency sweeps | Yes | — |
| `sparse` | Large $N$, GPU/TPU transient | Yes | — |

Start with `klu`.
