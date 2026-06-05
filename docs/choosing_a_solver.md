# Choosing a Solver

Circulax separates the **linear algebra backend** from the simulation algorithm.
The backend is selected when calling `compile_circuit`:

```python
circuit = compile_circuit(net_dict, models_map, backend="default")
```

Four backends are available. All expose the same interface — you can swap them without changing the rest of your simulation code.

---

## KLU Split Linear *(default)*

```python
circuit = compile_circuit(net_dict, models_map, backend="klu_split_linear")
```

Separates symbolic analysis (sparsity pattern, done once) from numeric factorisation (done once per transient step or Newton setup) and reuses the factorisation inside the frozen-Jacobian iteration. This is predictable for linear and mildly nonlinear circuits and is the default behind `backend="default"`.

**When to use:** Any circuit on CPU. The right default for almost all simulations.

**Trade-offs:** CPU-only (`klujax`). GPU falls back to Dense.

---

## KLU Split Refactor

```python
backend="klu_split"
```

Refactor-capable split KLU. It reuses symbolic analysis but can refresh the numeric factorisation during nonlinear iterations when the installed `klujax` backend supports it. `backend="klu_split_refactor"` is an explicit alias for this policy.

**When to use:** Strongly nonlinear circuits where full Newton convergence is more important than frozen-factor reuse.

**Trade-offs:** CPU-only (`klujax`) and may fall back to the linear split implementation if refactor support is unavailable.

---

## KLU

```python
backend="klu"
```

Non-split KLU: performs symbolic + numeric factorisation together on every Newton step. Slightly simpler code path but slower than split KLU for circuits requiring many Newton iterations.

**When to use:** Fallback if `klu_split` causes issues. Functionally identical results.

**Trade-offs:** Same as `klu_split`, but repeats symbolic analysis unnecessarily.

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

Memory scales as $O(N^2)$ and runtime as $O(N^3)$, so it becomes impractical quickly. For any real circuit, split KLU will outperform `dense`.

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
| `klu_split_linear` | All circuits on CPU — **default** | No | `klujax >= 0.5.0` |
| `klu_split` | Refactor-capable nonlinear solves | No | `klujax >= 0.5.0` |
| `klu` | Fallback (non-split) | No | `klujax` |
| `dense` | $N \lesssim 50$, GPU, HB frequency sweeps | Yes | — |
| `sparse` | Large $N$, GPU/TPU transient | Yes | — |

Start with `backend="default"` or `backend="klu_split_linear"`.
