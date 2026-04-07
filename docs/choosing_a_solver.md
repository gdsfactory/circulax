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

KLU is the **gold standard for circuit simulation** and is the default backend. It is the same direct sparse solver used by industry-standard tools such as SPICE, Spectre, and HSPICE. KLU performs a fill-reducing permutation (AMD ordering) followed by symbolic and numeric LU factorisation, making it highly efficient for the sparse, irregular matrices that arise in circuit simulation.

**When to use:**

- Any circuit on CPU — this is the right choice for the vast majority of simulations.
- DC operating-point solves, including strongly nonlinear circuits (diodes, transistors, MOSFETs).
- Transient simulation of circuits of any size.

**Trade-offs:**

CPU-only external library (`klujax`). GPU execution falls back to the Dense path.

---

## KLU Split *(experimental)*

```python
backend="klu_split"
```

An extended KLU interface that separates symbolic analysis (sparsity pattern, done once at setup) from numeric factorisation (done each Newton step). This avoids repeating the costly symbolic phase on every iteration, giving a measurable speedup for circuits where Newton convergence requires many steps.

!!! warning "Experimental"
    `klu_split` requires a custom build of `klujax` that exposes the split symbolic/numeric API. It is not available in the standard `klujax` release but is available in the main branch. Falls back silently to `klu` if the split interface is not present.

**When to use:**

- Circuits where `klu` is already the right choice but Newton convergence requires many iterations.
- Circuits with strongly nonlinear devices (diodes, transistors) where the symbolic sparsity pattern is fixed but numeric values change every step.

**Trade-offs:**

Same limitations as `klu`. Requires the extended `klujax` build.

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

Solves the sparse system iteratively using JAX's BiCGStab implementation. The sparsity pattern is pre-computed at setup time; only non-zero values are stored. Because BiCGStab is implemented entirely in JAX, it runs natively on GPU and TPU.

**When to use:**

- Large circuits ($N \gtrsim 2000$) on GPU or TPU where Dense runs out of memory.
- Large transient simulations where the sparsity of the circuit can be exploited for speed.

**Trade-offs:**

Iterative solvers can fail to converge for ill-conditioned systems. Convergence depends on problem conditioning and the diagonal preconditioner used internally. Not recommended for DC operating-point solves of strongly nonlinear circuits.

---

## Summary

| Backend | Best for | GPU | Requires |
|---------|----------|-----|---------|
| `klu` | All circuits on CPU — **default** | No | `klujax` |
| `klu_split` | CPU, many Newton steps per timestep | No | custom `klujax` build |
| `dense` | $N \lesssim 50$, GPU, HB frequency sweeps | Yes | — |
| `sparse` | Large $N$, GPU/TPU transient | Yes | — |

Start with `klu`. It is the industry-standard sparse solver for circuit simulation and the right choice for the overwhelming majority of cases.
