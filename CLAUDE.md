# CLAUDE.md

## Project Overview

Circulax is a differentiable, JAX-based circuit simulator supporting both electronic and photonic circuits. It formulates netlists as DAE (Differential Algebraic Equation) systems — F(y) + dQ/dt = 0 — and uses automatic differentiation for gradient-based optimization and inverse design.

## Commands

This project uses [Pixi](https://pixi.sh) as its package/task manager (conda-based).

```bash
# Testing
pixi run pytest_run          # Run the test suite

# Linting & formatting
pixi run lint_format         # Format with ruff
pixi run lint_check          # Lint with ruff

# Documentation
pixi run docs-serve          # Serve MkDocs docs locally

# Building
pixi run build               # Build sdist and wheel
```

To run a single test file or test: `pixi run python -m pytest tests/test_compiler.py -k "test_name"`

## Architecture

Circulax has a strict three-layer design separating **Physics**, **Topology**, and **Analysis**.

### Layer 1: Physics — `circulax/components/`

Components are plain Python functions decorated with `@component` or `@source` (from `base_component.py`). The decorators generate `CircuitComponent` subclasses (Equinox modules).

Functions receive:
- `signals`: Port voltages (dynamically constructed namedtuple)
- `s`: Internal state variables (namedtuple)
- `t`: Time (only for `@source`-decorated components)
- `**params`: Physical parameters (R, C, L, etc.)

Functions return `(f_dict, q_dict)`:
- `f_dict`: Flow/current equations mapped to ports
- `q_dict`: Storage terms (charge, flux) — these become dQ/dt in the DAE

Each component has two call interfaces:
- `__call__()`: Debug-friendly, accepts keyword arguments
- `solver_call()`: Flat array interface used during simulation (compatible with `jax.vmap`, `jax.jacfwd`)

### Layer 2: Topology — `circulax/compiler.py` and `circulax/netlist.py`

`compile_netlist(net_dict, models)` processes a SAX-format netlist dict:
1. `build_net_map` in `netlist.py`: Assigns unique node IDs to port strings (e.g., `"R1,p1"`) using union-find
2. Groups component instances by type (all Resistors together, all Capacitors, etc.)
3. Creates `ComponentGroup` objects (Equinox modules) with batched parameters and pre-computed indices

`ComponentGroup` is the core data structure passed to solvers:
- `params`: Batched JAX arrays of component parameters (shape: N × param_dims)
- `var_indices`: Indices into global state vector for each instance's ports/states
- `eq_indices`: Where to scatter each instance's residual into the global residual
- `jac_rows/jac_cols`: Pre-computed Jacobian sparsity pattern for sparse assembly

### Layer 3: Analysis — `circulax/solvers/`

**`linear.py`** — Strategy pattern for linear algebra:
- `DenseSolver`: JAX LU (`jnp.linalg.solve`), best for small circuits or GPU
- `KLUSplitLinear`/`KLUSplitQuadratic`: Sparse KLU with split symbolic/numeric via `klujax >= 5.0` (default)
- `KLUSolver`: Non-split KLU fallback via `klujax`
- `SparseSolver`: Iterative BiCGStab, best for transient on GPU
- All expose `solve_dc()` which runs Newton-Raphson (via Optimistix) to find DC operating point

**`assembly.py`** — Residual and Jacobian assembly:
- `assemble_system_real/complex`: Evaluates F(y) and J(y) via `jax.jacfwd` over vmapped component groups
- `assemble_residual_only_real/complex`: Residual-only variant (faster, for Newton inner loops)
- Complex photonic circuits use a 2N×2N block notation: `[RR RI; IR II]`

**`transient.py`** — Time-domain simulation:
- `VectorizedTransientSolver`: Wraps Diffrax's implicit ODE solver
- Implements Backward Euler time-stepping with Newton iterations per step
- The `setup_transient()` function (exported from `__init__.py`) returns a Diffrax `ODETerm`

### Netlist Format

Netlists follow the [SAX](https://flaport.github.io/sax/) format:

```python
{
    "instances": {
        "R1": {"component": "resistor", "settings": {"R": 100.0}},
    },
    "connections": {
        "R1,p1": "C1,p1",  # Short-hand for connecting ports
        "R1,p2": "GND,p1",
    },
    "ports": {"in": "R1,p1"},
}
```

`GND` is a special reserved instance name. Node IDs are assigned by `build_net_map` using union-find on the connections graph.

### Simulation Flow

```
compile_netlist(net_dict, models) → [ComponentGroup, ...]
    → analyze_circuit(groups) → CircuitLinearSolver
        → solve_dc()               # Newton-Raphson to DC operating point
        → setup_transient()        # Returns Diffrax ODETerm for time integration
        → setup_harmonic_balance() # Periodic steady-state via Harmonic Balance
```

### Photonic Circuits

Optical components in `photonic.py` are defined via S-parameters (scattering matrices). `s_transforms.py` converts these to Y-matrices (`s_to_y()`) for nodal analysis. Complex-valued fields are supported throughout — assembly handles real and complex paths separately.

## Key Files

| File | Role |
|------|------|
| `circulax/compiler.py` | Netlist→ComponentGroup compilation pipeline |
| `circulax/components/base_component.py` | `@component`/`@source` decorators and `CircuitComponent` base |
| `circulax/solvers/assembly.py` | Residual and Jacobian assembly (real & complex) |
| `circulax/solvers/linear.py` | Linear solver strategies and DC operating point |
| `circulax/solvers/transient.py` | Diffrax-compatible transient ODE solver |
| `circulax/solvers/harmonic_balance.py` | Harmonic Balance periodic steady-state solver |
| `circulax/s_transforms.py` | S-parameter ↔ Y-matrix conversion |
| `tests/conftest.py` | Shared test fixtures (LRC and optical netlists) |

## Code Style

- Ruff is used for linting/formatting with line-length 132
- Pre-commit hooks strip Jupyter notebook outputs before commits
- Components should return JAX-compatible (jittable, vmappable) computations — avoid Python control flow that depends on traced values
