# Copyright (c) 2026 Circulax contributors
# ruff: noqa: INP001
"""Measure native sample component tracing, compilation and warm evaluation.

Example::

    python -m benchmarks.sample_components circulax.components.bipolar BJT_PNP_Dynamic \
        --bias '[-2, -0.6, 0]' --settings '{"Tf": 1e-9}'

Run each component in a fresh process for comparable cold timings. Parameters
are passed as JAX leaves rather than closed-over constants, matching batched
circuit evaluation. This is a component benchmark, not a Newton-solver benchmark.
For optical models pass --complex-state and optionally --imag-bias: both input
quadratures and the full real-block Jacobian are then retained.
"""

import argparse
import importlib
import json
import platform
import time
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


def measure(
    module: str,
    name: str,
    settings: dict[str, Any],
    bias: list[float] | None,
    repeats: int,
    *,
    complex_state: bool = False,
    imag_bias: list[float] | None = None,
) -> dict[str, Any]:
    """Measure residual/charge and voltage-Jacobian evaluation separately."""
    cls = getattr(importlib.import_module(module), name)
    instance = cls(**settings)
    size = len(cls.ports) + len(cls.states)
    y = jnp.asarray(bias if bias is not None else [0.0] * size, dtype=jnp.float64)
    if y.shape != (size,):
        msg = f"Expected {size} values for {cls.ports + cls.states}; got shape {y.shape}"
        raise ValueError(msg)
    if imag_bias is not None and not complex_state:
        msg = "--imag-bias requires --complex-state"
        raise ValueError(msg)
    imaginary = jnp.asarray(imag_bias if imag_bias is not None else [0.0] * size, dtype=jnp.float64)
    if imaginary.shape != (size,):
        msg = f"Expected {size} imaginary bias values"
        raise ValueError(msg)
    if complex_state:
        y = jnp.concatenate([y, imaginary])
    params, static = eqx.partition(instance, eqx.is_inexact_array_like)

    def evaluate(v: jax.Array, p: Any) -> tuple[jax.Array, jax.Array]:
        physical = v[:size] + 1j * v[size:] if complex_state else v
        f, q = cls.solver_call(0.0, physical, eqx.combine(p, static))
        if complex_state:
            return jnp.concatenate([f.real, f.imag]), jnp.concatenate([q.real, q.imag])
        return f, q

    measurements = {}
    for label, fn in (("fq", evaluate), ("jacobian", jax.jacfwd(evaluate, argnums=0))):
        start = time.perf_counter()
        lowered = jax.jit(fn).lower(y, params)
        traced = time.perf_counter()
        compiled = lowered.compile()
        built = time.perf_counter()
        result = jax.block_until_ready(compiled(y, params))
        if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(result)):
            msg = f"{name} returned nonfinite {label} at the supplied bias"
            raise ValueError(msg)
        start_warm = time.perf_counter()
        for _ in range(repeats):
            jax.block_until_ready(compiled(y, params))
        measurements[label] = {
            "trace_seconds": traced - start,
            "compile_seconds": built - traced,
            "warm_seconds": (time.perf_counter() - start_warm) / repeats,
        }
    return {
        "component": f"{module}.{name}",
        "jax": jax.__version__,
        "python": platform.python_version(),
        "backend": jax.default_backend(),
        "variables": size,
        "real_unknowns": int(y.size),
        "complex_state": complex_state,
        "settings": settings,
        "bias": y.tolist(),
        "repeats": repeats,
        **measurements,
    }


def main() -> None:
    """Print one JSON result suitable for recording benchmark runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module")
    parser.add_argument("component")
    parser.add_argument("--settings", default="{}", type=json.loads)
    parser.add_argument("--bias", type=json.loads)
    parser.add_argument("--repeats", default=100, type=int)
    parser.add_argument("--complex-state", action="store_true", help="Benchmark full real-block F/Q and Jacobian")
    parser.add_argument("--imag-bias", type=json.loads, help="Imaginary physical state values; real values use --bias")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    jax.config.update("jax_enable_x64", True)  # noqa: FBT003
    result = measure(
        args.module,
        args.component,
        args.settings,
        args.bias,
        args.repeats,
        complex_state=args.complex_state,
        imag_bias=args.imag_bias,
    )
    print(json.dumps(result, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
