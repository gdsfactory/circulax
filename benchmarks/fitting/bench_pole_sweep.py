"""Benchmark the vmapped fixed-pole screening stage on the active two-port."""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import skrf

from circulax.fitting import FitOptions, vmap_pole_count_sweep
from circulax.fitting.sparam import _aaa_all_elements


def main() -> None:
    """Run the compiled and warm vmapped screening benchmark."""
    data = Path(__file__).parents[2] / "examples/fitting/data/190ghz_tx_measured.s2p"
    network = skrf.Network(data)
    sample_points = jnp.asarray(1j * 2 * np.pi * network.f)
    matrix = jnp.asarray(np.moveaxis(network.s, 0, -1))
    model, *_ = _aaa_all_elements(
        matrix,
        sample_points,
        FitOptions(N=0, asymp=2, weightparam=1),
        tol=1e-8,
        mmax=40,
        reciprocal=False,
        pole_selection="largest_response",
        verbose=False,
    )
    counts = np.arange(10, 32, 2)

    started = time.perf_counter()
    first = vmap_pole_count_sweep(model, network.f, network.s, counts)
    jax.block_until_ready(first.normalized_rmse)
    compile_and_run = time.perf_counter() - started

    started = time.perf_counter()
    second = vmap_pole_count_sweep(model, network.f, network.s, counts)
    jax.block_until_ready(second.normalized_rmse)
    warm = time.perf_counter() - started

    single = vmap_pole_count_sweep(model, network.f, network.s, [counts[0]])
    jax.block_until_ready(single.normalized_rmse)
    started = time.perf_counter()
    for count in counts:
        single = vmap_pole_count_sweep(model, network.f, network.s, [count])
        jax.block_until_ready(single.normalized_rmse)
    sequential = time.perf_counter() - started

    print(f"compile + run: {compile_and_run:.3f} s")  # noqa: T201
    print(f"warm run:      {warm:.3f} s")  # noqa: T201
    print(f"warm loop:     {sequential:.3f} s")  # noqa: T201
    for count, error, maximum, y_pole, condition in zip(
        counts,
        np.asarray(second.normalized_rmse),
        np.asarray(second.max_absolute_error),
        np.asarray(second.maximum_admittance_pole_real_part),
        np.asarray(second.transform_condition),
        strict=True,
    ):
        print(  # noqa: T201
            f"{count:2d} poles: NRMSE={error:.4%}, max |dS|={maximum:.4f}, "
            f"max Re(pY)={y_pole:.3e}, cond(V)={condition:.3e}"
        )


if __name__ == "__main__":
    main()
