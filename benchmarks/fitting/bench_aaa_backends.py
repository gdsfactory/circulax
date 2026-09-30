"""Compare AAA discovery alone; no pole cleanup, VF, or passivity in timings.

Run: pixi run python -m benchmarks.fitting.bench_aaa_backends
"""

# CLI benchmark intentionally prints its measurements.
# ruff: noqa: T201

import time

import jax
import jax.numpy as jnp
import numpy as np
import skrf

from circulax.fitting.aaa_jax import AAAResult, aaa_scalar_jax
from circulax.fitting.aaa_numpy import aaa_scalar_numpy


def main() -> None:
    """Time both backends on the same ring-slot responses and holdout."""
    network = skrf.data.ring_slot
    train = np.arange(len(network.f)) % 5 != 0
    z = 2j * np.pi * network.f[train]
    holdout = 2j * np.pi * network.f[~train]
    # Same reciprocal responses and stopping criterion for both backends.
    responses = network.s[train][:, (0, 1, 1), (0, 0, 1)].T
    targets = network.s[~train][:, (0, 1, 1), (0, 0, 1)].T
    z_device = jnp.asarray(z)
    device_responses = jnp.asarray(responses)
    jax.block_until_ready((z_device, device_responses))

    def numpy_run() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return [aaa_scalar_numpy(f, z, tol=1e-8, mmax=12) for f in responses]

    # Sequential calls match NumPy; batching is tested separately below.
    def jax_run() -> list[AAAResult]:
        return [aaa_scalar_jax(f, z_device, tol=1e-8, mmax=12) for f in device_responses]

    batched = jax.jit(jax.vmap(lambda f: aaa_scalar_jax(f, z_device, tol=1e-8, mmax=12)))
    for name, run in (("numpy", numpy_run), ("jax", jax_run),
                      ("jax batched", lambda: batched(device_responses))):
        durations = []
        for _ in range(11):
            start = time.perf_counter()
            result = run()
            jax.block_until_ready(result)
            durations.append(time.perf_counter() - start)
        if name == "jax batched":
            result = [jax.tree.map(lambda x, index=i: x[index], result) for i in range(len(responses))]
        compact = result if name == "numpy" else [
            tuple(np.asarray(x)[:int(r.count)] for x in r[:3]) for r in result
        ]
        predictions = []
        for w, nodes, values in compact:
            basis = 1 / (holdout[:, None] - nodes)
            predictions.append((basis @ (w * values)) / (basis @ w))
        error = np.linalg.norm(np.asarray(predictions) - targets) / np.linalg.norm(targets)
        print(f"{name}: first={1e3 * durations[0]:.3f} ms, warm median={1e3 * np.median(durations[1:]):.3f} ms, "
              f"supports={[len(r[0]) for r in compact]}, holdout NRMSE={error:.3e}")
    print(f"JAX {jax.__version__}; NumPy {np.__version__}; scikit-rf {skrf.__version__}; {jax.devices()}")
    print("First timings are first-use in this process, not isolated cold-process comparisons.")
    print("Barycentric AAA accuracy does not certify a stable, passive circuit realization.")


if __name__ == "__main__":
    main()
