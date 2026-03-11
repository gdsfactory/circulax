"""Tests for the Harmonic Balance solver: correctness, JIT, vmap, and grad."""

import jax
import jax.numpy as jnp
import pytest

from circulax.compiler import compile_netlist
from circulax.solvers import setup_harmonic_balance
from circulax.solvers.linear import analyze_circuit, backends

# One canonical backend name per unique solver implementation
_seen_classes: set = set()
_backends: list[str] = []
for _name, _cls in backends.items():
    if _name == "default":
        continue
    if _cls not in _seen_classes:
        _seen_classes.add(_cls)
        _backends.append(_name)

_NUM_HARMONICS = 3
_FREQ = 1e6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_real(simple_lrc_netlist, backend):
    net_dict, models_map = simple_lrc_netlist
    groups, sys_size, _ = compile_netlist(net_dict, models_map)
    linear_strat = analyze_circuit(groups, sys_size, backend=backend, is_complex=False)
    y_dc = linear_strat.solve_dc(groups, jnp.zeros(sys_size))
    run_hb = setup_harmonic_balance(groups, sys_size, freq=_FREQ, num_harmonics=_NUM_HARMONICS)
    return run_hb, y_dc, sys_size


def _setup_complex(simple_optical_netlist, backend):
    net_dict, models_map = simple_optical_netlist
    groups, sys_size, _ = compile_netlist(net_dict, models_map)
    linear_strat = analyze_circuit(groups, sys_size, backend=backend, is_complex=True)
    y_dc = linear_strat.solve_dc(groups, jnp.zeros(2 * sys_size))
    run_hb = setup_harmonic_balance(groups, sys_size, freq=_FREQ, num_harmonics=_NUM_HARMONICS, is_complex=True)
    return run_hb, y_dc, sys_size


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", _backends)
def test_hb_basic_real(simple_lrc_netlist, backend):
    """HB returns finite arrays with correct shapes for a real circuit."""
    run_hb, y_dc, sys_size = _setup_real(simple_lrc_netlist, backend)
    y_time, y_freq = run_hb(y_dc)

    K = 2 * _NUM_HARMONICS + 1
    assert y_time.shape == (K, sys_size)
    assert y_freq.shape == (_NUM_HARMONICS + 1, sys_size)
    assert jnp.isfinite(y_time).all()
    assert jnp.isfinite(jnp.abs(y_freq)).all()


@pytest.mark.parametrize("backend", _backends)
def test_hb_basic_complex(simple_optical_netlist, backend):
    """HB returns finite arrays with correct shapes for a complex circuit."""
    run_hb, y_dc, sys_size = _setup_complex(simple_optical_netlist, backend)
    y_time, y_freq = run_hb(y_dc)

    K = 2 * _NUM_HARMONICS + 1
    assert y_time.shape == (K, 2 * sys_size)
    assert y_freq.shape == (_NUM_HARMONICS + 1, 2 * sys_size)
    assert jnp.isfinite(y_time).all()
    assert jnp.isfinite(jnp.abs(y_freq)).all()


# ---------------------------------------------------------------------------
# JIT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", _backends)
def test_hb_jit_real(simple_lrc_netlist, backend):
    """jax.jit(run_hb) produces the same result as the eager call."""
    run_hb, y_dc, _ = _setup_real(simple_lrc_netlist, backend)

    y_time_ref, y_freq_ref = run_hb(y_dc)
    y_time_jit, y_freq_jit = jax.jit(run_hb)(y_dc)

    assert jnp.allclose(y_time_jit, y_time_ref, atol=1e-10)
    assert jnp.allclose(jnp.abs(y_freq_jit), jnp.abs(y_freq_ref), atol=1e-10)


@pytest.mark.parametrize("backend", _backends)
def test_hb_jit_complex(simple_optical_netlist, backend):
    """jax.jit(run_hb) works for the complex circuit path."""
    run_hb, y_dc, _ = _setup_complex(simple_optical_netlist, backend)

    y_time_ref, _ = run_hb(y_dc)
    y_time_jit, _ = jax.jit(run_hb)(y_dc)

    assert jnp.allclose(y_time_jit, y_time_ref, atol=1e-10)


# ---------------------------------------------------------------------------
# vmap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", _backends)
def test_hb_vmap_real(simple_lrc_netlist, backend):
    """jax.vmap(run_hb) batches over a stack of DC operating points."""
    run_hb, y_dc, sys_size = _setup_real(simple_lrc_netlist, backend)

    batch_size = 2
    y_batch = jnp.stack([y_dc] * batch_size)  # (batch_size, sys_size)
    y_time_batch, y_freq_batch = jax.vmap(run_hb)(y_batch)

    K = 2 * _NUM_HARMONICS + 1
    assert y_time_batch.shape == (batch_size, K, sys_size)
    assert y_freq_batch.shape == (batch_size, _NUM_HARMONICS + 1, sys_size)
    assert jnp.isfinite(y_time_batch).all()


@pytest.mark.parametrize("backend", _backends)
def test_hb_vmap_complex(simple_optical_netlist, backend):
    """jax.vmap(run_hb) batches over a stack of DC points for complex circuits."""
    run_hb, y_dc, sys_size = _setup_complex(simple_optical_netlist, backend)

    batch_size = 2
    y_batch = jnp.stack([y_dc] * batch_size)  # (batch_size, 2*sys_size)
    y_time_batch, _ = jax.vmap(run_hb)(y_batch)

    K = 2 * _NUM_HARMONICS + 1
    assert y_time_batch.shape == (batch_size, K, 2 * sys_size)
    assert jnp.isfinite(y_time_batch).all()


# ---------------------------------------------------------------------------
# grad
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", _backends)
def test_hb_grad_real(simple_lrc_netlist, backend):
    """jax.grad differentiates a scalar loss through run_hb w.r.t. y_dc."""
    run_hb, y_dc, _ = _setup_real(simple_lrc_netlist, backend)

    def loss(y):
        y_time, _ = run_hb(y)
        return jnp.sum(y_time**2)

    grad = jax.grad(loss)(y_dc)
    assert grad.shape == y_dc.shape
    assert jnp.isfinite(grad).all()


@pytest.mark.parametrize("backend", _backends)
def test_hb_grad_complex(simple_optical_netlist, backend):
    """jax.grad differentiates a scalar loss through run_hb for complex circuits."""
    run_hb, y_dc, _ = _setup_complex(simple_optical_netlist, backend)

    def loss(y):
        y_time, _ = run_hb(y)
        return jnp.sum(y_time**2)

    grad = jax.grad(loss)(y_dc)
    assert grad.shape == y_dc.shape
    assert jnp.isfinite(grad).all()
