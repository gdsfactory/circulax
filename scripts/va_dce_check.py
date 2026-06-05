"""Check whether XLA dead-code-eliminates Jacobian ops from the VA primal path.

The VA component's fast_physics function (when combined_fn is present) calls
_fast_combined which returns (f, q, j_f, j_q), then discards j_f and j_q.
This script measures whether XLA's DCE actually removes the Jacobian ops.

Strategy
--------
Instead of compiling PSP103 (~320 s JIT), we build a *synthetic* VA component
that has a nontrivial combined_fn:  a fake MOSFET-like physics with ~10 ops in
the primal path and ~40 ops in the Jacobian path (outer product of gradients).
We then lower both:

    a) primal_only  — fast_physics(y, params, t) -> (f, q)
    b) combined     — _fast_combined(y, params, t) -> (f, q, j_f, j_q)

and compare HLO text line counts.  If DCE works, (a) has ~50 % the ops of (b).
If DCE fails, (a) ≈ (b) in size (paying full combined cost for every residual
evaluation).

Usage
-----
    pixi run python scripts/va_dce_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Make sure circulax and bosdi are importable
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, "/home/cdaunt/code/bosdi/src")

from bosdi.circulax.va_component import va_component

# ---------------------------------------------------------------------------
# Synthetic "MOSFET-like" physics body
# ---------------------------------------------------------------------------
# We deliberately use transcendental ops (exp, tanh, log1p) so the HLO is
# large enough that the primal vs combined difference is clearly visible.
# The Jacobian body adds a full 2×2 outer-product (4 extra arrays).

PORTS = ("D", "G", "S", "B")
N_VARS = len(PORTS)   # no internal states for simplicity


def _physics_body(D, G, S, B, *, VTH0, KP, W, L):
    """Square-law MOSFET with saturation smoothing (nontrivial ops)."""
    vgs = G - S
    vds = D - S
    vbs = B - S
    vth = VTH0 - 0.1 * jnp.tanh(vbs)         # body effect (tanh)
    vov = vgs - vth
    vov_pos = jnp.where(vov > 0.0, vov, 0.0)
    beta = KP * W / L
    ids_lin = beta * (vov_pos * vds - 0.5 * vds ** 2)
    ids_sat = 0.5 * beta * vov_pos ** 2
    # Smoothly transition lin→sat using tanh of (vds - vov/2)
    t = jnp.tanh(10.0 * (vds - 0.5 * vov_pos))
    ids = 0.5 * (1.0 - t) * ids_lin + 0.5 * (1.0 + t) * ids_sat
    # Subthreshold leakage (exp)
    ids_leak = 1e-15 * jnp.exp(jnp.clip(vov, -20.0, 20.0) / 0.026)
    ids_total = ids + ids_leak
    # Gate capacitance (log1p for smoothness)
    q_g = 1e-15 * W * L * jnp.log1p(jnp.abs(vov))
    f = {"D": -ids_total, "G": 0.0, "S": ids_total, "B": 0.0}
    q = {"D": 0.0, "G": q_g, "S": -q_g, "B": 0.0}
    return f, q


def _jacobian_body(D, G, S, B, *, VTH0, KP, W, L):
    """Analytical Jacobian via jax.jacfwd on the physics body.

    Returns (j_f, j_q) arrays of shape (N_VARS, N_VARS) each.
    In a real VA-lowered component this comes from OpenVAF's mir_autodiff;
    here we use JAX AD to produce the same structure.
    """
    y = jnp.array([D, G, S, B])

    def _f_only(yy):
        vv = yy
        f_dict, _ = _physics_body(vv[0], vv[1], vv[2], vv[3],
                                   VTH0=VTH0, KP=KP, W=W, L=L)
        return jnp.array([f_dict.get(p, 0.0) for p in PORTS])

    def _q_only(yy):
        vv = yy
        _, q_dict = _physics_body(vv[0], vv[1], vv[2], vv[3],
                                   VTH0=VTH0, KP=KP, W=W, L=L)
        return jnp.array([q_dict.get(p, 0.0) for p in PORTS])

    j_f = jax.jacfwd(_f_only)(y)
    j_q = jax.jacfwd(_q_only)(y)
    return j_f, j_q


def _combined_body(D, G, S, B, *, VTH0, KP, W, L):
    """Return (f_dict, q_dict, j_f, j_q) from one shared evaluation."""
    f_dict, q_dict = _physics_body(D, G, S, B, VTH0=VTH0, KP=KP, W=W, L=L)
    j_f, j_q = _jacobian_body(D, G, S, B, VTH0=VTH0, KP=KP, W=W, L=L)
    return f_dict, q_dict, j_f, j_q


# ---------------------------------------------------------------------------
# Build a va_component class with the combined_fn installed
# ---------------------------------------------------------------------------

def _user_fn(signals, s, VTH0=0.4, KP=270e-6, W=10e-6, L=1e-6):
    return _physics_body(signals.D, signals.G, signals.S, signals.B,
                         VTH0=VTH0, KP=KP, W=W, L=L)


def _jac_fn(signals, s, VTH0=0.4, KP=270e-6, W=10e-6, L=1e-6):
    return _jacobian_body(signals.D, signals.G, signals.S, signals.B,
                          VTH0=VTH0, KP=KP, W=W, L=L)


def _comb_fn(signals, s, VTH0=0.4, KP=270e-6, W=10e-6, L=1e-6):
    return _combined_body(signals.D, signals.G, signals.S, signals.B,
                          VTH0=VTH0, KP=KP, W=W, L=L)


SyntheticMOS = va_component(
    ports=PORTS,
    states=(),
    jacobian_fn=_jac_fn,
    combined_fn=_comb_fn,
    differentiable_params=None,   # all params as JAX leaves so tracing is real
)(_user_fn)


# ---------------------------------------------------------------------------
# Instantiate and extract fast_physics / _combined_fn
# ---------------------------------------------------------------------------

mos = SyntheticMOS(VTH0=0.4, KP=270e-6, W=10e-6, L=1e-6)

# Sample inputs matching solver_call signature: (vars_vec, params, t)
y_sample = jnp.array([1.0, 0.8, 0.0, 0.0])   # [D, G, S, B]

# fast_physics is a static method on the class (replaced by _install_custom_jvp).
fast_physics = SyntheticMOS._fast_physics
combined_fn  = SyntheticMOS._combined_fn

print("=" * 60)
print("VA DCE CHECK")
print("=" * 60)
print(f"Component: {SyntheticMOS.__name__}")
print(f"Ports: {SyntheticMOS.ports}")
print(f"Has combined_fn: {SyntheticMOS._has_combined_fn}")
print()

# ---------------------------------------------------------------------------
# Lower both functions to HLO and measure op counts
# ---------------------------------------------------------------------------

def count_hlo_ops(hlo_text: str) -> dict[str, int]:
    """Return simple metrics from HLO text."""
    lines = hlo_text.splitlines()
    # Count non-empty, non-comment, non-brace lines as ops
    op_lines = [l for l in lines if (l.strip() and not l.strip().startswith("//")
                and "{" not in l) or "=" in l]
    # Count specific expensive op types
    exp_ops   = sum(1 for l in lines if "exponential" in l.lower())
    tanh_ops  = sum(1 for l in lines if "tanh" in l.lower())
    mul_ops   = sum(1 for l in lines if " multiply" in l.lower())
    dot_ops   = sum(1 for l in lines if " dot" in l.lower())
    total_ops = sum(1 for l in lines if " = " in l and "ROOT" not in l[:8])
    root_ops  = sum(1 for l in lines if "ROOT" in l)
    return {
        "total_assignments": total_ops,
        "root_outputs": root_ops,
        "exp_ops": exp_ops,
        "tanh_ops": tanh_ops,
        "multiply_ops": mul_ops,
        "dot_ops": dot_ops,
        "hlo_lines": len(lines),
    }


# (a) Primal-only: fast_physics -> (f, q)  [may call _fast_combined internally]
def _primal_wrapped(y, params, t):
    return fast_physics(y, params, t)

# (b) Full combined: _fast_combined -> (f, q, j_f, j_q)
def _combined_wrapped(y, params, t):
    return combined_fn(y, params, t)

primal_lowered   = jax.jit(_primal_wrapped).lower(y_sample, mos, 0.0)
combined_lowered = jax.jit(_combined_wrapped).lower(y_sample, mos, 0.0)

primal_hlo   = primal_lowered.as_text()
combined_hlo = combined_lowered.as_text()

p_stats = count_hlo_ops(primal_hlo)
c_stats = count_hlo_ops(combined_hlo)

print("HLO comparison (primal fast_physics  vs  full _fast_combined):")
print("-" * 60)
fmt = "{:<25s}  {:>8s}  {:>8s}  {:>8s}"
print(fmt.format("metric", "primal", "combined", "ratio"))
print("-" * 60)
for key in p_stats:
    pv = p_stats[key]
    cv = c_stats[key]
    ratio = pv / cv if cv > 0 else float("nan")
    print(fmt.format(key, str(pv), str(cv), f"{ratio:.2f}"))
print("-" * 60)
print()

# ---------------------------------------------------------------------------
# Conclusion
# ---------------------------------------------------------------------------
ratio = p_stats["total_assignments"] / max(c_stats["total_assignments"], 1)
if ratio < 0.65:
    verdict = "DCE IS WORKING — primal HLO is significantly smaller than combined."
elif ratio < 0.90:
    verdict = "DCE IS PARTIAL — some Jacobian ops are being eliminated but not all."
else:
    verdict = "DCE IS FAILING — primal pays full combined cost (Jacobian ops not eliminated)."

print(f"total_assignments ratio = {ratio:.3f}")
print(f"Verdict: {verdict}")
print()

# ---------------------------------------------------------------------------
# Extra: verify that fast_physics actually calls _fast_combined (not raw_physics)
# ---------------------------------------------------------------------------
print("Sanity: checking fast_physics primal output matches combined primal output...")
import jax

f_q_primal   = fast_physics(y_sample, mos, 0.0)
f_q_combined = combined_fn(y_sample, mos, 0.0)[:2]   # first 2 outputs

f_match = jnp.allclose(f_q_primal[0], f_q_combined[0], atol=1e-10)
q_match = jnp.allclose(f_q_primal[1], f_q_combined[1], atol=1e-10)
print(f"  f matches: {bool(f_match)}")
print(f"  q matches: {bool(q_match)}")
if bool(f_match) and bool(q_match):
    print("  OK: fast_physics routes through _fast_combined (outputs agree)")
else:
    print("  WARNING: outputs disagree — fast_physics may not be using combined_fn")
print()

# ---------------------------------------------------------------------------
# Dump HLO snippets for manual inspection (first 60 lines each)
# ---------------------------------------------------------------------------
print("=== Primal HLO (first 60 lines) ===")
for line in primal_hlo.splitlines()[:60]:
    print(line)
print()
print("=== Combined HLO (first 60 lines) ===")
for line in combined_hlo.splitlines()[:60]:
    print(line)
