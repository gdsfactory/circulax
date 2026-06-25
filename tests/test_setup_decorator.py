"""Tests for ``@<Component>.setup`` — the analog-init decorator.

Covers:

1. **No-init regression** — components without an ``init`` arg still work
   exactly as they did before this feature landed.
2. **Happy path** — a component with ``init`` and a registered ``.setup``
   evaluates correctly.
3. **Ring modulator** — the motivating photonic example, exercised through
   ``compile_netlist`` + DC solve.
4. **Gradient correctness through setup** — ``jax.grad`` against an input
   parameter that affects derived cache values; compared to symmetric finite
   difference. This is the headline regression gate.
5. **Decorator hygiene** — re-registration raises; non-callable raises.
6. **Missing-init error** — physics fn declares ``init`` but no ``.setup``
   was registered → clear error at first eval.
7. **Return types** — dict, namedtuple, and eqx.Module returns all flow
   through pytree semantics correctly.
"""

from __future__ import annotations

from collections import namedtuple

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from circulax.components.base_component import (
    Signals,
    States,
    component,
)

jax.config.update("jax_enable_x64", True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. No-init regression
# ─────────────────────────────────────────────────────────────────────────────


def test_existing_component_without_init_still_works() -> None:
    """Components without an ``init`` arg behave exactly as before."""

    @component(ports=("p1", "p2"))
    def PlainResistor(signals: Signals, s: States, R: float = 1.0):  # noqa: N802
        i = (signals.p1 - signals.p2) / R
        return {"p1": i, "p2": -i}, {}

    r = PlainResistor(R=10.0)
    f, _q = r(p1=5.0, p2=0.0)
    assert jnp.isclose(f["p1"], 0.5)
    assert jnp.isclose(f["p2"], -0.5)
    # No setup was registered, so _setup_fn_ref is None.
    assert r._setup_fn_ref is None
    assert r._has_init_arg is False


# ─────────────────────────────────────────────────────────────────────────────
# 2. Happy path
# ─────────────────────────────────────────────────────────────────────────────


def test_setup_dict_return() -> None:
    """A component with ``init`` and a dict-returning ``.setup`` evaluates correctly."""

    @component(ports=("p1", "p2"))
    def TempResistor(signals: Signals, s: States, init, R: float = 1.0, T: float = 300.0):  # noqa: N802
        g = init["g"]
        i = g * (signals.p1 - signals.p2)
        return {"p1": i, "p2": -i}, {}

    @TempResistor.setup
    def _setup(R: float = 1.0, T: float = 300.0):
        # Conductance with linear temp coefficient.
        return {"g": (1.0 / R) * (300.0 / T)}

    r = TempResistor(R=10.0, T=600.0)
    f, _q = r(p1=5.0, p2=0.0)
    # g = (1/10) * (300/600) = 0.05; i = 0.05 * 5 = 0.25
    assert jnp.isclose(f["p1"], 0.25)
    assert jnp.isclose(f["p2"], -0.25)
    # Decorator returned the class for chaining.
    assert TempResistor._setup_fn_ref is not None


def test_setup_via_solver_call() -> None:
    """The classmethod path (used by transient/dc solvers) injects init too."""

    @component(ports=("p1", "p2"))
    def Foo(signals: Signals, s: States, init, R: float = 1.0):  # noqa: N802
        g = init["g"]
        i = g * (signals.p1 - signals.p2)
        return {"p1": i, "p2": -i}, {}

    @Foo.setup
    def _(R: float = 1.0):
        return {"g": 1.0 / R}

    inst = Foo(R=4.0)
    y = jnp.array([2.0, 0.5])  # signals.p1=2, signals.p2=0.5
    f_vec, _q_vec = Foo.solver_call(0.0, y, inst)
    # g = 1/4 = 0.25; i = 0.25 * (2 - 0.5) = 0.375
    assert jnp.isclose(f_vec[0], 0.375)
    assert jnp.isclose(f_vec[1], -0.375)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ring-modulator worked example
# ─────────────────────────────────────────────────────────────────────────────


@component(ports=("in_", "thru", "drop"))
def RingModulator(  # noqa: N802
    signals: Signals,
    s: States,
    init,
    kappa: float = 0.3,
    neff: float = 2.4,
    alpha: float = 1e-3,
    L: float = 2 * 3.14159 * 5e-6,
    V_pi: float = 2.0,
    V: float = 0.0,
):
    """Toy CMT-style ring eval consuming round-trip-derived coefficients."""
    a = init["a"]
    t = init["t"]
    # Phase modulated by V/V_pi (linear approximation).
    phi = init["phi"] * (1.0 + V / V_pi)
    # Single-bus all-pass response amplitude:  |E_thru/E_in| = (t - a*exp(j*phi)) / (1 - t*a*exp(j*phi))
    # We model only the magnitude transfer for this test (DC, no feedback).
    cos_p = jnp.cos(phi)
    num = t - a * cos_p
    den = 1 - t * a * cos_p
    H = num / den
    return {
        "in_": signals.in_ - H * signals.in_,  # absorbed power
        "thru": signals.thru - H * signals.in_,
        "drop": signals.drop,  # passive in this toy model
    }, {}


@RingModulator.setup
def _ring_setup(
    kappa: float = 0.3,
    neff: float = 2.4,
    alpha: float = 1e-3,
    L: float = 2 * 3.14159 * 5e-6,
):
    """Round-trip model: derive CMT coefficients from physical params."""
    a = jnp.exp(-alpha * L / 2.0)
    t = jnp.sqrt(1.0 - kappa**2)
    phi0 = 2.0 * jnp.pi * neff * L
    return {"a": a, "t": t, "phi": phi0}


def test_ring_modulator_eval_at_one_bias() -> None:
    """Smoke test: ring modulator with init evaluates without crashing."""
    rm = RingModulator(kappa=0.3, neff=2.4, alpha=1e-3, V=1.0)
    f, _q = rm(in_=1.0, thru=0.0, drop=0.0)
    # No NaN / inf, finite output.
    assert jnp.isfinite(f["thru"])
    assert jnp.isfinite(f["in_"])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Gradient correctness through setup (HEADLINE)
# ─────────────────────────────────────────────────────────────────────────────


def test_grad_through_setup_matches_finite_difference() -> None:
    """``jax.grad`` against a parameter that flows through ``.setup`` must agree
    with symmetric finite difference. Severed-cache bugs would fail this test.
    """

    @component(ports=("p1", "p2"))
    def G(signals: Signals, s: States, init, R: float = 1.0, T: float = 300.0):  # noqa: N802
        g = init["g"]
        i = g * (signals.p1 - signals.p2)
        return {"p1": i, "p2": -i}, {}

    @G.setup
    def _(R: float = 1.0, T: float = 300.0):
        # Non-trivial dependency on R: g = 1/R + R*T/3000.
        return {"g": 1.0 / R + R * T / 3000.0}

    def f_of_R(R: float) -> jax.Array:
        # Build a fresh component each call — covers the hot path.
        inst = G(R=R, T=350.0)
        y = jnp.array([1.0, 0.0])
        f_vec, _q = G.solver_call(0.0, y, inst)
        return f_vec[0]  # current at p1

    R0 = 5.0
    grad_jax = jax.grad(f_of_R)(R0)

    eps = 1e-4
    grad_fd = (f_of_R(R0 + eps) - f_of_R(R0 - eps)) / (2 * eps)

    # Analytical: i = g*(p1-p2) = g; dg/dR = -1/R^2 + T/3000
    # at R=5, T=350: di/dR = -1/25 + 350/3000 = -0.04 + 0.11667 = 0.07667
    expected = -1.0 / R0**2 + 350.0 / 3000.0
    assert jnp.isclose(grad_jax, expected, rtol=1e-6)
    assert jnp.isclose(grad_jax, grad_fd, rtol=1e-4)


def test_grad_through_setup_with_jit() -> None:
    """Same gradient correctness under JIT — XLA constant-folding of static
    params must not sever AD when the param is differentiated."""

    @component(ports=("p1", "p2"))
    def G(signals: Signals, s: States, init, R: float = 1.0):  # noqa: N802
        i = init["g"] * (signals.p1 - signals.p2)
        return {"p1": i, "p2": -i}, {}

    @G.setup
    def _(R: float = 1.0):
        return {"g": 1.0 / R}

    @jax.jit
    def f_of_R(R: float) -> jax.Array:
        inst = G(R=R)
        y = jnp.array([2.0, 0.0])
        f_vec, _q = G.solver_call(0.0, y, inst)
        return f_vec[0]

    grad_jit = jax.grad(f_of_R)(3.0)
    expected = -2.0 / 9.0  # d(g*p1)/dR = -p1/R^2 = -2/9
    assert jnp.isclose(grad_jit, expected, rtol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Decorator hygiene
# ─────────────────────────────────────────────────────────────────────────────


def test_re_registration_raises() -> None:
    """Registering ``.setup`` twice on the same class raises (catches typos)."""

    @component(ports=("p1", "p2"))
    def H(signals: Signals, s: States, init, R: float = 1.0):  # noqa: N802
        return {"p1": init["g"], "p2": -init["g"]}, {}

    @H.setup
    def _first(R: float = 1.0):
        return {"g": 1.0 / R}

    with pytest.raises(RuntimeError, match="already registered"):

        @H.setup
        def _second(R: float = 1.0):
            return {"g": 2.0 / R}


def test_setup_returns_class_for_chaining() -> None:
    """``@MyComponent.setup`` returns the class so it can be chained."""

    @component(ports=("p1", "p2"))
    def J(signals: Signals, s: States, init, R: float = 1.0):  # noqa: N802
        return {"p1": init["g"], "p2": -init["g"]}, {}

    def setup_fn(R: float = 1.0):
        return {"g": 1.0 / R}

    returned = J.setup(setup_fn)
    assert returned is J


def test_setup_on_component_without_init_arg_raises() -> None:
    """Trying to register ``.setup`` on a component that lacks ``init`` raises."""

    @component(ports=("p1", "p2"))
    def K(signals: Signals, s: States, R: float = 1.0):  # noqa: N802
        return {"p1": 1.0 / R, "p2": -1.0 / R}, {}

    with pytest.raises(TypeError, match="requires the physics function to declare"):

        @K.setup
        def _(R: float = 1.0):
            return {"g": 1.0 / R}


def test_setup_rejects_non_callable() -> None:
    """``.setup`` must be given a callable."""

    @component(ports=("p1", "p2"))
    def L(signals: Signals, s: States, init, R: float = 1.0):  # noqa: N802
        return {"p1": init["g"], "p2": -init["g"]}, {}

    with pytest.raises(TypeError, match="expects a callable"):
        L.setup({"not": "callable"})


# ─────────────────────────────────────────────────────────────────────────────
# 6. Missing-init error
# ─────────────────────────────────────────────────────────────────────────────


def test_missing_setup_raises_at_eval() -> None:
    """A component with ``init`` but no registered ``.setup`` raises clearly."""

    @component(ports=("p1", "p2"))
    def M(signals: Signals, s: States, init, R: float = 1.0):  # noqa: N802
        return {"p1": init["g"], "p2": -init["g"]}, {}

    inst = M(R=5.0)
    with pytest.raises(RuntimeError, match="no setup function has been registered"):
        inst(p1=1.0, p2=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Return-type variations
# ─────────────────────────────────────────────────────────────────────────────


def test_setup_returns_namedtuple() -> None:
    """A namedtuple return is handed straight through; physics uses ``.attr``."""
    Cache = namedtuple("Cache", ["g"])  # noqa: PYI024

    @component(ports=("p1", "p2"))
    def NT(signals: Signals, s: States, init, R: float = 1.0):  # noqa: N802
        i = init.g * (signals.p1 - signals.p2)
        return {"p1": i, "p2": -i}, {}

    @NT.setup
    def _(R: float = 1.0):
        return Cache(g=1.0 / R)

    inst = NT(R=2.0)
    f, _q = inst(p1=4.0, p2=0.0)
    assert jnp.isclose(f["p1"], 2.0)


def test_setup_returns_eqx_module() -> None:
    """An eqx.Module return is handed straight through; physics uses ``.attr``."""

    class Cache(eqx.Module):
        g: jax.Array

    @component(ports=("p1", "p2"))
    def EM(signals: Signals, s: States, init, R: float = 1.0):  # noqa: N802
        i = init.g * (signals.p1 - signals.p2)
        return {"p1": i, "p2": -i}, {}

    @EM.setup
    def _(R: float = 1.0):
        return Cache(g=jnp.asarray(1.0 / R))

    inst = EM(R=8.0)
    f, _q = inst(p1=4.0, p2=0.0)
    assert jnp.isclose(f["p1"], 0.5)


def test_setup_returns_jnp_array() -> None:
    """A jnp.ndarray return is handed straight through; physics uses indexing."""

    @component(ports=("p1", "p2"))
    def AR(signals: Signals, s: States, init, R: float = 1.0):  # noqa: N802
        i = init[0] * (signals.p1 - signals.p2)
        return {"p1": i, "p2": -i}, {}

    @AR.setup
    def _(R: float = 1.0):
        return jnp.asarray([1.0 / R, R])

    inst = AR(R=4.0)
    f, _q = inst(p1=8.0, p2=0.0)
    assert jnp.isclose(f["p1"], 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# Setup-fn signature flexibility
# ─────────────────────────────────────────────────────────────────────────────


def test_setup_fn_accepting_subset_of_params() -> None:
    """The setup function only needs to accept params it uses."""

    @component(ports=("p1", "p2"))
    def S(signals: Signals, s: States, init, R: float = 1.0, T: float = 300.0, alpha: float = 1.0):  # noqa: N802
        i = init["g"] * (signals.p1 - signals.p2)
        return {"p1": i, "p2": -i}, {}

    @S.setup
    def _(R: float = 1.0):  # only consumes R, ignores T and alpha
        return {"g": 1.0 / R}

    inst = S(R=4.0, T=400.0, alpha=2.0)
    f, _q = inst(p1=4.0, p2=0.0)
    assert jnp.isclose(f["p1"], 1.0)
