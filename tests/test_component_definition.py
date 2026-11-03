"""Component definition must inspect structure without executing numerical work."""

from dataclasses import fields
from inspect import signature
from typing import Any, Never

import jax
import jax.numpy as jnp
import pytest

from circulax.components.base_component import PhysicsReturn, Signals, component, source


@pytest.mark.parametrize("decorator", [component, source])
def test_definition_and_construction_do_not_execute_physics(decorator: Any) -> None:
    calls = []

    def physics(signals: Signals, gain: float = 2.0) -> PhysicsReturn:
        calls.append(gain)
        return {"p": gain * signals.p}, {}

    def source_physics(signals: Signals, t: float, gain: float = 2.0) -> PhysicsReturn:
        calls.append(gain)
        return {"p": gain * signals.p + t}, {}

    model = decorator(ports=("p",))(source_physics if decorator is source else physics)
    assert calls == []
    assert "gain" in {field.name for field in fields(model)}
    assert signature(model).parameters["gain"].default == 2.0
    instance = model(gain=3.0)
    assert calls == []
    result, charges = instance(p=4.0)
    assert result == {"p": 12.0}
    assert charges == {}
    assert calls == [3.0]


def test_setup_and_delay_discovery_run_only_when_requested() -> None:
    calls = []

    @component(ports=("p",))
    def Delayed(signals: Signals, init: dict, gain: float = 2.0) -> PhysicsReturn:  # noqa: N802
        del gain
        calls.append("physics")
        return {"p": init["gain"] * signals.at_delay(0.5).p}, {}

    @Delayed.setup
    def setup(gain: float = 2.0) -> dict:
        calls.append("setup")
        return {"gain": gain}

    instance = Delayed(gain=3.0)
    assert calls == []
    assert Delayed.delay_values(instance) == (0.5,)
    assert calls == ["setup", "physics"]
    calls.clear()
    f, _ = Delayed.solver_call(0.0, jnp.asarray([4.0]), instance)
    assert jnp.allclose(f, jnp.asarray([12.0]))
    assert calls == ["setup", "physics"]


def test_body_errors_are_deferred_to_real_evaluation() -> None:
    @component(ports=("p",))
    def PositiveDefault(signals: Signals, scale: float = 0.0) -> PhysicsReturn:  # noqa: N802
        if scale == 0:
            msg = "scale must be nonzero"
            raise ValueError(msg)
        return {"p": signals.p / scale}, {}

    assert PositiveDefault(scale=2.0)(p=4.0)[0] == {"p": 2.0}
    instance = PositiveDefault()
    with pytest.raises(ValueError, match="scale must be nonzero"):
        instance(p=4.0)


def test_invalid_signatures_still_fail_at_definition() -> None:
    def missing_default(signals: Signals, gain: float) -> Never:
        del signals, gain
        msg = "signature validation must not execute physics"
        raise AssertionError(msg)

    with pytest.raises(TypeError, match="must have a default"):
        component(ports=("p",))(missing_default)

    def keyword_only_signals(*, signals: Signals, gain: float = 1.0) -> Never:
        del signals, gain
        msg = "signature validation must not execute physics"
        raise AssertionError(msg)

    with pytest.raises(TypeError, match="positional"):
        component(ports=("p",))(keyword_only_signals)


def test_definition_never_traces_or_compiles_jax() -> None:
    calls = []

    @component(ports=("p",))
    def Nonlinear(signals: Signals, gain: float = 2.0) -> PhysicsReturn:  # noqa: N802
        calls.append("physics")
        return {"p": jnp.sin(gain * signals.p)}, {}

    instance = Nonlinear()
    assert calls == []
    evaluated = jax.jit(lambda value: instance(p=value)[0]["p"])(0.5)
    assert jnp.allclose(evaluated, jnp.sin(1.0))
    assert calls == ["physics"]
