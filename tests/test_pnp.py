# Copyright (c) 2026 Circulax contributors
"""Polarity, charge, differentiability and circuit checks for PNP components."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax import compile_circuit
from circulax.components.base_component import CircuitComponent
from circulax.components.bipolar import BJT_PNP, BJT_PNP_Dynamic
from circulax.components.electronic import BJT_NPN, BJT_NPN_Dynamic, Resistor, VoltageSource


@pytest.mark.parametrize(("pnp", "npn"), [(BJT_PNP, BJT_NPN), (BJT_PNP_Dynamic, BJT_NPN_Dynamic)])
def test_polarity_and_current_conservation(pnp: type[CircuitComponent], npn: type[CircuitComponent]) -> None:
    # Forward active, reverse active, saturation, cutoff and common-mode shift.
    biases = jnp.array([[-2.0, -0.6, 0.0], [0.0, -0.6, -2.0], [-0.1, -0.65, 0.0], [-2.0, 0.2, 0.0], [1.0, 2.4, 3.0]])
    pmodel, nmodel = pnp(BetaF=73.0, BetaR=4.0), npn(BetaF=73.0, BetaR=4.0)
    evaluate = jax.jit(jax.vmap(lambda v: pnp.solver_call(0.0, v, pmodel)))
    pf, pq = evaluate(biases)
    nf, nq = jax.vmap(lambda v: npn.solver_call(0.0, -v, nmodel))(biases)
    np.testing.assert_allclose(pf, -nf, atol=1e-15)
    np.testing.assert_allclose(pq, -nq, atol=1e-25)
    np.testing.assert_allclose(pf.sum(axis=1), 0.0, atol=1e-15)
    np.testing.assert_allclose(pq.sum(axis=1), 0.0, atol=1e-25)


def test_charge_jacobian_and_ac_polarity() -> None:
    params = {"Cje": 2e-12, "Cjc": 3e-12, "Tf": 2e-9, "Tr": 1e-9}
    pmodel, nmodel = BJT_PNP_Dynamic(**params), BJT_NPN_Dynamic(**params)
    bias = jnp.array([-2.0, -0.6, 0.0])

    def jac(model: CircuitComponent, cls: type[CircuitComponent], v: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jax.jacfwd(lambda x: cls.solver_call(0.0, x, model))(v)

    pg, pc = jac(pmodel, BJT_PNP_Dynamic, bias)
    ng, nc = jac(nmodel, BJT_NPN_Dynamic, -bias)
    omega = 2 * jnp.pi * 1e8
    np.testing.assert_allclose(pg + 1j * omega * pc, ng + 1j * omega * nc, rtol=1e-12)
    np.testing.assert_allclose(pc.sum(axis=0), 0.0, atol=1e-25)
    np.testing.assert_allclose(pc.sum(axis=1), 0.0, atol=1e-25)
    assert pc[1, 1] > 0
    h = 1e-5
    direction = jnp.array([0.2, -0.1, 0.3])
    qp = BJT_PNP_Dynamic.solver_call(0.0, bias + h * direction, pmodel)[1]
    qm = BJT_PNP_Dynamic.solver_call(0.0, bias - h * direction, pmodel)[1]
    np.testing.assert_allclose((qp - qm) / (2 * h), pc @ direction, rtol=1e-7, atol=1e-24)


def test_parameter_derivatives() -> None:
    def current(saturation: jax.Array) -> jax.Array:
        model = BJT_PNP_Dynamic(Is=saturation, Tf=1e-9)
        return BJT_PNP_Dynamic.solver_call(0.0, jnp.array([-2.0, -0.6, 0.0]), model)[0][0]

    derivative = jax.jit(jax.grad(current))(1e-12)
    np.testing.assert_allclose(derivative, current(1e-12) / 1e-12, rtol=1e-12)
    assert derivative < 0


def test_common_emitter_circuit() -> None:
    net = {
        "instances": {
            "Q": {"component": "pnp", "settings": {"Is": 1e-14}},
            "R": {"component": "r", "settings": {"R": 1000.0}},
            "VE": {"component": "v", "settings": {"V": 1.0}},
            "VB": {"component": "v", "settings": {"V": 0.4}},
        },
        "connections": {"Q,c": "R,p1", "Q,e": "VE,p1", "Q,b": "VB,p1", "R,p2": "GND,p1", "VE,p2": "GND,p1", "VB,p2": "GND,p1"},
        "ports": {"out": "Q,c"},
    }
    circuit = compile_circuit(net, {"pnp": BJT_PNP_Dynamic, "r": Resistor, "v": VoltageSource})
    result = circuit.dc()
    collector = circuit.get_port_field(result, "Q,c")
    assert 0.1 < float(collector) < 0.15
    currents, _ = BJT_PNP_Dynamic(Is=1e-14)(c=collector, b=0.4, e=1.0)
    np.testing.assert_allclose(currents["c"] + collector / 1000.0, 0.0, atol=1e-10)
