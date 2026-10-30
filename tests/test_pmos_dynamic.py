# Copyright (c) 2026 Circulax contributors.
"""Generic PMOS polarity, charge and CMOS circuit tests."""

# ruff: noqa: ANN001, ANN201, FBT003

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax.components.electronic import NMOSDynamic
from circulax.components.mosfet import PMOSDynamic

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("bias", [[-2.0, 0.0, 0.0], [-0.1, -2.0, 0.0], [-2.0, -2.0, 0.0], [1.0, 0.0, 3.0]])
def test_current_charge_polarity_and_conservation(bias):
    y = jnp.array(bias)
    kwargs = {"Kp": 1e-5, "W": 20e-6, "L": 1e-6, "Vth": 1.0, "lam": 0.01, "Cox": 1e-3, "Cgd_ov": 2e-15, "Cgs_ov": 3e-15}
    n = NMOSDynamic(**kwargs)
    p = PMOSDynamic(**{**kwargs, "Vth": -1.0})
    nf, nq = n.solver_call(0.0, -y, n)
    pf, pq = jax.jit(lambda y: p.solver_call(0.0, y, p))(y)
    np.testing.assert_allclose(pf, -nf, rtol=1e-13)
    np.testing.assert_allclose(pq, -nq, rtol=1e-13)
    assert pf.sum() == 0
    assert pq.sum() == pytest.approx(0, abs=1e-28)
    g, c = jax.jacfwd(lambda y: p.solver_call(0.0, y, p))(y)
    assert np.isfinite(g).all()
    assert np.isfinite(c).all()


def test_overlap_capacitance_is_positive_and_parameter_gradient():
    # Cutoff isolates the two linear overlap branches from intrinsic charge.
    model = PMOSDynamic(Cgd_ov=2e-15, Cgs_ov=3e-15)
    y = jnp.array([-1.0, 0.0, 0.0])
    _, c = jax.jacfwd(lambda y: model.solver_call(0.0, y, model))(y)
    assert c[1, 1] == pytest.approx(5e-15, rel=1e-13, abs=1e-28)
    assert c[1, 0] == pytest.approx(-2e-15, rel=1e-13, abs=1e-28)
    assert c[1, 2] == pytest.approx(-3e-15, rel=1e-13, abs=1e-28)
    derivative = jax.jit(jax.grad(lambda width: PMOSDynamic(W=width)(d=-2.0, g=-2.0)[0]["d"]))(20e-6)
    assert np.isfinite(derivative)
    assert derivative < 0


@pytest.mark.parametrize("input_voltage", [0.0, 3.0])
def test_cmos_inverter_dc_and_finite_ac(input_voltage):
    from circulax import compile_circuit
    from circulax.components.electronic import Resistor, VoltageSource

    net = {
        "instances": {
            "P": {"component": "p"},
            "N": {"component": "n"},
            "VP": {"component": "v", "settings": {"V": 3.0}},
            "VI": {"component": "v", "settings": {"V": input_voltage}},
            "leak": {"component": "r", "settings": {"R": 1e9}},
            "GND": {"component": "ground"},
        },
        "connections": {
            "VP,p1": "P,s",
            "VI,p1": ("P,g", "N,g"),
            "P,d": ("N,d", "leak,p1"),
            "GND,p1": ("N,s", "VP,p2", "VI,p2", "leak,p2"),
        },
        "ports": {"out": "P,d"},
    }
    circuit = compile_circuit(net, {"p": PMOSDynamic, "n": NMOSDynamic, "v": VoltageSource, "r": Resistor})
    y = circuit.dc()
    expected = 3.0 if input_voltage == 0 else 0.0
    assert circuit.port(y, "P,d") == pytest.approx(expected, abs=2e-5)
    ac = circuit.ac_response(freqs=jnp.array([1e3, 1e6]), sources={"VI": 1.0})
    assert np.isfinite(ac["solution"]).all()
