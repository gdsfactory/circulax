"""Tests for the CoupledInductors and IdealTransformer primitives (issue #76)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import kfnetlist as kfnl
import pytest

from circulax import compile_circuit
from circulax.compiler import compile_netlist
from circulax.components.electronic import (
    CoupledInductors,
    IdealTransformer,
    Inductor,
    Resistor,
    VoltageSource,
    VoltageSourceAC,
)
from circulax.solvers import analyze_circuit, setup_ac_sweep

jax.config.update("jax_enable_x64", True)  # noqa: FBT003

Z0 = 50.0
FREQS = jnp.array([1e6, 1e7, 1e8])

MODELS = {
    "CoupledInductors": CoupledInductors,
    "IdealTransformer": IdealTransformer,
    "Inductor": Inductor,
    "Resistor": Resistor,
    "VDC": VoltageSource,
    "VAC": VoltageSourceAC,
    "ground": lambda: 0,
}


def _s_from_z(z: jnp.ndarray, z0: float = Z0) -> jnp.ndarray:
    eye = jnp.eye(z.shape[-1])
    return (z - z0 * eye) @ jnp.linalg.inv(z + z0 * eye)


def _z_coupled(freq: float, l1: float, l2: float, k: float) -> jnp.ndarray:
    m = k * jnp.sqrt(l1 * l2)
    return 1j * 2 * jnp.pi * freq * jnp.array([[l1, m], [m, l2]])


def _two_port(settings: dict, *, swap_secondary: bool = False) -> dict:
    """CoupledInductors with p2/s2 grounded; optionally swap the secondary terminals."""
    s_gnd, s_port = ("s1", "s2") if swap_secondary else ("s2", "s1")
    return {
        "instances": {"GND": {"component": "ground"}, "T1": {"component": "CoupledInductors", "settings": settings}},
        "connections": {"GND,p1": ("T1,p2", f"T1,{s_gnd}")},
        "ports": {"a": "T1,p1", "b": f"T1,{s_port}"},
    }


def _sp(net: dict, ports: list[str]) -> jnp.ndarray:
    circuit = compile_circuit(net, MODELS)
    return circuit.sp(ports=ports, freqs=FREQS, z0=Z0)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (CoupledInductors, {"L1": 0.0}),
        (CoupledInductors, {"L1": -1e-9}),
        (CoupledInductors, {"L2": -1e-9}),
        (CoupledInductors, {"k": 1.01}),
        (CoupledInductors, {"k": -1.01}),
        (IdealTransformer, {"n": 0.0}),
        (IdealTransformer, {"n": -2.0}),
    ],
)
def test_invalid_parameters_raise(cls: type, kwargs: dict) -> None:
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        cls(**kwargs)


def test_invalid_parameters_raise_through_compiler() -> None:
    net = _two_port({"L1": -1e-6})
    with pytest.raises(ValueError, match="L1"):
        compile_netlist(net, MODELS)
    net = {
        "instances": {"GND": {"component": "ground"}, "X": {"component": "IdealTransformer", "settings": {"n": -1.0}}},
        "connections": {"GND,p1": ("X,p2", "X,s2")},
    }
    with pytest.raises(ValueError, match="n"):
        compile_netlist(net, MODELS)


def test_valid_boundaries_and_tracers_accepted() -> None:
    CoupledInductors(k=1.0)
    CoupledInductors(k=-1.0)
    CoupledInductors(L1=jnp.array([1e-9, 2e-9]), k=jnp.array([0.1, -0.2]))
    # Traced parameters (grad / vmap paths) must not trip validation.
    jax.grad(lambda k: CoupledInductors(k=k).k)(0.5)
    jax.vmap(lambda n: IdealTransformer(n=n).n)(jnp.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# AC / S-parameters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0.9, 0.3, -0.6])
def test_mutual_coupling_matches_impedance_matrix(k: float) -> None:
    l1, l2 = 2e-6, 5e-6
    s = _sp(_two_port({"L1": l1, "L2": l2, "k": k}), ["T1,p1", "T1,s1"])
    expected = jnp.stack([_s_from_z(_z_coupled(f, l1, l2, k)) for f in FREQS])
    assert jnp.allclose(s, expected, rtol=1e-6, atol=1e-9)


def test_uncoupled_limit_matches_independent_inductors() -> None:
    l1, l2 = 2e-6, 5e-6
    coupled = _sp(_two_port({"L1": l1, "L2": l2, "k": 0.0}), ["T1,p1", "T1,s1"])
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "La": {"component": "Inductor", "settings": {"L": l1}},
            "Lb": {"component": "Inductor", "settings": {"L": l2}},
        },
        "connections": {"GND,p1": ("La,p2", "Lb,p2")},
        "ports": {"a": "La,p1", "b": "Lb,p1"},
    }
    independent = _sp(net, ["La,p1", "Lb,p1"])
    assert jnp.allclose(coupled, independent, rtol=1e-8, atol=1e-10)
    assert jnp.allclose(coupled[:, 0, 1], 0.0, atol=1e-10)


def test_polarity_swap_equals_negative_k() -> None:
    settings = {"L1": 1e-6, "L2": 3e-6, "k": 0.7}
    swapped = _sp(_two_port(settings, swap_secondary=True), ["T1,p1", "T1,s2"])
    negated = _sp(_two_port({**settings, "k": -0.7}), ["T1,p1", "T1,s1"])
    assert jnp.allclose(swapped, negated, rtol=1e-8, atol=1e-10)
    # And the sign is physical: S21 flips relative to the un-swapped dot orientation.
    plain = _sp(_two_port(settings), ["T1,p1", "T1,s1"])
    assert jnp.allclose(swapped[:, 1, 0], -plain[:, 1, 0], rtol=1e-8, atol=1e-10)


def test_batched_instances_keep_distinct_parameters() -> None:
    a = {"L1": 1e-6, "L2": 1e-6, "k": 0.3}
    b = {"L1": 4e-6, "L2": 2e-6, "k": -0.6}
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "TA": {"component": "CoupledInductors", "settings": a},
            "TB": {"component": "CoupledInductors", "settings": b},
        },
        "connections": {"GND,p1": ("TA,p2", "TA,s2", "TB,p2", "TB,s2")},
        "ports": {"a": "TA,p1", "b": "TA,s1", "c": "TB,p1", "d": "TB,s1"},
    }
    groups, _, _ = compile_netlist(net, MODELS)
    assert list(groups) == ["CoupledInductors"]
    assert groups["CoupledInductors"].var_indices.shape[0] == 2

    s = _sp(net, ["TA,p1", "TA,s1", "TB,p1", "TB,s1"])
    for f_idx, f in enumerate(FREQS):
        za, zb = (
            _z_coupled(f, l1=a["L1"], l2=a["L2"], k=a["k"]),
            _z_coupled(f, l1=b["L1"], l2=b["L2"], k=b["k"]),
        )
        assert jnp.allclose(s[f_idx, :2, :2], _s_from_z(za), rtol=1e-6, atol=1e-9)
        assert jnp.allclose(s[f_idx, 2:, 2:], _s_from_z(zb), rtol=1e-6, atol=1e-9)
        assert jnp.allclose(s[f_idx, :2, 2:], 0.0, atol=1e-9)
        assert jnp.allclose(s[f_idx, 2:, :2], 0.0, atol=1e-9)


def _loaded_transformer(n: float, r_load: float, *, source: str | None = None, r_src: float = 0.0) -> dict:
    inst = {
        "GND": {"component": "ground"},
        "X": {"component": "IdealTransformer", "settings": {"n": n}},
        "RL": {"component": "Resistor", "settings": {"R": r_load}},
    }
    conns = {"X,s1": "RL,p1", "GND,p1": ("X,p2", "X,s2", "RL,p2")}
    if source is None:
        return {"instances": inst, "connections": conns, "ports": {"in": "X,p1"}}
    inst["VS"] = {"component": source, "settings": {"V": 2.0, "freq": 1e6} if source == "VAC" else {"V": 1.0}}
    inst["RS"] = {"component": "Resistor", "settings": {"R": r_src}}
    conns["VS,p1"] = "RS,p1"
    conns["RS,p2"] = "X,p1"
    return {"instances": inst, "connections": {**conns, "GND,p1": ("X,p2", "X,s2", "RL,p2", "VS,p2")}}


@pytest.mark.parametrize("n", [0.5, 1.0, 4.0])
def test_ideal_transformer_input_impedance(n: float) -> None:
    r_load = 25.0
    s = _sp(_loaded_transformer(n, r_load), ["X,p1"])
    z_in = n**2 * r_load
    assert jnp.allclose(s[:, 0, 0], (z_in - Z0) / (z_in + Z0), rtol=1e-8)


def test_ideal_transformer_is_lossless() -> None:
    """Two-port S-matrix of the ideal transformer is unitary-like at matched ports (real, passive)."""
    n = 2.0
    net = {
        "instances": {"GND": {"component": "ground"}, "X": {"component": "IdealTransformer", "settings": {"n": n}}},
        "connections": {"GND,p1": ("X,p2", "X,s2")},
        "ports": {"a": "X,p1", "b": "X,s1"},
    }
    s = _sp(net, ["X,p1", "X,s1"])
    for sf in s:
        assert jnp.allclose(sf.conj().T @ sf, jnp.eye(2), atol=1e-8)


# ---------------------------------------------------------------------------
# Transient
# ---------------------------------------------------------------------------


def test_ideal_transformer_transient_voltage_and_current_ratio() -> None:
    n, r_load, r_src = 2.0, 40.0, 10.0
    net = _loaded_transformer(n, r_load, source="VAC", r_src=r_src)
    circuit = compile_circuit(net, MODELS)
    ts = jnp.linspace(0.0, 2e-6, 41)
    sol = circuit.transient(t0=0.0, t1=2e-6, dt0=1e-9, y0=jnp.zeros(circuit.sys_size), saveat=ts, max_steps=20000)
    v1 = circuit.port(sol.ys, "X,p1")
    v2 = circuit.port(sol.ys, "RL,p1")
    v_src = 2.0 * jnp.sin(2 * jnp.pi * 1e6 * ts)
    # v1 = n * v2
    assert jnp.allclose(v1, n * v2, atol=1e-6)
    # i1 = v2 / (n R_load) = (v_src - v1) / R_src  =>  v1 = v_src * n^2 R / (R_src + n^2 R)
    expected_v1 = v_src * n**2 * r_load / (r_src + n**2 * r_load)
    assert jnp.allclose(v1[1:], expected_v1[1:], atol=2e-3)


def test_coupled_inductor_rl_transient_matches_closed_form() -> None:
    l1, l2, k, r_src, r_load, v0 = 1e-6, 2e-6, 0.8, 10.0, 20.0, 1.0
    net = {
        "instances": {
            "GND": {"component": "ground"},
            "VS": {"component": "VDC", "settings": {"V": v0, "delay": 1e-12}},
            "RS": {"component": "Resistor", "settings": {"R": r_src}},
            "RL": {"component": "Resistor", "settings": {"R": r_load}},
            "T1": {"component": "CoupledInductors", "settings": {"L1": l1, "L2": l2, "k": k}},
        },
        "connections": {
            "VS,p1": "RS,p1",
            "RS,p2": "T1,p1",
            "T1,s1": "RL,p1",
            "GND,p1": ("VS,p2", "T1,p2", "T1,s2", "RL,p2"),
        },
    }
    circuit = compile_circuit(net, MODELS)
    t1 = 1e-6
    ts = jnp.linspace(0.0, t1, 21)
    sol = circuit.transient(t0=0.0, t1=t1, dt0=1e-10, saveat=ts, max_steps=50000)

    # x = [i1, i2], L x' = b - R x  with i2 positive into the dotted terminal s1.
    m = k * jnp.sqrt(l1 * l2)
    lm = jnp.array([[l1, m], [m, l2]])
    rm = jnp.diag(jnp.array([r_src, r_load]))
    a = -jnp.linalg.solve(lm, rm)
    x_ss = jnp.array([v0 / r_src, 0.0])
    x_t = jnp.stack([x_ss - jax.scipy.linalg.expm(a * t) @ x_ss for t in ts])  # x(0) = 0
    v_primary = v0 - r_src * x_t[:, 0]
    v_secondary = -r_load * x_t[:, 1]

    assert jnp.allclose(circuit.port(sol.ys, "T1,p1")[1:], v_primary[1:], atol=5e-3)
    assert jnp.allclose(circuit.port(sol.ys, "T1,s1")[1:], v_secondary[1:], atol=5e-3)


# ---------------------------------------------------------------------------
# kfnetlist composition
# ---------------------------------------------------------------------------


def test_kfnetlist_composition_two_distinct_instances() -> None:
    nl = kfnl.Netlist()
    nl.create_inst(name="GND", kcl="", component="ground")
    nl.create_inst(name="TA", kcl="", component="CoupledInductors", settings={"L1": 1e-6, "L2": 1e-6, "k": 0.5})
    nl.create_inst(name="TB", kcl="", component="CoupledInductors", settings={"L1": 2e-6, "L2": 8e-6, "k": 0.9})
    gnd = kfnl.PortRef(instance="GND", port="p1")
    nl.create_net(gnd, *[kfnl.PortRef(instance=i, port=p) for i in ("TA", "TB") for p in ("p2", "s2")])
    for name, (inst, port) in {"a": ("TA", "p1"), "b": ("TA", "s1"), "c": ("TB", "p1"), "d": ("TB", "s1")}.items():
        nl.create_port(name)
        nl.create_net(kfnl.NetlistPort(name), kfnl.PortRef(instance=inst, port=port))
    nl.sort()

    groups, sys_size, pmap = compile_netlist(nl, MODELS)
    assert groups["CoupledInductors"].var_indices.shape[0] == 2
    solver = analyze_circuit(groups, sys_size, backend="dense")
    y_dc = solver.solve_dc(groups, jnp.zeros(sys_size))
    run_ac = setup_ac_sweep(groups, sys_size, [pmap["TA,p1"], pmap["TA,s1"], pmap["TB,p1"], pmap["TB,s1"]], z0=Z0)
    s = run_ac(y_dc, FREQS)
    za = _z_coupled(FREQS[1], 1e-6, 1e-6, 0.5)
    zb = _z_coupled(FREQS[1], 2e-6, 8e-6, 0.9)
    assert jnp.allclose(s[1, :2, :2], _s_from_z(za), rtol=1e-6, atol=1e-9)
    assert jnp.allclose(s[1, 2:, 2:], _s_from_z(zb), rtol=1e-6, atol=1e-9)


def test_hierarchical_subcircuit_with_coupled_inductors() -> None:
    recnet = {
        "top": {
            "instances": {
                "SC1": {"component": "xfmr", "settings": {}},
                "GND": {"component": "ground"},
            },
            "connections": {"GND,p1": ("SC1,g",)},
            "ports": {"a": "SC1,a", "b": "SC1,b"},
        },
        "xfmr": {
            "instances": {
                "T": {"component": "CoupledInductors", "settings": {"L1": 1e-6, "L2": 4e-6, "k": 0.5}},
                "G": {"component": "ground"},
            },
            "connections": {"G,p1": ("T,p2", "T,s2")},
            "ports": {"a": "T,p1", "b": "T,s1", "g": "T,p2"},
        },
    }
    circuit = compile_circuit(recnet, MODELS)
    s = circuit.sp(ports=["SC1~T,p1", "SC1~T,s1"], freqs=FREQS, z0=Z0)
    expected = jnp.stack([_s_from_z(_z_coupled(f, 1e-6, 4e-6, 0.5)) for f in FREQS])
    assert jnp.allclose(s, expected, rtol=1e-6, atol=1e-9)
