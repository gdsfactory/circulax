"""Tests for transient parameter sensitivity via discrete adjoint.

Validates :func:`circulax.solvers.adjoint.transient_parameter_sensitivity`
and :func:`circulax.solvers.adjoint.transient_parameter_sensitivity_dense`
against finite-difference ground truth: perturb a parameter, re-run the
transient, measure how the loss changes, and compare to the adjoint gradient.

Circuit under test: OSDI Resistor || OSDI Capacitor from GND to node1,
driven by a 1 mA current step source.  This exercises both ∂F/∂p (via R)
and ∂Q/∂p (via C) in the discrete adjoint.

    Topology:
        CS (1 mA) → node1 → OSDI_R (1 kΩ) → GND
                       node1 → OSDI_C (1 nF)  → GND

    Exact solution: V(t) = I * R * (1 - exp(-t / (R*C)))
    At t >> RC: V → I * R = 1.0 V
"""

from __future__ import annotations

from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

_DATA_DIR = Path(__file__).parent / "data" / "va"
RESISTOR_OSDI = str(_DATA_DIR / "resistor.osdi")
CAPACITOR_OSDI = str(_DATA_DIR / "capacitor.osdi")


def _bosdi_available() -> bool:
    try:
        from bosdi.circulax import OsdiComponentGroup  # noqa: F401
        from osdi_jax import osdi_residual_eval  # noqa: F401
        from osdi_loader import load_osdi_model
        load_osdi_model(RESISTOR_OSDI)  # fails on macOS/Windows with Linux ELF binaries
        return True
    except (ImportError, RuntimeError, OSError):
        return False


pytestmark = pytest.mark.skipif(
    not _bosdi_available(), reason="bosdi/osdi_jax not available"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rc_osdi_circuit():
    """OSDI RC circuit: CS (1 mA) driving node1 through OSDI R=1kΩ || OSDI C=1nF to GND.

    Returns (groups, sys_size, port_map, OsdiResistor_descriptor, OsdiCapacitor_descriptor).
    """
    from circulax import osdi_component
    from circulax.compiler import compile_netlist
    from circulax.components.electronic import CurrentSource

    OsdiResistor = osdi_component(
        osdi_path=RESISTOR_OSDI,
        ports=("p", "n"),
        default_params={"$mfactor": 1.0, "R": 1000.0, "zeta": 0.0, "tnom": 300.15},
    )

    OsdiCapacitor = osdi_component(
        osdi_path=CAPACITOR_OSDI,
        ports=("p", "n"),
        default_params={"$mfactor": 1.0, "c": 1e-9},
    )

    models_map = {
        "cs": CurrentSource,
        "osdi_r": OsdiResistor,
        "osdi_c": OsdiCapacitor,
        "ground": lambda: 0,
    }

    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "CS": {"component": "cs", "settings": {"I": 1e-3}},
            "R1": {"component": "osdi_r", "settings": {"R": 1000.0}},
            "C1": {"component": "osdi_c", "settings": {"c": 1e-9}},
        },
        "connections": {
            "GND,p1": ("CS,p1", "R1,n", "C1,n"),
            "CS,p2": ("R1,p", "C1,p"),
        },
    }

    groups, sys_size, port_map = compile_netlist(net_dict, models_map)
    return groups, sys_size, port_map, OsdiResistor, OsdiCapacitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_osdi_key(groups: dict, name_hint: str) -> str:
    """Return the key of the OsdiComponentGroup whose key contains name_hint.

    Groups are keyed by component *type* (e.g. 'osdi_r', 'osdi_c'), not
    by instance name (e.g. 'R1', 'C1').  Use the model key substring as hint.
    """
    from bosdi.circulax import OsdiComponentGroup
    for k, g in groups.items():
        if isinstance(g, OsdiComponentGroup) and name_hint.lower() in k.lower():
            return k
    # Also accept first OSDI group if no hint matches
    for k, g in groups.items():
        if isinstance(g, OsdiComponentGroup):
            return k
    raise ValueError(f"No OsdiComponentGroup with name hint {name_hint!r} found in groups.")


def _run_transient(groups, sys_size, y0, t0, t1, dt0, n_saves):
    """Run transient simulation and return (y_traj, ts)."""
    from circulax.solvers import analyze_circuit, setup_transient

    solver = analyze_circuit(groups, sys_size, backend="dense")
    run_transient = setup_transient(groups, solver)

    ts = jnp.linspace(t0, t1, n_saves)
    saveat = diffrax.SaveAt(ts=ts)
    sol = run_transient(t0=t0, t1=t1, dt0=dt0, y0=y0, saveat=saveat, max_steps=50000)
    return sol.ys, ts


def _fd_transient_ground_truth(
    groups: dict,
    sys_size: int,
    y0: jax.Array,
    t0: float,
    t1: float,
    dt0: float,
    n_saves: int,
    loss_fn,
    osdi_group_key: str,
    pname: str,
    pcol: int,
    device_idx: int,
    eps_fd: float = 1e-4,
) -> float:
    """FD ground truth: perturb param, re-run transient, compute ∂loss/∂p.

    Uses central differences for O(h^2) accuracy.
    """
    from bosdi.circulax import OsdiComponentGroup

    group = groups[osdi_group_key]
    assert isinstance(group, OsdiComponentGroup)

    p_val = float(group.params[device_idx, pcol])
    h = eps_fd * abs(p_val) if abs(p_val) > 0 else eps_fd

    def _loss_at_param(p_new: float) -> float:
        import numpy as _np
        new_params = _np.array(jax.device_get(group.params))
        new_params[device_idx, pcol] = p_new
        new_group = group.with_params(jnp.array(new_params))
        groups_new = dict(groups)
        groups_new[osdi_group_key] = new_group

        y_traj, ts = _run_transient(groups_new, sys_size, y0, t0, t1, dt0, n_saves)
        return float(loss_fn(y_traj, ts))

    loss_plus = _loss_at_param(p_val + h)
    loss_minus = _loss_at_param(p_val - h)
    return (loss_plus - loss_minus) / (2.0 * h)


# ---------------------------------------------------------------------------
# Test: import
# ---------------------------------------------------------------------------


def test_adjoint_import() -> None:
    """transient_parameter_sensitivity and dense variant must be importable."""
    from circulax.solvers import transient_parameter_sensitivity, transient_parameter_sensitivity_dense  # noqa: F401
    from circulax.solvers.adjoint import transient_parameter_sensitivity as tps1  # noqa: F401


# ---------------------------------------------------------------------------
# Test: dense adjoint on OSDI resistor (∂F/∂R — resistive, no ∂Q/∂p)
# ---------------------------------------------------------------------------


def test_adjoint_dense_resistor_R(rc_osdi_circuit) -> None:
    """Discrete adjoint ∂loss/∂R matches FD ground truth (dense solver)."""
    from circulax.solvers.adjoint import transient_parameter_sensitivity_dense

    groups, sys_size, port_map, OsdiResistor, OsdiCapacitor = rc_osdi_circuit

    r_key = _find_osdi_key(groups, "osdi_r")

    # Use n_saves = n_steps + 1 so checkpoints coincide with BE steps.
    # The discrete adjoint is exact only when every BE step is a checkpoint.
    n_steps = 100
    t0, t1 = 0.0, 1e-6   # 1 µs = 1 RC (R=1kΩ, C=1nF) — full transient charging
    dt0 = t1 / n_steps
    n_saves = n_steps + 1

    # Start from zero volts so the RC circuit charges up during the simulation
    # (DC steady state = 1 V, so starting from 0 gives a full transient)
    y0 = jnp.zeros(sys_size)

    y_traj, ts = _run_transient(groups, sys_size, y0, t0, t1, dt0, n_saves)

    node_idx = port_map["CS,p2"]

    def loss_fn(y_traj, ts):
        # Loss = sum of V(node1)^2 over all checkpoints
        return jnp.sum(y_traj[:, node_idx] ** 2)

    grad = transient_parameter_sensitivity_dense(
        groups,
        y_traj,
        ts,
        loss_fn,
        osdi_group_key=r_key,
        param_names=["R"],
        model_descriptor=OsdiResistor,
    )

    assert "R" in grad
    assert grad["R"].shape == (1,)

    adjoint_grad = float(grad["R"][0])

    r_col = OsdiResistor._name_to_idx["r"]
    fd_grad = _fd_transient_ground_truth(
        groups, sys_size, y0, t0, t1, dt0, n_saves,
        loss_fn, r_key, "R", r_col, device_idx=0, eps_fd=1e-4,
    )

    print("\n[R gradient (dense)]")
    print(f"  Adjoint: {adjoint_grad:.6e}")
    print(f"  FD:      {fd_grad:.6e}")
    if abs(fd_grad) > 1e-30:
        rel_err = abs(adjoint_grad - fd_grad) / abs(fd_grad)
        print(f"  RelErr:  {rel_err:.4e}")

    # Accept < 1% relative error (FD has O(h^2) error with h=1e-4)
    assert abs(adjoint_grad - fd_grad) < 0.01 * abs(fd_grad) + 1e-20, (
        f"Adjoint {adjoint_grad:.6e} disagrees with FD {fd_grad:.6e}"
    )


# ---------------------------------------------------------------------------
# Test: dense adjoint on OSDI capacitor (∂Q/∂C — reactive, main path)
# ---------------------------------------------------------------------------


def test_adjoint_dense_capacitor_C(rc_osdi_circuit) -> None:
    """Discrete adjoint ∂loss/∂C matches FD ground truth (dense solver, reactive path)."""
    from circulax.solvers.adjoint import transient_parameter_sensitivity_dense

    groups, sys_size, port_map, OsdiResistor, OsdiCapacitor = rc_osdi_circuit

    c_key = _find_osdi_key(groups, "osdi_c")

    n_steps = 100
    t0, t1 = 0.0, 1e-6   # 1 µs (RC = 1 µs)
    dt0 = t1 / n_steps
    n_saves = n_steps + 1

    # Start from zero volts for a full transient charging sweep
    y0 = jnp.zeros(sys_size)

    y_traj, ts = _run_transient(groups, sys_size, y0, t0, t1, dt0, n_saves)

    node_idx = port_map["CS,p2"]

    def loss_fn(y_traj, ts):
        return jnp.sum(y_traj[:, node_idx] ** 2)

    grad = transient_parameter_sensitivity_dense(
        groups,
        y_traj,
        ts,
        loss_fn,
        osdi_group_key=c_key,
        param_names=["c"],
        model_descriptor=OsdiCapacitor,
    )

    assert "c" in grad
    assert grad["c"].shape == (1,)

    adjoint_grad = float(grad["c"][0])

    c_col = OsdiCapacitor._name_to_idx["c"]
    fd_grad = _fd_transient_ground_truth(
        groups, sys_size, y0, t0, t1, dt0, n_saves,
        loss_fn, c_key, "c", c_col, device_idx=0, eps_fd=1e-4,
    )

    print("\n[C gradient (dense, reactive)]")
    print(f"  Adjoint: {adjoint_grad:.6e}")
    print(f"  FD:      {fd_grad:.6e}")
    if abs(fd_grad) > 1e-30:
        rel_err = abs(adjoint_grad - fd_grad) / abs(fd_grad)
        print(f"  RelErr:  {rel_err:.4e}")

    # Accept < 1% relative error
    assert abs(adjoint_grad - fd_grad) < 0.01 * abs(fd_grad) + 1e-20, (
        f"Adjoint {adjoint_grad:.6e} disagrees with FD {fd_grad:.6e}"
    )


# ---------------------------------------------------------------------------
# Test: final-state-only loss function (single-arg form)
# ---------------------------------------------------------------------------


def test_adjoint_dense_final_state_loss(rc_osdi_circuit) -> None:
    """Discrete adjoint works when loss_fn depends only on final state y[-1]."""
    from circulax.solvers.adjoint import transient_parameter_sensitivity_dense

    groups, sys_size, port_map, OsdiResistor, OsdiCapacitor = rc_osdi_circuit

    r_key = _find_osdi_key(groups, "osdi_r")

    n_steps = 50
    t0, t1 = 0.0, 5e-7
    dt0 = t1 / n_steps
    n_saves = n_steps + 1

    # Start from zero for meaningful transient
    y0 = jnp.zeros(sys_size)

    y_traj, ts = _run_transient(groups, sys_size, y0, t0, t1, dt0, n_saves)

    node_idx = port_map["CS,p2"]

    # Single-argument loss: only depends on final state
    def loss_final(y_final):
        return y_final[node_idx] ** 2

    # Two-argument wrapper for FD
    def loss_fn_2(y_traj, ts):
        return y_traj[-1, node_idx] ** 2

    grad = transient_parameter_sensitivity_dense(
        groups,
        y_traj,
        ts,
        loss_final,  # single-arg form
        osdi_group_key=r_key,
        param_names=["R"],
        model_descriptor=OsdiResistor,
    )

    r_col = OsdiResistor._name_to_idx["r"]
    fd_grad = _fd_transient_ground_truth(
        groups, sys_size, y0, t0, t1, dt0, n_saves,
        loss_fn_2, r_key, "R", r_col, device_idx=0, eps_fd=1e-4,
    )

    adjoint_grad = float(grad["R"][0])

    print("\n[R gradient (final-state loss)]")
    print(f"  Adjoint: {adjoint_grad:.6e}")
    print(f"  FD:      {fd_grad:.6e}")
    if abs(fd_grad) > 1e-30:
        rel_err = abs(adjoint_grad - fd_grad) / abs(fd_grad)
        print(f"  RelErr:  {rel_err:.4e}")

    # Accept < 2% relative error (slightly more generous for final-state-only loss,
    # which uses a shorter sim with more BE discretisation error in the coupling term)
    assert abs(adjoint_grad - fd_grad) < 0.02 * abs(fd_grad) + 1e-20, (
        f"Adjoint {adjoint_grad:.6e} disagrees with FD {fd_grad:.6e}"
    )


# ---------------------------------------------------------------------------
# Test: KLU-backed adjoint matches dense
# ---------------------------------------------------------------------------


def test_adjoint_klu_matches_dense(rc_osdi_circuit) -> None:
    """KLU-backed adjoint gradient matches dense variant for OSDI resistor."""
    from circulax.solvers import analyze_circuit
    from circulax.solvers.adjoint import transient_parameter_sensitivity, transient_parameter_sensitivity_dense

    groups, sys_size, port_map, OsdiResistor, OsdiCapacitor = rc_osdi_circuit

    r_key = _find_osdi_key(groups, "osdi_r")

    n_steps = 50
    t0, t1 = 0.0, 5e-7
    dt0 = t1 / n_steps
    n_saves = n_steps + 1

    # Use KLU solver for the trajectory, starting from zero
    solver_klu = analyze_circuit(groups, sys_size, backend="klu_split_linear")
    y0 = jnp.zeros(sys_size)

    y_traj, ts = _run_transient(groups, sys_size, y0, t0, t1, dt0, n_saves)

    node_idx = port_map["CS,p2"]

    def loss_fn(y_traj, ts):
        return jnp.sum(y_traj[:, node_idx] ** 2)

    grad_klu = transient_parameter_sensitivity(
        groups,
        solver_klu,
        y_traj,
        ts,
        loss_fn,
        osdi_group_key=r_key,
        param_names=["R"],
        model_descriptor=OsdiResistor,
    )

    grad_dense = transient_parameter_sensitivity_dense(
        groups,
        y_traj,
        ts,
        loss_fn,
        osdi_group_key=r_key,
        param_names=["R"],
        model_descriptor=OsdiResistor,
    )

    klu_val = float(grad_klu["R"][0])
    dense_val = float(grad_dense["R"][0])
    print("\n[KLU vs dense R gradient]")
    print(f"  KLU:   {klu_val:.8e}")
    print(f"  Dense: {dense_val:.8e}")

    assert jnp.allclose(grad_klu["R"], grad_dense["R"], rtol=1e-5), (
        f"KLU gradient {klu_val:.6e} differs from dense {dense_val:.6e}"
    )


# ---------------------------------------------------------------------------
# Test: error handling
# ---------------------------------------------------------------------------


def test_adjoint_wrong_key_raises(rc_osdi_circuit) -> None:
    """Raises ValueError for a missing OSDI group key."""
    from circulax.solvers.adjoint import transient_parameter_sensitivity_dense

    groups, sys_size, port_map, OsdiResistor, _ = rc_osdi_circuit

    y_traj = jnp.zeros((5, sys_size))
    ts = jnp.linspace(0, 1e-6, 5)

    with pytest.raises(ValueError, match="not found"):
        transient_parameter_sensitivity_dense(
            groups, y_traj, ts,
            lambda yt, t: jnp.sum(yt),
            osdi_group_key="nonexistent",
            param_names=["R"],
            model_descriptor=OsdiResistor,
        )


def test_adjoint_wrong_param_raises(rc_osdi_circuit) -> None:
    """Raises ValueError for unknown parameter name."""
    from circulax.solvers.adjoint import transient_parameter_sensitivity_dense

    groups, sys_size, port_map, OsdiResistor, _ = rc_osdi_circuit
    r_key = _find_osdi_key(groups, "osdi_r")

    y_traj = jnp.zeros((5, sys_size))
    ts = jnp.linspace(0, 1e-6, 5)

    with pytest.raises(ValueError, match="not"):
        transient_parameter_sensitivity_dense(
            groups, y_traj, ts,
            lambda yt, t: jnp.sum(yt),
            osdi_group_key=r_key,
            param_names=["NONEXISTENT"],
            model_descriptor=OsdiResistor,
        )
