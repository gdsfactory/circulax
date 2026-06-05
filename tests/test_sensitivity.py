"""Tests for DC parameter sensitivity via implicit differentiation.

Validates :func:`circulax.solvers.sensitivity.dc_parameter_sensitivity` and
:func:`circulax.solvers.sensitivity.dc_parameter_sensitivity_dense` against
finite-difference ground truth: perturb a parameter, re-run the DC solve,
measure how the loss changes, and compare to our adjoint gradient.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

# Paths to OSDI test binaries
_DATA_DIR = Path(__file__).parent / "data" / "va"
DIODE_OSDI = str(_DATA_DIR / "diode.osdi")
RESISTOR_OSDI = str(_DATA_DIR / "resistor.osdi")


def _bosdi_available() -> bool:
    try:
        from bosdi.circulax import OsdiComponentGroup  # noqa: F401
        from osdi_jax import osdi_residual_eval  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _bosdi_available(), reason="bosdi/osdi_jax not available"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def resistor_osdi_circuit():
    """OSDI resistor circuit: CS (1 mA) driving node1 through OSDI R=1kΩ to GND.

    Circuit: CS+ → node1 → OSDI_R → GND, CS- → GND.
    At DC: V_node1 = I_CS × R = 1e-3 × 1000 = 1.0 V.
    Loss: L(y) = y[node1]^2 = 1.0 → ∂L/∂R = 2 × V × I² × ... (non-trivial gradient).

    Returns (groups, sys_size, port_map, descriptor) so tests have access
    to the OsdiModelDescriptor for param name resolution.
    """
    from circulax import osdi_component
    from circulax.compiler import compile_netlist
    from circulax.components.electronic import CurrentSource

    OsdiResistor = osdi_component(
        osdi_path=RESISTOR_OSDI,
        ports=("p", "n"),
        # Provide complete defaults; $mfactor=0 disables the device
        default_params={"$mfactor": 1.0, "R": 1000.0, "zeta": 0.0, "tnom": 300.15},
    )

    models_map = {
        "cs": CurrentSource,
        "osdi_r": OsdiResistor,
        "ground": lambda: 0,
    }

    # CS forces I=1mA into node1; OSDI R connects node1 to GND.
    # V_node1 = I × R at DC.
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "CS": {"component": "cs", "settings": {"I": 1e-3}},
            "R1": {"component": "osdi_r", "settings": {"R": 1000.0}},
        },
        "connections": {
            "GND,p1": ("CS,p1", "R1,n"),
            "CS,p2": "R1,p",
        },
    }

    groups, sys_size, port_map = compile_netlist(net_dict, models_map)
    return groups, sys_size, port_map, OsdiResistor


@pytest.fixture
def diode_osdi_circuit():
    """OSDI diode circuit: 1 mA current source forcing current through diode || 10kΩ.

    Using a current source avoids Newton divergence from the exponential I-V of
    a stiff forward-biased diode driven by a voltage source.

    Returns (groups, sys_size, port_map, descriptor).
    Loss: L(y) = sum(y^2)
    """
    from circulax import osdi_component
    from circulax.compiler import compile_netlist
    from circulax.components.electronic import CurrentSource, Resistor

    OsdiDiode = osdi_component(
        osdi_path=DIODE_OSDI,
        ports=("A", "C"),
        # All 15 params must be specified; unspecified keys default to 0.0
        # which disables the device ($mfactor=0) or breaks model math (Tnom=0).
        default_params={
            "$mfactor": 1.0,
            "area": 1.0,
            "Is": 1e-12,
            "N": 1.5,
            "Rs": 0.0,
            "BV": 100.0,
            "IBV": 1e-6,
            "XTI": 3.0,
            "EG": 1.11,
            "Tnom": 300.15,
            "Cjo": 0.0,
            "Vj": 1.0,
            "M": 0.5,
            "FC": 0.5,
            "TT": 0.0,
        },
    )

    models_map = {
        "cs": CurrentSource,
        "r": Resistor,
        "osdi_d": OsdiDiode,
        "ground": lambda: 0,
    }

    # CS (1 mA) forces current into node1.  D1 and R1=10kΩ are in parallel
    # from node1 to GND.
    net_dict = {
        "instances": {
            "GND": {"component": "ground"},
            "CS": {"component": "cs", "settings": {"I": 1e-3}},
            "R1": {"component": "r", "settings": {"R": 10000.0}},
            "D1": {"component": "osdi_d", "settings": {}},
        },
        "connections": {
            "GND,p1": ("CS,p1", "R1,p2", "D1,C"),
            "CS,p2": ("R1,p1", "D1,A"),
        },
    }

    groups, sys_size, port_map = compile_netlist(net_dict, models_map)
    return groups, sys_size, port_map, OsdiDiode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_osdi_key(groups: dict) -> str:
    """Return the key of the first OsdiComponentGroup in groups."""
    from bosdi.circulax import OsdiComponentGroup
    for k, g in groups.items():
        if isinstance(g, OsdiComponentGroup):
            return k
    raise ValueError("No OsdiComponentGroup found in groups.")


def _fd_ground_truth(
    groups: dict,
    sys_size: int,
    loss_fn,
    osdi_group_key: str,
    descriptor,
    pname: str,
    device_idx: int,
    eps_fd: float = 1e-4,
) -> float:
    """Compute ∂loss/∂p_k[device_idx] by re-running the full DC solve.

    This is the gold standard for validating the adjoint gradient.
    Central differences are used for O(h^2) accuracy.
    """
    from bosdi.circulax import OsdiComponentGroup

    from circulax.solvers import analyze_circuit

    group = groups[osdi_group_key]
    assert isinstance(group, OsdiComponentGroup)

    # Find param column via the descriptor's name-to-index map
    col = descriptor._name_to_idx.get(pname.lower())
    assert col is not None, f"Param {pname!r} not in descriptor._name_to_idx"

    p_val = float(group.params[device_idx, col])
    # Pure relative step: h = eps_fd * |p_val|.
    # For params like Is=1e-12, this gives h=1e-16 (safe).
    # For p_val=0, fall back to eps_fd as absolute step.
    h = eps_fd * abs(p_val) if abs(p_val) > 0 else eps_fd

    def _solve_with_param(p_new: float) -> float:
        import numpy as _np
        new_params = _np.array(jax.device_get(group.params))
        new_params[device_idx, col] = p_new
        new_group = group.with_params(jnp.array(new_params))
        groups_new = dict(groups)
        groups_new[osdi_group_key] = new_group

        solver = analyze_circuit(groups_new, sys_size, backend="dense")
        y0 = jnp.zeros(sys_size)
        y = solver.solve_dc(groups_new, y0)
        return float(loss_fn(y))

    loss_plus = _solve_with_param(p_val + h)
    loss_minus = _solve_with_param(p_val - h)
    return (loss_plus - loss_minus) / (2.0 * h)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_import() -> None:
    """dc_parameter_sensitivity and dense variant must be importable."""
    from circulax.solvers import dc_parameter_sensitivity, dc_parameter_sensitivity_dense  # noqa: F401
    from circulax.solvers.sensitivity import dc_parameter_sensitivity as dps1  # noqa: F401


# ---------------------------------------------------------------------------
# Dense variant tests (no KLU handle required)
# ---------------------------------------------------------------------------


def test_sensitivity_dense_resistor(resistor_osdi_circuit) -> None:
    """Adjoint gradient for OSDI R w.r.t. R parameter matches FD ground truth."""
    from circulax.solvers import analyze_circuit
    from circulax.solvers.sensitivity import dc_parameter_sensitivity_dense

    groups, sys_size, port_map, descriptor = resistor_osdi_circuit
    osdi_key = _find_osdi_key(groups)

    solver = analyze_circuit(groups, sys_size, backend="dense")
    y_star = solver.solve_dc(groups, jnp.zeros(sys_size))

    # Voltage at R anode = CS+ node; V_node = I_CS × R = 1 mA × 1 kΩ = 1 V
    node_idx = port_map["CS,p2"]

    def loss_fn(y):
        return y[node_idx] ** 2

    grad = dc_parameter_sensitivity_dense(
        groups,
        y_star,
        loss_fn,
        osdi_group_key=osdi_key,
        param_names=["R"],
        model_descriptor=descriptor,
    )

    assert "R" in grad
    assert grad["R"].shape == (1,)  # one device instance

    adjoint_grad = float(grad["R"][0])

    # FD ground truth: re-run DC with perturbed R
    fd_grad = _fd_ground_truth(
        groups, sys_size, loss_fn, osdi_key, descriptor, "R", device_idx=0, eps_fd=1e-4
    )

    print("\nResistor R sensitivity:")
    print(f"  Adjoint gradient: {adjoint_grad:.8e}")
    print(f"  FD ground truth:  {fd_grad:.8e}")
    if abs(fd_grad) > 1e-30:
        rel_err = abs(adjoint_grad - fd_grad) / abs(fd_grad)
        print(f"  Relative error:   {rel_err:.4e}")

    # Adjoint and FD should agree to better than 0.5%
    assert abs(adjoint_grad - fd_grad) < 0.005 * abs(fd_grad) + 1e-20, (
        f"Adjoint gradient {adjoint_grad:.6e} disagrees with FD {fd_grad:.6e}"
    )


def test_sensitivity_dense_diode_Is(diode_osdi_circuit) -> None:
    """Adjoint gradient for OSDI diode w.r.t. Is parameter matches FD ground truth."""
    from circulax.solvers import analyze_circuit
    from circulax.solvers.sensitivity import dc_parameter_sensitivity_dense

    groups, sys_size, port_map, descriptor = diode_osdi_circuit
    osdi_key = _find_osdi_key(groups)

    solver = analyze_circuit(groups, sys_size, backend="dense")
    y_star = solver.solve_dc(groups, jnp.zeros(sys_size))

    def loss_fn(y):
        return jnp.sum(y ** 2)

    grad = dc_parameter_sensitivity_dense(
        groups,
        y_star,
        loss_fn,
        osdi_group_key=osdi_key,
        param_names=["Is"],
        model_descriptor=descriptor,
    )

    assert "Is" in grad
    assert grad["Is"].shape == (1,)

    adjoint_grad = float(grad["Is"][0])

    fd_grad = _fd_ground_truth(
        groups, sys_size, loss_fn, osdi_key, descriptor, "Is", device_idx=0, eps_fd=1e-4
    )

    print("\nDiode Is sensitivity:")
    print(f"  Adjoint gradient: {adjoint_grad:.8e}")
    print(f"  FD ground truth:  {fd_grad:.8e}")
    if abs(fd_grad) > 1e-30:
        rel_err = abs(adjoint_grad - fd_grad) / abs(fd_grad)
        print(f"  Relative error:   {rel_err:.4e}")

    # Is is extremely small (1e-12); use a generous absolute tolerance
    assert abs(adjoint_grad - fd_grad) < 0.01 * (abs(fd_grad) + 1e-25) + 1e-30


def test_sensitivity_dense_diode_N(diode_osdi_circuit) -> None:
    """Adjoint gradient for OSDI diode w.r.t. N parameter matches FD ground truth."""
    from circulax.solvers import analyze_circuit
    from circulax.solvers.sensitivity import dc_parameter_sensitivity_dense

    groups, sys_size, port_map, descriptor = diode_osdi_circuit
    osdi_key = _find_osdi_key(groups)

    solver = analyze_circuit(groups, sys_size, backend="dense")
    y_star = solver.solve_dc(groups, jnp.zeros(sys_size))

    def loss_fn(y):
        return jnp.sum(y ** 2)

    grad = dc_parameter_sensitivity_dense(
        groups,
        y_star,
        loss_fn,
        osdi_group_key=osdi_key,
        param_names=["N"],
        model_descriptor=descriptor,
    )

    assert "N" in grad
    assert grad["N"].shape == (1,)

    adjoint_grad = float(grad["N"][0])

    fd_grad = _fd_ground_truth(
        groups, sys_size, loss_fn, osdi_key, descriptor, "N", device_idx=0, eps_fd=1e-4
    )

    print("\nDiode N sensitivity:")
    print(f"  Adjoint gradient: {adjoint_grad:.8e}")
    print(f"  FD ground truth:  {fd_grad:.8e}")
    if abs(fd_grad) > 1e-30:
        rel_err = abs(adjoint_grad - fd_grad) / abs(fd_grad)
        print(f"  Relative error:   {rel_err:.4e}")

    assert abs(adjoint_grad - fd_grad) < 0.01 * (abs(fd_grad) + 1e-20) + 1e-30


# ---------------------------------------------------------------------------
# KLU variant test
# ---------------------------------------------------------------------------


def test_sensitivity_klu_matches_dense(resistor_osdi_circuit) -> None:
    """KLU-based adjoint gradient matches dense variant for OSDI resistor."""
    from circulax.solvers import analyze_circuit
    from circulax.solvers.sensitivity import dc_parameter_sensitivity, dc_parameter_sensitivity_dense

    groups, sys_size, port_map, descriptor = resistor_osdi_circuit
    osdi_key = _find_osdi_key(groups)

    solver_klu = analyze_circuit(groups, sys_size, backend="klu_split_linear")
    y_star = solver_klu.solve_dc(groups, jnp.zeros(sys_size))

    anode_idx = port_map["CS,p2"]

    def loss_fn(y):
        return y[anode_idx] ** 2

    grad_klu = dc_parameter_sensitivity(
        groups,
        solver_klu,
        y_star,
        loss_fn,
        osdi_group_key=osdi_key,
        param_names=["R"],
        model_descriptor=descriptor,
    )

    grad_dense = dc_parameter_sensitivity_dense(
        groups,
        y_star,
        loss_fn,
        osdi_group_key=osdi_key,
        param_names=["R"],
        model_descriptor=descriptor,
    )

    klu_val = float(grad_klu["R"][0])
    dense_val = float(grad_dense["R"][0])
    print(f"\nR gradient: KLU={klu_val:.8e}, dense={dense_val:.8e}")

    assert jnp.allclose(grad_klu["R"], grad_dense["R"], rtol=1e-5), (
        f"KLU gradient {klu_val:.6e} differs from dense {dense_val:.6e}"
    )


# ---------------------------------------------------------------------------
# Multi-param test
# ---------------------------------------------------------------------------


def test_sensitivity_multi_param(resistor_osdi_circuit) -> None:
    """Can compute gradients for multiple params in one call."""
    from circulax.solvers import analyze_circuit
    from circulax.solvers.sensitivity import dc_parameter_sensitivity_dense

    groups, sys_size, port_map, descriptor = resistor_osdi_circuit
    osdi_key = _find_osdi_key(groups)

    solver = analyze_circuit(groups, sys_size, backend="dense")
    y_star = solver.solve_dc(groups, jnp.zeros(sys_size))

    def loss_fn(y):
        return jnp.sum(y ** 2)

    grad = dc_parameter_sensitivity_dense(
        groups,
        y_star,
        loss_fn,
        osdi_group_key=osdi_key,
        param_names=["R", "zeta"],
        model_descriptor=descriptor,
    )

    assert set(grad.keys()) == {"R", "zeta"}
    assert grad["R"].shape == (1,)
    assert grad["zeta"].shape == (1,)
    print(f"\nMulti-param: R={float(grad['R'][0]):.4e}, zeta={float(grad['zeta'][0]):.4e}")


# ---------------------------------------------------------------------------
# param_to_col explicit mapping
# ---------------------------------------------------------------------------


def test_sensitivity_param_to_col_explicit(resistor_osdi_circuit) -> None:
    """param_to_col kwarg works as explicit column override (no descriptor needed)."""
    from circulax.solvers import analyze_circuit
    from circulax.solvers.sensitivity import dc_parameter_sensitivity_dense

    groups, sys_size, port_map, descriptor = resistor_osdi_circuit
    osdi_key = _find_osdi_key(groups)

    solver = analyze_circuit(groups, sys_size, backend="dense")
    y_star = solver.solve_dc(groups, jnp.zeros(sys_size))

    anode_idx = port_map["CS,p2"]

    def loss_fn(y):
        return y[anode_idx] ** 2

    # OSDI resistor param order: [$mfactor=0, R=1, zeta=2, tnom=3]
    grad_explicit = dc_parameter_sensitivity_dense(
        groups,
        y_star,
        loss_fn,
        osdi_group_key=osdi_key,
        param_names=["R"],
        param_to_col={"R": 1},
    )

    grad_descriptor = dc_parameter_sensitivity_dense(
        groups,
        y_star,
        loss_fn,
        osdi_group_key=osdi_key,
        param_names=["R"],
        model_descriptor=descriptor,
    )

    assert jnp.allclose(grad_explicit["R"], grad_descriptor["R"], rtol=1e-10), (
        f"explicit {float(grad_explicit['R'][0]):.6e} vs descriptor {float(grad_descriptor['R'][0]):.6e}"
    )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_sensitivity_wrong_key_raises(resistor_osdi_circuit) -> None:
    """Raises ValueError for a missing OSDI group key."""
    from circulax.solvers import analyze_circuit
    from circulax.solvers.sensitivity import dc_parameter_sensitivity_dense

    groups, sys_size, _, descriptor = resistor_osdi_circuit
    solver = analyze_circuit(groups, sys_size, backend="dense")
    y_star = solver.solve_dc(groups, jnp.zeros(sys_size))

    with pytest.raises(ValueError, match="not found"):
        dc_parameter_sensitivity_dense(
            groups,
            y_star,
            lambda y: jnp.sum(y),
            osdi_group_key="nonexistent_key",
            param_names=["R"],
            model_descriptor=descriptor,
        )


def test_sensitivity_wrong_param_raises(resistor_osdi_circuit) -> None:
    """Raises ValueError for unknown parameter name."""
    from circulax.solvers import analyze_circuit
    from circulax.solvers.sensitivity import dc_parameter_sensitivity_dense

    groups, sys_size, _, descriptor = resistor_osdi_circuit
    osdi_key = _find_osdi_key(groups)

    solver = analyze_circuit(groups, sys_size, backend="dense")
    y_star = solver.solve_dc(groups, jnp.zeros(sys_size))

    with pytest.raises(ValueError, match="not"):
        dc_parameter_sensitivity_dense(
            groups,
            y_star,
            lambda y: jnp.sum(y),
            osdi_group_key=osdi_key,
            param_names=["NONEXISTENT_PARAM"],
            model_descriptor=descriptor,
        )
