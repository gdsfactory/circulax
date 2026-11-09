"""Assembly regression tests against the full equality-constrained OSDI DAE."""

import shutil
import subprocess

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax import compile_netlist
from circulax.solvers.assembly import _assemble_osdi_gc_separate, _assemble_osdi_group


@pytest.fixture(scope="module")
def binaries(tmp_path_factory):
    osdi_jax = pytest.importorskip("osdi_jax")
    if not hasattr(osdi_jax.OsdiBatchHandle, "topology"):
        pytest.skip("requires experimental BOSDI topology API")
    compiler = shutil.which("openvaf-r")
    if compiler is None:
        pytest.skip("requires OpenVAF")
    root = tmp_path_factory.mktemp("topology")
    sources = {
        "rc": """module rc(p,n);
  inout p,n; electrical p,n,mid;
  parameter real r=0; parameter real c=1e-12;
  analog begin
    if (r>0) I(p,mid) <+ V(p,mid)/r*$temperature/300*$simparam("scale",1);
    else V(p,mid) <+ 0;
    I(mid,n) <+ V(mid,n)/2000 + ddt(c*V(mid,n));
  end
endmodule""",
        "branch": """module conditional_branch(p,n);
  inout p,n; electrical p,n,mid;
  parameter integer enabled=0; parameter real r=1000;
  analog begin
    if (enabled) V(p,mid) <+ r*I(p,mid);
    else V(p,mid) <+ 0;
    I(mid,n) <+ V(mid,n)/r + ddt(1e-12*V(mid,n));
  end
endmodule""",
        "variant": """module variant(p,n);
  inout p,n; electrical p,n,mid;
  parameter real r=1000;
  analog begin
    if (r>0 && $temperature<310 && $simparam("collapse",0)==0 && !analysis("ac"))
      I(p,mid) <+ V(p,mid)/r;
    else V(p,mid) <+ 0;
    I(mid,n) <+ V(mid,n)/2000 + ddt(1e-12*V(mid,n));
  end
endmodule""",
        "terminal_resistor": """module terminal_resistor(p,n);
  inout p,n; electrical p,n;
  parameter real r=1000;
  analog begin
    if (r==0) V(p,n) <+ 0;
    else V(p,n) <+ r*I(p,n);
  end
endmodule""",
        "shorts": """module shorts(p,n);
  inout p,n; electrical p,n;
  parameter integer mode=0;
  analog begin
    if (mode==1) V(p,n) <+ 0; else I(p,n) <+ V(p,n)/1000;
    if (mode==2) V(p) <+ 0;
  end
endmodule""",
    }
    result = {}
    for name, source in sources.items():
        va, binary = root / f"{name}.va", root / f"{name}.osdi"
        va.write_text('`include "disciplines.vams"\n' + source)
        subprocess.run([compiler, str(va), "-o", str(binary)], check=True, capture_output=True)
        result[name] = binary
    return result


def build(binary, analysis, reduced=True, settings=None, temperature=300, simparams=None):
    from bosdi.circulax import osdi_component

    desc = osdi_component(str(binary), ("p", "n"), analysis=analysis, temperature=temperature, simparams=simparams)
    settings = settings or [{"r": 0}, {"r": 1000}]
    net = {
        "instances": {
            **{f"d{i}": {"component": "device", "settings": value} for i, value in enumerate(settings)},
            "GND": {"component": "ground"},
        },
        "connections": {
            **{f"d{i},n": "GND,p1" for i in range(len(settings))},
            **{f"d{i},p": "d0,p" for i in range(1, len(settings))},
        },
        "ports": {"out": "d0,p"},
    }
    return compile_netlist(net, {"device": desc}, osdi_reduction=reduced), desc


def dense(group, values, size):
    result = np.zeros((size, size), dtype=np.asarray(values).dtype)
    np.add.at(result, (np.asarray(group.jac_rows), np.asarray(group.jac_cols)), np.asarray(values).ravel())
    return result


def vector(group, values, size):
    result = np.zeros(size)
    np.add.at(result, np.asarray(group.eq_indices).ravel(), np.asarray(values).ravel())
    return result


@pytest.mark.parametrize("analysis", ["dc", "tran", "ac"])
@pytest.mark.parametrize("kind", ["rc", "branch"])
def test_projection_and_solutions(binaries, analysis, kind):
    settings = [{"enabled": 0}, {"enabled": 1}] if kind == "branch" else None
    (groups, size, ports), desc = build(binaries[kind], analysis, settings=settings)
    (full_groups, full_size, _), _ = build(binaries[kind], analysis, False, settings=settings)
    group, full = groups["device"], full_groups["device"]
    topology = group.handle.topology
    assert size < full_size
    assert group.var_indices.shape == full.var_indices.shape
    assert group.states.shape == full.states.shape
    # Projection P gathers reduced values into all full unknowns, including aliases.
    P = np.zeros((full_size, size))
    for raw_full, raw_reduced in zip(np.asarray(full.var_indices).ravel(), np.asarray(group.var_indices).ravel(), strict=True):
        if raw_reduced:
            P[raw_full, raw_reduced] = 1
    y = np.linspace(0.1, 0.4, size)
    y[0] = 0.73  # ground must be literal zero at native evaluation
    y_full = P @ y
    yf = jnp.asarray(y_full)
    yr = jnp.asarray(y)
    f, q, j = _assemble_osdi_group(yr, group, 1, 1e-10)
    f0, q0, j0 = _assemble_osdi_group(yf, full, 1, 1e-10)
    # Native physical equations already occupy representatives. Equality rows
    # must be selected away (not numerically canceled against tiny stamps).
    mask = np.asarray(topology.physical_row_mask) & (np.asarray(group.var_indices) != 0)
    np.testing.assert_allclose(vector(group, f, size), vector(group, np.where(mask, f0, 0), size), atol=1e-15)
    np.testing.assert_allclose(vector(group, q, size), vector(group, np.where(mask, q0, 0), size), atol=1e-25)
    expected = np.zeros((size, size))
    raw = np.asarray(j0).reshape(len(settings or [0, 1]), group.num_nodes, group.num_nodes)
    for d in range(raw.shape[0]):
        for r, c in np.ndindex(raw.shape[1:]):
            gr, gc = np.asarray(group.var_indices)[d, [r, c]]
            if mask[d, r] and gc:
                expected[gr, gc] += raw[d, r, c]
    np.testing.assert_allclose(dense(group, j, size), expected, atol=1e-14)
    fres, qres, _ = _assemble_osdi_group(yr, group, 1, 1e-10, residual_only=True)
    np.testing.assert_allclose(fres, f, atol=1e-15)
    np.testing.assert_allclose(qres, q, atol=1e-25)
    g, c = _assemble_osdi_gc_separate(jnp.zeros(size), group)
    gf, cf = _assemble_osdi_gc_separate(jnp.zeros(full_size), full)
    G, C = dense(group, g, size), dense(group, c, size)
    GF, CF = dense(full, gf, full_size), dense(full, cf, full_size)
    # Solve the full equality-constrained DAE independently, then compare.
    previous = np.zeros(size)
    previous_full = np.zeros(full_size)
    for coefficient in [0] if analysis == "dc" else [2j * np.pi * 1e9] if analysis == "ac" else [1e10] * 8:
        A = G + coefficient * C
        AF = GF + coefficient * CF
        rhs = coefficient * C @ previous
        rhsf = coefficient * CF @ previous_full
        A[0] = 0
        A[0, 0] = 1
        rhs[0] = 0
        AF[0] = 0
        AF[0, 0] = 1
        rhsf[0] = 0
        node = ports["d0,p"]
        full_node = np.asarray(full.var_indices)[0, 0]
        A[node] = 0
        A[node, node] = 1
        rhs[node] = 1
        AF[full_node] = 0
        AF[full_node, full_node] = 1
        rhsf[full_node] = 1
        previous = np.linalg.solve(A, rhs)
        previous_full = np.linalg.solve(AF, rhsf)
        np.testing.assert_allclose(P @ previous, previous_full, atol=1e-12)
    # Finite difference Jacobian includes repeated global indices.
    if analysis != "dc":
        eps = 1e-6
        numeric = []
        for i in range(size):
            yp = y.copy()
            ym = y.copy()
            yp[i] += eps
            ym[i] -= eps
            fp, qp, _ = _assemble_osdi_group(jnp.asarray(yp), group, 1, 1e-10)
            fm, qm, _ = _assemble_osdi_group(jnp.asarray(ym), group, 1, 1e-10)
            numeric.append(vector(group, (fp - fm + (qp - qm) / 1e-10) / (2 * eps), size))
        np.testing.assert_allclose(np.array(numeric).T, expected, atol=1e-11)


def test_updates_and_binding(binaries):
    (groups, size, _), desc = build(binaries["rc"], "tran")
    group = groups["device"]
    col = desc.param_names.index("r")
    changed = group.with_params(group.params.at[1, col].set(2000))
    assert changed.handle.handle_id != group.handle.handle_id
    _assemble_osdi_group(jnp.zeros(size), changed, 1, 1)
    with pytest.raises(ValueError, match="changes topology"):
        group.with_params(group.params.at[0, col].set(1000))
    with pytest.raises(TypeError, match="Traced"):
        jax.jit(lambda x: group.with_params(group.params.at[1, col].set(x)).params)(2000.0)
    with pytest.raises(ValueError, match="configured handle"):
        group.without_handle()
    stale = eqx.tree_at(lambda g: g.params, group, group.params.at[1, col].set(2000))
    with pytest.raises(Exception, match="with_params"):
        _assemble_osdi_group(jnp.zeros(size), stale, 1, 1)[0].block_until_ready()
    # Ordinary reduced assembly remains JIT-compatible after pytree flattening.
    jax.jit(lambda g: _assemble_osdi_group(jnp.zeros(size), g, 1, 1)[0])(group).block_until_ready()


def test_terminal_unions(binaries):
    from bosdi.circulax import osdi_component

    desc = osdi_component(str(binaries["shorts"]), ("p", "n"))
    net = {
        "instances": {
            "a": {"component": "s", "settings": {"mode": 1}},
            "b": {"component": "s", "settings": {"mode": 2}},
            "r": {"component": "s"},
            "GND": {"component": "ground"},
        },
        "connections": {"a,n": "b,n", "b,p": "r,p", "r,n": "a,p"},
        "ports": {"out": "r,p"},
    }
    groups, size, ports = compile_netlist(net, {"s": desc})
    assert ports["a,p"] == ports["a,n"] == ports["b,n"]
    assert ports["b,p"] == 0
    assert ports["r,p"] == 0
    assert size == 2  # setup grounds r,p; a/b terminals union; flow slots inactive
    assert np.any(np.asarray(groups["s"].var_indices) == 0)
    # Historical full raw allocation cannot safely represent external KCL at
    # setup-shorted terminals. Reject that opt-out explicitly, even when a
    # traced update could activate a candidate later.
    with pytest.raises(ValueError, match="terminal-short candidates"):
        compile_netlist(net, {"s": desc}, osdi_reduction=False)


def test_registration_and_old_api(binaries, monkeypatch):
    (groups, _, _), desc = build(binaries["rc"], "tran", temperature=330, simparams={"scale": 2})
    group = groups["device"]
    np.testing.assert_array_equal(group.representatives, group.handle.topology.raw_to_representative)
    import osdi_jax

    original = osdi_jax.osdi_setup_batch

    class OldHandle:
        def __init__(self, handle):
            self.handle = handle

    monkeypatch.setattr(osdi_jax, "osdi_setup_batch", lambda *args: OldHandle(original(*args)))
    (_, full_size, _), _ = build(binaries["rc"], "tran")
    assert full_size == 4


def test_public_transient_ac_and_history(binaries):
    from bosdi.circulax import osdi_component

    from circulax import compile_circuit
    from circulax.components.electronic import SmoothPulse
    from circulax.solvers.adjoint import _build_jq_total, _jq_matvec_klu
    from circulax.solvers.assembly import assemble_gc_real
    from circulax.solvers.transient import _compute_history

    desc = osdi_component(str(binaries["rc"]), ("p", "n"), analysis="tran")
    net = {
        "instances": {
            "v": {"component": "v", "settings": {"V": 1, "delay": 1e-9, "tr": 1e-9}},
            "a": {"component": "rc", "settings": {"r": 0}},
            "b": {"component": "rc", "settings": {"r": 1000}},
            "GND": {"component": "ground"},
        },
        "connections": {"v,p1": "a,p", "v,p2": "GND,p1", "a,n": "GND,p1", "b,p": "a,p", "b,n": "GND,p1"},
        "ports": {"out": "a,p"},
    }
    circuits = [
        compile_circuit(net, {"v": SmoothPulse, "rc": desc}, backend="dense", g_leak=0, osdi_reduction=reduce)
        for reduce in [True, False]
    ]
    outputs = []
    for circuit in circuits:
        ydc = circuit.dc()
        outputs.append((np.asarray(circuit.port(ydc, "out")), np.asarray(circuit.sp(ports="out", freqs=jnp.array([1e6, 1e9])))))
        sol = circuit.transient(
            t0=0, t1=1e-8, dt0=1e-10, y0=jnp.zeros(circuit.sys_size), saveat=jnp.linspace(0, 1e-8, 20), max_steps=110
        )
        assert np.isfinite(sol.ys).all()
        outputs[-1] += (np.asarray(circuit.port(sol.ys, "out")), np.asarray(sol.ys[:, circuit.port_map["b,__osdi_2"]]))
        group = circuit.groups["rc"]
        y = jnp.linspace(0, 1, circuit.sys_size)
        q = _compute_history(circuit.groups, y, 0, circuit.sys_size)
        expected_q = jnp.zeros(circuit.sys_size)
        _, local_q, _ = _assemble_osdi_group(y, group, 1, 1, residual_only=True)
        expected_q = expected_q.at[group.eq_indices].add(local_q)
        np.testing.assert_allclose(q, expected_q, atol=1e-25)
        _, cs = assemble_gc_real(y, circuit.groups)
        rows = np.concatenate([np.asarray(circuit.groups[k].jac_rows).ravel() for k in sorted(circuit.groups)])
        cols = np.concatenate([np.asarray(circuit.groups[k].jac_cols).ravel() for k in sorted(circuit.groups)])
        expected_c = np.zeros((circuit.sys_size, circuit.sys_size))
        np.add.at(expected_c, (rows, cols), cs)
        np.testing.assert_allclose(_build_jq_total(circuit.groups, y, circuit.sys_size), expected_c, atol=1e-25)
        np.testing.assert_allclose(_jq_matvec_klu(circuit.groups, y, 1e-10, y), expected_c.T @ y / 1e-10, atol=1e-15)
    for reduced, full in zip(*outputs, strict=True):
        np.testing.assert_allclose(reduced, full, rtol=1e-9, atol=1e-12)


def test_analysis_and_settings_cannot_reuse_topology(binaries):
    from bosdi.circulax import osdi_component

    from circulax import compile_circuit

    desc = osdi_component(str(binaries["variant"]), ("p", "n"), analysis="tran", temperature=300)
    net = {
        "instances": {"r": {"component": "r"}, "GND": {"component": "ground"}},
        "connections": {"r,n": "GND,p1"},
        "ports": {"out": "r,p"},
    }
    circuit = compile_circuit(net, {"r": desc}, backend="dense", g_leak=0)
    ac = circuit.prepare_analysis("ac_response")
    changed = circuit.set_simparams({"collapse": 1})
    warm = compile_circuit(
        net, {"r": osdi_component(str(binaries["variant"]), ("p", "n"), analysis="tran", temperature=330)}, backend="dense"
    )
    assert len({c.groups["r"].model_id for c in (circuit, ac, changed, warm)}) == 4
    # OpenVAF retains these runtime-conditioned slots. Follow actual maps,
    # without predicting collapse from temperature or analysis expressions.
    for c in (circuit, ac, changed, warm):
        group = c.groups["r"]
        mapping = group.handle.topology.raw_to_representative
        np.testing.assert_array_equal(mapping, [[0, 1, 2]])
        assert c.sys_size == 3
    # A registration factory returning a different actual map must fail before
    # any old operating point or sparse factorisation can be reused.
    collapsed = compile_circuit(net, {"r": osdi_component(str(binaries["rc"]), ("p", "n"), default_params={"r": 0})})
    expanded = compile_circuit(net, {"r": osdi_component(str(binaries["rc"]), ("p", "n"), default_params={"r": 1000})})
    collapsed._analysis_factory = lambda _mode: expanded
    with pytest.raises(ValueError, match="topology"):
        collapsed.prepare_analysis("ac_response")


def test_small_physical_stamp_survives_equality_rows(binaries):
    from circulax.osdi_topology import matrix_values

    (groups, _, _), _ = build(binaries["rc"], "tran")
    group = groups["device"]
    mapping = group.handle.topology.raw_to_representative
    raw = np.zeros((2, group.num_nodes, group.num_nodes))
    raw[0, 0, 0] = 1e-24
    redundant = int(np.flatnonzero(mapping[0] != np.arange(group.num_nodes))[0])
    raw[0, redundant, redundant] = 1
    raw[0, redundant, mapping[0, redundant]] = -1
    values = matrix_values(group, jnp.asarray(raw))
    assert float(jnp.sum(values)) == pytest.approx(1e-24, rel=1e-15, abs=0)


def test_reduced_parameter_derivatives(binaries):
    from circulax.osdi_topology import checked_residual, gather
    from circulax.solvers.adjoint import _compute_transient_fd_gradients
    from circulax.solvers.sensitivity import _compute_fd_gradients

    (groups, size, _), desc = build(binaries["rc"], "tran", settings=[{"r": 0, "c": 1e-12}, {"r": 1000, "c": 1e-12}])
    group = groups["device"]
    ccol = desc.param_names.index("c")
    rcol = desc.param_names.index("r")
    y = jnp.linspace(0, 1, size)
    lam = jnp.arange(size, dtype=jnp.float64)
    dc_grad = _compute_fd_gradients(group, y, lam, ["c"], [ccol], 1e-8, None)
    np.testing.assert_allclose(dc_grad["c"], 0, atol=1e-15)
    with pytest.raises(ValueError, match="changes topology"):
        _compute_fd_gradients(group, y, lam, ["r"], [rcol], 1e-8, None)
    acc = np.zeros((1, 2))
    dt = 1e-10
    _compute_transient_fd_gradients(group, y, jnp.zeros(size), lam, dt, size, ["c"], [ccol], 1e-8, acc)
    zero_c = group.params.at[:, ccol].set(0)
    _, q, _ = checked_residual(group, gather(group, y), group.params)
    _, qzero, _ = checked_residual(group, gather(group, y), zero_c)
    expected = -jnp.sum((q - qzero) / 1e-12 / dt * lam[group.eq_indices], axis=1)
    np.testing.assert_allclose(acc[0], expected, rtol=1e-12, atol=1e-8)


def test_public_passive_ac_matches_analytic_mixed_batch(binaries):
    from bosdi.circulax import osdi_component
    from circulax import compile_circuit

    desc = osdi_component(str(binaries["rc"]), ("p", "n"), analysis="tran")
    net = {
        "instances": {
            "a": {"component": "r", "settings": {"r": 0}},
            "b": {"component": "r", "settings": {"r": 1000}},
            "GND": {"component": "ground"},
        },
        "connections": {"a,n": "GND,p1", "b,n": "GND,p1", "b,p": "a,p"},
        "ports": {"out": "a,p"},
    }
    frequency = jnp.asarray([1e3, 1e6, 1e9])
    admittance = 1 / 2000 + 2j * jnp.pi * frequency * 1e-12
    expected = 2 / (1 + 50 * (admittance + 1 / (1000 + 1 / admittance))) - 1
    for reduced in (True, False):
        circuit = compile_circuit(net, {"r": desc}, backend="dense", g_leak=0, osdi_reduction=reduced)
        actual = circuit.sp(ports="out", freqs=frequency)[:, 0, 0]
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-13)


def test_initial_identity_with_possible_terminal_shorts_keeps_update_guard(binaries):
    from bosdi.circulax import osdi_component
    from circulax.osdi_topology import ReducedOsdiGroup

    desc = osdi_component(str(binaries["terminal_resistor"]), ("p", "n"))
    net = {
        "instances": {"r": {"component": "r", "settings": {"r": 1000}}, "GND": {"component": "ground"}},
        "connections": {"r,n": "GND,p1"},
        "ports": {"out": "r,p"},
    }
    groups, _, _ = compile_netlist(net, {"r": desc})
    group = groups["r"]
    np.testing.assert_array_equal(group.handle.topology.raw_to_representative, [[0, 1, 2]])
    assert isinstance(group, ReducedOsdiGroup)
    col = desc.param_names.index("r")
    changed = group.with_params(group.params.at[0, col].set(2000))
    np.testing.assert_array_equal(changed.var_indices, group.var_indices)
    with pytest.raises(ValueError, match="changes topology"):
        group.with_params(group.params.at[0, col].set(0))
    with pytest.raises(ValueError, match="terminal-short candidates"):
        compile_netlist(net, {"r": desc}, osdi_reduction=False)
