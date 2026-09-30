"""Unit tests for Phase 2 — passivity assessment."""

import jax
import jax.numpy as jnp
import numpy as np

from circulax.fitting.passivity.check import (
    find_violation_bands,
    find_violation_extrema,
    passivity_sweep_Y,
)
from circulax.fitting.passivity.eigtrack import track_eigenvalues
from circulax.fitting.types import SSModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _1port_ss(pole: complex, residue: complex, D_val: complex) -> SSModel:
    """Minimal 1-port SSModel: Y(s) = residue/(s - pole) + D_val."""
    return SSModel(
        A=jnp.array([pole], dtype=jnp.complex128),
        B=jnp.array([[1.0 + 0j]], dtype=jnp.complex128),
        C=jnp.array([[residue]], dtype=jnp.complex128),
        D=jnp.array([[D_val]], dtype=jnp.complex128),
        E=jnp.array([[0.0 + 0j]], dtype=jnp.complex128),
    )


def _2port_ss(poles, R11, R22, R12, D11, D22, D12) -> SSModel:
    """2-port SSModel with N poles."""
    poles = np.asarray(poles, dtype=np.complex128)
    N = len(poles)
    Nc = 2
    A_diag = np.tile(poles, Nc).astype(np.complex128)

    B = np.kron(np.eye(Nc), np.ones((N, 1))).astype(np.complex128)

    # residues: (Nc, Nc, N)
    res = np.zeros((Nc, Nc, N), dtype=np.complex128)
    res[0, 0] = R11
    res[1, 1] = R22
    res[0, 1] = R12
    res[1, 0] = R12
    C = res.reshape(Nc, Nc * N)

    D = np.array([[D11, D12], [D12, D22]], dtype=np.complex128)
    E = np.zeros((Nc, Nc), dtype=np.complex128)

    return SSModel(
        A=jnp.array(A_diag),
        B=jnp.array(B),
        C=jnp.array(C),
        D=jnp.array(D),
        E=jnp.array(E),
    )


# ---------------------------------------------------------------------------
# TestEigtrack
# ---------------------------------------------------------------------------


class TestEigtrack:
    """Tests for track_eigenvalues (intercheig3 equivalent)."""

    def test_identity_permutation(self):
        """Already-aligned eigenvectors → output identical to input."""
        Nc = 3
        V = jnp.eye(Nc, dtype=jnp.complex128)
        vals = jnp.array([1.0, 2.0, 3.0])
        v_out, V_out = track_eigenvalues(vals, V, V)
        np.testing.assert_allclose(np.array(v_out), [1.0, 2.0, 3.0], atol=1e-14)
        np.testing.assert_allclose(np.abs(np.array(V_out)), np.eye(Nc), atol=1e-14)

    def test_swap_permutation(self):
        """Nc=2 swap: new col 0 ↔ col 1 relative to old."""
        V_old = jnp.eye(2, dtype=jnp.complex128)
        V_new = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
        vals_new = jnp.array([2.0, 1.0])  # new col 0 has λ=2, col 1 has λ=1
        v_out, _ = track_eigenvalues(vals_new, V_new, V_old)
        # V_new[:,0]=e1 → matches old col 1 → output channel 1 gets λ=2
        # V_new[:,1]=e0 → matches old col 0 → output channel 0 gets λ=1
        np.testing.assert_allclose(float(v_out[0]), 1.0, atol=1e-14)
        np.testing.assert_allclose(float(v_out[1]), 2.0, atol=1e-14)

    def test_3cycle_permutation(self):
        """Nc=3 cycle: new col k = old col (k+1) mod 3."""
        # V_new columns: [e1, e2, e0]
        V_old = jnp.eye(3, dtype=jnp.complex128)
        V_new = jnp.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=jnp.complex128)
        vals_new = jnp.array([10.0, 20.0, 30.0])
        # new col 0=e1 → old col 1; new col 1=e2 → old col 2; new col 2=e0 → old col 0
        v_out, _ = track_eigenvalues(vals_new, V_new, V_old)
        # output channel 0 = new col matched to old col 0 = new col 2 → λ=30
        # output channel 1 = new col matched to old col 1 = new col 0 → λ=10
        # output channel 2 = new col matched to old col 2 = new col 1 → λ=20
        np.testing.assert_allclose(float(v_out[0]), 30.0, atol=1e-14)
        np.testing.assert_allclose(float(v_out[1]), 10.0, atol=1e-14)
        np.testing.assert_allclose(float(v_out[2]), 20.0, atol=1e-14)

    def test_output_shapes(self):
        Nc = 4
        V = jnp.eye(Nc, dtype=jnp.complex128)
        vals = jnp.arange(Nc, dtype=jnp.float64)
        v_out, V_out = track_eigenvalues(vals, V, V)
        assert v_out.shape == (Nc,)
        assert V_out.shape == (Nc, Nc)

    def test_jit_compilable(self):
        """track_eigenvalues must survive jit compilation."""
        jit_fn = jax.jit(track_eigenvalues)
        V = jnp.eye(2, dtype=jnp.complex128)
        vals = jnp.array([1.0, 2.0])
        v_out, V_out = jit_fn(vals, V, V)
        assert v_out.shape == (2,)

    def test_complex_eigenvectors(self):
        """Works with genuinely complex (non-real) eigenvectors."""
        # Unitary rotation by π/4 in complex plane
        c, s_ = np.cos(np.pi / 4), np.sin(np.pi / 4)
        V_old = jnp.array([[c, -s_], [s_, c]], dtype=jnp.complex128)
        # New = old rotated by a small angle (nearly identical)
        eps = 0.01
        V_new = jnp.array([[c + eps * 1j, -s_], [s_, c - eps * 1j]], dtype=jnp.complex128)
        vals = jnp.array([1.0, 2.0])
        v_out, V_out = track_eigenvalues(vals, V_new, V_old)
        assert v_out.shape == (2,)


# ---------------------------------------------------------------------------
# TestPassivitySweep
# ---------------------------------------------------------------------------


class TestPassivitySweep:
    def setup_method(self):
        self.Ns = 60
        self.s = 1j * jnp.logspace(2, 6, self.Ns)

    def test_passive_1port_real_pole(self):
        """Positive residue at stable pole → Re(Y) > 0 → passive."""
        # Y(s) = 1000/(s+1000) + 0.001  → Re(Y(jω)) = 1000²/(ω²+10⁶) + 0.001 > 0
        ss = _1port_ss(pole=-1000.0, residue=1000.0, D_val=0.001)
        gmin, is_passive = passivity_sweep_Y(ss, self.s)
        assert is_passive
        assert float(jnp.min(gmin)) > 0.0

    def test_non_passive_1port_negative_residue(self):
        """Negative residue at stable pole → Re(Y) < 0 → non-passive."""
        # Y(s) = -1000/(s+1000) → Re(Y(jω)) = -1000²/(ω²+10⁶) < 0
        ss = _1port_ss(pole=-1000.0, residue=-1000.0, D_val=0.0)
        gmin, is_passive = passivity_sweep_Y(ss, self.s)
        assert not is_passive
        assert float(jnp.min(gmin)) < 0.0

    def test_gmin_shape(self):
        ss = _1port_ss(pole=-1000.0, residue=1000.0, D_val=0.001)
        gmin, _ = passivity_sweep_Y(ss, self.s)
        assert gmin.shape == (self.Ns,)

    def test_passive_2port(self):
        """2-port with PD residue matrices and positive D → passive."""
        poles = np.array([-1e3, -1e4], dtype=np.complex128)
        # Symmetric positive-definite residues: diagonally dominant
        ss = _2port_ss(
            poles=poles,
            R11=[5e3, 2e4],
            R22=[3e3, 1e4],
            R12=[1e2, 5e2],
            D11=0.01,
            D22=0.01,
            D12=0.001,
        )
        gmin, is_passive = passivity_sweep_Y(ss, self.s)
        assert is_passive, f"Expected passive but gmin_min={float(jnp.min(gmin)):.3e}"

    def test_non_passive_2port_off_diagonal(self):
        """2-port with large off-diagonal residues → non-passive."""
        poles = np.array([-1e3], dtype=np.complex128)
        ss = _2port_ss(
            poles=poles,
            R11=[1e3],
            R22=[1e3],
            R12=[5e3],  # |R12| > R11, R22
            D11=0.0,
            D22=0.0,
            D12=0.0,
        )
        gmin, is_passive = passivity_sweep_Y(ss, self.s)
        assert not is_passive


# ---------------------------------------------------------------------------
# TestViolationBands
# ---------------------------------------------------------------------------


class TestViolationBands:
    def setup_method(self):
        self.omega = np.linspace(1e2, 1e6, 200)

    def test_no_violations(self):
        gmin = np.ones(200) * 0.1
        assert find_violation_bands(gmin, self.omega) == []

    def test_single_band_middle(self):
        gmin = np.ones(200) * 0.1
        gmin[80:120] = -0.05
        bands = find_violation_bands(gmin, self.omega)
        assert len(bands) == 1
        w1, w2 = bands[0]
        assert w1 <= self.omega[80]
        assert w2 >= self.omega[119]

    def test_two_separate_bands(self):
        gmin = np.ones(200) * 0.1
        gmin[30:60] = -0.02
        gmin[140:170] = -0.03
        bands = find_violation_bands(gmin, self.omega)
        assert len(bands) == 2

    def test_violation_extends_to_end(self):
        gmin = np.ones(200) * 0.1
        gmin[170:] = -0.05
        bands = find_violation_bands(gmin, self.omega)
        assert len(bands) == 1
        _, w2 = bands[0]
        assert w2 > self.omega[-1]

    def test_violation_starts_at_dc(self):
        gmin = np.ones(200) * 0.1
        gmin[:30] = -0.05
        bands = find_violation_bands(gmin, self.omega)
        assert len(bands) == 1
        w1, _ = bands[0]
        assert w1 >= 0.0

    def test_tolg_tolerance(self):
        """Values within TOLG of zero are treated as passive."""
        gmin = np.ones(200) * 0.1
        gmin[50:80] = -0.5e-6  # within default TOLG=1e-6
        bands = find_violation_bands(gmin, self.omega, TOLG=1e-6)
        assert bands == []

    def test_all_violating(self):
        gmin = -np.ones(200) * 0.1
        bands = find_violation_bands(gmin, self.omega)
        assert len(bands) == 1


# ---------------------------------------------------------------------------
# TestViolationExtrema
# ---------------------------------------------------------------------------


class TestViolationExtrema:
    def setup_method(self):
        self.Ns = 60
        self.s = 1j * jnp.logspace(2, 6, self.Ns)
        self.omega = np.array(jnp.imag(self.s))
        # Non-passive 1-port
        self.ss = _1port_ss(pole=-1000.0, residue=-1000.0, D_val=0.0)

    def test_returns_nonempty_for_non_passive(self):
        gmin, _ = passivity_sweep_Y(self.ss, self.s)
        bands = find_violation_bands(np.array(gmin), self.omega)
        pairs = find_violation_extrema(self.ss, self.omega, bands)
        assert len(pairs) > 0

    def test_extrema_have_negative_lambda(self):
        gmin, _ = passivity_sweep_Y(self.ss, self.s)
        bands = find_violation_bands(np.array(gmin), self.omega)
        pairs = find_violation_extrema(self.ss, self.omega, bands)
        for p in pairs:
            assert p["lambda_min"] < 0.0

    def test_extrema_omega_in_band(self):
        gmin, _ = passivity_sweep_Y(self.ss, self.s)
        bands = find_violation_bands(np.array(gmin), self.omega)
        pairs = find_violation_extrema(self.ss, self.omega, bands)
        for p, (w1, w2) in zip(pairs, bands):
            assert w1 <= p["omega"] <= w2 * 1.01  # allow 1% tolerance on clamped edge

    def test_eigvec_shape(self):
        gmin, _ = passivity_sweep_Y(self.ss, self.s)
        bands = find_violation_bands(np.array(gmin), self.omega)
        pairs = find_violation_extrema(self.ss, self.omega, bands)
        Nc = int(self.ss.D.shape[0])
        for p in pairs:
            assert p["eigvec"].shape == (Nc,)

    def test_group_index_in_range(self):
        gmin, _ = passivity_sweep_Y(self.ss, self.s)
        bands = find_violation_bands(np.array(gmin), self.omega)
        pairs = find_violation_extrema(self.ss, self.omega, bands)
        Nc = int(self.ss.D.shape[0])
        for p in pairs:
            assert 0 <= p["group"] < Nc

    def test_empty_for_passive_model(self):
        ss_passive = _1port_ss(pole=-1000.0, residue=1000.0, D_val=0.001)
        gmin, is_passive = passivity_sweep_Y(ss_passive, self.s)
        assert is_passive
        bands = find_violation_bands(np.array(gmin), self.omega)
        assert bands == []
        pairs = find_violation_extrema(ss_passive, self.omega, bands)
        assert pairs == []
