"""Unit tests for circulax.fitting.utils."""

import jax.numpy as jnp
import numpy as np

from circulax.fitting.utils import (
    build_dk,
    compute_cindex,
    compute_rmserr,
    compute_weights,
    init_poles_logcmplx,
    sort_poles,
    stack_upper_triangle,
    unstack_upper_triangle,
)

# ---------------------------------------------------------------------------
# compute_cindex
# ---------------------------------------------------------------------------


class TestCindex:
    def test_all_real(self):
        poles = np.array([-1.0, -2.0, -3.0])
        c = compute_cindex(poles)
        assert list(c) == [0, 0, 0]

    def test_single_complex_pair(self):
        poles = np.array([-1.0 - 10j, -1.0 + 10j])
        c = compute_cindex(poles)
        assert list(c) == [1, 2]

    def test_mixed(self):
        # real, complex pair, real
        poles = np.array([-5.0, -1.0 - 10j, -1.0 + 10j, -50.0])
        c = compute_cindex(poles)
        assert list(c) == [0, 1, 2, 0]

    def test_two_complex_pairs(self):
        poles = np.array([-1 - 1j, -1 + 1j, -2 - 2j, -2 + 2j])
        c = compute_cindex(poles)
        assert list(c) == [1, 2, 1, 2]


# ---------------------------------------------------------------------------
# sort_poles
# ---------------------------------------------------------------------------


class TestSortPoles:
    def test_real_poles_first(self):
        poles = np.array([-1 - 10j, -1 + 10j, -5.0 + 0j])
        result = sort_poles(poles)
        assert result[0].imag == 0.0  # real pole first
        assert result[1].imag < 0  # negative-imag of pair
        assert result[2].imag > 0  # positive-imag of pair

    def test_conjugate_pairs_adjacent(self):
        # Two conjugate pairs — should be interleaved (pair1, pair2)
        poles = np.array([-1 - 1j, -1 + 1j, -10 - 5j, -10 + 5j])
        result = sort_poles(poles)
        c = compute_cindex(result)
        assert list(c) == [1, 2, 1, 2], f"Bad cindex {c} for poles {result}"

    def test_conjugation_cleanup(self):
        # A pole with tiny imaginary part should become real
        tol_pole = np.array([-100.0 + 1e-16j])
        result = sort_poles(tol_pole)
        assert result[0].imag == 0.0


# ---------------------------------------------------------------------------
# build_dk
# ---------------------------------------------------------------------------


class TestBuildDk:
    def setup_method(self):
        self.Ns = 10
        self.s = 1j * np.linspace(1, 100, self.Ns)
        self.s_jax = jnp.array(self.s)

    def test_real_pole_shape(self):
        poles = np.array([-10.0 + 0j])
        cindex = compute_cindex(poles)
        Dk = build_dk(self.s_jax, jnp.array(poles), cindex, offs=0)
        assert Dk.shape == (self.Ns, 1)

    def test_real_pole_values(self):
        poles = np.array([-10.0 + 0j])
        cindex = compute_cindex(poles)
        Dk = build_dk(self.s_jax, jnp.array(poles), cindex, offs=0)
        expected = 1.0 / (self.s - (-10.0))
        np.testing.assert_allclose(np.array(Dk[:, 0]), expected, rtol=1e-10)

    def test_complex_pair_shape_with_offs(self):
        poles = np.array([-1 - 10j, -1 + 10j])
        cindex = compute_cindex(poles)
        Dk = build_dk(self.s_jax, jnp.array(poles), cindex, offs=1)
        assert Dk.shape == (self.Ns, 3)  # 2 poles + 1 constant col

    def test_complex_pair_relationship(self):
        # First col: 1/(s-a) + 1/(s-a*)
        # Second col: 1j*(1/(s-a*) - 1/(s-a)) = i/(s-a) - i/(s-a*)
        a = -1.0 - 10j
        poles = np.array([a, np.conj(a)])
        cindex = compute_cindex(poles)
        Dk = build_dk(self.s_jax, jnp.array(poles), cindex, offs=0)

        col0_expected = 1.0 / (self.s - a) + 1.0 / (self.s - np.conj(a))
        col1_expected = 1j / (self.s - a) - 1j / (self.s - np.conj(a))
        np.testing.assert_allclose(np.array(Dk[:, 0]), col0_expected, rtol=1e-10)
        np.testing.assert_allclose(np.array(Dk[:, 1]), col1_expected, rtol=1e-10)

    def test_offs2_extra_cols(self):
        poles = np.array([-5.0 + 0j])
        cindex = compute_cindex(poles)
        Dk = build_dk(self.s_jax, jnp.array(poles), cindex, offs=2)
        assert Dk.shape == (self.Ns, 3)
        np.testing.assert_allclose(np.array(Dk[:, 1]), np.ones(self.Ns), rtol=1e-12)
        np.testing.assert_allclose(np.array(Dk[:, 2]), np.array(self.s), rtol=1e-12)


# ---------------------------------------------------------------------------
# Upper-triangle stacking
# ---------------------------------------------------------------------------


class TestTriangleStacking:
    def setup_method(self):
        Nc, Ns = 3, 10
        rng = np.random.default_rng(42)
        H = rng.standard_normal((Nc, Nc, Ns)) + 1j * rng.standard_normal((Nc, Nc, Ns))
        # Make symmetric
        H = (H + H.transpose(1, 0, 2)) / 2
        self.H = jnp.array(H)
        self.Nc = Nc

    def test_stack_shape(self):
        f = stack_upper_triangle(self.H)
        nnn = self.Nc * (self.Nc + 1) // 2
        assert f.shape == (nnn, self.H.shape[2])

    def test_roundtrip(self):
        f = stack_upper_triangle(self.H)
        H_rec = unstack_upper_triangle(f, self.Nc)
        np.testing.assert_allclose(np.array(H_rec), np.array(self.H), atol=1e-14)


# ---------------------------------------------------------------------------
# Pole initialisation
# ---------------------------------------------------------------------------


class TestPoleInit:
    def setup_method(self):
        self.s = 1j * jnp.logspace(1, 5, 50)

    def test_logcmplx_count(self):
        poles = init_poles_logcmplx(self.s, N=8)
        assert len(poles) == 8

    def test_logcmplx_stability(self):
        poles = init_poles_logcmplx(self.s, N=8)
        assert np.all(np.real(poles) < 0), "All poles should have negative real part"

    def test_logcmplx_conjugate_pairs(self):
        poles = init_poles_logcmplx(self.s, N=8)
        c = compute_cindex(poles)
        # All should be complex pairs
        assert np.all(c != 0), "All poles should be complex pairs"


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


class TestComputeWeights:
    def setup_method(self):
        Nc, Ns = 2, 20
        H = jnp.ones((Nc, Nc, Ns), dtype=jnp.complex128) * 2.0
        self.H = H
        self.Nc = Nc
        self.Ns = Ns

    def test_uniform_weight(self):
        w = compute_weights(self.H, 1)
        assert w.shape == (1, self.Ns)
        np.testing.assert_allclose(np.array(w), 1.0, atol=1e-12)

    def test_inverse_mag_weight(self):
        w = compute_weights(self.H, 2)
        nnn = self.Nc * (self.Nc + 1) // 2
        assert w.shape == (nnn, self.Ns)
        # H=2 everywhere → weight = 1/2 everywhere
        np.testing.assert_allclose(np.array(w), 0.5, atol=1e-12)

    def test_nonreciprocal_weights_include_every_ordered_response(self):
        w = compute_weights(self.H, 2, reciprocal=False)
        assert w.shape == (self.Nc**2, self.Ns)

    def test_frobenius_weight(self):
        w = compute_weights(self.H, 4)
        assert w.shape == (1, self.Ns)


# ---------------------------------------------------------------------------
# RMS error
# ---------------------------------------------------------------------------


class TestRmserr:
    def test_zero_error(self):
        H = jnp.ones((10, 2, 2), dtype=jnp.complex128)
        err = compute_rmserr(H, H)
        assert abs(err) < 1e-15

    def test_known_error(self):
        H = jnp.zeros((1, 1, 1), dtype=jnp.complex128)
        Hfit = jnp.ones((1, 1, 1), dtype=jnp.complex128)
        err = compute_rmserr(H, Hfit)
        assert abs(err - 1.0) < 1e-12
