"""Tests for the AAA rational approximation driver (circulax.fitting/aaa.py)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from circulax.fitting.aaa import _aaa_poles, aaa_driver, aaa_scalar
from circulax.fitting.types import FitOptions, SSModel, VFModel, eval_model

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rational(z, poles, residues, D=0.0):
    """Evaluate H(z) = sum(R_m/(z - p_m)) + D."""
    H = np.full(z.shape, D, dtype=np.complex128)
    for p, R in zip(poles, residues):
        H = H + R / (z - p)
    return H


def _make_bigH_1port(s, poles, residues, D=0.0):
    """Build (1, 1, Ns) bigH for a single-port rational function."""
    H = _make_rational(np.asarray(s), poles, residues, D)
    return jnp.array(H[None, None, :])


def _make_bigH_2port(s, poles, R11, R22, R12, D11=0.0, D22=0.0, D12=0.0):
    """Build (2, 2, Ns) bigH for a two-port rational function."""
    s_np = np.asarray(s)
    Y11 = _make_rational(s_np, poles, R11, D11)
    Y22 = _make_rational(s_np, poles, R22, D22)
    Y12 = _make_rational(s_np, poles, R12, D12)
    bigH = np.array(
        [
            [Y11, Y12],
            [Y12, Y22],
        ]
    )  # (2, 2, Ns)
    return jnp.array(bigH)


# ---------------------------------------------------------------------------
# Test 1: aaa_scalar recovers a known rational function
# ---------------------------------------------------------------------------


class TestAaaScalarMatchesRational:
    """aaa_scalar should reconstruct an exact rational function."""

    def setup_method(self):
        # Three poles: two complex conjugate + one real
        self.true_poles = [-1e3 - 2e3j, -1e3 + 2e3j, -5e3]
        self.true_residues = [1e4 + 2e4j, 1e4 - 2e4j, 3e3]
        self.Ns = 200
        self.s = 1j * np.logspace(2, 6, self.Ns)
        self.f = _make_rational(self.s, self.true_poles, self.true_residues)

    def _eval_barycentric(self, w, zj, fj, z):
        """Evaluate barycentric rational approximation at arbitrary points."""
        z = np.asarray(z)
        r = np.empty_like(z, dtype=np.complex128)
        for k, zk in enumerate(z):
            cauchy = 1.0 / (zk - zj)
            N_val = cauchy @ (w * fj)
            D_val = cauchy @ w
            if abs(D_val) < 1e-300:
                # zk is very close to a support node — use nearby value
                closest = np.argmin(np.abs(zk - zj))
                r[k] = fj[closest]
            else:
                r[k] = N_val / D_val
        return r

    def test_reconstructs_rational(self):
        """AAA should reproduce the rational function to within tol."""
        w, zj, fj = aaa_scalar(self.f, self.s, tol=1e-10, mmax=100)

        # Evaluate at the original sample points
        r = self._eval_barycentric(w, zj, fj, self.s)

        # Compare only at non-support points (support points are exact by construction)
        support_set = set(map(complex, zj))
        non_sup_mask = np.array([complex(si) not in support_set for si in self.s])

        rel_err = np.abs(r[non_sup_mask] - self.f[non_sup_mask]) / (np.abs(self.f[non_sup_mask]) + 1e-300)
        assert np.max(rel_err) < 1e-6, f"Max relative error {np.max(rel_err):.2e} exceeds 1e-6"

    def test_aaa_converges(self):
        """aaa_scalar should converge (return non-empty weights)."""
        w, zj, fj = aaa_scalar(self.f, self.s, tol=1e-10, mmax=100)
        assert len(w) > 0
        assert len(zj) == len(w)
        assert len(fj) == len(w)


# ---------------------------------------------------------------------------
# Test 2: _aaa_poles recovers known poles
# ---------------------------------------------------------------------------


class TestAaaPolesCorrect:
    """_aaa_poles should recover the poles of a simple rational function."""

    def test_two_real_poles(self):
        """f(z) = 1/(z+1) + 1/(z+2) has poles at -1 and -2."""
        # Sample on imaginary axis — poles at -1, -2 (real half-plane)
        s = 1j * np.linspace(0.1, 10.0, 500)
        f = 1.0 / (s + 1) + 1.0 / (s + 2)

        w, zj, fj = aaa_scalar(f, s, tol=1e-12, mmax=100)
        pols = _aaa_poles(w, zj)

        # Find poles closest to -1 and -2
        true_poles = [-1.0, -2.0]
        for tp in true_poles:
            dists = np.abs(pols - tp)
            assert np.min(dists) < 0.01, f"No pole found near {tp}: closest distance = {np.min(dists):.4f}, poles = {pols}"

    def test_complex_pole_pair(self):
        """f(z) = 1/(z+1+2j) + 1/(z+1-2j) has poles at -1±2j."""
        s = 1j * np.linspace(0.1, 10.0, 500)
        p1 = -1.0 + 2.0j
        p2 = -1.0 - 2.0j
        f = 1.0 / (s - p1) + 1.0 / (s - p2)

        w, zj, fj = aaa_scalar(f, s, tol=1e-12, mmax=100)
        pols = _aaa_poles(w, zj)

        for tp in [p1, p2]:
            dists = np.abs(pols - tp)
            assert np.min(dists) < 0.01, f"No pole found near {tp}: closest = {np.min(dists):.4f}"


# ---------------------------------------------------------------------------
# Test 3: aaa_driver on 1-port
# ---------------------------------------------------------------------------


class TestAaaDriver1Port:
    """aaa_driver should fit a 1-port synthetic admittance."""

    def setup_method(self):
        # Two complex conjugate pairs in left half-plane
        self.poles = [-500.0 - 1000j, -500.0 + 1000j, -5000.0 - 8000j, -5000.0 + 8000j]
        self.residues = [1e5 + 2e5j, 1e5 - 2e5j, 3e5 + 1e5j, 3e5 - 1e5j]
        self.Ns = 200
        self.s = 1j * jnp.logspace(2, 5, self.Ns)
        self.bigH = _make_bigH_1port(self.s, self.poles, self.residues, D=0.001)
        self.opts = FitOptions(N=4, asymp=2, weightparam=1)

    def test_rmserr_small(self):
        """RMS error should be small for exact rational data."""
        _, _, rmserr, _ = aaa_driver(self.bigH, self.s, self.opts, tol=1e-10, mmax=100, verbose=False)
        assert rmserr < 1e-4, f"rmserr = {rmserr:.2e} exceeds 1e-4"

    def test_eval_model_consistent(self):
        """eval_model on the returned SSModel should be self-consistent."""
        model, ss, _, bigHfit = aaa_driver(self.bigH, self.s, self.opts, tol=1e-10, mmax=100, verbose=False)
        Hfit_eval = eval_model(self.s, ss)  # (Ns, 1, 1)
        Hfit_eval = jnp.moveaxis(Hfit_eval, 0, -1)  # (1, 1, Ns)
        np.testing.assert_allclose(np.array(bigHfit), np.array(Hfit_eval), atol=1e-10)


# ---------------------------------------------------------------------------
# Test 4: aaa_driver on 2-port
# ---------------------------------------------------------------------------


class TestAaaDriver2Port:
    """aaa_driver should fit a 2-port synthetic admittance matrix."""

    def setup_method(self):
        self.poles = [-1e3 - 2e3j, -1e3 + 2e3j, -8e3 - 5e3j, -8e3 + 5e3j]
        self.Ns = 200
        self.s = 1j * jnp.logspace(2, 5, self.Ns)
        self.bigH = _make_bigH_2port(
            self.s,
            poles=self.poles,
            R11=[2e4 + 1e4j, 2e4 - 1e4j, 5e4 + 3e4j, 5e4 - 3e4j],
            R22=[3e4 + 2e4j, 3e4 - 2e4j, 4e4 + 1e4j, 4e4 - 1e4j],
            R12=[-1e4 - 5e3j, -1e4 + 5e3j, -2e4 - 1e4j, -2e4 + 1e4j],
            D11=0.01,
            D22=0.02,
            D12=0.005,
        )
        self.opts = FitOptions(N=4, asymp=2, weightparam=1)

    def test_rmserr_small(self):
        """RMS error should be < 1e-4 for exact rational 2-port data."""
        _, _, rmserr, _ = aaa_driver(self.bigH, self.s, self.opts, tol=1e-10, mmax=100, verbose=False)
        assert rmserr < 1e-4, f"rmserr = {rmserr:.2e} exceeds 1e-4"

    def test_bigHfit_shape(self):
        """bigHfit should have shape (2, 2, Ns)."""
        _, _, _, bigHfit = aaa_driver(self.bigH, self.s, self.opts, tol=1e-10, mmax=100, verbose=False)
        assert bigHfit.shape == (2, 2, self.Ns)

    def test_poles_stable(self):
        """All VFModel poles should have non-positive real part."""
        model, _, _, _ = aaa_driver(self.bigH, self.s, self.opts, tol=1e-10, mmax=100, verbose=False)
        poles_np = np.array(model.poles)
        assert np.all(poles_np.real <= 0.0), f"Unstable poles found: {poles_np[poles_np.real > 0]}"

    def test_eval_model_matches_bigHfit(self):
        """eval_model should match bigHfit from aaa_driver."""
        model, ss, _, bigHfit = aaa_driver(self.bigH, self.s, self.opts, tol=1e-10, mmax=100, verbose=False)
        Hfit_eval = eval_model(self.s, ss)  # (Ns, 2, 2)
        Hfit_eval = jnp.moveaxis(Hfit_eval, 0, -1)  # (2, 2, Ns)
        np.testing.assert_allclose(np.array(bigHfit), np.array(Hfit_eval), atol=1e-10)


# ---------------------------------------------------------------------------
# Test 5: Return type checks
# ---------------------------------------------------------------------------


class TestAaaDriverReturnsVFModel:
    """aaa_driver return types and shapes."""

    def setup_method(self):
        self.Nc = 2
        self.Ns = 100
        self.s = 1j * jnp.logspace(3, 6, self.Ns)
        poles = [-1e4 - 3e4j, -1e4 + 3e4j]
        self.bigH = _make_bigH_2port(
            self.s,
            poles=poles,
            R11=[1e5 + 2e5j, 1e5 - 2e5j],
            R22=[2e5 + 1e5j, 2e5 - 1e5j],
            R12=[-5e4 - 1e5j, -5e4 + 1e5j],
        )
        self.opts = FitOptions(N=2, asymp=2, weightparam=1)

    def test_isinstance_vfmodel(self):
        """Return value should be a VFModel."""
        model, ss, rmserr, bigHfit = aaa_driver(self.bigH, self.s, self.opts, verbose=False)
        assert isinstance(model, VFModel)

    def test_isinstance_ssmodel(self):
        """Second return value should be a SSModel."""
        model, ss, rmserr, bigHfit = aaa_driver(self.bigH, self.s, self.opts, verbose=False)
        assert isinstance(ss, SSModel)

    def test_bigHfit_shape(self):
        """bigHfit should have shape (Nc, Nc, Ns)."""
        model, ss, rmserr, bigHfit = aaa_driver(self.bigH, self.s, self.opts, verbose=False)
        assert bigHfit.shape == (self.Nc, self.Nc, self.Ns)

    def test_rmserr_is_float(self):
        """rmserr should be a Python float."""
        model, ss, rmserr, bigHfit = aaa_driver(self.bigH, self.s, self.opts, verbose=False)
        assert isinstance(rmserr, float)
