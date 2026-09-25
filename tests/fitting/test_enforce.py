"""Tests for Phase 3 — passivity enforcement."""

import jax.numpy as jnp
import numpy as np

from circulax.fitting.passivity.check import passivity_sweep_Y
from circulax.fitting.passivity.enforce import (
    _basis_real_at,
    _build_basis,
    _build_constraint_matrix,
    _build_ls_system,
    enforce_passivity,
)
from circulax.fitting.types import FitOptions, SSModel, VFModel, vfmodel_to_ss
from circulax.fitting.utils import compute_cindex

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _opts(**kwargs):
    defaults = dict(N=4, asymp=2, TOLG=1e-6, TOLD=1e-3, TOLE=1e-12, Niter_out=20, nu=1e-3)
    defaults.update(kwargs)
    return FitOptions(**defaults)


def _make_vfmodel_1port(pole, residue, D_val):
    """1-port VFModel: Y(s) = residue/(s-pole) + D."""
    return VFModel(
        poles=jnp.array([pole], dtype=jnp.complex128),
        residues=jnp.array([[[residue]]], dtype=jnp.complex128),
        D=jnp.array([[D_val]], dtype=jnp.float64),
        E=jnp.array([[0.0]], dtype=jnp.float64),
    )


def _make_non_passive_1port():
    """Non-passive 1-port: Y(s) = -1000/(s+1000) — Re(Y) < 0 everywhere."""
    return _make_vfmodel_1port(pole=-1000.0, residue=-1000.0, D_val=0.0)


def _make_passive_1port():
    """Passive 1-port: Y(s) = 1000/(s+1000) + 0.001 — Re(Y) > 0."""
    return _make_vfmodel_1port(pole=-1000.0, residue=1000.0, D_val=0.001)


def _sweep():
    return 1j * jnp.logspace(2, 6, 60)


# ---------------------------------------------------------------------------
# TestBasisFunctions
# ---------------------------------------------------------------------------


class TestBasisFunctions:
    def test_real_pole_basis_shape(self):
        s = np.array([1j, 2j, 3j])
        poles = np.array([-10.0 + 0j])
        cindex = compute_cindex(poles)
        M = _build_basis(s, poles, cindex, Dflag=False, Eflag=False)
        assert M.shape == (3, 1)

    def test_real_pole_basis_values(self):
        s = np.array([1j * 10.0])
        poles = np.array([-10.0 + 0j])
        cindex = compute_cindex(poles)
        M = _build_basis(s, poles, cindex, Dflag=False, Eflag=False)
        expected = 1.0 / (1j * 10.0 - (-10.0))
        np.testing.assert_allclose(M[0, 0], expected, rtol=1e-12)

    def test_D_column(self):
        s = np.array([1j, 2j])
        poles = np.array([-5.0 + 0j])
        cindex = compute_cindex(poles)
        M = _build_basis(s, poles, cindex, Dflag=True, Eflag=False)
        assert M.shape == (2, 2)
        np.testing.assert_allclose(M[:, 1], [1.0, 1.0], atol=1e-14)

    def test_E_column(self):
        s = np.array([1j * 100.0, 1j * 200.0])
        poles = np.array([-5.0 + 0j])
        cindex = compute_cindex(poles)
        M = _build_basis(s, poles, cindex, Dflag=True, Eflag=True)
        assert M.shape == (2, 3)
        np.testing.assert_allclose(M[:, 2], s, atol=1e-14)

    def test_real_part_at_freq(self):
        poles = np.array([-10.0 + 0j])
        cindex = compute_cindex(poles)
        sk = 1j * 10.0
        dum = _basis_real_at(sk, poles, cindex, Dflag=False, Eflag=False)
        assert dum.shape == (1,)
        # real(1/(j10 + 10)) = real((10-j10)/(100+100)) = 10/200 = 0.05
        np.testing.assert_allclose(dum[0], 0.05, rtol=1e-12)

    def test_complex_pair_basis(self):
        a = -10.0 + 100j
        poles = np.array([a, np.conj(a)])
        cindex = compute_cindex(poles)
        s = np.array([1j * 50.0])
        M = _build_basis(s, poles, cindex, Dflag=False, Eflag=False)
        # First col: 1/(s-a) + 1/(s-conj(a))
        expected_0 = 1 / (s[0] - a) + 1 / (s[0] - np.conj(a))
        np.testing.assert_allclose(M[0, 0], expected_0, rtol=1e-12)


# ---------------------------------------------------------------------------
# TestLSSystem
# ---------------------------------------------------------------------------


class TestLSSystem:
    def test_rsub_shape(self):
        poles = np.array([-1e3, -1e4], dtype=complex)
        cindex = compute_cindex(poles)
        s = np.array(1j * np.logspace(2, 5, 30))
        N = 2
        Nc = 1
        Ndum = N + 0  # no D/E
        nnn = 1
        Rsub, bigEscale = _build_ls_system(
            s,
            poles,
            cindex,
            Nc=Nc,
            Dflag=False,
            Eflag=False,
            nnn=nnn,
            Ns_orig=30,
            weightfactor=1e-3,
        )
        assert Rsub.shape == (Ndum, Ndum)
        assert bigEscale.shape == (nnn * Ndum,)

    def test_rsub_upper_triangular(self):
        poles = np.array([-1e3], dtype=complex)
        cindex = compute_cindex(poles)
        s = np.array(1j * np.logspace(2, 5, 20))
        Rsub, _ = _build_ls_system(
            s,
            poles,
            cindex,
            Nc=1,
            Dflag=False,
            Eflag=False,
            nnn=1,
            Ns_orig=20,
        )
        # R from QR is upper-triangular
        np.testing.assert_allclose(np.tril(Rsub, -1), 0, atol=1e-12)

    def test_bigescale_positive(self):
        poles = np.array([-1e3, -1e4], dtype=complex)
        cindex = compute_cindex(poles)
        s = np.array(1j * np.logspace(2, 5, 30))
        _, bigEscale = _build_ls_system(
            s,
            poles,
            cindex,
            Nc=2,
            Dflag=False,
            Eflag=False,
            nnn=3,
            Ns_orig=30,
        )
        assert np.all(bigEscale > 0)


# ---------------------------------------------------------------------------
# TestConstraintMatrix
# ---------------------------------------------------------------------------


class TestConstraintMatrix:
    def _simple_1port_ss(self):
        pole = -1000.0 + 0j
        ss = SSModel(
            A=jnp.array([pole], dtype=jnp.complex128),
            B=jnp.array([[1.0 + 0j]]),
            C=jnp.array([[-1000.0 + 0j]]),  # negative → non-passive
            D=jnp.array([[0.0 + 0j]]),
            E=jnp.array([[0.0 + 0j]]),
        )
        return ss

    def test_shape(self):
        ss = self._simple_1port_ss()
        poles = np.array([-1000.0 + 0j])
        cindex = compute_cindex(poles)
        Nc = 1
        N = 1
        nnn = 1
        Ndum = 1
        violpairs = [{"group": 0, "omega": 1e4, "lambda_min": -0.5, "eigvec": np.array([1.0 + 0j])}]
        bigB, bigc, offsB = _build_constraint_matrix(
            violpairs,
            poles,
            cindex,
            Nc=Nc,
            N=N,
            Dflag=False,
            Eflag=False,
            nnn=nnn,
            TOLG=1e-6,
            TOLD=1e-3,
            TOLE=1e-12,
            alpha=1.0,
            ss=ss,
        )
        assert bigB.shape[1] == nnn * Ndum
        assert offsB == 1
        assert bigc[0] < 0  # -TOLG + alpha * (-0.5) < 0

    def test_no_violpairs_no_rows(self):
        ss = self._simple_1port_ss()
        poles = np.array([-1000.0 + 0j])
        cindex = compute_cindex(poles)
        bigB, bigc, offsB = _build_constraint_matrix(
            [],
            poles,
            cindex,
            Nc=1,
            N=1,
            Dflag=False,
            Eflag=False,
            nnn=1,
            TOLG=1e-6,
            TOLD=1e-3,
            TOLE=1e-12,
            alpha=1.0,
            ss=ss,
        )
        assert offsB == 0


# ---------------------------------------------------------------------------
# TestEnforcePassivity
# ---------------------------------------------------------------------------


class TestEnforcePassivity:
    def setup_method(self):
        self.s = _sweep()
        self.omega = np.array(jnp.imag(self.s))
        self.opts = _opts(Niter_out=20, TOLG=1e-6, TOLD=1e-3, nu=1e-3)

    def test_passive_model_unchanged(self):
        """A passive model should require 0 iterations and remain passive."""
        model = _make_passive_1port()
        new_model, gmin = enforce_passivity(model, self.s, self.opts, verbose=False)
        _, is_passive = passivity_sweep_Y(vfmodel_to_ss(new_model, 1), self.s)
        assert is_passive

    def test_non_passive_1port_becomes_passive(self):
        """Non-passive 1-port should converge to passive after enforcement."""
        model = _make_non_passive_1port()
        _, is_passive_before = passivity_sweep_Y(vfmodel_to_ss(model, 1), self.s)
        assert not is_passive_before

        new_model, gmin = enforce_passivity(model, self.s, self.opts, verbose=False)
        _, is_passive_after = passivity_sweep_Y(vfmodel_to_ss(new_model, 1), self.s)
        assert is_passive_after, f"Still non-passive: gmin_min={float(np.min(gmin)):.3e}"

    def test_gmin_nonnegative_after_enforcement(self):
        model = _make_non_passive_1port()
        _, gmin = enforce_passivity(model, self.s, self.opts, verbose=False)
        assert float(np.min(gmin)) >= -self.opts.TOLG * 10, f"gmin_min={float(np.min(gmin)):.3e} after enforcement"

    def test_returns_vfmodel(self):
        model = _make_non_passive_1port()
        new_model, gmin = enforce_passivity(model, self.s, self.opts, verbose=False)
        assert isinstance(new_model, VFModel)
        assert gmin.shape == (len(self.s),)

    def test_poles_unchanged(self):
        """Enforcement only perturbs residues, not poles."""
        model = _make_non_passive_1port()
        new_model, _ = enforce_passivity(model, self.s, self.opts, verbose=False)
        np.testing.assert_allclose(np.array(new_model.poles), np.array(model.poles), atol=1e-14)

    def test_mildly_non_passive_2port(self):
        """2-port with slightly large off-diagonal residue should become passive."""
        # Y(s) with pole at -1e3: residue matrix has R12 slightly too large.
        pole = -1000.0 + 0j
        R11, R22, R12 = 2e3, 2e3, 1.8e3  # off-diagonal close to limit

        model = VFModel(
            poles=jnp.array([pole], dtype=jnp.complex128),
            residues=jnp.array([[[R11], [R12]], [[R12], [R22]]], dtype=jnp.complex128),
            D=jnp.array([[0.001, 0.0], [0.0, 0.001]], dtype=jnp.float64),
            E=jnp.array([[0.0, 0.0], [0.0, 0.0]], dtype=jnp.float64),
        )
        new_model, gmin = enforce_passivity(model, self.s, self.opts, verbose=False)
        assert float(np.min(gmin)) >= -self.opts.TOLG * 10
