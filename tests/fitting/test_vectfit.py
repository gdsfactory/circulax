"""Integration-style tests: single VF iteration and end-to-end driver."""

import jax.numpy as jnp
import numpy as np

from circulax.fitting.driver import vfdriver
from circulax.fitting.types import FitOptions
from circulax.fitting.utils import init_poles_logcmplx


def _make_rational_admittance(poles, residues, D, s):
    """Evaluate H(s) = sum(R_m/(s-a_m)) + D for a 1-port."""
    H = jnp.zeros(s.shape, dtype=jnp.complex128)
    for a, R in zip(poles, residues):
        H = H + R / (s - a)
    return H + D


def _make_2port(s, poles, R11, R22, R12, D11, D22, D12):
    Y11 = _make_rational_admittance(poles, R11, D11, s)
    Y22 = _make_rational_admittance(poles, R22, D22, s)
    Y12 = _make_rational_admittance(poles, R12, D12, s)
    bigH = jnp.stack(
        [
            jnp.stack([Y11, Y12], axis=0),
            jnp.stack([Y12, Y22], axis=0),
        ],
        axis=0,
    )
    return bigH


class TestVfdriver:
    """End-to-end driver tests using synthetic rational data."""

    def setup_method(self):
        self.Ns = 100
        self.s = 1j * jnp.logspace(2, 6, self.Ns)
        self.true_poles = [-1e3, -1e4]
        self.bigH = _make_2port(
            self.s,
            poles=self.true_poles,
            R11=[1e3, 2e4],
            R22=[3e3, 1e4],
            R12=[-5e2, -1e4],
            D11=0.001,
            D22=0.002,
            D12=0.0005,
        )

    def test_proper_rational_fit(self):
        """Fitting exact rational data should give near-machine-precision error."""
        poles = init_poles_logcmplx(self.s, N=4)
        opts = FitOptions(N=4, Niter1=3, Niter2=4, asymp=2)
        _, _, err, _ = vfdriver(self.bigH, self.s, poles, opts, verbose=False)
        assert err < 1e-10, f"RMS error {err:.2e} too large for exact rational data"

    def test_poles_converge_to_true(self):
        """Fitted poles should include the two true poles."""
        poles = init_poles_logcmplx(self.s, N=4)
        opts = FitOptions(N=4, Niter1=3, Niter2=5, asymp=2)
        model, _, _, _ = vfdriver(self.bigH, self.s, poles, opts, verbose=False)

        fitted = sorted(np.abs(np.real(np.array(model.poles))))
        # Two true poles plus two extra poles (which should be very large or negligible)
        # Check at least two poles are within 10% of true
        found = set()
        for true_p in [1e3, 1e4]:
            for p in fitted:
                if abs(p - true_p) / true_p < 0.1:
                    found.add(true_p)
                    break
        assert len(found) == 2, f"Only found {found} of true poles in {fitted}"

    def test_output_shapes(self):
        """Check VFModel and SSModel have correct array shapes."""
        N = 6
        poles = init_poles_logcmplx(self.s, N=N)
        opts = FitOptions(N=N, Niter1=2, Niter2=2, asymp=2)
        model, ss, _, Hfit = vfdriver(self.bigH, self.s, poles, opts, verbose=False)
        Nc = 2

        assert model.poles.shape == (N,)
        assert model.residues.shape == (Nc, Nc, N)
        assert model.D.shape == (Nc, Nc)
        assert model.E.shape == (Nc, Nc)
        assert ss.A.shape == (Nc * N,)
        assert ss.B.shape == (Nc * N, Nc)
        assert ss.C.shape == (Nc, Nc * N)
        assert Hfit.shape == (Nc, Nc, self.Ns)

    def test_hfit_matches_model_eval(self):
        """Hfit from vfdriver should match direct model evaluation."""
        from circulax.fitting.types import eval_model

        poles = init_poles_logcmplx(self.s, N=4)
        opts = FitOptions(N=4, Niter1=2, Niter2=2, asymp=2)
        model, ss, _, Hfit = vfdriver(self.bigH, self.s, poles, opts, verbose=False)

        Hfit_direct = eval_model(self.s, ss)  # (Ns, Nc, Nc)
        Hfit_direct = jnp.moveaxis(Hfit_direct, 0, -1)  # (Nc, Nc, Ns)
        np.testing.assert_allclose(np.array(Hfit), np.array(Hfit_direct), atol=1e-12)

    def test_reciprocal_complex_fit_preserves_transpose_symmetry(self):
        """Reciprocity mirrors residues without conjugating them."""
        poles = np.array([-2e3 - 3e4j, -2e3 + 3e4j])
        bigH = _make_2port(
            self.s,
            poles=poles,
            R11=[1e3 + 2e2j, 1e3 - 2e2j],
            R22=[2e3 + 3e2j, 2e3 - 3e2j],
            R12=[-5e2 + 1e2j, -5e2 - 1e2j],
            D11=0.001,
            D22=0.002,
            D12=0.0005,
        )
        opts = FitOptions(N=2, Niter1=3, Niter2=4, asymp=2)
        model, _, error, fitted = vfdriver(bigH, self.s, poles, opts, verbose=False)

        assert error < 1e-10
        np.testing.assert_allclose(model.residues[0, 1], model.residues[1, 0], atol=1e-12)
        np.testing.assert_allclose(fitted[0, 1], fitted[1, 0], atol=1e-12)

    def test_none_poles_auto_init(self):
        """Passing poles=None should auto-generate logarithmically-spaced poles."""
        opts = FitOptions(N=4, Niter1=2, Niter2=2, asymp=2)
        model, _, err, _ = vfdriver(self.bigH, self.s, None, opts, verbose=False)
        assert err < 1.0  # just verify it runs without error

    def test_single_port(self):
        """1-port (Nc=1) fitting."""
        true_poles = [-500.0, -5000.0]
        Y = _make_rational_admittance(true_poles, [1e2, 1e3], 0.01, self.s)
        bigH1 = Y[None, None, :]  # (1, 1, Ns)

        opts = FitOptions(N=4, Niter1=3, Niter2=4, asymp=2)
        model, _, err, _ = vfdriver(bigH1, self.s, None, opts, verbose=False)
        assert err < 1e-10

    def test_asymp3_improper(self):
        """Fitting improper data (E term) with asymp=3."""
        # Y = 1/R + s*C (improper — linear in s)
        C_val, R_val = 1e-6, 1e3
        Y = 1 / R_val + self.s * C_val
        bigH1 = Y[None, None, :]

        opts = FitOptions(N=4, Niter1=3, Niter2=4, asymp=3)
        model, _, err, _ = vfdriver(bigH1, self.s, None, opts, verbose=False)
        assert err < 1e-5, f"asymp=3 fit error {err:.2e} too large"
