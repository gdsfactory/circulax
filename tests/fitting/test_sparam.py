"""Tests for S-parameter preprocessing with delay de-embedding."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from circulax.fitting import FitOptions, VFModel, aaa_driver, eval_model
from circulax.fitting.sparam import (
    CausalityError,
    CausalityWarning,
    _y_to_s,
    deembed_delay,
    embed_delay,
    evaluate_sparameter_model,
    extract_group_delay,
    fit_with_delay,
    s_to_y,
    scattering_state_space_to_admittance,
    scattering_state_space_to_admittance_jax,
    vmap_scattering_state_space_to_admittance,
    y_to_s,
)
from circulax.fitting.types import vfmodel_to_ss


def _synth_delay_rational_2port(freqs, tau, poles=None, residue_matrices=None, Y_const=None, z0=50.0):
    """Synthesize S-parameters from a known rational Y-model plus delay.

    Builds Y(s) = Y_const + sum_k R_k/(s - p_k), converts to S, then adds
    delay via embed_delay. Guarantees the poles exist in Y after de-embedding.

    When poles is None, returns a pure delay line (constant Y).
    """
    from circulax.fitting.sparam import _y_to_s, embed_delay

    Ns = len(freqs)
    s_vals = 1j * 2.0 * np.pi * freqs

    if Y_const is None:
        Y_const = np.array([[0.02, -0.019], [-0.019, 0.02]])

    S_deemb = np.zeros((Ns, 2, 2), dtype=np.complex128)
    for k in range(Ns):
        Y = Y_const.astype(np.complex128).copy()
        if poles is not None:
            for p, R in zip(poles, residue_matrices):
                Y = Y + R / (s_vals[k] - p)
        S_deemb[k] = _y_to_s(Y, z0)

    tau_arr = np.array([tau, tau])
    S = embed_delay(S_deemb, freqs, tau_arr)
    return S


class TestStoYRoundTrip:
    """Verify s_to_y and _y_to_s are inverses."""

    def test_identity_roundtrip(self):
        S = np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.complex128)
        Y = s_to_y(S, z0=50.0)
        S_back = _y_to_s(Y, z0=50.0)
        np.testing.assert_allclose(S_back, S, atol=1e-12)

    def test_matches_circulax_formula(self):
        S = np.array([[0.05, 0.8 * np.exp(-1j * 0.3)], [0.8 * np.exp(-1j * 0.3), 0.05]], dtype=np.complex128)
        z0 = 50.0 + 1e-12j
        n = S.shape[0]
        I = np.eye(n, dtype=np.complex128)
        Y_ref = (I - S) @ np.linalg.inv(z0 * S + np.conj(z0) * I)
        Y = s_to_y(S, z0)
        np.testing.assert_allclose(Y, Y_ref, atol=1e-14)

    def test_state_space_s_to_y_transform_preserves_response(self):
        poles = jnp.array([-2e9 + 3e9j, -2e9 - 3e9j])
        residues = jnp.array(
            [
                [[2e8 + 1e8j, 2e8 - 1e8j], [3e7 - 2e7j, 3e7 + 2e7j]],
                [[-6e7 + 1e7j, -6e7 - 1e7j], [1e8 + 4e7j, 1e8 - 4e7j]],
            ]
        )
        model = VFModel(poles, residues, jnp.array([[0.1, 0.02], [-0.04, 0.05]]), jnp.zeros((2, 2)))
        scattering_ss = vfmodel_to_ss(model, 2)
        admittance_ss, condition = scattering_state_space_to_admittance(scattering_ss)
        freqs = np.linspace(1e7, 2e9, 100)
        s = jnp.asarray(1j * 2 * np.pi * freqs)
        expected_S = np.asarray(eval_model(s, scattering_ss))
        transformed_S = evaluate_sparameter_model(admittance_ss, freqs, np.zeros(2))

        assert np.isfinite(condition)
        np.testing.assert_allclose(transformed_S, expected_S, rtol=1e-10, atol=1e-10)

    def test_state_space_s_to_y_transform_is_jittable_and_vmappable(self):
        poles = jnp.array([-2e9 + 3e9j, -2e9 - 3e9j])
        residues = jnp.array(
            [
                [[2e8 + 1e8j, 2e8 - 1e8j], [3e7 - 2e7j, 3e7 + 2e7j]],
                [[-6e7 + 1e7j, -6e7 - 1e7j], [1e8 + 4e7j, 1e8 - 4e7j]],
            ]
        )
        model = VFModel(
            poles,
            residues,
            jnp.array([[0.1, 0.02], [-0.04, 0.05]]),
            jnp.zeros((2, 2)),
        )
        scattering_ss = vfmodel_to_ss(model, 2)

        compiled_ss, compiled_condition = scattering_state_space_to_admittance_jax(
            scattering_ss,
            jnp.asarray(50.0),
        )
        batched_scattering_ss = jax.tree.map(
            lambda value: jnp.stack([value, value]),
            scattering_ss,
        )
        batched_ss, batched_condition = vmap_scattering_state_space_to_admittance(
            batched_scattering_ss,
            jnp.asarray(50.0),
        )

        assert batched_ss.A.shape == (2, len(compiled_ss.A))
        np.testing.assert_allclose(batched_condition, compiled_condition, rtol=1e-12)
        freqs = np.linspace(1e7, 2e9, 50)
        expected_S = np.asarray(eval_model(1j * 2 * jnp.pi * freqs, scattering_ss))
        for batch_index in range(2):
            candidate = jax.tree.map(
                lambda value, index=batch_index: value[index],
                batched_ss,
            )
            transformed_S = evaluate_sparameter_model(candidate, freqs, np.zeros(2))
            np.testing.assert_allclose(transformed_S, expected_S, rtol=1e-10, atol=1e-10)


class TestDelayExtraction:
    """Tests for extract_group_delay."""

    def test_known_delay(self):
        """Extract delay from pure delay S21 = exp(-j*omega*tau)."""
        tau_true = 1e-9
        freqs = np.linspace(1e6, 10e9, 200)
        Ns = len(freqs)
        S = np.zeros((Ns, 2, 2), dtype=np.complex128)
        omega = 2.0 * np.pi * freqs
        S[:, 0, 1] = 0.9 * np.exp(-1j * omega * tau_true)
        S[:, 1, 0] = S[:, 0, 1]
        S[:, 0, 0] = 0.01
        S[:, 1, 1] = 0.01

        tau = extract_group_delay(S, freqs)
        np.testing.assert_allclose(tau, tau_true, rtol=1e-3)

    def test_zero_delay(self):
        """No delay → tau ≈ 0."""
        freqs = np.linspace(1e6, 10e9, 100)
        Ns = len(freqs)
        S = np.zeros((Ns, 2, 2), dtype=np.complex128)
        S[:, 0, 1] = 0.9
        S[:, 1, 0] = 0.9
        S[:, 0, 0] = 0.01
        S[:, 1, 1] = 0.01

        tau = extract_group_delay(S, freqs)
        np.testing.assert_allclose(tau, 0.0, atol=1e-15)

    def test_nonnegative_clamp(self):
        """Negative tau (non-causal) is clamped to 0."""
        freqs = np.linspace(1e6, 10e9, 100)
        Ns = len(freqs)
        S = np.zeros((Ns, 2, 2), dtype=np.complex128)
        omega = 2.0 * np.pi * freqs
        S[:, 0, 1] = 0.9 * np.exp(1j * omega * 1e-9)
        S[:, 1, 0] = S[:, 0, 1]

        tau = extract_group_delay(S, freqs)
        assert np.all(tau >= 0.0)

    def test_scale_factor(self):
        """Scale < 1 under-estimates delay."""
        tau_true = 1e-9
        freqs = np.linspace(1e6, 10e9, 200)
        Ns = len(freqs)
        S = np.zeros((Ns, 2, 2), dtype=np.complex128)
        omega = 2.0 * np.pi * freqs
        S[:, 0, 1] = 0.9 * np.exp(-1j * omega * tau_true)
        S[:, 1, 0] = S[:, 0, 1]

        tau = extract_group_delay(S, freqs, scale=0.95)
        np.testing.assert_allclose(tau, 0.95 * tau_true, rtol=1e-3)


class TestDeembedEmbed:
    """Tests for deembed_delay / embed_delay round-trip."""

    def test_roundtrip(self):
        """deembed then embed recovers original S."""
        tau_true = 1e-9
        freqs = np.linspace(1e6, 10e9, 100)
        Ns = len(freqs)
        omega = 2.0 * np.pi * freqs
        S = np.zeros((Ns, 2, 2), dtype=np.complex128)
        S[:, 0, 1] = 0.9 * np.exp(-1j * omega * tau_true)
        S[:, 1, 0] = S[:, 0, 1]
        S[:, 0, 0] = 0.01
        S[:, 1, 1] = 0.01

        tau = np.array([tau_true, tau_true])
        S_deemb = deembed_delay(S, freqs, tau)
        S_back = embed_delay(S_deemb, freqs, tau)
        np.testing.assert_allclose(S_back, S, atol=1e-12)

    def test_deembed_removes_phase_rotation(self):
        """After de-embedding, S21 should have nearly constant phase."""
        tau_true = 1e-9
        freqs = np.linspace(1e6, 10e9, 100)
        Ns = len(freqs)
        omega = 2.0 * np.pi * freqs
        S = np.zeros((Ns, 2, 2), dtype=np.complex128)
        S[:, 0, 1] = 0.9 * np.exp(-1j * omega * tau_true)
        S[:, 1, 0] = S[:, 0, 1]

        tau = np.array([tau_true, tau_true])
        S_deemb = deembed_delay(S, freqs, tau)

        phase_range = np.ptp(np.unwrap(np.angle(S_deemb[:, 0, 1])))
        assert phase_range < 0.01, f"Phase range after de-embed: {phase_range:.4f} rad"


class TestFitWithDelay:
    """Integration tests for the full pipeline."""

    @pytest.fixture
    def synth_data(self):
        """Synthesize delay + rational Y-model with known poles."""
        tau = 1e-9
        freqs = np.linspace(1e6, 20e9, 300)
        poles = [-1e9 + 1j * 6e10, -1e9 - 1j * 6e10, -5e9]
        R1 = np.array([[5e7 + 2e7j, -4e7 - 1e7j], [-4e7 - 1e7j, 5e7 + 2e7j]])
        R2 = R1.conj()
        R3 = np.array([[1e8, -8e7], [-8e7, 1e8]])
        S = _synth_delay_rational_2port(
            freqs,
            tau,
            poles=poles,
            residue_matrices=[R1, R2, R3],
        )
        return S, freqs, tau

    def test_pole_reduction(self, synth_data):
        """De-embedded fit uses far fewer poles than raw fit."""
        S, freqs, tau_true = synth_data
        opts = FitOptions(N=0, asymp=2, weightparam=2)

        _, tau_fit, meta = fit_with_delay(
            S,
            freqs,
            z0=50.0,
            opts=opts,
            tol=1e-6,
            enforce_passive=False,
            verbose=False,
        )
        poles_deemb = meta["pole_count"]

        Y_raw = np.stack([s_to_y(S[k], 50.0) for k in range(len(freqs))])
        bigH_raw = jnp.array(np.moveaxis(Y_raw, 0, -1))
        s_pts = jnp.array(1j * 2.0 * np.pi * freqs)
        raw_model, _, _, _ = aaa_driver(
            bigH_raw,
            s_pts,
            opts,
            tol=1e-6,
            verbose=False,
        )
        poles_raw = len(np.asarray(raw_model.poles))

        np.testing.assert_allclose(tau_fit, tau_true, rtol=0.1)
        assert poles_deemb <= 20, f"De-embedded fit used {poles_deemb} poles (expected ≤20)"
        assert meta["rmserr_S"] < 0.1, f"S-domain RMS error: {meta['rmserr_S']:.2e}"
        ratio = poles_raw / max(poles_deemb, 1)
        assert ratio >= 5, f"Expected ≥5× pole reduction, got {poles_raw} raw / {poles_deemb} de-embedded = {ratio:.1f}×"

    def test_returns_valid_ssmodel(self, synth_data):
        """Returned SSModel has correct shapes and finite values."""
        S, freqs, _ = synth_data
        ss, tau, meta = fit_with_delay(
            S,
            freqs,
            z0=50.0,
            tol=1e-6,
            enforce_passive=False,
            verbose=False,
        )
        Nc = 2
        N = meta["pole_count"]
        assert ss.A.shape == (Nc * N,)
        assert ss.B.shape == (Nc * N, Nc)
        assert ss.C.shape == (Nc, Nc * N)
        assert ss.D.shape == (Nc, Nc)
        assert ss.E.shape == (Nc, Nc)
        assert np.all(np.isfinite(np.asarray(ss.A)))

    def test_y_domain_max_poles_refits_a_reduced_model(self, synth_data):
        """Contribution pruning is available for the simulation Y-model."""
        S, freqs, _ = synth_data
        _, _, full = fit_with_delay(
            S,
            freqs,
            tol=1e-6,
            enforce_passive=False,
            causality="ignore",
            verbose=False,
        )
        ss, _, reduced = fit_with_delay(
            S,
            freqs,
            tol=1e-6,
            max_poles=2,
            enforce_passive=False,
            causality="ignore",
            verbose=False,
        )

        assert reduced["pole_count"] <= 2
        assert reduced["pole_count"] < full["pole_count"]
        assert ss.A.shape == (2 * reduced["pole_count"],)

    def test_nonreciprocal_fit_keeps_ordered_responses_distinct(self):
        """Active networks can fit Y12 and Y21 without mirroring them."""
        freqs = np.linspace(1e8, 10e9, 180)
        s = 1j * 2 * np.pi * freqs
        poles = np.array([-8e8 + 1j * 2e10, -8e8 - 1j * 2e10, -4e9])
        residues = np.array(
            [
                [[4e7 + 1e7j, 4e7 - 1e7j, 8e7], [1e6, 1e6, 2e6]],
                [[-8e7 - 2e7j, -8e7 + 2e7j, -1.6e8], [2e7, 2e7, 4e7]],
            ],
            dtype=np.complex128,
        )
        D = np.array([[0.03, 0.001], [-0.025, 0.02]])
        Y = D[None, ...] + np.sum(
            residues[None, ...] / (s[:, None, None, None] - poles[None, None, None, :]),
            axis=-1,
        )
        S = np.stack([y_to_s(value) for value in Y])

        ss, tau, metadata = fit_with_delay(
            S,
            freqs,
            tol=1e-7,
            mmax=20,
            enforce_passive=False,
            reciprocal=False,
            verbose=False,
        )
        fitted = evaluate_sparameter_model(ss, freqs, tau)

        assert metadata["delay_mode"] == "none"
        assert not metadata["reciprocal"]
        assert not np.allclose(fitted[:, 0, 1], fitted[:, 1, 0])
        assert np.linalg.norm(fitted - S) / np.linalg.norm(S) < 1.5e-1

    def test_nonreciprocal_passivity_enforcement_is_rejected(self):
        freqs = np.linspace(1e8, 1e9, 20)
        S = np.zeros((len(freqs), 2, 2), dtype=np.complex128)

        with pytest.raises(ValueError, match="requires reciprocal=True"):
            fit_with_delay(S, freqs, reciprocal=False, enforce_passive=True, verbose=False)

    def test_passivity_pipeline_reports_scalar_margin(self, synth_data):
        """Passivity enforcement reports the worst sweep margin as a scalar."""
        S, freqs, _ = synth_data

        _, _, meta = fit_with_delay(
            S,
            freqs,
            z0=50.0,
            tol=1e-6,
            enforce_passive=True,
            verbose=False,
        )

        assert np.isscalar(meta["passivity_margin"])
        assert np.isfinite(meta["passivity_margin"])

    def test_overestimate_delay_warning(self):
        """Over-estimating delay produces RHP poles that get flipped."""
        tau = 1e-9
        freqs = np.linspace(1e6, 20e9, 300)
        poles = [-1e9 + 1j * 6e10, -1e9 - 1j * 6e10, -5e9]
        R1 = np.array([[5e7 + 2e7j, -4e7 - 1e7j], [-4e7 - 1e7j, 5e7 + 2e7j]])
        R2 = R1.conj()
        R3 = np.array([[1e8, -8e7], [-8e7, 1e8]])
        S = _synth_delay_rational_2port(
            freqs,
            tau,
            poles=poles,
            residue_matrices=[R1, R2, R3],
        )

        with pytest.warns(CausalityWarning, match="right-half-plane"):
            _, _, meta = fit_with_delay(
                S,
                freqs,
                z0=50.0,
                delay_scale=1.05,
                tol=1e-6,
                enforce_passive=False,
                verbose=False,
            )
        assert meta["pole_flips"] > 0, f"Over-estimated delay should produce RHP poles, got pole_flips={meta['pole_flips']}"
        assert meta["causality"]["status"] == "warning"
        assert meta["causality"]["max_raw_pole_real"] > 0

        with pytest.raises(CausalityError, match="right-half-plane"):
            fit_with_delay(
                S,
                freqs,
                z0=50.0,
                delay_scale=1.05,
                tol=1e-6,
                enforce_passive=False,
                causality="error",
                verbose=False,
            )
