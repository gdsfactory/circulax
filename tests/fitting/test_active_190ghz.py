"""Release-regression benchmark for the scikit-rf 190 GHz active device."""

from pathlib import Path

import numpy as np
import skrf

from circulax.fitting import evaluate_sparameter_model, fit_with_delay


def test_s_domain_fit_matches_skrf_published_accuracy() -> None:
    data = Path(__file__).parents[2] / "examples/fitting/data/190ghz_tx_measured.s2p"
    network = skrf.Network(data)

    ss, delay, metadata = fit_with_delay(
        network.s,
        network.f,
        tol=1e-8,
        mmax=40,
        reciprocal=False,
        enforce_passive=False,
        fit_domain="s",
        s_refinement_iterations=4,
        max_poles=20,
        pole_count_candidates=(14, 16, 18, 20, 22),
        verbose=False,
    )
    prediction = evaluate_sparameter_model(ss, network.f, delay)
    error = prediction - network.s
    normalized_rmse = np.linalg.norm(error) / np.linalg.norm(network.s)

    assert metadata["fit_domain"] == "s"
    assert metadata["delay_mode"] == "none"
    assert metadata["aaa_pole_count"] == 50
    assert metadata["pole_count"] == 20
    assert metadata["state_count"] == 40
    assert np.max(np.real(ss.A)) < 0.0
    sweep = metadata["pole_sweep"]
    np.testing.assert_array_equal(np.asarray(sweep.retained_counts), [14, 16, 18, 20, 22])
    assert np.all(np.diff(np.asarray(sweep.normalized_rmse)) <= 0.0)
    assert normalized_rmse <= 1.3e-2
    assert np.max(np.abs(error)) <= 3e-2
