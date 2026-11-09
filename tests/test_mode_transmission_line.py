"""Mode-solver inputs must recover known physical line impedances and delays."""

import numpy as np
import pytest

from circulax.components import mode_transmission_line

C0 = 299_792_458.0


def test_functions_scalars_and_arrays_agree() -> None:
    f = np.r_[0.0, np.geomspace(1e6, 100e9, 80)]
    a, n, z = 250.0, 2.2, 55.0
    functions = mode_transmission_line(
        f, attenuation_db_per_m=lambda f: np.full_like(f, a), n_eff=lambda _f: n, impedance=lambda _f: z
    )
    arrays = mode_transmission_line(f, attenuation_db_per_m=np.full_like(f, a), n_eff=np.full_like(f, n), impedance=z)
    gamma = np.log(10) / 20 * a + 2j * np.pi * f * n / C0
    np.testing.assert_allclose(functions.series.evaluate(f), gamma * z, atol=1e-8)
    np.testing.assert_allclose(functions.shunt.evaluate(f), gamma / z, atol=1e-10)
    np.testing.assert_allclose(functions.series.evaluate(f), arrays.series.evaluate(f), atol=1e-10)
    # Recover the RF group index from propagation, not the filter cutoff.
    propagation = np.sqrt(functions.series.evaluate(f) * functions.shunt.evaluate(f))
    np.testing.assert_allclose(C0 * np.gradient(propagation.imag, 2 * np.pi * f), n, atol=1e-9)


def test_complex_frequency_dependent_impedance_from_known_rlc() -> None:
    f = np.geomspace(1e6, 30e9, 120)
    s = 2j * np.pi * f
    poles = np.array([2e8, 3e10])
    z_prime = 800 + s * 400e-9 + 8000 * s / (s + poles[0])
    y_prime = 0.02 + s * 120e-12 + 2 * s / (s + poles[1])
    gamma = np.sqrt(z_prime * y_prime)
    impedance = np.sqrt(z_prime / y_prime)
    line = mode_transmission_line(
        f,
        attenuation_db_per_m=gamma.real * 20 / np.log(10),
        n_eff=gamma.imag * C0 / (2 * np.pi * f),
        impedance=impedance,
        poles=poles,
        tolerance=1e-8,
    )
    np.testing.assert_allclose(line.series.evaluate(f), z_prime, atol=1e-8)
    np.testing.assert_allclose(line.shunt.evaluate(f), y_prime, atol=1e-10)


@pytest.mark.parametrize(
    "change",
    [
        {"attenuation_db_per_m": -1},
        {"n_eff": 0},
        {"n_eff": 1j},
        {"impedance": -50},
        {"impedance": 50 + 1j},
        {"attenuation_db_per_m": lambda f: np.full_like(f, np.nan)},
        {"n_eff": lambda f: np.ones(len(f) + 1)},
        {"frequencies": [1e9, 1e8]},
    ],
)
def test_invalid_mode_data(change: dict) -> None:
    kwargs = {"frequencies": np.r_[0.0, np.geomspace(1e6, 1e9, 30)], "attenuation_db_per_m": 250.0, "n_eff": 2.2, "impedance": 50.0}
    with pytest.raises(ValueError):
        mode_transmission_line(**(kwargs | change))
