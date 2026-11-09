"""Correctness guards for the stress benchmark's digital result validation."""

import numpy as np

from benchmarks.c6288.bench_circulax import check_product


def test_product_requires_valid_logic_levels() -> None:
    expected = 0xFFFE0001
    voltages = np.asarray([1.2 * ((expected >> bit) & 1) for bit in range(32)])
    assert check_product(voltages, expected)["valid_logic"]
    # Thresholding alone would accept this unknown logic level as a one.
    voltages[0] = 0.7
    result = check_product(voltages, expected)
    assert result["observed"] == result["expected"]
    assert not result["valid_logic"]
    assert result["invalid_bits"] == [0]
    voltages[0] = np.nan
    assert not check_product(voltages, expected)["valid_logic"]


def test_vacask_statistics_without_elapsed_time() -> None:
    from benchmarks.c6288.bench_vacask import parse_stats

    stats = parse_stats("""Stats:
  Number of unknowns:              25380
  Eval and load time:              58.6404
  NR solver iterations:            3503
  NR solver time:                  63.4205
  Accepted timepoints:             1022
  Rejected timepoints:             9
""")
    assert stats["nr_solver_s"] == 63.4205
    assert stats["accepted_timepoints"] == 1022
    assert stats["unknowns"] == 25380
    assert "sim_reported_s" not in stats
