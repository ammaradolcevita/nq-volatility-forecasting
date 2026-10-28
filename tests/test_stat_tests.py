"""Tests for the forecast-comparison tests (src/nqvol/dm.py)."""
import math

import numpy as np

from nqvol.dm import (clark_west, dm_test, holm_adjust, long_run_variance,
                      newey_west_lag)


def test_newey_west_lag_formula():
    # floor(4 * (T / 100) ** (2 / 9)): 4 for T = 100, 6 for T = 1000
    assert newey_west_lag(100) == 4
    assert newey_west_lag(1000) == 6


def test_long_run_variance_without_lags_is_the_plain_variance():
    d = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    assert math.isclose(long_run_variance(d, lag=0), np.var(d))


def test_dm_test_identical_forecasts_is_undefined():
    e = np.array([0.5, -0.2, 0.1, 0.3, -0.4])
    stat, p, *_ = dm_test(e, e)
    assert math.isnan(stat) and math.isnan(p)


def test_dm_test_finds_the_clearly_better_model():
    rng = np.random.default_rng(0)
    e_good = rng.normal(0, 1.0, 500)
    e_bad = rng.normal(0, 2.0, 500)
    stat, p, loss_good, loss_bad, _ = dm_test(e_good, e_bad)
    assert stat < 0           # negative: model 1 has the smaller loss
    assert p < 0.01
    assert loss_good < loss_bad


def test_clark_west_detects_a_useful_extra_predictor():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1.0, 500)
    y = x + rng.normal(0, 0.5, 500)
    stat, p = clark_west(y, p_small=np.zeros(500), p_large=x)
    assert stat > 0
    assert p < 0.01


def test_holm_adjust_known_values():
    # sorted: 0.01 * 3 = 0.03, 0.03 * 2 = 0.06, 0.04 * 1 = 0.04 -> raised to 0.06
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert np.allclose(adjusted, [0.03, 0.06, 0.06])


def test_holm_adjust_never_lowers_a_p_value_and_caps_at_one():
    p = np.array([0.001, 0.2, 0.5, 0.9])
    adjusted = holm_adjust(p)
    assert np.all(adjusted >= p)
    assert np.all(adjusted <= 1.0)


def test_holm_adjust_leaves_undefined_tests_out_of_the_family():
    adjusted = holm_adjust([0.01, float("nan")])
    assert math.isclose(adjusted[0], 0.01)   # family size is 1, not 2
    assert math.isnan(adjusted[1])
