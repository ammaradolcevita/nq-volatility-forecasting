"""Tests for the forecast evaluation metrics (src/nqvol/metrics.py).

Each test uses tiny numbers where the right answer can be worked out by hand.
"""
import math

from nqvol.metrics import adjusted_r2, mae, r2, rmse


def test_perfect_forecast_has_zero_error_and_r2_of_one():
    y = [1.0, 2.0, 3.0, 4.0]
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert r2(y, y) == 1.0


def test_forecasting_the_mean_gives_r2_of_zero():
    y = [1.0, 2.0, 3.0]
    assert abs(r2(y, [2.0, 2.0, 2.0])) < 1e-12


def test_mae_and_rmse_known_values():
    y = [1.0, 2.0, 3.0]
    p = [1.0, 2.0, 4.0]          # one error of size 1 out of three
    assert math.isclose(mae(y, p), 1 / 3)
    assert math.isclose(rmse(y, p), math.sqrt(1 / 3))


def test_adjusted_r2_known_value():
    # 1 - (1 - 0.5) * (11 - 1) / (11 - 1 - 1) = 1 - 0.5 * 10 / 9
    assert math.isclose(adjusted_r2(0.5, n=11, k=1), 1 - 0.5 * 10 / 9)
