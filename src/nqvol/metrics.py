"""Forecast evaluation metrics."""
import numpy as np


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mape(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(np.mean(np.abs((y - p) / y)) * 100)


def r2(y, p):
    """Standard coefficient of determination (benchmark: mean of the evaluated sample)."""
    y, p = np.asarray(y), np.asarray(p)
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def adjusted_r2(r2_value, n, k):
    """Adjusted R^2 for n observations and k predictors (excluding the intercept)."""
    return float(1 - (1 - r2_value) * (n - 1) / (n - k - 1))


def summary(y, p):
    return {"R2": r2(y, p), "MAE": mae(y, p), "RMSE": rmse(y, p), "MAPE": mape(y, p)}
