"""Diebold-Mariano (1995) test of equal predictive accuracy.

Differences from the version used in the submitted manuscript:
  * the long-run variance of the loss differential is estimated with a
    Newey-West (Bartlett-kernel) HAC estimator, so autocorrelation in the loss
    differential is accounted for;
  * the Harvey, Leybourne & Newbold (1997) small-sample correction is applied
    and p-values come from a Student-t distribution with T-1 degrees of freedom.
"""
import numpy as np
from scipy import stats


def newey_west_lag(T):
    """Automatic bandwidth: floor(4 * (T/100)^(2/9)) (Newey & West, 1994)."""
    return int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))


def long_run_variance(d, lag):
    d = np.asarray(d, dtype=float)
    T = len(d)
    dc = d - d.mean()
    lrv = np.dot(dc, dc) / T
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)            # Bartlett weight
        gamma_k = np.dot(dc[k:], dc[:-k]) / T
        lrv += 2.0 * w * gamma_k
    return lrv


def dm_test(e1, e2, loss="SE", h=1, lag=None):
    """Test H0: E[L(e1)] = E[L(e2)].

    e1, e2 : forecast errors (actual - forecast) of model 1 and model 2.
    Returns (DM statistic, two-sided p-value, mean loss 1, mean loss 2, lag).
    A negative statistic means model 1 has the smaller expected loss.
    """
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    if loss == "SE":
        l1, l2 = e1 ** 2, e2 ** 2
    elif loss == "AE":
        l1, l2 = np.abs(e1), np.abs(e2)
    else:
        raise ValueError("loss must be 'SE' or 'AE'")
    d = l1 - l2
    T = len(d)
    if np.allclose(e1, e2, rtol=0, atol=1e-8):
        # identical forecasts (e.g. a Lasso that sets the extra coefficient to zero): test undefined
        return float("nan"), float("nan"), float(l1.mean()), float(l2.mean()), lag
    if lag is None:
        lag = max(newey_west_lag(T), h - 1)
    lrv = long_run_variance(d, lag)
    dm = d.mean() / np.sqrt(lrv / T)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_hln = dm * hln
    p = 2 * stats.t.sf(np.abs(dm_hln), df=T - 1)
    return float(dm_hln), float(p), float(l1.mean()), float(l2.mean()), lag


def clark_west(y, p_small, p_large, lag=None):
    """Clark & West (2007) test for nested models.

    H0: the larger model does not improve on the smaller (nested) one.
    Adjusted loss differential f_t = e_small^2 - (e_large^2 - (p_small - p_large)^2);
    one-sided test of E[f_t] > 0 with a Newey-West t-statistic.
    Returns (CW statistic, one-sided p-value).
    """
    y, ps, pl = (np.asarray(a, float) for a in (y, p_small, p_large))
    f = (y - ps) ** 2 - ((y - pl) ** 2 - (ps - pl) ** 2)
    T = len(f)
    if lag is None:
        lag = newey_west_lag(T)
    stat = f.mean() / np.sqrt(long_run_variance(f, lag) / T)
    return float(stat), float(stats.t.sf(stat, df=T - 1))


def holm_adjust(pvalues):
    """Holm (1979) step-down adjustment for multiple comparisons."""
    p = np.asarray(pvalues, float)
    adj = np.full(len(p), np.nan)
    valid = np.flatnonzero(~np.isnan(p))   # undefined tests are excluded from the family
    m = len(valid)
    order = valid[np.argsort(p[valid])]
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj
