"""Model definitions.

Fixed hyperparameters are those of the submitted version (Table 4 of the manuscript).
Changes relative to the submitted code (see docs/CHANGES.md):
  * cross-validated regularisation (Ridge, Lasso, Elastic Net, Adaptive Lasso)
    uses TimeSeriesSplit(5) instead of cv=5 (unshuffled, contiguous KFold, in
    which some validation folds precede their training data);
  * LightGBM gets subsample_freq=1 so that the reported subsample=0.8 is
    actually applied (with LightGBM's default subsample_freq=0 it is ignored);
  * no test data is passed to any fit() call.
"""
import numpy as np
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import (BayesianRidge, ElasticNetCV, HuberRegressor,
                                  LassoCV, LinearRegression, RidgeCV,
                                  TheilSenRegressor)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from . import config as C

ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
RIDGE_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
L1_RATIOS = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]


def tscv():
    return TimeSeriesSplit(n_splits=5)


# ---------------------------------------------------------------------------
# Econometric models (statsmodels OLS with Newey-West HAC standard errors)
# ---------------------------------------------------------------------------
class OLSHAC(BaseEstimator, RegressorMixin):
    """OLS on a column subset with Newey-West HAC covariance (floor(N^(1/3)) lags).

    log=True estimates the model in logs (LogHAR) and retransforms forecasts
    with the Duan (1983) smearing estimator.
    """

    def __init__(self, columns=None, log=False):
        self.columns = columns
        self.log = log

    def _design(self, X):
        Z = X[self.columns] if self.columns is not None else X
        if self.log:
            Z = np.log(Z)
        return sm.add_constant(Z, has_constant="add")

    def fit(self, X, y):
        y = np.asarray(y, float)
        yy = np.log(y) if self.log else y
        Z = self._design(X)
        self.nw_lags_ = int(np.floor(len(y) ** (1 / 3)))
        self.result_ = sm.OLS(yy, Z).fit(cov_type="HAC", cov_kwds={"maxlags": self.nw_lags_})
        if self.log:
            self.smearing_ = float(np.mean(np.exp(self.result_.resid)))
        return self

    def predict(self, X):
        pred = np.asarray(self.result_.predict(self._design(X)))
        if self.log:
            pred = np.exp(pred) * self.smearing_
        return pred

    @property
    def n_predictors_(self):
        return len(self.result_.params) - 1


class AdaptiveLasso(BaseEstimator, RegressorMixin):
    """Two-step adaptive Lasso (Zou, 2006): OLS weights w_j = 1/|b_j|, then a
    weighted Lasso with alpha chosen by time-series cross-validation."""

    def __init__(self, alphas=ALPHAS, gamma=1.0):
        self.alphas = alphas
        self.gamma = gamma

    def fit(self, X, y):
        X = np.asarray(X, float)
        init = LinearRegression().fit(X, y)
        self.weights_ = 1.0 / (np.abs(init.coef_) ** self.gamma + 1e-8)
        self.lasso_ = LassoCV(alphas=self.alphas, cv=tscv(), max_iter=10000,
                              random_state=C.SEED).fit(X / self.weights_, y)
        self.coef_ = self.lasso_.coef_ / self.weights_
        self.intercept_ = self.lasso_.intercept_
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.coef_ + self.intercept_


def _scaled(estimator):
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def build_models(har_columns=C.HAR_FEATURES):
    """Return an ordered dict name -> (category, factory) for the 18 non-LSTM models.

    `har_columns` are the persistence columns used by the pure HAR / LogHAR
    benchmarks. HAR-X and all ML models use every column of the X they receive,
    so the caller controls the (common) predictor set.
    """
    s = C.SEED
    return {
        # Econometric benchmarks
        "HAR": ("Econometric", lambda: OLSHAC(columns=har_columns)),
        "HAR-X": ("Econometric", lambda: OLSHAC(columns=None)),
        "LogHAR": ("Econometric", lambda: OLSHAC(columns=har_columns, log=True)),
        # Linear (standardised inputs, scaler fitted on training data only)
        "Linear Regression": ("Linear", lambda: _scaled(LinearRegression())),
        "Ridge": ("Linear", lambda: _scaled(RidgeCV(alphas=RIDGE_ALPHAS, cv=tscv(),
                                                     scoring="neg_mean_absolute_error"))),
        "Lasso": ("Linear", lambda: _scaled(LassoCV(alphas=ALPHAS, cv=tscv(), max_iter=10000,
                                                     random_state=s))),
        "Elastic Net": ("Linear", lambda: _scaled(ElasticNetCV(alphas=ALPHAS, l1_ratio=L1_RATIOS,
                                                               cv=tscv(), max_iter=10000,
                                                               random_state=s))),
        "Huber": ("Linear", lambda: _scaled(HuberRegressor(epsilon=1.35, max_iter=10000))),
        "Adaptive Lasso": ("Linear", lambda: _scaled(AdaptiveLasso())),
        "Bayesian Ridge": ("Linear", lambda: _scaled(BayesianRidge(max_iter=300, tol=1e-6,
                                                                   compute_score=True))),
        "Theil-Sen": ("Linear", lambda: _scaled(TheilSenRegressor(max_subpopulation=10000,
                                                                  n_subsamples=1000,
                                                                  max_iter=300,
                                                                  random_state=s))),
        # Tree-based (raw inputs)
        "Decision Tree": ("Tree", lambda: DecisionTreeRegressor(max_depth=10, min_samples_split=20,
                                                                 min_samples_leaf=10, random_state=s)),
        "Random Forest": ("Tree", lambda: RandomForestRegressor(n_estimators=200, max_depth=15,
                                                                 min_samples_split=10,
                                                                 min_samples_leaf=5,
                                                                 max_features="sqrt",
                                                                 random_state=s, n_jobs=-1)),
        "Extra Trees": ("Tree", lambda: ExtraTreesRegressor(n_estimators=200, max_depth=15,
                                                             min_samples_split=10,
                                                             min_samples_leaf=5,
                                                             max_features="sqrt",
                                                             random_state=s, n_jobs=-1)),
        "Gradient Boosting": ("Tree", lambda: GradientBoostingRegressor(n_estimators=200,
                                                                         learning_rate=0.05,
                                                                         max_depth=5,
                                                                         min_samples_split=2,
                                                                         min_samples_leaf=1,
                                                                         subsample=0.8,
                                                                         random_state=s)),
        "XGBoost": ("Tree", lambda: XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                                  subsample=0.8, colsample_bytree=0.8,
                                                  reg_alpha=0.1, reg_lambda=1.0,
                                                  random_state=s, verbosity=0)),
        "LightGBM": ("Tree", lambda: LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                                    subsample=0.8, subsample_freq=1,
                                                    colsample_bytree=0.8, reg_alpha=0.1,
                                                    reg_lambda=1.0, random_state=s, verbose=-1)),
        "CatBoost": ("Tree", lambda: CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6,
                                                        subsample=0.8, reg_lambda=1.0,
                                                        random_seed=s, verbose=0,
                                                        allow_writing_files=False)),
    }


MODEL_ORDER = ["HAR", "HAR-X", "LogHAR",
               "Linear Regression", "Ridge", "Lasso", "Elastic Net", "Huber",
               "Adaptive Lasso", "Bayesian Ridge", "Theil-Sen",
               "Decision Tree", "Random Forest", "Extra Trees", "Gradient Boosting",
               "XGBoost", "LightGBM", "CatBoost", "LSTM"]

CATEGORY = {**{m: c for m, (c, _) in build_models().items()}, "LSTM": "Deep learning"}
