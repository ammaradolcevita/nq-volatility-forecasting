"""Fit every model on one train/test split and collect forecasts."""
import time

import numpy as np

from . import config as C
from .data import RATIO_FEATURES
from .lstm import LSTMEnsemble
from .models import MODEL_ORDER, build_models

SPECS = {
    # main specification: target and predictors in index points
    "main": dict(features=C.FEATURES, target=C.TARGET, har_columns=C.HAR_FEATURES, rescale=None),
    # scale-free specification: target and range predictors divided by har_monthly
    "ratio": dict(features=RATIO_FEATURES, target="ratio_target",
                  har_columns=["r_har_daily", "r_har_weekly"], rescale=C.RATIO_NORMALISER),
}


def fit_predict_all(df, train_mask, test_mask, spec="main", extra_features=(),
                    include_lstm=True, lstm_seeds=C.LSTM_SEEDS_MAIN, models=None,
                    verbose=True):
    """Return {model_name: dict(train_pred, test_pred, model, seconds)}.

    All predictions are returned on the original scale (index points).
    Every model uses the identical predictor list `features + extra_features`
    (the pure HAR / LogHAR benchmarks use only the persistence columns).
    """
    cfg = SPECS[spec]
    features = list(cfg["features"]) + list(extra_features)
    X = df[features]
    y = df[cfg["target"]].to_numpy(float)
    scale = df[cfg["rescale"]].to_numpy(float) if cfg["rescale"] else np.ones(len(df))

    wanted = models or MODEL_ORDER
    out = {}
    for name, (category, factory) in build_models(cfg["har_columns"]).items():
        if name not in wanted:
            continue
        t0 = time.time()
        est = factory().fit(X[train_mask], y[train_mask])
        out[name] = dict(
            category=category,
            model=est,
            train_pred=est.predict(X[train_mask]) * scale[train_mask],
            test_pred=est.predict(X[test_mask]) * scale[test_mask],
            train_index=np.flatnonzero(train_mask),
            seconds=time.time() - t0,
        )
        if verbose:
            print(f"  {name:<18} fitted in {out[name]['seconds']:6.1f}s", flush=True)

    if include_lstm and "LSTM" in wanted:
        t0 = time.time()
        dfx = df.copy()
        ens = LSTMEnsemble(features, seeds=lstm_seeds).fit(dfx, train_mask, cfg["target"])
        tr_idx = ens.train_index_
        tr_mask_lstm = np.zeros(len(df), bool)
        tr_mask_lstm[tr_idx] = True
        seed_test = ens.predict_by_seed(dfx, test_mask) * scale[test_mask]
        seed_train = ens.predict_by_seed(dfx, tr_mask_lstm) * scale[tr_mask_lstm]
        out["LSTM"] = dict(
            category="Deep learning",
            model=ens,
            train_pred=seed_train.mean(axis=0),
            test_pred=seed_test.mean(axis=0),
            train_index=tr_idx,
            seed_test_preds=seed_test,
            seed_train_preds=seed_train,
            best_epochs=ens.best_epochs_,
            seconds=time.time() - t0,
        )
        if verbose:
            print(f"  {'LSTM':<18} fitted in {out['LSTM']['seconds']:6.1f}s "
                  f"({len(lstm_seeds)} seeds, best epochs {ens.best_epochs_})", flush=True)
    return out
