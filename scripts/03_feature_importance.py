"""Feature importance with ONE method for all model families (Table 6; reviewer #6).

Permutation importance on the test set (Breiman, 2001; Fisher et al., 2019):
increase in test MSE when one predictor is randomly shuffled across test days,
averaged over 20 shuffles. Shares are normalised to 100% within each model
(negative values set to 0 before normalising). Native importances (standardised
coefficients, impurity/gain) are saved as a supplementary table.

Usage: python scripts/03_feature_importance.py [main|ratio]
"""
import sys

import _common  # noqa: F401
import numpy as np
import pandas as pd

from nqvol import config as C
from nqvol.data import add_ratio_columns, chronological_split, load_dataset
from nqvol.models import CATEGORY, MODEL_ORDER
from nqvol.pipeline import SPECS, fit_predict_all
from nqvol.report import TABLES, save_table

N_REPEATS = 20


def group_of(feature):
    base = feature[2:] if feature.startswith("r_") else feature
    for g, cols in C.FEATURE_GROUPS.items():
        if base in cols:
            return g
    return "Other"


def main(spec="main"):
    sfx = "" if spec == "main" else f"_{spec}"
    df = load_dataset()
    if spec == "ratio":
        df = add_ratio_columns(df)
    features = list(SPECS[spec]["features"])
    scale_col = SPECS[spec]["rescale"]
    train_mask, test_mask = chronological_split(df)
    y_test = df.loc[test_mask, C.TARGET].to_numpy(float)
    scale = df.loc[test_mask, scale_col].to_numpy(float) if scale_col else np.ones(test_mask.sum())

    print(f"Fitting models (spec={spec}) ...")
    out = fit_predict_all(df, train_mask, test_mask, spec=spec, verbose=False)

    def predictor(name):
        model = out[name]["model"]
        if name == "LSTM":
            return lambda d: model.predict(d, test_mask) * scale
        return lambda d: model.predict(d.loc[test_mask, features]) * scale

    rng = np.random.default_rng(C.SEED)
    test_rows = np.flatnonzero(test_mask)
    perm_orders = [rng.permutation(len(test_rows)) for _ in range(N_REPEATS)]

    rows = []
    for name in MODEL_ORDER:
        f = predictor(name)
        base_pred = f(df)
        assert np.allclose(base_pred, out[name]["test_pred"]), name
        base_mse = np.mean((y_test - base_pred) ** 2)
        for feat in features:
            deltas = []
            col = df.columns.get_loc(feat)
            original = df.iloc[test_rows, col].to_numpy().copy()
            for order in perm_orders:
                dperm = df.copy()
                dperm.iloc[test_rows, col] = original[order]
                deltas.append(np.mean((y_test - f(dperm)) ** 2) - base_mse)
            rows.append({"Model": name, "Category": CATEGORY[name], "Feature": feat, "Group": group_of(feat),
                         "Delta MSE": float(np.mean(deltas)), "SD": float(np.std(deltas, ddof=1)),
                         "Base MSE": base_mse})
        print(f"  {name:<18} done", flush=True)

    long = pd.DataFrame(rows)
    long["Positive"] = long["Delta MSE"].clip(lower=0)
    long["Share (%)"] = 100 * long["Positive"] / long.groupby("Model")["Positive"].transform("sum")
    long["Delta MSE / Base MSE (%)"] = 100 * long["Delta MSE"] / long["Base MSE"]
    TABLES.mkdir(parents=True, exist_ok=True)
    long.to_csv(TABLES / f"permutation_importance_long{sfx}.csv", index=False)

    share = long.pivot(index="Feature", columns="Model", values="Share (%)").reindex(index=features, columns=MODEL_ORDER)
    groups = long.groupby(["Group", "Model"])["Share (%)"].sum().unstack().reindex(columns=MODEL_ORDER)
    groups.index = [f"Group: {g}" for g in groups.index]
    table = pd.concat([share, groups]).reset_index().rename(columns={"index": "Feature"})
    pd.set_option("display.width", 300)
    print(table.round(1).to_string(index=False))
    save_table(table, f"table6_permutation_importance{sfx}",
               caption="Permutation importance on the test set (share of total, \\%)",
               label="tab:importance", float_format="%.1f",
               notes="Increase in test MSE when a predictor is shuffled (20 repetitions), normalised to 100\\% per model.")

    # Supplementary: native importances for the family representatives
    native = {}
    hx = out["HAR-X"]["model"].result_
    Xtr = df.loc[train_mask, features]
    native["HAR-X |b*sd(x)|"] = (hx.params[features].abs() * Xtr.std()).values
    en = out["Elastic Net"]["model"].named_steps["model"]
    native["Elastic Net |b| (std. inputs)"] = np.abs(en.coef_)
    cb = out["CatBoost"]["model"]
    native["CatBoost gain"] = cb.get_feature_importance()
    rf = out["Random Forest"]["model"]
    native["Random Forest impurity"] = rf.feature_importances_
    nat = pd.DataFrame(native, index=features)
    nat = 100 * nat / nat.sum()
    save_table(nat.reset_index().rename(columns={"index": "Feature"}), f"native_importance{sfx}",
               caption="Native (model-specific) importance measures, share of total (\\%)",
               label="tab:native_importance", float_format="%.1f")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
