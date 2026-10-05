"""Main chronological split: train 2009-2021, test 2022-2025.

Outputs
  results/tables/table3_sample_split.*      sample split (Table 3)
  results/tables/table5_main_results.*      train and test metrics, all 19 models (Table 5; reviewer #2, #4, #7)
  results/tables/lstm_seed_stability.*      LSTM metrics across seeds
  results/tables/har_coefficients.*         HAR / HAR-X / LogHAR coefficients with Newey-West t-statistics
  results/predictions/main_test_predictions.csv
  results/predictions/main_train_predictions.csv
"""
import _common  # noqa: F401
import numpy as np
import pandas as pd

from nqvol import config as C
from nqvol import metrics as M
from nqvol.data import add_ratio_columns, chronological_split, load_dataset, split_summary
from nqvol.models import MODEL_ORDER
from nqvol.pipeline import SPECS, fit_predict_all
from nqvol.report import save_predictions, save_table


def main(spec="main"):
    sfx = "" if spec == "main" else f"_{spec}"
    df = load_dataset()
    if spec == "ratio":
        df = add_ratio_columns(df)
    train_mask, test_mask = chronological_split(df)
    y = df[C.TARGET].to_numpy(float)

    split = split_summary(df, train_mask, test_mask)
    print(split.to_string(index=False))
    save_table(split, "table3_sample_split", caption="Chronological sample split",
               label="tab:split", float_format="%.2f")

    print(f"\nFitting all models on the main split (spec={spec}) ...")
    out = fit_predict_all(df, train_mask, test_mask, spec=spec)

    y_test = y[test_mask]
    mse_har = np.mean((y_test - out["HAR"]["test_pred"]) ** 2)

    rows = []
    for name in MODEL_ORDER:
        r = out[name]
        y_tr = y[r["train_index"]]
        k = len(SPECS[spec]["har_columns"]) if name in ("HAR", "LogHAR") else len(SPECS[spec]["features"])
        tr = M.summary(y_tr, r["train_pred"])
        te = M.summary(y_test, r["test_pred"])
        rows.append({
            "Model": name, "Category": r["category"],
            "N train": len(y_tr), "N test": int(test_mask.sum()), "k": k,
            "Train R2": tr["R2"], "Train adj. R2": M.adjusted_r2(tr["R2"], len(y_tr), k),
            "Train MAE": tr["MAE"], "Train RMSE": tr["RMSE"],
            "Test R2": te["R2"],
            "Test R2_OOS (vs HAR)": 1 - np.mean((y_test - r["test_pred"]) ** 2) / mse_har,
            "Test MAE": te["MAE"], "Test RMSE": te["RMSE"], "Test MAPE": te["MAPE"],
            "R2 gap (train - test)": tr["R2"] - te["R2"],
        })
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print("\n", table.round(4).to_string(index=False))
    save_table(table, "table5_main_results" + sfx,
               caption="Out-of-sample forecast accuracy, chronological split (train 2009--2021, test 2022--2025)",
               label="tab:main",
               notes=f"HAR-X and all machine-learning models use the same {len(SPECS[spec]['features'])} predictors; all models are evaluated on the same test days. "
                     f"HAR and LogHAR use only the {len(SPECS[spec]['har_columns'])} persistence columns {', '.join(SPECS[spec]['har_columns'])}. "
                     "LSTM: average forecast of 10 networks trained with different seeds.")

    # LSTM stability across seeds
    seed_rows = []
    for seed, p in zip(C.LSTM_SEEDS_MAIN, out["LSTM"]["seed_test_preds"]):
        s = M.summary(y_test, p)
        seed_rows.append({"Seed": seed, **s})
    seeds = pd.DataFrame(seed_rows)
    seeds["best epoch"] = out["LSTM"]["best_epochs"]
    agg = seeds.drop(columns="Seed").agg(["mean", "std", "min", "max"]).reset_index().rename(columns={"index": "Seed"})
    seeds = pd.concat([seeds, agg], ignore_index=True)
    print("\nLSTM across seeds\n", seeds.round(4).to_string(index=False))
    save_table(seeds, "lstm_seed_stability" + sfx, caption="LSTM test metrics by seed", label="tab:lstm_seeds")

    # HAR-family coefficients with Newey-West t-statistics
    coef_rows = []
    for name in ["HAR", "HAR-X", "LogHAR"]:
        res = out[name]["model"].result_
        for var in res.params.index:
            coef_rows.append({"Model": name, "Variable": var, "Coefficient": res.params[var],
                              "NW t-stat": res.tvalues[var], "p-value": res.pvalues[var]})
        coef_rows.append({"Model": name, "Variable": "In-sample R2", "Coefficient": res.rsquared,
                          "NW t-stat": np.nan, "p-value": np.nan})
        coef_rows.append({"Model": name, "Variable": "NW lags", "Coefficient": out[name]["model"].nw_lags_,
                          "NW t-stat": np.nan, "p-value": np.nan})
        if name == "LogHAR":
            coef_rows.append({"Model": name, "Variable": "Duan smearing factor",
                              "Coefficient": out[name]["model"].smearing_, "NW t-stat": np.nan, "p-value": np.nan})
    coefs = pd.DataFrame(coef_rows)
    save_table(coefs, "har_coefficients" + sfx, caption="HAR-family estimates (Newey-West HAC)", label="tab:har_coef")

    # predictions
    test_pred = pd.DataFrame({"date": df.loc[test_mask, "date"].dt.date.values, "actual": y_test})
    for name in MODEL_ORDER:
        test_pred[name] = out[name]["test_pred"]
    save_predictions(test_pred, "main_test_predictions" + sfx)

    train_pred = pd.DataFrame({"date": df.loc[train_mask, "date"].dt.date.values, "actual": y[train_mask]})
    train_idx = np.flatnonzero(train_mask)
    for name in MODEL_ORDER:
        s = pd.Series(np.nan, index=train_idx)
        s.loc[out[name]["train_index"]] = out[name]["train_pred"]
        train_pred[name] = s.values
    save_predictions(train_pred, "main_train_predictions" + sfx)
    print("\nSaved tables and predictions to results/.")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
