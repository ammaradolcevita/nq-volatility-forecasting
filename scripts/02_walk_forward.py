"""Expanding-window (walk-forward) evaluation of all 19 models (Table 7).

Usage: python scripts/02_walk_forward.py [main|ratio]

Outputs
  results/tables/table7_walk_forward[_ratio].*       R^2 by fold, mean, sd, mean MAE
  results/tables/walk_forward_long[_ratio].csv        every model x fold x metric
  results/predictions/walk_forward_predictions[_ratio].csv
"""
import sys

import _common  # noqa: F401
import numpy as np
import pandas as pd

from nqvol import config as C
from nqvol import metrics as M
from nqvol.data import add_ratio_columns, fold_masks, load_dataset
from nqvol.models import CATEGORY, MODEL_ORDER
from nqvol.pipeline import fit_predict_all
from nqvol.report import TABLES, save_predictions, save_table


def main(spec="main"):
    sfx = "" if spec == "main" else f"_{spec}"
    df = load_dataset()
    if spec == "ratio":
        df = add_ratio_columns(df)
    y = df[C.TARGET].to_numpy(float)

    long_rows, preds = [], []
    for fold in C.WALK_FORWARD_FOLDS:
        tr, te = fold_masks(df, fold)
        label = f"{fold[1]}-{str(fold[2])[-2:]}"
        print(f"\nFold train<= {fold[0]} -> test {fold[1]}-{fold[2]}  (N train {tr.sum()}, N test {te.sum()})")
        out = fit_predict_all(df, tr, te, spec=spec, lstm_seeds=C.LSTM_SEEDS_WF, verbose=False)
        fp = pd.DataFrame({"fold": label, "date": df.loc[te, "date"].dt.date.values, "actual": y[te]})
        for name in MODEL_ORDER:
            s = M.summary(y[te], out[name]["test_pred"])
            long_rows.append({"Fold": label, "Train end": fold[0], "N train": int(tr.sum()),
                              "N test": int(te.sum()), "Model": name, "Category": CATEGORY[name], **s})
            fp[name] = out[name]["test_pred"]
        preds.append(fp)
        best = max(MODEL_ORDER, key=lambda m: long_rows[-len(MODEL_ORDER) + MODEL_ORDER.index(m)]["R2"])
        print(f"  best R2 in fold: {best}")

    long = pd.DataFrame(long_rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    long.to_csv(TABLES / f"walk_forward_long{sfx}.csv", index=False)
    save_predictions(pd.concat(preds, ignore_index=True), f"walk_forward_predictions{sfx}")

    r2 = long.pivot(index="Model", columns="Fold", values="R2").reindex(MODEL_ORDER)
    mae = long.pivot(index="Model", columns="Fold", values="MAE").reindex(MODEL_ORDER)
    table = r2.copy()
    table.insert(0, "Category", [CATEGORY[m] for m in table.index])
    table["Mean R2"] = r2.mean(axis=1)
    table["SD R2"] = r2.std(axis=1, ddof=1)
    table["Mean MAE"] = mae.mean(axis=1)
    table["Folds beating HAR (R2)"] = (r2.gt(r2.loc["HAR"], axis=1)).sum(axis=1)
    table = table.reset_index()
    pd.set_option("display.width", 250)
    print("\n", table.round(3).to_string(index=False))
    save_table(table, f"table7_walk_forward{sfx}",
               caption="Expanding-window out-of-sample $R^2$ by fold",
               label="tab:walkforward", float_format="%.3f",
               notes="Columns are test windows; each fold trains on all data up to the year before its test window. "
                     "Identical test days for all models; identical predictors for HAR-X and all machine-learning models. LSTM: average of 5 seeds per fold.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
