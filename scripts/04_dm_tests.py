"""Diebold-Mariano tests on the main-split test forecasts (Table 8).

Usage: python scripts/04_dm_tests.py [main|ratio]      (run 01_main_split.py first)

Panel A: every model against the HAR benchmark (plus the Clark-West test for nested models);
         Linear Regression is omitted because it is identical to HAR-X.
Panel B: pairwise comparison of the best model of each family (lowest test MSE).
Panel C: the model pairs reported in the submitted manuscript.
Long-run variance: Newey-West (Bartlett) with bandwidth floor(4(T/100)^(2/9)); Harvey-Leybourne-Newbold
small-sample correction; Student-t p-values; Holm-adjusted p-values within each panel/loss.
A negative DM statistic means that model A has the lower expected loss.
"""
import itertools
import sys

import _common  # noqa: F401
import numpy as np
import pandas as pd

from nqvol.dm import clark_west, dm_test, holm_adjust
from nqvol.models import CATEGORY, MODEL_ORDER
from nqvol.report import PREDICTIONS, TABLES, save_table

PAPER_PAIRS = [("Elastic Net", "CatBoost"), ("Elastic Net", "HAR-X"), ("HAR-X", "CatBoost"),
               ("HAR-X", "HAR"), ("HAR-X", "LSTM"), ("CatBoost", "LSTM"), ("Elastic Net", "LSTM"),
               ("Elastic Net", "Theil-Sen"), ("CatBoost", "XGBoost")]


def run_pairs(pred, pairs, panel):
    y = pred["actual"].to_numpy(float)
    rows = []
    for loss in ["SE", "AE"]:
        block = []
        for a, b in pairs:
            ea, eb = y - pred[a].to_numpy(float), y - pred[b].to_numpy(float)
            stat, p, la, lb, lag = dm_test(ea, eb, loss=loss)
            row = {"Panel": panel, "Loss": loss, "Model A": a, "Model B": b,
                   "Mean loss A": la, "Mean loss B": lb,
                   "Loss difference A vs B (%)": 100 * (la / lb - 1),
                   "DM": stat, "p-value": p, "NW lag": lag}
            if b == "HAR" and loss == "SE" and (CATEGORY[a] == "Linear" or a == "HAR-X"):
                # nested linear models only (their predictor set contains the HAR components)
                cw, cwp = clark_west(y, pred["HAR"], pred[a])
                row.update({"Clark-West stat (vs HAR)": cw, "Clark-West p (one-sided)": cwp})
            block.append(row)
        adj = holm_adjust([r["p-value"] for r in block])
        for r, pa in zip(block, adj):
            r["Holm p-value"] = pa
            better = r["Model A"] if r["DM"] < 0 else r["Model B"]
            r["Better (raw p<0.05)"] = better if r["p-value"] < 0.05 else "none"
            r["Better (Holm p<0.05)"] = better if pa < 0.05 else "none"
        rows += block
    return rows


def main(spec="main"):
    sfx = "" if spec == "main" else f"_{spec}"
    pred = pd.read_csv(PREDICTIONS / f"main_test_predictions{sfx}.csv")
    y = pred["actual"].to_numpy(float)
    mse = {m: np.mean((y - pred[m]) ** 2) for m in MODEL_ORDER}

    champions = {}
    for fam in ["Econometric", "Linear", "Tree", "Deep learning"]:
        members = [m for m in MODEL_ORDER if CATEGORY[m] == fam]
        champions[fam] = min(members, key=mse.get)
    print("Family champions (lowest test MSE):", champions)

    rows = []
    # Linear Regression is omitted from panel A: with the common predictor set it is the same
    # OLS model as HAR-X (identical forecasts), and counting it twice would distort the Holm family.
    rows += run_pairs(pred, [(m, "HAR") for m in MODEL_ORDER if m not in ("HAR", "Linear Regression")],
                      "A: model vs HAR")
    rows += run_pairs(pred, list(itertools.combinations(champions.values(), 2)), "B: family champions")
    rows += run_pairs(pred, PAPER_PAIRS, "C: pairs in submitted manuscript")
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(table.round(4).to_string(index=False))
    save_table(table, f"table8_dm_tests{sfx}", caption="Diebold--Mariano tests of equal predictive accuracy",
               label="tab:dm",
               notes="Newey--West long-run variance (Bartlett kernel, bandwidth $\\lfloor 4(T/100)^{2/9} \\rfloor$), Harvey--Leybourne--Newbold correction, "
                     "Student-$t$ p-values; Holm adjustment within each panel and loss function. "
                     "Negative DM: model A more accurate. Clark--West (2007) statistics for nested comparisons "
                     "against HAR (squared-error loss, one-sided).")

    # full matrix of SE-loss DM statistics (appendix)
    mat = pd.DataFrame(index=MODEL_ORDER, columns=MODEL_ORDER, dtype=float)
    for a in MODEL_ORDER:
        for b in MODEL_ORDER:
            if a != b:
                mat.loc[a, b] = dm_test(y - pred[a], y - pred[b], loss="SE")[0]
    mat.to_csv(TABLES / f"dm_matrix_SE{sfx}.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
