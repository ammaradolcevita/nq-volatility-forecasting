"""Robustness checks.

Usage: python scripts/05_robustness.py [main|ratio]   (run 01_main_split.py first)

(1) Adding `year` as a predictor (reviewer #9). For the 17 models that use the full
    predictor set (all except HAR and LogHAR): in-sample adjusted R^2, out-of-sample R^2,
    change in test MSE and MAE, and Diebold-Mariano tests of the model with versus
    without `year`. Out-of-sample R^2 does not mechanically fall when a predictor is
    removed, so the DM test is the relevant evidence.
(2) [main spec only] The submitted "regime-stratified" split re-run with the new common
    predictor set, with and without `year`. This quantifies how much of the accuracy
    reported in the submitted manuscript came from the split design (reviewer #2).
"""
import sys

import _common  # noqa: F401
import numpy as np
import pandas as pd

from nqvol import config as C
from nqvol import metrics as M
from nqvol.data import add_ratio_columns, chronological_split, load_dataset
from nqvol.dm import dm_test
from nqvol.models import CATEGORY, MODEL_ORDER
from nqvol.pipeline import SPECS, fit_predict_all
from nqvol.report import PREDICTIONS, save_table


def regime_stratified_masks(df):
    """Replicates the split of the submitted manuscript: years whose mean range exceeds 80
    points form the 'high' regime; the first 80% of each regime (in time order) is used
    for training and the last 20% for testing."""
    year_mean = df.groupby(df["date"].dt.year)[C.TARGET].mean()
    high_years = year_mean[year_mean > 80].index
    high = df["date"].dt.year.isin(high_years).values
    train = np.zeros(len(df), bool)
    for regime in (~high, high):
        idx = np.flatnonzero(regime)
        train[idx[: int(len(idx) * 0.8)]] = True
    return train, ~train


def year_check(df, spec, sfx):
    train_mask, test_mask = chronological_split(df)
    y = df[C.TARGET].to_numpy(float)
    y_test = y[test_mask]
    base_test = pd.read_csv(PREDICTIONS / f"main_test_predictions{sfx}.csv")
    base_train = pd.read_csv(PREDICTIONS / f"main_train_predictions{sfx}.csv")
    print(f"Fitting models with `year` added (spec={spec}) ...")
    out = fit_predict_all(df, train_mask, test_mask, spec=spec, extra_features=["year"], verbose=False)
    k0 = len(SPECS[spec]["features"])
    rows = []
    for name in MODEL_ORDER:
        if name in ("HAR", "LogHAR"):
            continue  # these use only the persistence components
        p0, p1 = base_test[name].to_numpy(float), out[name]["test_pred"]
        tr0 = base_train[name].to_numpy(float)
        ok = ~np.isnan(tr0)
        ytr = base_train["actual"].to_numpy(float)
        r2tr0 = M.r2(ytr[ok], tr0[ok])
        r2tr1 = M.r2(y[out[name]["train_index"]], out[name]["train_pred"])
        n = ok.sum()
        se = dm_test(y_test - p1, y_test - p0, loss="SE")
        ae = dm_test(y_test - p1, y_test - p0, loss="AE")
        rows.append({
            "Model": name, "Category": CATEGORY[name],
            "Train adj. R2 without year": M.adjusted_r2(r2tr0, n, k0),
            "Train adj. R2 with year": M.adjusted_r2(r2tr1, n, k0 + 1),
            "Test R2 without year": M.r2(y_test, p0),
            "Test R2 with year": M.r2(y_test, p1),
            "Test MSE change with year (%)": 100 * (np.mean((y_test - p1) ** 2) / np.mean((y_test - p0) ** 2) - 1),
            "Test MAE change with year (%)": 100 * (np.mean(np.abs(y_test - p1)) / np.mean(np.abs(y_test - p0)) - 1),
            "DM (SE) with vs without": se[0], "p (SE)": se[1],
            "DM (AE) with vs without": ae[0], "p (AE)": ae[1],
        })
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))
    save_table(table, f"table9_year_robustness{sfx}",
               caption="Adding \\texttt{year} as a predictor: in-sample adjusted and out-of-sample fit",
               label="tab:year",
               notes="Negative DM statistic: the specification with year has lower expected loss. "
                     "Newey--West/HLN Diebold--Mariano test on the 2022--2025 test period.")


def split_decomposition(df):
    y = df[C.TARGET].to_numpy(float)
    rows = []
    designs = {
        "Regime-stratified split (submitted), with year": (regime_stratified_masks(df), ["year"]),
        "Regime-stratified split (submitted), without year": (regime_stratified_masks(df), []),
        "Chronological split (revised), with year": (chronological_split(df), ["year"]),
        "Chronological split (revised), without year": (chronological_split(df), []),
    }
    for label, ((tr, te), extra) in designs.items():
        print(f"\n{label}: N train {tr.sum()}, N test {te.sum()}")
        out = fit_predict_all(df, tr, te, spec="main", extra_features=extra, verbose=False)
        for name in MODEL_ORDER:
            s = M.summary(y[te], out[name]["test_pred"])
            rows.append({"Design": label, "Model": name, "Category": CATEGORY[name], **s})
    long = pd.DataFrame(rows)
    wide = long.pivot(index="Model", columns="Design", values="R2").reindex(MODEL_ORDER)[list(designs)]
    wide.insert(0, "Category", [CATEGORY[m] for m in wide.index])
    wide = wide.reset_index()
    print(wide.round(3).to_string(index=False))
    long.to_csv(C.RESULTS_DIR / "tables" / "split_decomposition_long.csv", index=False)
    save_table(wide, "split_decomposition_r2",
               caption="Test $R^2$ under the submitted and the revised sample split",
               label="tab:split_decomp", float_format="%.3f",
               notes="All designs use the revised common predictor set. The regime-stratified split mixes "
                     "2017--2019 and 2024--2025 test days and trains on 2020--2024.")


def main(spec="main"):
    sfx = "" if spec == "main" else f"_{spec}"
    df = load_dataset()
    if spec == "ratio":
        df = add_ratio_columns(df)
    year_check(df, spec, sfx)
    if spec == "main":
        split_decomposition(df)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main")
