"""Session summaries versus individual 30-minute bars (reviewer comment #3).

Usage: python scripts/07_raw_bars.py [path/to/NQ.txt] [main|ratio]
       (default path: environment variable NQ_RAW_PATH; default spec: ratio)

The raw Kibot 30-minute bars are licensed and are NOT part of this repository.
This script reads them from the path given, builds the high-low range of every
individual bar in the Asia and London windows and compares three predictor sets:

  S    summary : the paper's predictors (asia_range, asia_dir, london_range, london_dir + common set)
  B    bars    : asia_range and london_range replaced by the 15 bar ranges
                 (asia_bar1..asia_bar9, london_bar1..london_bar6); asia_dir, london_dir kept
  S+B  both    : the paper's predictors plus the 15 bar ranges

Bar definitions (bar START times, US Eastern), identical to the code that built
data/ml_dataset1.csv:
  Asia  of day D: bars 20:00, 20:30, ..., 23:30 of the previous calendar day and the
        00:00 bar of day D (asia_bar1 = 20:00 ... asia_bar9 = 00:00). Bars at or after
        20:00 are assigned to the next calendar day, so Monday's Asia session is the
        Sunday-evening session (Friday-evening bars fall on Saturday and are unused).
  London of day D: bars 02:00 ... 04:30 of day D (london_bar1 = 02:00 ... london_bar6 = 04:30).
A missing bar gets range 0 (counted and reported).

Step 1 validates the bars: the session aggregates recomputed from them must reproduce
asia_range, asia_dir, london_range, london_dir of data/ml_dataset1.csv.
Models: HAR-X (OLS), Elastic Net, Random Forest, CatBoost, LightGBM with the
manuscript hyper-parameters (nqvol.models.build_models). The LSTM is not run
(too slow for three predictor sets x six splits on a 2-core machine).

Outputs
  results/tables/table_raw_bars[_ratio].csv/.tex        main split + walk-forward mean + DM vs S
  results/tables/raw_bars_walk_forward_long[_ratio].csv every model x set x fold
  results/tables/raw_bars_evidence[_ratio].csv          validation and correlation evidence
"""
import os
import sys
import time
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import pandas as pd

from nqvol import config as C
from nqvol import metrics as M
from nqvol.data import (RATIO_FEATURES, add_ratio_columns, chronological_split,
                        fold_masks, load_dataset)
from nqvol.dm import dm_test
from nqvol.models import build_models
from nqvol.report import TABLES, save_table

ASIA_SLOTS = ["20:00", "20:30", "21:00", "21:30", "22:00", "22:30", "23:00", "23:30", "00:00"]
LONDON_SLOTS = ["02:00", "02:30", "03:00", "03:30", "04:00", "04:30"]
ASIA_BARS = [f"asia_bar{i}" for i in range(1, len(ASIA_SLOTS) + 1)]
LONDON_BARS = [f"london_bar{i}" for i in range(1, len(LONDON_SLOTS) + 1)]
BARS = ASIA_BARS + LONDON_BARS

MODELS = ["HAR-X", "Elastic Net", "Random Forest", "CatBoost", "LightGBM"]
SETS = ["S", "B", "S+B"]
SET_LABEL = {"S": "S (session summaries)", "B": "B (15 bar ranges)", "S+B": "S+B (both)"}


# ---------------------------------------------------------------------------
# Raw bars
# ---------------------------------------------------------------------------
def read_bars(path):
    raw = pd.read_csv(path, header=None,
                      names=["d", "t", "open", "high", "low", "close", "volume"],
                      dtype={"d": str, "t": str})
    raw["dt"] = pd.to_datetime(raw["d"] + " " + raw["t"], format="%m/%d/%Y %H:%M")
    raw["slot"] = raw["dt"].dt.strftime("%H:%M")
    return raw.sort_values("dt").reset_index(drop=True)


def session_bars(raw, slots, next_day_from=None):
    """Bars of one session window with the trading day they belong to."""
    s = raw[raw["slot"].isin(slots)].copy()
    s["date"] = s["dt"].dt.normalize()
    if next_day_from is not None:   # Asia: evening bars belong to the next calendar day
        s.loc[s["dt"].dt.hour >= next_day_from, "date"] += pd.Timedelta(days=1)
    return s


def aggregate(s):
    """Session range and direction exactly as in the original code."""
    g = s.sort_values("dt").groupby("date")
    return pd.DataFrame({"range": g["high"].max() - g["low"].min(),
                         "dir": np.sign(g["close"].last() - g["open"].first()),
                         "n_bars": g.size()})


def bar_ranges(s, slots, names, dates):
    w = (s.assign(rng=s["high"] - s["low"])
          .pivot(index="date", columns="slot", values="rng")
          .reindex(index=dates, columns=slots))
    w.columns = names
    return w


def build_bar_features(raw, dates):
    asia = session_bars(raw, ASIA_SLOTS, next_day_from=20)
    london = session_bars(raw, LONDON_SLOTS)
    wide = pd.concat([bar_ranges(asia, ASIA_SLOTS, ASIA_BARS, dates),
                      bar_ranges(london, LONDON_SLOTS, LONDON_BARS, dates)], axis=1)
    missing = wide.isna()
    agg = pd.concat([aggregate(asia).add_prefix("asia_"),
                     aggregate(london).add_prefix("london_")], axis=1).reindex(dates)
    return wide.fillna(0.0), missing, agg


# ---------------------------------------------------------------------------
# Validation and descriptive evidence
# ---------------------------------------------------------------------------
def validate(full, agg, missing):
    rows = []
    print("\nVALIDATION: session aggregates recomputed from the bars vs data/ml_dataset1.csv "
          f"({len(full)} days)")
    for col in ["asia_range", "asia_dir", "london_range", "london_dir"]:
        diff = (full[col].to_numpy(float) - agg[col].to_numpy(float))
        ok = np.abs(diff) <= 1e-9
        ok &= ~np.isnan(diff)
        print(f"  {col:<13} exact match (|diff| <= 1e-9): {ok.sum()}/{len(ok)} = {ok.mean():.6f}"
              f"   max |diff| = {np.nanmax(np.abs(diff)):.3g}")
        rows.append({"Item": f"match rate {col}", "Value": ok.mean(), "Count": int(ok.sum()), "N": len(ok)})
        if (~ok).any():
            print(full.loc[~ok, ["date", col]].assign(recomputed=agg[col].to_numpy()[~ok]).head(20).to_string(index=False))
    miss_days = missing.any(axis=1)
    print(f"\n  Days with at least one missing bar: {int(miss_days.sum())} "
          f"(Asia {int(missing[ASIA_BARS].any(axis=1).sum())}, London {int(missing[LONDON_BARS].any(axis=1).sum())}); "
          f"missing bars in total: {int(missing.values.sum())} of {missing.size}. Filled with range 0.")
    for d, r in missing[miss_days].iterrows():
        print(f"    {d.date()}  missing: {', '.join(r.index[r.values])}")
    rows.append({"Item": "days with missing bars", "Value": float(miss_days.sum()),
                 "Count": int(miss_days.sum()), "N": len(miss_days)})
    rows.append({"Item": "missing bars (total)", "Value": float(missing.values.sum()),
                 "Count": int(missing.values.sum()), "N": int(missing.size)})
    return rows


def correlation_evidence(df, spec, train_mask):
    """Corr(sum / max of bar ranges, session range) and collinearity among the bar ranges."""
    pre = "r_" if spec == "ratio" else ""
    rows = []
    print(f"\nEVIDENCE (spec={spec}; variables as they enter the models, full modelling sample "
          f"and training sample)")
    for sample, m in [("all", np.ones(len(df), bool)), ("train", train_mask)]:
        d = df.loc[m]
        for sess, bars in [("asia", ASIA_BARS), ("london", LONDON_BARS)]:
            b = d[[pre + c for c in bars]]
            sr = d[pre + f"{sess}_range"]
            for stat, v in [("sum", b.sum(axis=1)), ("max", b.max(axis=1))]:
                c = np.corrcoef(v, sr)[0, 1]
                rows.append({"Item": f"corr({stat} of {sess} bar ranges, {sess}_range) [{sample}]",
                             "Value": c, "Count": np.nan, "N": int(m.sum())})
            cb = b.corr().to_numpy()
            iu = np.triu_indices_from(cb, 1)
            rows.append({"Item": f"mean pairwise corr among {len(bars)} {sess} bars [{sample}]",
                         "Value": cb[iu].mean(), "Count": np.nan, "N": int(m.sum())})
        b = d[[pre + c for c in BARS]]
        cb = b.corr().to_numpy()
        iu = np.triu_indices_from(cb, 1)
        rows.append({"Item": f"mean pairwise corr among 15 bars [{sample}]", "Value": cb[iu].mean(),
                     "Count": np.nan, "N": int(m.sum())})
        rows.append({"Item": f"min pairwise corr among 15 bars [{sample}]", "Value": cb[iu].min(),
                     "Count": np.nan, "N": int(m.sum())})
        rows.append({"Item": f"max pairwise corr among 15 bars [{sample}]", "Value": cb[iu].max(),
                     "Count": np.nan, "N": int(m.sum())})
        # variance inflation factors of the bar ranges within the B design (without calendar/news)
        Xb = b.to_numpy(float)
        Xb = (Xb - Xb.mean(0)) / Xb.std(0)
        vif = np.diag(np.linalg.inv(np.corrcoef(Xb, rowvar=False)))
        rows.append({"Item": f"mean VIF of the 15 bar ranges (among themselves) [{sample}]",
                     "Value": vif.mean(), "Count": np.nan, "N": int(m.sum())})
        rows.append({"Item": f"max VIF of the 15 bar ranges (among themselves) [{sample}]",
                     "Value": vif.max(), "Count": np.nan, "N": int(m.sum())})
        tgt = d["ratio_target" if spec == "ratio" else C.TARGET]
        for name, v in [("asia_range", d[pre + "asia_range"]), ("london_range", d[pre + "london_range"])]:
            rows.append({"Item": f"corr({name}, target) [{sample}]", "Value": np.corrcoef(v, tgt)[0, 1],
                         "Count": np.nan, "N": int(m.sum())})
        bc = [np.corrcoef(d[pre + c], tgt)[0, 1] for c in BARS]
        rows.append({"Item": f"mean corr(single bar range, target) [{sample}]", "Value": float(np.mean(bc)),
                     "Count": np.nan, "N": int(m.sum())})
        rows.append({"Item": f"max corr(single bar range, target) [{sample}]", "Value": float(np.max(bc)),
                     "Count": np.nan, "N": int(m.sum())})
    for r in rows:
        print(f"  {r['Item']:<72} {r['Value']: .4f}")
    return rows


# ---------------------------------------------------------------------------
# Predictor sets and model fitting
# ---------------------------------------------------------------------------
def predictor_sets(spec):
    if spec == "ratio":
        base, pre = list(RATIO_FEATURES), "r_"
    else:
        base, pre = list(C.FEATURES), ""
    bars = [pre + c for c in BARS]
    a, l = pre + "asia_range", pre + "london_range"
    b_set = []
    for f in base:   # replace the two session ranges in place by their bars
        if f == a:
            b_set += [pre + c for c in ASIA_BARS]
        elif f == l:
            b_set += [pre + c for c in LONDON_BARS]
        else:
            b_set.append(f)
    return {"S": base, "B": b_set, "S+B": base + bars}


def fit_predict(df, features, target, scale, train_mask, test_mask, factories):
    X = df[features]
    y = df[target].to_numpy(float)
    out = {}
    for name in MODELS:
        est = factories[name]().fit(X[train_mask], y[train_mask])
        out[name] = dict(train_pred=est.predict(X[train_mask]) * scale[train_mask],
                         test_pred=est.predict(X[test_mask]) * scale[test_mask])
    return out


def main(path, spec="ratio"):
    t_start = time.time()
    sfx = "" if spec == "main" else f"_{spec}"
    print(f"Raw bars: {os.path.basename(path)}\nSpecification: {spec}")

    raw = read_bars(path)
    print(f"Read {len(raw)} bars, {raw['dt'].min()} -> {raw['dt'].max()}")

    # validation on every day of the published dataset (before dropping the HAR burn-in)
    full = pd.read_csv(C.DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    wide, missing, agg = build_bar_features(raw, pd.DatetimeIndex(full["date"]))
    evidence = validate(full, agg.reset_index(drop=True), missing)

    # modelling dataset (same rows as every other script)
    df = load_dataset()
    df = df.merge(wide.rename_axis("date").reset_index(), on="date", how="left", validate="1:1")
    assert df[BARS].notna().all().all()
    m_days = missing.loc[pd.DatetimeIndex(df["date"])].any(axis=1)
    print(f"  In the modelling sample ({len(df)} days): {int(m_days.sum())} days with missing bars")
    evidence.append({"Item": "days with missing bars (modelling sample)", "Value": float(m_days.sum()),
                     "Count": int(m_days.sum()), "N": len(df)})
    if spec == "ratio":
        df = add_ratio_columns(df)
        for c in BARS:
            df["r_" + c] = df[c] / df[C.RATIO_NORMALISER]
        target, scale = "ratio_target", df[C.RATIO_NORMALISER].to_numpy(float)
    else:
        target, scale = C.TARGET, np.ones(len(df))
    y = df[C.TARGET].to_numpy(float)
    train_mask, test_mask = chronological_split(df)
    print(f"  N train {train_mask.sum()}, N test {test_mask.sum()}")

    evidence += correlation_evidence(df, spec, train_mask)

    sets = predictor_sets(spec)
    for s, f in sets.items():
        print(f"\nSet {s} (k={len(f)}): {', '.join(f)}")
    har_cols = ["r_har_daily", "r_har_weekly"] if spec == "ratio" else C.HAR_FEATURES
    factories = {n: fac for n, (_, fac) in build_models(har_cols).items()}

    # ---------------- main chronological split ----------------
    print("\nMAIN SPLIT (train <= 2021-12-31, test 2022-01-03 .. 2025-11-25)")
    main_out = {}
    for s, feats in sets.items():
        t0 = time.time()
        main_out[s] = fit_predict(df, feats, target, scale, train_mask, test_mask, factories)
        print(f"  set {s:<4} fitted in {time.time() - t0:6.1f}s", flush=True)

    # sanity check: S must reproduce Table 5 (and, below, Table 7)
    t5 = pd.read_csv(TABLES / f"table5_main_results{sfx}.csv").set_index("Model")
    y_te = y[test_mask]
    print(f"\nSanity check: set S vs results/tables/table5_main_results{sfx}.csv (test R2)")
    for name in MODELS:
        r2s = M.r2(y_te, main_out["S"][name]["test_pred"])
        ref = float(t5.loc[name, "Test R2"])
        print(f"  {name:<14} S {r2s:.10f}   Table 5 {ref:.10f}   diff {r2s - ref: .2e}"
              f"   {'OK' if abs(r2s - ref) < 1e-9 else 'MISMATCH'}")

    # ---------------- walk-forward ----------------
    print("\nWALK-FORWARD (5 expanding folds)")
    wf_rows = []
    for fold in C.WALK_FORWARD_FOLDS:
        tr, te = fold_masks(df, fold)
        label = f"{fold[1]}-{str(fold[2])[-2:]}"
        t0 = time.time()
        for s, feats in sets.items():
            out = fit_predict(df, feats, target, scale, tr, te, factories)
            for name in MODELS:
                wf_rows.append({"Fold": label, "Train end": fold[0], "N train": int(tr.sum()),
                                "N test": int(te.sum()), "Model": name, "Set": s,
                                **M.summary(y[te], out[name]["test_pred"])})
        print(f"  fold {label}: {time.time() - t0:6.1f}s", flush=True)
    wf = pd.DataFrame(wf_rows)
    t7_path = TABLES / f"walk_forward_long{sfx}.csv"
    if t7_path.is_file():
        t7 = wf[wf.Set == "S"].merge(pd.read_csv(t7_path), on=["Fold", "Model"], suffixes=("", "_t7"))
        print(f"  Sanity check: set S vs walk_forward_long{sfx}.csv: {len(t7)} model-folds, "
              f"max |R2 diff| = {(t7['R2'] - t7['R2_t7']).abs().max():.2e}")
    TABLES.mkdir(parents=True, exist_ok=True)
    wf.to_csv(TABLES / f"raw_bars_walk_forward_long{sfx}.csv", index=False)

    # ---------------- table ----------------
    rows = []
    for name in MODELS:
        e_s = y_te - main_out["S"][name]["test_pred"]
        wf_s = wf[(wf.Model == name) & (wf.Set == "S")].set_index("Fold")["R2"]
        for s in SETS:
            r = main_out[s][name]
            y_tr = y[train_mask]
            k = len(sets[s])
            tr_ = M.summary(y_tr, r["train_pred"])
            te_ = M.summary(y_te, r["test_pred"])
            wf_r2 = wf[(wf.Model == name) & (wf.Set == s)].set_index("Fold")["R2"]
            row = {"Model": name, "Predictors": s, "k": k,
                   "Train R2": tr_["R2"], "Train adj. R2": M.adjusted_r2(tr_["R2"], len(y_tr), k),
                   "Test R2": te_["R2"], "Test MAE": te_["MAE"], "Test RMSE": te_["RMSE"],
                   "WF mean R2": wf_r2.mean(), "WF folds > S": np.nan,
                   "DM SE vs S": np.nan, "p SE": np.nan, "DM AE vs S": np.nan, "p AE": np.nan}
            if s != "S":
                e = y_te - r["test_pred"]
                row["DM SE vs S"], row["p SE"], *_ = dm_test(e, e_s, loss="SE")
                row["DM AE vs S"], row["p AE"], *_ = dm_test(e, e_s, loss="AE")
                row["WF folds > S"] = int((wf_r2 > wf_s.reindex(wf_r2.index)).sum())
            rows.append(row)
    table = pd.DataFrame(rows)
    table["WF folds > S"] = table["WF folds > S"].astype("Int64")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\nRESULTS (metrics in index points; DM < 0 = set has lower loss than S)")
    print(table.round(4).to_string(index=False))

    wf_piv = wf.pivot_table(index=["Model", "Set"], columns="Fold", values="R2").reindex(
        pd.MultiIndex.from_product([MODELS, SETS]))
    wf_piv["Mean"] = wf_piv.mean(axis=1)
    print("\nWalk-forward R2 by fold")
    print(wf_piv.round(4).to_string())

    notes = ("S: the paper's predictors (asia_range, asia_dir, london_range, london_dir, ndog_range, "
             "HAR components, month, weekday, impact_medium, impact_high). "
             "B: asia_range and london_range replaced by the high-low ranges of the 9 Asia and 6 London "
             "30-minute bars. S+B: S plus the 15 bar ranges. "
             + ("Range variables divided by har_monthly; forecasts rescaled to index points. " if spec == "ratio" else "")
             + "Test: 2022-01-03 to 2025-11-25 (969 days). WF: mean out-of-sample R2 over the 5 expanding "
             "folds; WF folds > S: folds in which the set beats S. DM: Diebold-Mariano statistic "
             "(Newey-West, HLN) of the set against S under squared (SE) and absolute (AE) loss; "
             "negative values favour the set.")
    save_table(table, f"table_raw_bars{sfx}", caption="Session summaries versus individual 30-minute bars",
               label="tab:rawbars", notes=notes)
    ev = pd.DataFrame(evidence)
    ev.to_csv(TABLES / f"raw_bars_evidence{sfx}.csv", index=False)
    print(f"\nSaved table_raw_bars{sfx}.csv/.tex, raw_bars_walk_forward_long{sfx}.csv, "
          f"raw_bars_evidence{sfx}.csv to results/tables/")
    print(f"Total runtime: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    args = sys.argv[1:]
    spec = "ratio"
    if args and args[-1] in ("main", "ratio"):
        spec = args.pop()
    path = args[0] if args else os.environ.get("NQ_RAW_PATH")
    if not path or not Path(path).is_file():
        sys.exit("Raw Kibot file not found: pass the path to NQ.txt or set NQ_RAW_PATH "
                 "(the licensed bars are not distributed with this repository).")
    main(path, spec)
