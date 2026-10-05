"""Gap midpoint retracement (RQ2 / H1), computed from the raw Kibot bars.

Usage: python scripts/08_gap_retracement.py [path/to/NQ.txt]   (or set NQ_RAW_PATH)

Definition (as in the manuscript): the overnight gap of day t runs from the close of the
previous trading day's 16:00 bar to the open of day t's 09:30 bar. The gap midpoint is "reached" if it lies
within the low-high range of the bars starting 09:30, 10:00 and 10:30 ET (09:30-11:00 ET).
Days: the 4,021 trading days of data/ml_dataset1.csv.

Output: results/tables/gap_retracement.csv, results/tables/gap_retracement_quintiles.csv
"""
import os
import sys

import _common  # noqa: F401
import numpy as np
import pandas as pd
from scipy import stats

from nqvol import config as C
from nqvol.report import TABLES


def main(path):
    raw = pd.read_csv(path, header=None, names=["d", "t", "o", "h", "l", "c", "v"])
    raw["d"] = pd.to_datetime(raw["d"], format="%m/%d/%Y")
    raw["t"] = raw["t"].str.strip()
    ds = pd.read_csv(C.DATA_PATH, parse_dates=["date"]).set_index("date")

    days = sorted(set(raw.loc[raw.t == "09:30", "d"]) & set(raw.loc[raw.t == "16:00", "d"]))
    prev = {days[i]: days[i - 1] for i in range(1, len(days))}
    close16 = raw[raw.t == "16:00"].drop_duplicates("d").set_index("d")["c"]
    open930 = raw[raw.t == "09:30"].drop_duplicates("d").set_index("d")["o"]
    ny = raw[raw.t.isin(["09:30", "10:00", "10:30"])].groupby("d").agg(h=("h", "max"), l=("l", "min"))

    rows = []
    for d in ds.index:
        pc, op = close16[prev[d]], open930[d]
        mid = (pc + op) / 2
        rows.append({"date": d, "gap": abs(op - pc), "hit": bool(ny.loc[d, "l"] <= mid <= ny.loc[d, "h"])})
    r = pd.DataFrame(rows)
    assert np.allclose(r["gap"].values, ds["ndog_range"].values), "gap does not reproduce the dataset"

    nz = r[r.gap > 0].copy()
    out = []
    for label, sub in [("All days", r), ("Non-zero gaps", nz)]:
        k, n = int(sub.hit.sum()), len(sub)
        test = stats.binomtest(k, n, 0.5, alternative="greater")
        out.append({"Sample": label, "Days": n, "Midpoint reached": k, "Rate (%)": 100 * k / n,
                    "Binomial p (H0: 50%, one-sided)": test.pvalue})
    summary = pd.DataFrame(out)
    nz["Quintile"] = pd.qcut(nz.gap, 5, labels=[1, 2, 3, 4, 5])
    q = nz.groupby("Quintile", observed=True).agg(Days=("hit", "size"), Rate=("hit", "mean"),
                                                  Min_gap=("gap", "min"), Max_gap=("gap", "max"))
    q["Rate"] *= 100
    q = q.rename(columns={"Rate": "Rate (%)", "Min_gap": "Min gap (points)", "Max_gap": "Max gap (points)"})
    print(summary.to_string(index=False))
    print(q.round(2).to_string())
    TABLES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLES / "gap_retracement.csv", index=False)
    q.reset_index().to_csv(TABLES / "gap_retracement_quintiles.csv", index=False)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("NQ_RAW_PATH", "")
    if not path or not os.path.exists(path):
        sys.exit("Raw Kibot file not found: pass its path or set NQ_RAW_PATH.")
    main(path)
