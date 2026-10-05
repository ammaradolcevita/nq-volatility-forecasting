"""List the weekdays that are in the raw Kibot file but not in the modelling dataset, with the reason.

Usage: python scripts/00_excluded_days.py [path/to/NQ.txt]   (or set NQ_RAW_PATH)

A weekday is a candidate if the raw file has a 09:30 ET bar for it (28 Sep 2009 - 25 Nov 2025,
the last day of the dataset). The reasons are explained in data/README.md.
Output: data/excluded_days.csv
"""
import os
import sys

import _common  # noqa: F401
import pandas as pd

from nqvol import config as C

POST_HOLIDAY_NO_OVERNIGHT = ["2011-12-27", "2012-01-03", "2012-12-26", "2013-01-02", "2013-12-26", "2014-01-02"]
LIMIT_LOCKED = ["2020-03-16", "2020-03-18"]
NO_NEWS_RECORD = ["2024-12-31", "2025-10-13", "2025-11-11"]   # no entry in the 2025 calendar records


def main(path):
    raw = pd.read_csv(path, header=None, names=["d", "t", "o", "h", "l", "c", "v"])
    raw["d"] = pd.to_datetime(raw["d"], format="%m/%d/%Y")
    raw["t"] = raw["t"].str.strip()
    ds = set(pd.read_csv(C.DATA_PATH, parse_dates=["date"])["date"])
    has_open = set(raw.loc[raw.t == "09:30", "d"])
    has_close = set(raw.loc[raw.t == "16:00", "d"])
    first = pd.Timestamp("2009-09-28")
    cand = sorted(d for d in has_open if d.weekday() < 5 and first <= d <= max(ds))
    rows = []
    for d in cand:
        if d in ds:
            continue
        s = str(d.date())
        if d == first:
            reason = "first day of the raw file (no previous close for the opening gap)"
        elif d not in has_close:
            reason = "U.S. holiday or shortened session (no regular 16:00 ET close)"
        elif s in POST_HOLIDAY_NO_OVERNIGHT:
            reason = "session after a holiday without overnight trading (no Asia/London session)"
        elif s in LIMIT_LOCKED:
            reason = "overnight trading locked at the CME price limit (zero London-session range)"
        elif s in NO_NEWS_RECORD:
            reason = "no entry in the calendar records collected from 30 Dec 2024 (no news classification)"
        else:
            reason = "UNCLASSIFIED"
        rows.append({"date": s, "reason": reason})
    out = pd.DataFrame(rows)
    assert not (out["reason"] == "UNCLASSIFIED").any(), out[out.reason == "UNCLASSIFIED"]
    assert len(cand) - len(out) == len(ds), "candidate days minus exclusions must equal the dataset"
    out.to_csv(C.ROOT / "data" / "excluded_days.csv", index=False)
    print(f"Weekdays with a 09:30 bar: {len(cand)}; in dataset: {len(ds)}; excluded: {len(out)}")
    print(out["reason"].value_counts().to_string())


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("NQ_RAW_PATH", "")
    if not path or not os.path.exists(path):
        sys.exit("Raw Kibot file not found: pass its path or set NQ_RAW_PATH.")
    main(path)
