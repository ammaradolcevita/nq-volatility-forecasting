"""Data loading, HAR component construction and sample splits."""
import numpy as np
import pandas as pd

from . import config as C


def load_dataset(path=C.DATA_PATH) -> pd.DataFrame:
    """Load the daily modelling dataset and add the HAR persistence components.

    HAR components (Corsi, 2009) are built from past NY-session ranges only:
        har_daily   = range on day t-1
        har_weekly  = mean range over days t-5 ... t-1
        har_monthly = mean range over days t-22 ... t-1
    The first 22 observations have no complete monthly window and are dropped,
    so that every model is evaluated on the same rows.
    """
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    y = df[C.TARGET]
    df["har_daily"] = y.shift(1)
    df["har_weekly"] = y.rolling(5, min_periods=5).mean().shift(1)
    df["har_monthly"] = y.rolling(22, min_periods=22).mean().shift(1)
    df = df.dropna(subset=C.HAR_FEATURES).reset_index(drop=True)
    return df


def chronological_split(df: pd.DataFrame, train_end: str = C.TRAIN_END):
    """Boolean masks for a strictly chronological train/test split."""
    train_mask = (df["date"] <= pd.Timestamp(train_end)).values
    test_mask = ~train_mask
    return train_mask, test_mask


def fold_masks(df: pd.DataFrame, fold):
    """Masks for one expanding-window fold (last train year, first/last test year)."""
    train_last, test_first, test_last = fold
    year = df["date"].dt.year
    train_mask = (year <= train_last).values
    test_mask = ((year >= test_first) & (year <= test_last)).values
    return train_mask, test_mask


def add_ratio_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Scale-free specification: divide target and range-type predictors by the
    trailing 22-day mean range (har_monthly), which is known before the open."""
    out = df.copy()
    norm = out[C.RATIO_NORMALISER]
    out["ratio_target"] = out[C.TARGET] / norm
    for col in C.RATIO_SCALED_COLUMNS:
        out["r_" + col] = out[col] / norm
    return out


RATIO_FEATURES = (["r_har_daily", "r_har_weekly", "r_asia_range", "asia_dir",
                   "r_london_range", "london_dir", "r_ndog_range"]
                  + C.CALENDAR_FEATURES + C.NEWS_FEATURES)


def split_summary(df, train_mask, test_mask) -> pd.DataFrame:
    rows = []
    for name, m in [("Train", train_mask), ("Test", test_mask)]:
        part = df.loc[m]
        rows.append({
            "Set": name,
            "Start": part["date"].min().date(),
            "End": part["date"].max().date(),
            "N": int(m.sum()),
            "Share (%)": round(100 * m.sum() / len(df), 1),
            "Mean range": round(part[C.TARGET].mean(), 2),
            "Std range": round(part[C.TARGET].std(), 2),
        })
    return pd.DataFrame(rows)
