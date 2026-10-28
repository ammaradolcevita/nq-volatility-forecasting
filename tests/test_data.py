"""Tests for sample splits and the look-ahead rules (src/nqvol/data.py).

The first group uses small made-up data. The second group uses the real dataset
in data/ and checks the rules the paper relies on.
"""
import numpy as np
import pandas as pd

from nqvol import config as C
from nqvol.data import (add_ratio_columns, chronological_split, fold_masks,
                        load_dataset)


def _daily_frame(start, end):
    return pd.DataFrame({"date": pd.date_range(start, end, freq="D")})


def test_chronological_split_puts_every_training_day_before_every_test_day():
    df = _daily_frame("2020-01-01", "2022-12-31")
    train, test = chronological_split(df, train_end="2021-12-31")
    assert df.loc[train, "date"].max() < df.loc[test, "date"].min()


def test_chronological_split_masks_do_not_overlap_and_cover_every_row():
    df = _daily_frame("2020-01-01", "2022-12-31")
    train, test = chronological_split(df, train_end="2021-12-31")
    assert not np.any(train & test)
    assert np.all(train | test)


def test_walk_forward_fold_trains_on_the_past_only():
    df = _daily_frame("2010-01-01", "2020-12-31")
    train, test = fold_masks(df, (2014, 2015, 2016))
    assert df.loc[train, "date"].max().year == 2014
    assert df.loc[test, "date"].min().year == 2015
    assert df.loc[test, "date"].max().year == 2016
    assert df.loc[train, "date"].max() < df.loc[test, "date"].min()


def test_ratio_columns_divide_by_the_trailing_monthly_range():
    df = pd.DataFrame({
        C.TARGET: [100.0, 200.0],
        "har_daily": [50.0, 100.0],
        "har_weekly": [60.0, 120.0],
        "har_monthly": [100.0, 200.0],
        "asia_range": [10.0, 20.0],
        "london_range": [30.0, 40.0],
        "ndog_range": [5.0, 8.0],
    })
    out = add_ratio_columns(df)
    assert out["ratio_target"].tolist() == [1.0, 1.0]
    assert out["r_london_range"].tolist() == [0.3, 0.2]
    assert C.TARGET in df.columns and "ratio_target" not in df.columns  # input untouched


# --- real dataset ---------------------------------------------------------

def test_har_components_use_only_past_days():
    df = load_dataset()
    # yesterday's range: each row's har_daily equals the previous row's target
    assert np.allclose(df["har_daily"].iloc[1:].values, df[C.TARGET].iloc[:-1].values)
    # the weekly component is the mean of the five previous days, today excluded
    i = 100
    assert np.isclose(df["har_weekly"].iloc[i], df[C.TARGET].iloc[i - 5:i].mean())


def test_real_split_is_strictly_chronological():
    df = load_dataset()
    train, test = chronological_split(df)
    assert df.loc[train, "date"].max() < df.loc[test, "date"].min()
    assert df["date"].is_monotonic_increasing
    assert df["date"].is_unique


def test_every_model_gets_complete_predictor_rows():
    df = load_dataset()
    assert not df[C.FEATURES + [C.TARGET]].isna().any().any()
