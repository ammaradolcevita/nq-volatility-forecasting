"""Central configuration: one place that defines the sample split, the predictor
set and the random seeds used by every model family.

Keeping these definitions in a single module guarantees that all 19 models are
trained and evaluated on exactly the same rows, and that HAR-X and every
machine-learning model use exactly the same predictors (the HAR and LogHAR
benchmarks use the persistence components only; reviewer comments #2 and #7).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "ml_dataset1.csv"
RESULTS_DIR = ROOT / "results"

SEED = 42
TARGET = "ny_range"  # NY session (09:30-11:00 ET) high-low range, index points

# ---------------------------------------------------------------------------
# Sample split: strictly chronological.
# Every training observation precedes every test observation.
# Training covers both the low-volatility years (2009-2019) and the
# high-volatility years 2020-2021; the test period 2022-2025 contains the 2022
# drawdown and the 2023-2025 recovery.
# ---------------------------------------------------------------------------
TRAIN_END = "2021-12-31"

# Expanding-window (walk-forward) folds: (last training year, first test year, last test year)
WALK_FORWARD_FOLDS = [
    (2014, 2015, 2016),
    (2016, 2017, 2018),
    (2018, 2019, 2020),
    (2020, 2021, 2022),
    (2022, 2023, 2025),
]

# ---------------------------------------------------------------------------
# Predictors: ONE common set used by every model (HAR-X, linear, tree, LSTM).
# impact_low is the omitted reference category of the news dummies
# (impact_low + impact_medium + impact_high == 1 on every day).
# `year` is excluded from the main specification (see robustness script).
# ---------------------------------------------------------------------------
HAR_FEATURES = ["har_daily", "har_weekly", "har_monthly"]
SESSION_FEATURES = ["asia_range", "asia_dir", "london_range", "london_dir", "ndog_range"]
CALENDAR_FEATURES = ["month", "weekday"]
NEWS_FEATURES = ["impact_medium", "impact_high"]

FEATURES = HAR_FEATURES + SESSION_FEATURES + CALENDAR_FEATURES + NEWS_FEATURES

FEATURE_GROUPS = {
    "Persistence (HAR)": HAR_FEATURES,
    "Session": SESSION_FEATURES,
    "Calendar": CALENDAR_FEATURES,
    "Macro news": NEWS_FEATURES,
}

# Scale-free (primary) specification: target and range-type predictors are
# divided by the trailing 22-day mean range (har_monthly), known before the open.
RATIO_NORMALISER = "har_monthly"
RATIO_SCALED_COLUMNS = ["har_daily", "har_weekly", "asia_range", "london_range", "ndog_range"]

# LSTM
LSTM_SEQUENCE_LENGTH = 5          # days t-4 ... t (all inputs of day t are known before 09:30 ET)
LSTM_SEEDS_MAIN = list(range(42, 52))   # 10 seeds on the main split
LSTM_SEEDS_WF = list(range(42, 47))     # 5 seeds per walk-forward fold
