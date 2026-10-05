# Session-based forecasting of E-mini NASDAQ-100 NY-session ranges

Replication code for *"Intraday Spillovers Across Global Sessions and Machine-Learning Forecasts
of U.S. Equity Index Futures Ranges"* (revised version, 2026).

The target is the high–low range of the E-mini NASDAQ-100 futures (NQ) during the first
90 minutes of the New York session (09:30–11:00 ET). Predictors describe what happened
before the open: the Asia and London sessions, the opening gap, recent NY ranges
(HAR persistence components), the calendar, and scheduled U.S. macro announcements.
Nineteen models are compared: 3 econometric (HAR, HAR-X, LogHAR), 8 linear,
7 tree-based and 1 LSTM.

## Design principles of the revision

- **Strictly chronological evaluation.** Train 2009–2021, test 2022–2025, plus five
  expanding-window folds. No test observation precedes a training observation.
- **One common predictor set.** HAR-X and every machine-learning model (linear, tree, LSTM)
  use the same 12 predictors; the pure HAR and LogHAR benchmarks use the persistence components
  only. All 19 models are scored on the same test days. The set is defined once, in
  `src/nqvol/config.py`.
- **One shared code base.** All scripts import the same data, split, model and metric code
  from `src/nqvol`, so the model families cannot drift apart.

`docs/CHANGES.md` lists every change from the submitted version and the reviewer comment it answers.

## Repository layout

```
data/            ml_dataset1.csv (daily modelling dataset), data dictionary and sample rule (data/README.md),
                 excluded_days.csv (every weekday not in the dataset, with the reason)
src/nqvol/       config.py    split dates, predictor set, seeds
                 data.py      loading, HAR components, splits, scale-free variables
                 models.py    18 models with the hyper-parameters of Table 4
                 lstm.py      LSTM ensemble
                 pipeline.py  fits all models on a split
                 metrics.py   R2, adjusted R2, MAE, RMSE, MAPE
                 dm.py        Diebold-Mariano test (Newey-West, HLN) and Holm adjustment
scripts/         00_excluded_days.py       data/excluded_days.csv (needs raw Kibot bars)
                 01_main_split.py          Table 3, Table 5 (train and test metrics), HAR coefficients
                 02_walk_forward.py        Table 7 (all models, 5 folds)
                 03_feature_importance.py  Table 6 (permutation importance, all models)
                 04_dm_tests.py            Table 8 (Diebold-Mariano)
                 05_robustness.py          Table 9 (year), split decomposition
                 07_raw_bars.py            Table 11: session summaries vs individual 30-minute bars (needs raw Kibot bars)
                 08_gap_retracement.py     gap midpoint retracement rate, quintiles, binomial test (needs raw Kibot bars)
results/tables/       every table as .csv and .tex (suffix _ratio = scale-free specification)
results/predictions/  test and train forecasts of every model
results/logs/         console output of each script
docs/            CHANGES.md
```

## Specifications

The **ratio** specification is the primary specification of the revised manuscript; the **main**
(points) specification is reported as a robustness check. File names keep the code labels:
tables without a suffix belong to the points specification, tables ending in `_ratio` to the
primary scale-free one. In both, the forecast target is the NY-session range in index points.

- **main** (robustness) — target and range predictors in index points (as in the submitted manuscript).
- **ratio** (primary) — scale-free: the target and range predictors are divided by the trailing 22-day
  mean range (`har_monthly`, known before the open) and forecasts are converted back to
  points. `har_monthly` becomes the normaliser, so all models share 11 predictors; the HAR
  benchmark uses `har_daily/har_monthly` and `har_weekly/har_monthly`. The index level grew strongly over 2009–2025 and the mean NY range rose with it (from 17 points in 2009 to 213 points in 2025); tree ensembles
  and the LSTM cannot extrapolate beyond the range of the training target, which this
  specification removes. All metrics are in index points in both specifications.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh            # about 1.5 hours on a 2-core CPU
# or a single step, e.g.
python scripts/01_main_split.py main
python scripts/01_main_split.py ratio
```

Results were produced with Python 3.11 on Linux x86-64 (versions in `requirements.txt`).
Tree ensembles with random subsampling (notably XGBoost) can differ slightly across
operating systems and library versions.

## Raw-bar comparison (reviewer #3)

`scripts/07_raw_bars.py` compares the session summaries (`asia_range`, `london_range`) with the
high–low ranges of the 15 individual 30-minute bars they aggregate (9 Asia, 6 London), for HAR-X,
Elastic Net, Random Forest, CatBoost and LightGBM (main split, 5 walk-forward folds,
Diebold–Mariano tests). It first checks that the bars reproduce every session variable in
`data/ml_dataset1.csv`. It needs the licensed Kibot file, so set `NQ_RAW_PATH`
(`run_all.sh` skips the step otherwise):

```bash
export NQ_RAW_PATH=/path/to/NQ.txt
python scripts/07_raw_bars.py ratio      # or: python scripts/07_raw_bars.py /path/to/NQ.txt main
```

Outputs: `results/tables/table_raw_bars[_ratio].*`, `raw_bars_walk_forward_long[_ratio].csv`,
`raw_bars_evidence[_ratio].csv` (validation and correlation evidence). About 30 seconds per specification.

Two further steps use the same file: `scripts/00_excluded_days.py` writes `data/excluded_days.csv`
(the weekdays of the raw file that are not in the dataset, with the reason) and checks that the
remaining days are exactly the dataset; `scripts/08_gap_retracement.py` computes the gap midpoint
retracement statistic (RQ2) and checks that the gaps rebuilt from the bars match `ndog_range`.

## Data

See `data/README.md`. The raw 30-minute NQ bars come from Kibot and are licensed, so only the
derived daily dataset is included.
