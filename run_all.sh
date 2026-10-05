#!/usr/bin/env bash
# Reproduce every table in results/ (about 1.5 hours on a 2-core CPU).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results/logs
# List of excluded days with reasons (data/excluded_days.csv): needs the licensed Kibot 30-minute bars
if [[ -n "${NQ_RAW_PATH:-}" && -f "${NQ_RAW_PATH}" ]]; then
  python scripts/00_excluded_days.py "$NQ_RAW_PATH" 2>&1 | tee results/logs/00_excluded_days.log
else
  echo "Skipping 00_excluded_days: set NQ_RAW_PATH to the licensed Kibot NQ.txt file to run it."
fi
for spec in main ratio; do
  for step in 01_main_split 02_walk_forward 03_feature_importance 04_dm_tests 05_robustness; do
    python "scripts/${step}.py" "$spec" 2>&1 | grep -v -E "absl|oneDNN|^WARNING: All log" | tee "results/logs/${step}_${spec}.log"
  done
  # Raw-bar comparison (reviewer #3): needs the licensed Kibot 30-minute bars (not in this repository)
  if [[ -n "${NQ_RAW_PATH:-}" && -f "${NQ_RAW_PATH}" ]]; then
    python scripts/07_raw_bars.py "$NQ_RAW_PATH" "$spec" 2>&1 | grep -v -E "absl|oneDNN|^WARNING: All log" | tee "results/logs/07_raw_bars_${spec}.log"
  else
    echo "Skipping 07_raw_bars ($spec): set NQ_RAW_PATH to the licensed Kibot NQ.txt file to run it."
  fi
done
# Gap midpoint retracement (RQ2): also needs the raw Kibot bars
if [[ -n "${NQ_RAW_PATH:-}" && -f "${NQ_RAW_PATH}" ]]; then
  python scripts/08_gap_retracement.py "$NQ_RAW_PATH" 2>&1 | tee results/logs/08_gap_retracement.log
else
  echo "Skipping 08_gap_retracement: set NQ_RAW_PATH to the licensed Kibot NQ.txt file to run it."
fi
# Manuscript tables (manuscript/tables/*.tex) from results/tables/*.csv, if the manuscript sources are present
if [[ -f manuscript/tools/make_tables.py ]]; then
  python manuscript/tools/make_tables.py
fi
