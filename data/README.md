# Data

## Source

Intraday 30-minute OHLCV bars for the continuous E-mini NASDAQ-100 futures contract (NQ),
27 September 2009 – 26 November 2025, purchased from Kibot (https://www.kibot.com).
The raw bars are licensed and are not redistributed here; they can be purchased from Kibot.
Timestamps are US Eastern time and refer to the start of each bar.

Scheduled U.S. macroeconomic announcements and their expected-impact ratings
(low / medium / high) are taken from the Forex Factory economic calendar
(https://www.forexfactory.com/calendar), which marks them with a yellow, orange or red
folder icon. Records for 2009–2024 come from a historical export of the calendar; records
from 30 December 2024 onward were collected from the calendar website with the same rule.

## `ml_dataset1.csv`

One row per trading day (4,021 days, 29 Sep 2009 – 25 Nov 2025), built from the raw bars. A day is included if the NY target, both overnight sessions, the opening gap and the news classification are available. Of the 4,167 weekdays between 28 Sep 2009 and 25 Nov 2025 that have a 09:30 ET bar in the raw file, 146 are not in the dataset. [`excluded_days.csv`](excluded_days.csv) lists each of them with its reason; `scripts/00_excluded_days.py` finds the excluded weekdays in the raw Kibot file, assigns the reasons (days without a 16:00 ET bar are detected from the file; the 12 other days are listed in the script) and checks that the remaining days are exactly the 4,021 days of the dataset.

| Reason | Days |
|---|---|
| U.S. exchange holiday or shortened session without a regular 16:00 ET close (Martin Luther King Jr. Day, Presidents' Day, Memorial Day, Juneteenth, 3–5 July, Labor Day, Thanksgiving and the following day, 24 December) | 134 |
| Session after a holiday without overnight trading, so no Asia/London session (2011-12-27, 2012-01-03, 2012-12-26, 2013-01-02, 2013-12-26, 2014-01-02) | 6 |
| No entry in the calendar records collected from 30 December 2024 onward, so no news classification (2024-12-31, 2025-10-13, 2025-11-11) | 3 |
| Overnight trading locked at the CME price limit during the March 2020 sell-off, so the London-session range is zero (2020-03-16, 2020-03-18) | 2 |
| First day of the raw file, so no previous close for the opening gap (2009-09-28) | 1 |
| **Total** | **146** |

Lags and rolling windows are computed over consecutive rows (previous available trading day). The opening gap is the exception: it is measured from the last session in the raw file with a regular 16:00 ET close, which can be a day excluded for another reason (e.g. 2025-10-13).

| Column | Definition |
|---|---|
| `date` | trading day |
| `ny_range` | **target**: high − low of the NY session, 09:30–11:00 ET (index points) |
| `ny_range_lag1` | `ny_range` of the previous trading day |
| `asia_range`, `asia_dir` | high − low, and sign of (last close − first open), of the Asia session (20:00–00:00 ET): the 30-minute bars from 20:00 to 00:00 ET, starting on the previous evening |
| `london_range`, `london_dir` | same for the London session: bars starting 02:00–04:30 ET (02:00–05:00 ET) |
| `ndog_range` | opening gap (called `gap_range` in the paper): absolute difference between the close of the previous trading day's 16:00 bar and the open of the current day's 09:30 bar; the previous trading day is the last session with a regular 16:00 ET close (after a shortened session, e.g. the day after Thanksgiving, the last regular session) |
| `impact_low/medium/high` | one-hot encoding of the highest expected impact (high = red, medium = orange, low = yellow folder) among scheduled USD announcements between 08:30 and 11:00 ET; days without announcements are coded as low impact |
| `year`, `month` (0–11), `month_sin`, `month_cos`, `weekday` (0 = Monday) | calendar variables |

The NY-session target uses bars starting 09:30–10:30 ET (09:30–11:00 ET).

All predictors of day *t* are known at the NY open (09:30 ET): the session variables and lagged ranges before it, the opening gap at the opening print, and the news-impact category from the published calendar.

Kibot bars are labelled by their start time: from 2016 onward the last bar of each day starts at 16:30 and the first bar of the new session at 18:00 ET, matching the CME Globex trading hours (17:00 close, 18:00 reopen).
