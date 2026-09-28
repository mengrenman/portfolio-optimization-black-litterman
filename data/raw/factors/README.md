Fama-French factor returns, bundled for the performance-attribution regressions in this repo:
`scripts/factor_attribution.py` (cross-case report) and the per-case-study
`factor_attribution.csv` written by `scripts/run_case_study.py`. They are not used anywhere in
the Black-Litterman equilibrium-return or views logic. See `src/portfolio_bl/data/factors.py`
for the loader and `portfolio_bl.data.load_fama_french` for the public entry point.

## Schema

All three files share the same shape: a `date` column followed by one column
per factor, with every value published as a **percent** (`0.85` means
`0.85%`, not `0.0085`). `load_factor_csv` is the only place in the codebase
that divides by 100 to convert to decimal.

`ff3_daily.csv` / `ff5_daily.csv`:
- `date` (`YYYY-MM-DD`)
- `Mkt-RF`, `SMB`, `HML` (`ff3`) or `Mkt-RF`, `SMB`, `HML`, `RMW`, `CMA`
  (`ff5`) -- excess-return factors
- `RF` -- the one-month T-bill return expressed as a simple daily rate

`ff3_monthly.csv`:
- `date` (`YYYY-MM`)
- `Mkt-RF`, `SMB`, `HML`, `RF` -- same factors, monthly

The `capm` model in `factors.py` (the market factor alone) reads `Mkt-RF` from `ff3_daily.csv`;
there is no separate CAPM-only file.

## Source and vintage

Downloaded 2026-09-25 from Kenneth French's data library
(<https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/>), vintage
"built using the 202607 CRSP database":

| File | sha256 |
|---|---|
| `F-F_Research_Data_Factors_daily_CSV.zip` | `1916d331c2c51d2aee3d00215897d2b8e5995cb387f1f4569ba46bff5fb049a8` |
| `F-F_Research_Data_5_Factors_2x3_daily_CSV.zip` | `478350b8d60831351fbf754bc593fc0112b013552c24639c10902e1f26d7f306` |
| `F-F_Research_Data_Factors_CSV.zip` (monthly, used only for its `RF` column) | `b840dba55d319f4818fc7300e65c52eff5f64870c8d495fa58ff5d4cd749f5eb` |

French revises the whole history with every CRSP vintage (corporate-action
and delisting corrections, universe reconstitutions), so refreshing this
bundle from a newer vintage changes every downstream number, not just the
newest rows -- re-run the full case-study pipeline after a refresh rather than
assuming old outputs still hold.

## Current bundled dataset

Built by `scripts/fetch_fama_french.py`, trimmed to dates on or after
`2018-01-01`:

| File | Rows | Coverage |
|---|---|---|
| `ff3_daily.csv` | 2,156 | 2018-01-02 -> 2026-07-31 |
| `ff5_daily.csv` | 2,156 | 2018-01-02 -> 2026-07-31 |
| `ff3_monthly.csv` | 103 | 2018-01 -> 2026-07 |

## The 3-factor and 5-factor `SMB` are not interchangeable

Both files have a column named `SMB` ("small minus big"), but they are built
from different sorts and are not the same series. The 3-factor `SMB` comes
from a single sort on size crossed with book-to-market (2x3 portfolios). The
5-factor `SMB` is the *average* of three separate size sorts -- one crossed
with book-to-market, one with operating profitability (`OP`), one with
investment (`Inv`) -- so it nets out some of the size effect's correlation
with profitability and investment. Do not mix `smb` from `ff3_daily.csv` into
a computation that otherwise uses `ff5_daily.csv`, or vice versa; use one
file's factors together.

## Why the daily risk-free rate is derived from the monthly file

`load_fama_french(..., derive_daily_rf=True)` (the default) does not use the
published daily `RF` column as-is. It replaces it with, for each calendar
month present in the daily file, `(1 + RF_monthly) ** (1 / n_days) - 1`,
which by construction compounds exactly back to that month's published
monthly `RF`.

The reason is that the published daily `RF` is rounded to the nearest 0.01%
(one basis point) before publication. Because short-term T-bill rates move
slowly, that rounding is systematic, not noise that averages out: over the
2018-08-01 to 2025-12-31 backtest window, compounding the published daily
`RF` (252 trading days/year) annualizes to **2.81%**, versus **2.59%** for
the monthly T-bill series it is supposed to reproduce (12 months/year). In
2025 alone, compounding the published daily `RF` gives **5.13%**, against
the monthly file's published **4.25%** for that year -- an overstatement of
**0.88 percentage points** in a single year. All four figures were
recomputed directly from the bundled files (see
`_with_derived_daily_rf`/`load_fama_french` in `factors.py`); if a refreshed
vintage moves them, recompute rather than trusting this file.

The derived series is close to the published one, not a different rate: rounded to the
published two decimal places, the derived rate matches the published daily `RF` on about 98% of
trading days in the bundled window (2,114 of 2,156, verified directly from the bundled files),
and by construction compounds exactly (to floating-point precision) to the monthly rate every
month -- which the published daily series, being rounded, does not quite do.

## Refreshing

```
python scripts/fetch_fama_french.py
```

Add `--start YYYY-MM-DD` to change the trim date, or `--source-dir DIR` to
build from zip files already on disk instead of hitting the network (used to
generate this bundle without a network call; see the script's docstring).
