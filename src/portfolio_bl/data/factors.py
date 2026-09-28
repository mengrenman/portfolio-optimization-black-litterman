"""Loaders for the bundled Fama-French factor data.

See ``data/raw/factors/README.md`` for the data's provenance, schema, and an
explanation of why the daily risk-free rate is re-derived from the monthly
file by default.
"""

from __future__ import annotations

import calendar
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

#: Regressor columns for each supported factor model, in the order they are
#: expressed in the equity-style literature. ``"rf"`` is not a regressor and
#: is appended separately by :func:`load_fama_french`. ``"capm"`` is the
#: market factor alone, taken from the three-factor file.
FACTOR_MODELS: dict[str, tuple[str, ...]] = {
    "capm": ("mkt_rf",),
    "ff3": ("mkt_rf", "smb", "hml"),
    "ff5": ("mkt_rf", "smb", "hml", "rmw", "cma"),
}

# Bundled daily file each model's columns are read from.
_MODEL_SOURCE = {"capm": "ff3", "ff3": "ff3", "ff5": "ff5"}

_REQUIRED_COLUMNS = {"date", "Mkt-RF", "RF"}
_DAILY_DATE_LENGTH = 10  # "YYYY-MM-DD"
_MONTHLY_DATE_LENGTH = 7  # "YYYY-MM"
_MIN_ROWS_FOR_UNIT_GUARD = 20
_DAILY_MKT_RF_STD_RANGE = (0.001, 0.1)  # market daily vol is ~1%, in decimal
# Largest plausible |rf| in decimal: 0.1% a day (about 29% a year) and 2% a
# month, both well above the 1981 T-bill peak. A larger value means the file
# was scaled twice or is not in percent.
_RF_ABS_MAX = {True: 0.001, False: 0.02}
# A month's first trading day is never later than the 4th calendar day, and its
# last is never more than 4 days before month end.
_MONTH_EDGE_DAYS = 4


def load_factor_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate one bundled Fama-French factor CSV.

    Detects whether ``path`` holds daily or monthly data from its date
    format (10-character ``YYYY-MM-DD`` for daily, 7-character ``YYYY-MM``
    for monthly; monthly dates become month-start timestamps), renames
    ``Mkt-RF`` to ``mkt_rf`` and lowercases the remaining columns, and
    divides every value by 100 to convert from the published percent to a
    decimal fraction. This is the single place in the codebase that performs
    that conversion.

    Args:
        path: Path to a bundled factor CSV file (e.g.
            ``data/raw/factors/ff3_daily.csv``).

    Returns:
        A DataFrame indexed by a ``DatetimeIndex`` named ``"date"``, sorted
        ascending, with float columns in decimal units.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be read for any other reason.
        ValueError: If the ``date`` or ``Mkt-RF``/``RF`` columns are missing,
            a date cannot be parsed, any value is non-numeric or missing,
            dates are duplicated, or (for a daily file with at least
            :data:`_MIN_ROWS_FOR_UNIT_GUARD` rows) the standard deviation of
            ``mkt_rf`` after conversion falls outside the range expected for
            daily market returns, suggesting the file is not in percent; or
            any ``rf`` value exceeds a plausible magnitude (0.1% a day, 2% a
            month) after conversion.
    """
    path = Path(path)
    try:
        raw = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Factor file not found: {path}") from None
    except Exception as exc:
        raise RuntimeError(f"Failed to load factor file {path}: {exc}") from exc

    missing = _REQUIRED_COLUMNS.difference(raw.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Factor file {path} is missing required column(s): {missing_str}")

    date_str = raw["date"].astype(str).str.strip()
    date_length = len(date_str.iloc[0]) if len(date_str) else _DAILY_DATE_LENGTH
    is_daily = date_length == _DAILY_DATE_LENGTH
    if is_daily:
        parsed_dates = pd.to_datetime(date_str, format="%Y-%m-%d", errors="coerce")
    elif date_length == _MONTHLY_DATE_LENGTH:
        parsed_dates = pd.to_datetime(date_str + "-01", format="%Y-%m-%d", errors="coerce")
    else:
        raise ValueError(f"Factor file {path} has an unrecognized date format: {date_str.iloc[0]!r}")

    if parsed_dates.isna().any():
        raise ValueError(f"Factor file {path} contains unparseable date(s).")
    if parsed_dates.duplicated().any():
        raise ValueError(f"Factor file {path} contains duplicate date(s).")

    rename_map = {c: ("mkt_rf" if c == "Mkt-RF" else c.lower()) for c in raw.columns if c != "date"}
    out = raw.rename(columns=rename_map)

    value_columns = list(rename_map.values())
    for column in value_columns:
        numeric = pd.to_numeric(out[column], errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"Factor file {path} has non-numeric or missing value(s) in column {column!r}.")
        out[column] = numeric / 100.0

    out = out.drop(columns="date")
    out.index = pd.DatetimeIndex(parsed_dates.to_numpy(), name="date")
    out = out.sort_index()

    if is_daily and len(out) >= _MIN_ROWS_FOR_UNIT_GUARD:
        mkt_rf_std = out["mkt_rf"].std()
        low, high = _DAILY_MKT_RF_STD_RANGE
        if not (low <= mkt_rf_std <= high):
            raise ValueError(
                f"Factor file {path}: mkt_rf standard deviation ({mkt_rf_std:.6f}) is outside "
                f"the expected daily range [{low}, {high}]; the file looks like it is not in percent."
            )

    rf_max = _RF_ABS_MAX[is_daily]
    if (out["rf"].abs() > rf_max).any():
        raise ValueError(
            f"Factor file {path}: an rf value exceeds {rf_max} in absolute value after "
            "conversion; the file looks like it is not in percent."
        )

    logger.debug("Loaded %d factor row(s) from %s.", len(out), path)
    return out


def load_fama_french(
    directory: str | Path,
    model: str = "ff3",
    *,
    derive_daily_rf: bool = True,
) -> pd.DataFrame:
    """Load a daily Fama-French factor frame for one model.

    Args:
        directory: Directory holding the bundled ``ff3_daily.csv`` and
            ``ff5_daily.csv`` files and, when ``derive_daily_rf`` is
            ``True``, ``ff3_monthly.csv``.
        model: A key of :data:`FACTOR_MODELS` (``"capm"``, ``"ff3"`` or
            ``"ff5"``). ``"capm"`` reads the market factor from
            ``ff3_daily.csv``.
        derive_daily_rf: When ``True`` (default), replace the published daily
            risk-free rate with one derived from ``ff3_monthly.csv``: for
            each calendar month present in the daily file, ``rf_daily =
            (1 + RF_monthly) ** (1 / n_days_in_month) - 1``, so that
            compounding it over the month exactly reproduces the monthly
            rate. The published daily rate is rounded to 0.01% a day, which
            matters because T-bill rates move slowly; see
            ``data/raw/factors/README.md`` for the size of the effect. When
            ``False``, the published daily rate is returned unchanged.

    Returns:
        A factor frame: a DataFrame with a ``DatetimeIndex`` named
        ``"date"`` and float columns, in decimal units, exactly
        ``FACTOR_MODELS[model] + ("rf",)`` in that order.

    Raises:
        ValueError: If ``model`` is not a key of :data:`FACTOR_MODELS`; if
            (via :func:`load_factor_csv`) either input file fails
            validation; or if ``derive_daily_rf`` is ``True`` and the daily
            file's first row falls later than the 4th calendar day of its
            month, or its last row more than 4 days before the end of its
            month, either of which would make that month's row count -- and
            so its derived rate -- wrong.
        FileNotFoundError: If either input file does not exist.
    """
    if model not in FACTOR_MODELS:
        valid = ", ".join(sorted(FACTOR_MODELS))
        raise ValueError(f"Unknown factor model {model!r}. Valid models: {valid}.")

    directory = Path(directory)
    columns = (*FACTOR_MODELS[model], "rf")

    daily = load_factor_csv(directory / f"{_MODEL_SOURCE[model]}_daily.csv")[list(columns)]

    if derive_daily_rf:
        monthly = load_factor_csv(directory / "ff3_monthly.csv")
        daily = _with_derived_daily_rf(daily, monthly["rf"])

    daily.index.name = "date"
    return daily[list(columns)]


def _with_derived_daily_rf(daily: pd.DataFrame, monthly_rf: pd.Series) -> pd.DataFrame:
    """Replace ``daily["rf"]`` with a rate derived from ``monthly_rf``.

    Args:
        daily: Daily factor frame, sorted ascending by date.
        monthly_rf: Monthly risk-free rate, indexed by month-start
            timestamps.

    Returns:
        ``daily`` with ``rf`` replaced by the derived rate. Rows whose
        calendar month has no matching entry in ``monthly_rf`` are dropped.

    Raises:
        ValueError: If the first daily row falls later than the 4th calendar
            day of its month, if the last row that has a monthly rate falls
            more than 4 days before the end of its month, or if no daily row
            has a monthly rate.
    """
    first_date = daily.index[0]
    if first_date.day > _MONTH_EDGE_DAYS:
        raise ValueError(
            f"Daily factor frame starts on {first_date.date()}, later than the 4th calendar "
            "day of its month. That first month would have an incomplete row count, so its "
            "derived risk-free rate would be wrong."
        )

    monthly_by_period = monthly_rf.copy()
    monthly_by_period.index = monthly_by_period.index.to_period("M")

    month_key = daily.index.to_period("M")
    missing_months = sorted(set(month_key.unique()) - set(monthly_by_period.index))
    if missing_months:
        keep = ~month_key.isin(missing_months)
        logger.warning(
            "Dropping %d daily row(s) with no matching monthly risk-free rate for month(s): %s.",
            int((~keep).sum()),
            ", ".join(str(m) for m in missing_months),
        )
        daily = daily.loc[keep].copy()
        month_key = month_key[keep]
    if daily.empty:
        raise ValueError("No daily factor row has a matching monthly risk-free rate.")

    last_date = daily.index[-1]
    days_in_month = calendar.monthrange(last_date.year, last_date.month)[1]
    if days_in_month - last_date.day > _MONTH_EDGE_DAYS:
        raise ValueError(
            f"Daily factor frame ends on {last_date.date()}, more than {_MONTH_EDGE_DAYS} days "
            "before the end of its month. That last month would have an incomplete row count, "
            "so its derived risk-free rate would be wrong."
        )

    n_per_month = month_key.value_counts()
    derived_rf = [
        (1.0 + monthly_by_period.loc[period]) ** (1.0 / n_per_month.loc[period]) - 1.0 for period in month_key
    ]
    daily = daily.copy()
    daily["rf"] = pd.Series(derived_rf, index=daily.index)
    return daily
