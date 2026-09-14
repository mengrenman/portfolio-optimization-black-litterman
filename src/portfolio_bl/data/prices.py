from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_PRICE_COLUMNS = {"date", "ticker", "close"}


def load_prices_csv(path: str | Path) -> pd.DataFrame:
    """Load and clean a prices CSV file.

    Parses historical price data. Dates are coerced to datetime; tickers are
    uppercased; close prices are coerced to float. Rows with non-positive or
    missing prices are dropped and a warning is emitted with the count of
    affected rows.

    Args:
        path: Path to the prices CSV file.

    Returns:
        A cleaned DataFrame sorted by ``date`` and ``ticker``.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be read for any other reason.
        ValueError: If required columns are missing or the cleaned dataset is
            empty after filtering.
    """
    path = Path(path)
    try:
        prices = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Prices file not found: {path}") from None
    except Exception as exc:
        raise RuntimeError(f"Failed to load prices from {path}: {exc}") from exc

    missing = REQUIRED_PRICE_COLUMNS.difference(prices.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required price columns: {missing_str}")

    out = prices.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")

    n_before = len(out)
    out = out.dropna(subset=["date", "ticker", "close"])
    out = out[out["close"] > 0].copy()
    n_dropped = n_before - len(out)
    if n_dropped > 0:
        logger.warning(
            "Dropped %d price row(s) due to missing or non-positive close prices.", n_dropped
        )

    if out.empty:
        raise ValueError("Price dataset is empty after cleaning.")

    logger.debug("Loaded %d price rows from %s.", len(out), path)
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def to_return_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert a long-format price DataFrame to a wide-format return matrix.

    Pivots prices from long ``(date, ticker, close)`` format to wide format,
    then computes period-over-period percentage returns. Duplicate
    ``(date, ticker)`` entries are resolved by keeping the last value; a
    warning is emitted if any duplicates are found.

    Args:
        prices: Cleaned price DataFrame as returned by :func:`load_prices_csv`.

    Returns:
        A DataFrame indexed by date with tickers as columns, containing
        percentage returns. All-NaN rows are dropped.

    Raises:
        ValueError: If the resulting return matrix is empty.
    """
    dup_mask = prices.duplicated(subset=["date", "ticker"], keep=False)
    n_dups = dup_mask.sum()
    if n_dups > 0:
        logger.warning(
            "Found %d duplicate (date, ticker) price row(s); keeping the last entry for each pair.",
            n_dups,
        )

    matrix = (
        prices.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
        .sort_index()
        .sort_index(axis=1)
    )
    returns = matrix.pct_change().dropna(how="all")
    if returns.empty:
        raise ValueError("Return matrix is empty; not enough observations in prices.")

    return returns


def monthly_rebalance_dates(index: pd.DatetimeIndex, frequency: str = "ME") -> pd.DatetimeIndex:
    """Extract period-end rebalance dates from a datetime index.

    Groups the index by the given frequency and returns the last date in each
    period.

    Args:
        index: DatetimeIndex to extract rebalance dates from.
        frequency: Pandas offset alias (e.g. ``'ME'`` for month-end, ``'QE'``
            for quarter-end).

    Returns:
        A DatetimeIndex of rebalance dates.
    """
    series = pd.Series(index=index, data=index)
    grouped = series.groupby(pd.Grouper(freq=frequency)).last().dropna()
    return pd.DatetimeIndex(grouped.values)
