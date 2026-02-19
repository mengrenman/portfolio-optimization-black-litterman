from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_PRICE_COLUMNS = {"date", "ticker", "close"}



def load_prices_csv(path: str | Path) -> pd.DataFrame:
    prices = pd.read_csv(path)
    missing = REQUIRED_PRICE_COLUMNS.difference(prices.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required price columns: {missing_str}")

    out = prices.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out["close"] = pd.to_numeric(out["close"], errors="coerce")

    out = out.dropna(subset=["date", "ticker", "close"])
    out = out[out["close"] > 0].copy()

    if out.empty:
        raise ValueError("Price dataset is empty after cleaning.")

    return out.sort_values(["date", "ticker"]).reset_index(drop=True)



def to_return_matrix(prices: pd.DataFrame) -> pd.DataFrame:
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
    series = pd.Series(index=index, data=index)
    grouped = series.groupby(pd.Grouper(freq=frequency)).last().dropna()
    return pd.DatetimeIndex(grouped.values)
