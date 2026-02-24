from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    """Infer the number of return periods per year from a datetime index.

    Estimates the data frequency based on the median number of calendar days
    between consecutive dates:

    - ≤ 2 days  → 252 (daily)
    - ≤ 10 days → 52  (weekly)
    - ≤ 40 days → 12  (monthly)
    - > 40 days → 4   (quarterly)

    Args:
        index: DatetimeIndex of return dates.

    Returns:
        Estimated number of periods per year.

    Warns:
        UserWarning: If fewer than 3 observations are present; frequency cannot
            be reliably inferred and defaults to 12 (monthly). Annualised
            metrics may be inaccurate.
    """
    if len(index) < 3:
        warnings.warn(
            f"Only {len(index)} observation(s) in index; cannot reliably infer frequency. "
            "Defaulting to 12 (monthly). Annualised metrics may be inaccurate.",
            UserWarning,
            stacklevel=2,
        )
        return 12

    deltas = index.to_series().diff().dt.days.dropna()
    if deltas.empty:
        return 12

    median_days = deltas.median()
    if median_days <= 2:
        return 252
    if median_days <= 10:
        return 52
    if median_days <= 40:
        return 12
    return 4


def annualized_return(returns: pd.Series, periods_per_year: int) -> float:
    """Compute annualised geometric return.

    Args:
        returns: Period return series.
        periods_per_year: Number of return periods per calendar year.

    Returns:
        Annualised geometric return, or NaN if the series is empty.
    """
    if returns.empty:
        return float("nan")
    growth = float((1.0 + returns).prod())
    n = len(returns)
    return growth ** (periods_per_year / n) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int) -> float:
    """Compute annualised return volatility.

    Args:
        returns: Period return series.
        periods_per_year: Number of return periods per calendar year.

    Returns:
        Annualised volatility (standard deviation), or NaN if the series is
        empty.
    """
    if returns.empty:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series, periods_per_year: int, risk_free_rate: float = 0.0
) -> float:
    """Compute annualised Sharpe ratio.

    Args:
        returns: Period return series.
        periods_per_year: Number of return periods per calendar year.
        risk_free_rate: Annualised risk-free rate used as the hurdle.

    Returns:
        Sharpe ratio, or NaN if volatility is zero or not finite.
    """
    ann_ret = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    if not np.isfinite(ann_vol) or ann_vol <= 0:
        return float("nan")
    return float((ann_ret - risk_free_rate) / ann_vol)


def sortino_ratio(
    returns: pd.Series, periods_per_year: int, risk_free_rate: float = 0.0
) -> float:
    """Compute annualised Sortino ratio.

    Unlike the Sharpe ratio, Sortino penalises only downside volatility
    (returns below zero). Returns NaN — rather than +Inf — when no negative
    returns exist. This edge case is common in short bull-market windows and
    returning NaN keeps downstream comparisons and CSV outputs well-defined.

    Args:
        returns: Period return series.
        periods_per_year: Number of return periods per calendar year.
        risk_free_rate: Annualised risk-free rate used as the hurdle.

    Returns:
        Sortino ratio, or NaN if the series is empty, has no downside returns,
        or if downside volatility is not finite.
    """
    if returns.empty:
        return float("nan")

    downside = returns[returns < 0]
    if downside.empty:
        return float("nan")

    downside_vol = downside.std(ddof=1) * np.sqrt(periods_per_year)
    if not np.isfinite(downside_vol) or downside_vol <= 0:
        return float("nan")

    ann_ret = annualized_return(returns, periods_per_year)
    return float((ann_ret - risk_free_rate) / downside_vol)


def max_drawdown(returns: pd.Series) -> float:
    """Compute maximum peak-to-trough drawdown.

    Args:
        returns: Period return series.

    Returns:
        Maximum drawdown as a negative fraction (e.g. -0.20 for a 20% drop),
        or NaN if the series is empty.
    """
    if returns.empty:
        return float("nan")

    nav = (1.0 + returns).cumprod()
    peak = nav.cummax()
    drawdown = nav / peak - 1.0
    return float(drawdown.min())


def concentration_hhi(weight_history: pd.DataFrame) -> float:
    """Compute the mean Herfindahl-Hirschman Index (HHI) of portfolio concentration.

    HHI is defined as the sum of squared portfolio weights. A value of 1/n
    represents an equal-weight portfolio; a value of 1.0 represents full
    concentration in a single asset.

    Args:
        weight_history: DataFrame of rebalance-date portfolio weights
            (dates × tickers).

    Returns:
        Mean HHI across all rebalance dates, or NaN if the history is empty.
    """
    if weight_history.empty:
        return float("nan")

    normalized = weight_history.div(weight_history.sum(axis=1), axis=0).fillna(0.0)
    hhi = (normalized**2).sum(axis=1).mean()
    return float(hhi)


def average_turnover(weight_history: pd.DataFrame) -> float:
    """Compute mean one-way portfolio turnover across rebalances.

    Turnover is defined as half the sum of absolute weight changes at each
    rebalance, averaged over all rebalance intervals. A value of 0.10 means
    10% of the portfolio is replaced on average per rebalance.

    Args:
        weight_history: DataFrame of rebalance-date portfolio weights.

    Returns:
        Mean one-way turnover, or 0.0 if fewer than two rebalances exist.
    """
    if weight_history.shape[0] <= 1:
        return 0.0

    delta = weight_history.diff().abs().sum(axis=1) / 2.0
    return float(delta.iloc[1:].mean())


def summarize_strategy(
    returns: pd.Series,
    weight_history: pd.DataFrame,
    periods_per_year: int,
) -> dict[str, float]:
    """Compute a standard set of performance metrics for a strategy.

    Args:
        returns: Portfolio period return series.
        weight_history: Rebalance-date weight history DataFrame.
        periods_per_year: Number of return periods per calendar year.

    Returns:
        A dictionary with keys: ``annual_return``, ``annual_volatility``,
        ``sharpe``, ``sortino``, ``max_drawdown``, ``hhi``, ``avg_turnover``.
    """
    return {
        "annual_return": annualized_return(returns, periods_per_year),
        "annual_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "sortino": sortino_ratio(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "hhi": concentration_hhi(weight_history),
        "avg_turnover": average_turnover(weight_history),
    }
