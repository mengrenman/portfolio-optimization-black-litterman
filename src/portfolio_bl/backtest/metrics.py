from __future__ import annotations

import numpy as np
import pandas as pd



def infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    if len(index) < 3:
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
    if returns.empty:
        return float("nan")
    growth = float((1.0 + returns).prod())
    n = len(returns)
    return growth ** (periods_per_year / n) - 1.0



def annualized_volatility(returns: pd.Series, periods_per_year: int) -> float:
    if returns.empty:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))



def sharpe_ratio(returns: pd.Series, periods_per_year: int, risk_free_rate: float = 0.0) -> float:
    ann_ret = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    if not np.isfinite(ann_vol) or ann_vol <= 0:
        return float("nan")
    return float((ann_ret - risk_free_rate) / ann_vol)



def sortino_ratio(returns: pd.Series, periods_per_year: int, risk_free_rate: float = 0.0) -> float:
    if returns.empty:
        return float("nan")

    downside = returns[returns < 0]
    if downside.empty:
        return float("inf")

    downside_vol = downside.std(ddof=1) * np.sqrt(periods_per_year)
    if not np.isfinite(downside_vol) or downside_vol <= 0:
        return float("nan")

    ann_ret = annualized_return(returns, periods_per_year)
    return float((ann_ret - risk_free_rate) / downside_vol)



def max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")

    nav = (1.0 + returns).cumprod()
    peak = nav.cummax()
    drawdown = nav / peak - 1.0
    return float(drawdown.min())



def concentration_hhi(weight_history: pd.DataFrame) -> float:
    if weight_history.empty:
        return float("nan")

    normalized = weight_history.div(weight_history.sum(axis=1), axis=0).fillna(0.0)
    hhi = (normalized**2).sum(axis=1).mean()
    return float(hhi)



def average_turnover(weight_history: pd.DataFrame) -> float:
    if weight_history.shape[0] <= 1:
        return 0.0

    delta = weight_history.diff().abs().sum(axis=1) / 2.0
    return float(delta.iloc[1:].mean())



def summarize_strategy(
    returns: pd.Series,
    weight_history: pd.DataFrame,
    periods_per_year: int,
) -> dict[str, float]:
    return {
        "annual_return": annualized_return(returns, periods_per_year),
        "annual_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "sortino": sortino_ratio(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "hhi": concentration_hhi(weight_history),
        "avg_turnover": average_turnover(weight_history),
    }
