from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    returns: pd.Series
    nav: pd.Series
    weight_history: pd.DataFrame



def rolling_backtest(
    returns: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    lookback_periods: int,
    weight_fn: Callable[[pd.DataFrame, pd.Timestamp], pd.Series],
    initial_nav: float = 1.0,
) -> BacktestResult:
    if returns.empty:
        raise ValueError("Returns matrix is empty.")

    returns = returns.sort_index().copy()
    all_dates = returns.index

    rebalance_dates = pd.DatetimeIndex(sorted(set(pd.to_datetime(rebalance_dates))))
    rebalance_dates = rebalance_dates.intersection(all_dates)
    if rebalance_dates.empty:
        raise ValueError("No rebalance dates intersect with return index.")

    eligible_rebalances = [d for d in rebalance_dates if all_dates.get_loc(d) >= lookback_periods]
    if not eligible_rebalances:
        raise ValueError("No rebalance date has enough lookback observations.")

    weights_by_date: dict[pd.Timestamp, pd.Series] = {}
    portfolio_returns: list[tuple[pd.Timestamp, float]] = []
    nav_points: list[tuple[pd.Timestamp, float]] = []
    nav_value = initial_nav

    for i, reb_date in enumerate(eligible_rebalances):
        reb_idx = all_dates.get_loc(reb_date)
        train = returns.iloc[reb_idx - lookback_periods : reb_idx]

        weights = weight_fn(train, reb_date).reindex(returns.columns).fillna(0.0)
        if weights.sum() <= 0:
            weights = pd.Series(1.0 / len(returns.columns), index=returns.columns)
        else:
            weights = weights / weights.sum()

        weights_by_date[reb_date] = weights

        next_reb_date = eligible_rebalances[i + 1] if i + 1 < len(eligible_rebalances) else None
        if next_reb_date is not None:
            # Hold current weights through the next rebalance timestamp.
            end_idx = all_dates.get_loc(next_reb_date) + 1
        else:
            end_idx = len(all_dates)

        # Apply the rebalance weights starting from the next period to avoid look-ahead.
        start_idx = reb_idx + 1
        for t in range(start_idx, end_idx):
            date = all_dates[t]
            step_return = float(np.dot(weights.to_numpy(), returns.iloc[t].fillna(0.0).to_numpy()))
            nav_value *= 1.0 + step_return
            portfolio_returns.append((date, step_return))
            nav_points.append((date, nav_value))

    returns_series = pd.Series(
        [r for _, r in portfolio_returns],
        index=pd.DatetimeIndex([d for d, _ in portfolio_returns]),
        name="portfolio_return",
    )
    nav_series = pd.Series(
        [v for _, v in nav_points],
        index=pd.DatetimeIndex([d for d, _ in nav_points]),
        name="nav",
    )
    weight_history = pd.DataFrame(weights_by_date).T
    weight_history.index.name = "rebalance_date"

    # Preserve consistent column ordering.
    weight_history = weight_history.reindex(columns=returns.columns).fillna(0.0)

    return BacktestResult(
        returns=returns_series,
        nav=nav_series,
        weight_history=weight_history,
    )
