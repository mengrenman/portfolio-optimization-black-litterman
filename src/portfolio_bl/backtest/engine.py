from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for the outputs of a single rolling backtest.

    Attributes:
        returns: Portfolio period returns indexed by date.
        nav: Cumulative net-asset-value series starting at ``initial_nav``.
        weight_history: Portfolio weights recorded at each rebalance decision
            date. **Note:** the index holds the *decision* date (when weights
            were computed), not the first date those weights took effect.
            Weights are applied starting from the period *following* each
            rebalance date to avoid look-ahead bias.
    """

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
    """Run a walk-forward rolling backtest.

    At each eligible rebalance date the weight function is called with the
    preceding ``lookback_periods`` rows of returns and the rebalance timestamp.
    The resulting weights are then applied to all periods *after* the rebalance
    date (no look-ahead bias). If the weight function returns all-zero or
    negative weights the backtest falls back to an equal-weight portfolio and
    emits a warning.

    Args:
        returns: Return matrix (dates × tickers). Must not be empty.
        rebalance_dates: Candidate rebalance dates. Dates not present in the
            returns index are silently dropped.
        lookback_periods: Minimum number of historical periods required before
            a rebalance is considered eligible.
        weight_fn: Callable that accepts ``(train_returns, rebalance_date)``
            and returns a Series of portfolio weights indexed by ticker.
        initial_nav: Starting net-asset-value (default 1.0).

    Returns:
        A :class:`BacktestResult` containing the return series, NAV series,
        and weight history.

    Raises:
        ValueError: If ``returns`` is empty, no rebalance dates intersect the
            return index, or no rebalance date has sufficient lookback history.
    """
    if returns.empty:
        raise ValueError("Returns matrix is empty.")

    returns = returns.sort_index().copy()
    all_dates = returns.index

    rebalance_dates = pd.DatetimeIndex(sorted(set(pd.to_datetime(rebalance_dates))))
    rebalance_dates = rebalance_dates.intersection(all_dates)
    if rebalance_dates.empty:
        raise ValueError("No rebalance dates intersect with return index.")

    # Build an ordered list of rebalance dates so that ``lookback_periods``
    # is interpreted as the number of *rebalance intervals* (e.g. months),
    # not the number of rows in the daily return matrix.
    reb_list = list(rebalance_dates)
    eligible_rebalances = [
        d for idx, d in enumerate(reb_list) if idx >= lookback_periods
    ]
    if not eligible_rebalances:
        raise ValueError("No rebalance date has enough lookback observations.")

    logger.debug(
        "Starting backtest: %d eligible rebalance(s), lookback=%d rebalance periods, tickers=%d.",
        len(eligible_rebalances),
        lookback_periods,
        len(returns.columns),
    )

    weights_by_date: dict[pd.Timestamp, pd.Series] = {}
    portfolio_returns: list[tuple[pd.Timestamp, float]] = []
    nav_points: list[tuple[pd.Timestamp, float]] = []
    nav_value = initial_nav

    for i, reb_date in enumerate(eligible_rebalances):
        reb_idx = all_dates.get_loc(reb_date)
        # Look back ``lookback_periods`` rebalance dates to find the start of
        # the training window, then slice all daily rows in between.
        reb_pos = reb_list.index(reb_date)
        lookback_start_date = reb_list[reb_pos - lookback_periods]
        lookback_start_idx = all_dates.get_loc(lookback_start_date)
        train = returns.iloc[lookback_start_idx : reb_idx]

        weights = weight_fn(train, reb_date).reindex(returns.columns).fillna(0.0)
        if weights.sum() <= 0:
            logger.warning(
                "Weight function returned all non-positive weights at %s; "
                "falling back to equal-weight.",
                reb_date.date(),
            )
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

    logger.debug("Backtest complete: %d return observation(s).", len(returns_series))
    return BacktestResult(
        returns=returns_series,
        nav=nav_series,
        weight_history=weight_history,
    )
