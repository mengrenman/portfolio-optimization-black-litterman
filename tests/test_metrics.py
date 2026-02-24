from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from portfolio_bl.backtest.metrics import (
    annualized_return,
    annualized_volatility,
    average_turnover,
    concentration_hhi,
    infer_periods_per_year,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    summarize_strategy,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MONTHLY_DATES = pd.to_datetime(
    ["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31"]
)

WEIGHT_HISTORY = pd.DataFrame(
    {
        "AAPL": [0.5, 0.6, 0.4, 0.5, 0.5],
        "MSFT": [0.5, 0.4, 0.6, 0.5, 0.5],
    },
    index=MONTHLY_DATES,
)


# ---------------------------------------------------------------------------
# infer_periods_per_year
# ---------------------------------------------------------------------------


def test_infer_periods_per_year_monthly() -> None:
    idx = pd.date_range("2024-01-31", periods=12, freq="ME")
    assert infer_periods_per_year(idx) == 12


def test_infer_periods_per_year_daily() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    assert infer_periods_per_year(idx) == 252


def test_infer_periods_per_year_quarterly() -> None:
    idx = pd.date_range("2020-03-31", periods=8, freq="QE")
    assert infer_periods_per_year(idx) == 4


def test_infer_periods_per_year_warns_with_few_observations() -> None:
    idx = pd.DatetimeIndex(["2025-01-31", "2025-02-28"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = infer_periods_per_year(idx)
    assert result == 12
    assert len(w) == 1
    assert issubclass(w[0].category, UserWarning)
    assert "reliably infer" in str(w[0].message)


def test_infer_periods_per_year_empty_warns() -> None:
    idx = pd.DatetimeIndex([])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = infer_periods_per_year(idx)
    assert result == 12
    assert len(w) == 1


# ---------------------------------------------------------------------------
# annualized_return
# ---------------------------------------------------------------------------


def test_annualized_return_empty() -> None:
    assert np.isnan(annualized_return(pd.Series([], dtype=float), 12))


def test_annualized_return_positive_growth() -> None:
    returns = pd.Series([0.01] * 12)
    ann = annualized_return(returns, 12)
    assert ann > 0


# ---------------------------------------------------------------------------
# sortino_ratio
# ---------------------------------------------------------------------------


def test_sortino_ratio_is_nan_when_all_returns_positive() -> None:
    """Sortino should return NaN (not +Inf) when there are no downside returns."""
    returns = pd.Series([0.01, 0.02, 0.03, 0.01, 0.02])
    result = sortino_ratio(returns, periods_per_year=12)
    assert np.isnan(result), f"Expected NaN, got {result}"


def test_sortino_ratio_empty_series() -> None:
    assert np.isnan(sortino_ratio(pd.Series([], dtype=float), 12))


def test_sortino_ratio_with_downside() -> None:
    returns = pd.Series([0.02, -0.05, 0.01, -0.02, 0.03])
    result = sortino_ratio(returns, periods_per_year=12)
    assert np.isfinite(result)


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------


def test_sharpe_ratio_nan_when_zero_vol() -> None:
    returns = pd.Series([0.01, 0.01, 0.01])
    result = sharpe_ratio(returns, periods_per_year=12)
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------


def test_max_drawdown_is_negative_when_path_drops() -> None:
    returns = pd.Series([0.02, -0.05, 0.01, -0.02, 0.03])
    mdd = max_drawdown(returns)
    assert mdd < 0


def test_max_drawdown_zero_when_always_up() -> None:
    returns = pd.Series([0.01, 0.02, 0.03])
    mdd = max_drawdown(returns)
    assert np.isclose(mdd, 0.0)


def test_max_drawdown_empty() -> None:
    assert np.isnan(max_drawdown(pd.Series([], dtype=float)))


# ---------------------------------------------------------------------------
# concentration_hhi
# ---------------------------------------------------------------------------


def test_hhi_equal_weight_is_one_over_n() -> None:
    weights = pd.DataFrame({"A": [0.5], "B": [0.5]})
    hhi = concentration_hhi(weights)
    assert np.isclose(hhi, 0.5)


def test_hhi_full_concentration_is_one() -> None:
    weights = pd.DataFrame({"A": [1.0], "B": [0.0]})
    hhi = concentration_hhi(weights)
    assert np.isclose(hhi, 1.0)


# ---------------------------------------------------------------------------
# average_turnover
# ---------------------------------------------------------------------------


def test_average_turnover_no_change_is_zero() -> None:
    weights = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]})
    assert np.isclose(average_turnover(weights), 0.0)


def test_average_turnover_single_row_is_zero() -> None:
    weights = pd.DataFrame({"A": [0.6], "B": [0.4]})
    assert average_turnover(weights) == 0.0


# ---------------------------------------------------------------------------
# summarize_strategy
# ---------------------------------------------------------------------------


def test_summarize_strategy_contains_expected_keys() -> None:
    returns = pd.Series([0.01, 0.0, -0.01, 0.02, 0.01])
    summary = summarize_strategy(returns, WEIGHT_HISTORY, periods_per_year=12)

    expected_keys = {
        "annual_return",
        "annual_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "hhi",
        "avg_turnover",
    }
    assert expected_keys.issubset(summary)
    assert np.isfinite(summary["hhi"])


def test_summarize_strategy_all_positive_returns_sortino_is_nan() -> None:
    """Verify the sortino NaN fix propagates correctly through summarize_strategy."""
    returns = pd.Series([0.01, 0.02, 0.03, 0.01])
    summary = summarize_strategy(returns, WEIGHT_HISTORY.iloc[:4], periods_per_year=12)
    assert np.isnan(summary["sortino"])
