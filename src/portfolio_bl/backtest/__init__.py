"""Backtesting engine and performance metrics."""

from portfolio_bl.backtest.attribution import (
    FactorRegression,
    RegressionResult,
    attribution_table,
    factor_regression,
    newey_west_lags,
    ols_newey_west,
)
from portfolio_bl.backtest.engine import BacktestResult, rolling_backtest
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

__all__ = [
    "BacktestResult",
    "FactorRegression",
    "RegressionResult",
    "annualized_return",
    "annualized_volatility",
    "attribution_table",
    "average_turnover",
    "concentration_hhi",
    "factor_regression",
    "infer_periods_per_year",
    "max_drawdown",
    "newey_west_lags",
    "ols_newey_west",
    "rolling_backtest",
    "sharpe_ratio",
    "sortino_ratio",
    "summarize_strategy",
]
