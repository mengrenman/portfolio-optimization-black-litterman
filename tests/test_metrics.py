from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_bl.backtest.metrics import max_drawdown, summarize_strategy



def test_max_drawdown_is_negative_when_path_drops() -> None:
    returns = pd.Series([0.02, -0.05, 0.01, -0.02, 0.03])
    mdd = max_drawdown(returns)
    assert mdd < 0



def test_summarize_strategy_contains_expected_keys() -> None:
    returns = pd.Series([0.01, 0.0, -0.01, 0.02, 0.01])
    weights = pd.DataFrame(
        {
            "AAPL": [0.5, 0.6, 0.4],
            "MSFT": [0.5, 0.4, 0.6],
        },
        index=pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"]),
    )

    summary = summarize_strategy(returns, weights, periods_per_year=12)

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
