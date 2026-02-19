from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from portfolio_bl.config import load_config
from portfolio_bl.pipeline import run_case_study



def test_pipeline_smoke(tmp_path: Path) -> None:
    disclosures = pd.DataFrame(
        {
            "person": ["Warren Buffett", "Warren Buffett", "Warren Buffett", "Nancy Pelosi"],
            "as_of_date": ["2025-03-31", "2025-03-31", "2025-03-31", "2025-03-31"],
            "ticker": ["AAPL", "MSFT", "XOM", "AAPL"],
            "value_usd": [100.0, 80.0, 20.0, 50.0],
        }
    )

    dates = pd.date_range("2024-01-31", periods=18, freq="ME")
    rows = []
    for i, date in enumerate(dates):
        rows.extend(
            [
                {"date": date, "ticker": "AAPL", "close": 100 + 2.0 * i},
                {"date": date, "ticker": "MSFT", "close": 90 + 1.5 * i},
                {"date": date, "ticker": "XOM", "close": 70 + 0.8 * i},
            ]
        )
    prices = pd.DataFrame(rows)

    disclosures_path = tmp_path / "disclosures.csv"
    prices_path = tmp_path / "prices.csv"
    config_path = tmp_path / "config.yaml"

    disclosures.to_csv(disclosures_path, index=False)
    prices.to_csv(prices_path, index=False)

    config = {
        "data": {
            "disclosures_path": str(disclosures_path),
            "prices_path": str(prices_path),
        },
        "backtest": {
            "lookback_periods": 6,
            "rebalance_frequency": "ME",
            "risk_aversion": 2.5,
            "tau": 0.05,
        },
        "case_studies": {
            "buffett": {
                "person_label": "Warren Buffett",
                "disclosure_aliases": ["warren buffett", "buffett"],
            }
        },
    }

    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    app_config = load_config(config_path)
    result = run_case_study(app_config, person_key="buffett")

    assert len(result.universe) == 3
    assert "black_litterman" in result.summary.index
    assert not result.summary.empty
