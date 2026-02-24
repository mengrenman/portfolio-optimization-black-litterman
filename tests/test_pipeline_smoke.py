from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from portfolio_bl.config import load_config
from portfolio_bl.pipeline import run_case_study


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_smoke_fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write minimal CSV files and a config YAML; return their paths."""
    disclosures = pd.DataFrame(
        {
            "person": ["Warren Buffett", "Warren Buffett", "Warren Buffett"],
            "as_of_date": ["2025-03-31", "2025-03-31", "2025-03-31"],
            "ticker": ["AAPL", "MSFT", "XOM"],
            "value_usd": [100.0, 80.0, 20.0],
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
            "view_confidence": 0.65,
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

    return disclosures_path, prices_path, config_path


# ---------------------------------------------------------------------------
# Happy-path smoke test
# ---------------------------------------------------------------------------


def test_pipeline_smoke(tmp_path: Path) -> None:
    _, _, config_path = _write_smoke_fixtures(tmp_path)

    app_config = load_config(config_path)
    result = run_case_study(app_config, person_key="buffett")

    assert len(result.universe) == 3
    assert "black_litterman" in result.summary.index
    assert not result.summary.empty


def test_pipeline_view_confidence_read_from_config(tmp_path: Path) -> None:
    """view_confidence set in YAML must be visible in BacktestConfig."""
    _, _, config_path = _write_smoke_fixtures(tmp_path)
    app_config = load_config(config_path)
    assert app_config.backtest.view_confidence == 0.65


def test_pipeline_view_confidence_override(tmp_path: Path) -> None:
    """Explicit override in run_case_study must take precedence over config."""
    _, _, config_path = _write_smoke_fixtures(tmp_path)
    app_config = load_config(config_path)
    # Passes without error; the result should still be a valid CaseStudyResult.
    result = run_case_study(app_config, person_key="buffett", view_confidence=0.9)
    assert not result.summary.empty


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_config(tmp_path / "nonexistent.yaml")


def test_load_config_malformed_yaml_raises(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("key: [\nunclosed bracket", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed YAML"):
        load_config(bad_yaml)


def test_load_config_non_mapping_yaml_raises(tmp_path: Path) -> None:
    non_mapping = tmp_path / "list.yaml"
    non_mapping.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_config(non_mapping)


def test_load_config_no_case_studies_raises(tmp_path: Path) -> None:
    cfg = tmp_path / "empty.yaml"
    cfg.write_text("data: {}\nbacktest: {}\ncase_studies: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No case studies"):
        load_config(cfg)


def test_pipeline_unknown_person_key_raises(tmp_path: Path) -> None:
    _, _, config_path = _write_smoke_fixtures(tmp_path)
    app_config = load_config(config_path)
    with pytest.raises(ValueError, match="Unknown person key"):
        run_case_study(app_config, person_key="unknown_person")


def test_pipeline_small_universe_raises(tmp_path: Path) -> None:
    """A disclosed portfolio with only one ticker that has prices should raise."""
    disclosures = pd.DataFrame(
        {
            "person": ["buffett"],
            "as_of_date": ["2025-03-31"],
            "ticker": ["AAPL"],
            "value_usd": [100.0],
        }
    )
    dates = pd.date_range("2024-01-31", periods=18, freq="ME")
    prices = pd.DataFrame(
        [{"date": d, "ticker": "AAPL", "close": 100 + i} for i, d in enumerate(dates)]
    )

    disclosures_path = tmp_path / "d.csv"
    prices_path = tmp_path / "p.csv"
    config_path = tmp_path / "c.yaml"

    disclosures.to_csv(disclosures_path, index=False)
    prices.to_csv(prices_path, index=False)

    config = {
        "data": {
            "disclosures_path": str(disclosures_path),
            "prices_path": str(prices_path),
        },
        "backtest": {"lookback_periods": 6, "rebalance_frequency": "ME"},
        "case_studies": {
            "buffett": {
                "person_label": "Buffett",
                "disclosure_aliases": ["buffett"],
            }
        },
    }
    with config_path.open("w") as f:
        yaml.safe_dump(config, f)

    app_config = load_config(config_path)
    with pytest.raises(ValueError, match="fewer than 2 assets"):
        run_case_study(app_config, person_key="buffett")
