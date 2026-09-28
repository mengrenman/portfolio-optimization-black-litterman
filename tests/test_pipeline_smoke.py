from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from portfolio_bl.backtest.metrics import (
    annualized_return,
    annualized_risk_free,
    annualized_volatility,
    infer_periods_per_year,
    risk_free_per_period,
    summarize_strategy,
)
from portfolio_bl.config import load_config
from portfolio_bl.data.factors import load_fama_french
from portfolio_bl.data.prices import load_prices_csv
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


def _write_smoke_fixtures_with_factors_dir(tmp_path: Path) -> Path:
    """Write the same shape of fixtures as :func:`_write_smoke_fixtures`, plus a
    synthetic ``data.factors_dir``.

    Price (and so rebalance/return) dates are business-month-end (``BME``)
    rather than plain calendar month-end, and the bundled ``ff3_daily.csv``
    covers every business day of every month the prices span (not just one
    row per month). Both are required for
    ``load_fama_french(factors_dir, "capm")`` -- which ``run_case_study``
    now calls with its default ``derive_daily_rf=True`` whenever
    ``factors_dir`` is configured -- to have a complete first and last month,
    and for every strategy return date to have a matching daily risk-free
    rate (no filling needed).

    Returns:
        Path to the written config YAML.
    """
    disclosures = pd.DataFrame(
        {
            "person": ["Warren Buffett", "Warren Buffett", "Warren Buffett"],
            "as_of_date": ["2025-03-31", "2025-03-31", "2025-03-31"],
            "ticker": ["AAPL", "MSFT", "XOM"],
            "value_usd": [100.0, 80.0, 20.0],
        }
    )

    dates = pd.date_range("2024-01-01", periods=18, freq="BME")
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
    factors_dir = tmp_path / "factors"
    factors_dir.mkdir()

    disclosures.to_csv(disclosures_path, index=False)
    prices.to_csv(prices_path, index=False)

    first_month_start = dates.min().replace(day=1)
    last_month_end = dates.max() + pd.offsets.MonthEnd(0)
    daily_dates = pd.bdate_range(first_month_start, last_month_end)

    rng = np.random.default_rng(7)
    n = len(daily_dates)
    pd.DataFrame(
        {
            "date": daily_dates.strftime("%Y-%m-%d"),
            "Mkt-RF": rng.normal(0.05, 0.5, n),
            "SMB": rng.normal(0.0, 0.3, n),
            "HML": rng.normal(0.0, 0.3, n),
            "RF": np.full(n, 0.008),  # 0.008% per day, published units
        }
    ).to_csv(factors_dir / "ff3_daily.csv", index=False)

    months = pd.period_range(first_month_start, last_month_end, freq="M")
    pd.DataFrame(
        {
            "date": [str(m) for m in months],
            "Mkt-RF": rng.normal(1.0, 1.0, len(months)),
            "SMB": rng.normal(0.0, 0.5, len(months)),
            "HML": rng.normal(0.0, 0.5, len(months)),
            "RF": np.full(len(months), 0.2),  # 0.2% per month, published units
        }
    ).to_csv(factors_dir / "ff3_monthly.csv", index=False)

    config = {
        "data": {
            "disclosures_path": str(disclosures_path),
            "prices_path": str(prices_path),
            "factors_dir": str(factors_dir),
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

    return config_path


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
# Risk-free-rate wiring (data.factors_dir)
# ---------------------------------------------------------------------------


def test_pipeline_risk_free_rate_zero_without_factors_dir(tmp_path: Path) -> None:
    _, _, config_path = _write_smoke_fixtures(tmp_path)
    app_config = load_config(config_path)
    assert app_config.factors_dir is None

    result = run_case_study(app_config, person_key="buffett")
    assert result.risk_free_rate == 0.0
    assert result.risk_free_series is None

    periods_per_year = infer_periods_per_year(
        next(iter(result.strategy_results.values())).returns.index
    )
    expected_summary = pd.DataFrame(
        {
            name: summarize_strategy(sr.returns, sr.weight_history, periods_per_year=periods_per_year)
            for name, sr in result.strategy_results.items()
        }
    ).T
    expected_summary.index.name = "strategy"
    pd.testing.assert_frame_equal(result.summary, expected_summary)


def test_pipeline_with_factors_dir_charges_risk_free_rate(tmp_path: Path) -> None:
    """This fixture's prices are MONTHLY (BME) -- exactly the case the
    reindex-based risk-free charge got wrong: reindexing the daily T-bill
    series directly onto monthly return dates kept only one of ~21 days'
    rates per month, charging roughly 1/21 of the true monthly rate.
    ``risk_free_rate`` must reflect the true annualized rate of the
    synthetic T-bill -- a constant 0.2% a month (``ff3_monthly.csv`` in
    :func:`_write_smoke_fixtures_with_factors_dir`), i.e. about
    ``(1.002)**12 - 1`` -- not that ~1/21-of-it value.
    """
    config_path = _write_smoke_fixtures_with_factors_dir(tmp_path)
    app_config = load_config(config_path)
    assert app_config.factors_dir is not None

    result = run_case_study(app_config, person_key="buffett")
    assert result.risk_free_rate > 0.0

    prices = load_prices_csv(app_config.prices_path)
    price_dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    daily_risk_free = load_fama_french(app_config.factors_dir, "capm")["rf"]
    # The pipeline converts only the charged window: from the price date that
    # opens the earliest strategy return onward.
    first_return = min(r.returns.index[0] for r in result.strategy_results.values())
    opening = price_dates[price_dates < first_return][-1]
    risk_free_series = risk_free_per_period(daily_risk_free, price_dates[price_dates >= opening])

    periods_per_year = infer_periods_per_year(
        next(iter(result.strategy_results.values())).returns.index
    )
    assert periods_per_year == 12

    disclosed = result.strategy_results["disclosed"]
    expected_risk_free_rate = annualized_risk_free(
        risk_free_series, disclosed.returns.index, periods_per_year
    )
    assert result.risk_free_rate == pytest.approx(expected_risk_free_rate)

    true_annual_rate = 1.002**12 - 1.0
    assert result.risk_free_rate == pytest.approx(true_annual_rate, rel=1e-6)

    # The old, wrong approach: reindexing the raw daily series directly onto
    # the monthly return dates, keeping one arbitrary day's rate per month.
    # annualized_risk_free's frequency guard now refuses this outright.
    with pytest.raises(ValueError, match="finer than the return periods"):
        annualized_risk_free(daily_risk_free, disclosed.returns.index, periods_per_year)

    for name, sr in result.strategy_results.items():
        rf_annual = annualized_risk_free(risk_free_series, sr.returns.index, periods_per_year)
        ann_ret = annualized_return(sr.returns, periods_per_year)
        ann_vol = annualized_volatility(sr.returns, periods_per_year)
        expected_sharpe = (ann_ret - rf_annual) / ann_vol
        assert result.summary.loc[name, "sharpe"] == pytest.approx(expected_sharpe)

    assert result.risk_free_series is not None
    pd.testing.assert_series_equal(result.risk_free_series, risk_free_series)


def test_pipeline_converts_only_the_charged_window(tmp_path: Path) -> None:
    """Factor files that begin after the prices but before the first charged
    return must not trip the extension cap. The lookback months are never
    charged, so the run succeeds and charges exactly what full coverage would.
    """
    config_path = _write_smoke_fixtures_with_factors_dir(tmp_path)
    app_config = load_config(config_path)
    full = run_case_study(app_config, person_key="buffett")

    factors_dir = app_config.factors_dir
    daily = pd.read_csv(factors_dir / "ff3_daily.csv", dtype=str)
    daily[daily["date"] >= "2024-05-01"].to_csv(factors_dir / "ff3_daily.csv", index=False)
    monthly = pd.read_csv(factors_dir / "ff3_monthly.csv", dtype=str)
    monthly[monthly["date"] >= "2024-05"].to_csv(factors_dir / "ff3_monthly.csv", index=False)

    # Converting over the whole price calendar would now exceed the cap ...
    price_dates = pd.DatetimeIndex(sorted(load_prices_csv(app_config.prices_path)["date"].unique()))
    trimmed = load_fama_french(factors_dir, "capm")["rf"]
    with pytest.raises(ValueError, match="max_extension_days"):
        risk_free_per_period(trimmed, price_dates)

    # ... but the pipeline converts only from the opening of the first return.
    late = run_case_study(app_config, person_key="buffett")
    assert late.risk_free_rate == pytest.approx(full.risk_free_rate, rel=1e-12)
    pd.testing.assert_frame_equal(late.summary, full.summary)


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
