"""Tests for the factor-loader/regression wiring: config, pipeline, script glue.

Self-contained: every fixture is synthetic and written to ``tmp_path``. Per
the suite's invariant, these tests never read ``data/raw/`` or
``configs/case_studies.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from portfolio_bl.backtest.attribution import attribution_table
from portfolio_bl.config import load_config
from portfolio_bl.data.factors import load_fama_french
from portfolio_bl.pipeline import run_case_study

# ---------------------------------------------------------------------------
# load_config: data.factors_dir
# ---------------------------------------------------------------------------


def _write_minimal_config(tmp_path: Path, factors_dir: Path | None) -> Path:
    """Write the smallest config YAML that ``load_config`` accepts.

    Args:
        tmp_path: Directory to write the fixtures into.
        factors_dir: When given, written as an absolute ``data.factors_dir``
            value; when ``None``, the key is omitted entirely.

    Returns:
        Path to the written config YAML.
    """
    disclosures_path = tmp_path / "disclosures.csv"
    prices_path = tmp_path / "prices.csv"
    disclosures_path.write_text(
        "person,as_of_date,ticker,value_usd\nBuffett,2025-01-01,AAPL,100\n", encoding="utf-8"
    )
    prices_path.write_text("date,ticker,close\n2024-01-01,AAPL,100\n2024-02-01,AAPL,101\n", encoding="utf-8")

    data_cfg: dict[str, str] = {
        "disclosures_path": str(disclosures_path),
        "prices_path": str(prices_path),
    }
    if factors_dir is not None:
        data_cfg["factors_dir"] = str(factors_dir)

    config = {
        "data": data_cfg,
        "case_studies": {"buffett": {"person_label": "Buffett", "disclosure_aliases": ["buffett"]}},
    }
    config_path = tmp_path / "config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return config_path


def test_load_config_factors_dir_absent_is_none(tmp_path: Path) -> None:
    config_path = _write_minimal_config(tmp_path, factors_dir=None)
    app_config = load_config(config_path)
    assert app_config.factors_dir is None


def test_load_config_factors_dir_present_is_resolved_absolute_path(tmp_path: Path) -> None:
    factors_dir = tmp_path / "raw_factors"
    factors_dir.mkdir()
    config_path = _write_minimal_config(tmp_path, factors_dir=factors_dir)
    app_config = load_config(config_path)
    assert app_config.factors_dir == factors_dir.resolve()
    assert app_config.factors_dir.is_absolute()


# ---------------------------------------------------------------------------
# End-to-end: run_case_study + load_fama_french + attribution_table
# ---------------------------------------------------------------------------


def _write_pipeline_and_factor_fixtures(tmp_path: Path) -> tuple[Path, pd.DatetimeIndex]:
    """Build tiny synthetic prices/disclosures/factor files.

    Follows the same fixture shape as
    ``tests/test_pipeline_smoke.py::_write_smoke_fixtures`` (one person, three
    tickers, monthly price dates), except price dates are business-month-end
    (``BME``) rather than plain calendar month-end. The bundled
    ``ff3_daily.csv``/``ff5_daily.csv`` cover every business day of every
    month the prices span (not just one row per month), and a matching
    ``ff3_monthly.csv`` is included too: ``run_case_study`` now calls
    ``load_fama_french(factors_dir, "capm")`` with its default
    ``derive_daily_rf=True`` whenever ``factors_dir`` is configured, which
    requires a complete first and last month (see
    ``portfolio_bl.data.factors.load_fama_french``). Using business-day dates
    for both prices and factors also means every strategy return date has an
    exact matching daily factor row, so the regression sample stays fully
    overlapping.

    Args:
        tmp_path: Directory to write the fixtures into.

    Returns:
        A tuple of ``(config_path, price_dates)``.
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
    rng = np.random.default_rng(0)
    base_prices = {"AAPL": 100.0, "MSFT": 90.0, "XOM": 70.0}
    rows = []
    for i, date in enumerate(dates):
        for ticker, base in base_prices.items():
            # A gentle random walk (not a pure linear trend) so the return
            # series has nonzero variance in every direction, which the mean-
            # variance and Black-Litterman weight functions need.
            price = base * (1.0 + 0.01 * i + 0.02 * rng.standard_normal())
            rows.append({"date": date, "ticker": ticker, "close": price})
    prices = pd.DataFrame(rows)

    disclosures_path = tmp_path / "disclosures.csv"
    prices_path = tmp_path / "prices.csv"
    factors_dir = tmp_path / "factors"
    factors_dir.mkdir()
    config_path = tmp_path / "config.yaml"

    disclosures.to_csv(disclosures_path, index=False)
    prices.to_csv(prices_path, index=False)

    # Synthetic FF3/FF5 daily factor files, in PERCENT units (load_factor_csv
    # divides by 100), covering every business day of every month the price
    # dates span -- not just the price dates themselves -- so the derived
    # daily risk-free rate (see ff3_monthly.csv below) has a complete first
    # and last month, while every price date (itself a business day) still
    # gets an exact matching factor row.
    first_month_start = dates.min().replace(day=1)
    last_month_end = dates.max() + pd.offsets.MonthEnd(0)
    daily_dates = pd.bdate_range(first_month_start, last_month_end)
    date_strs = daily_dates.strftime("%Y-%m-%d")
    n = len(daily_dates)
    ff_common = {
        "date": date_strs,
        "Mkt-RF": rng.normal(0.05, 0.5, n),
        "SMB": rng.normal(0.0, 0.3, n),
        "HML": rng.normal(0.0, 0.3, n),
        "RF": np.full(n, 0.01),
    }
    pd.DataFrame(ff_common).to_csv(factors_dir / "ff3_daily.csv", index=False)

    ff5 = dict(ff_common)
    ff5["RMW"] = rng.normal(0.0, 0.2, n)
    ff5["CMA"] = rng.normal(0.0, 0.2, n)
    # Preserve the pinned column order (factors before RF).
    ff5_ordered = pd.DataFrame(ff5)[["date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]]
    ff5_ordered.to_csv(factors_dir / "ff5_daily.csv", index=False)

    # Monthly risk-free file used to derive the daily rf (derive_daily_rf=True,
    # the pipeline's default): one row per calendar month spanned by the
    # daily files above.
    months = pd.period_range(first_month_start, last_month_end, freq="M")
    pd.DataFrame(
        {
            "date": [str(m) for m in months],
            "Mkt-RF": rng.normal(1.0, 1.0, len(months)),
            "SMB": rng.normal(0.0, 0.5, len(months)),
            "HML": rng.normal(0.0, 0.5, len(months)),
            "RF": np.full(len(months), 0.2),
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

    return config_path, dates


def test_end_to_end_factor_attribution_wiring(tmp_path: Path) -> None:
    config_path, price_dates = _write_pipeline_and_factor_fixtures(tmp_path)
    app_config = load_config(config_path)
    assert app_config.factors_dir is not None

    result = run_case_study(app_config, person_key="buffett")
    # run_case_study calls load_fama_french(factors_dir, "capm") with its
    # default derive_daily_rf=True to charge Sharpe/Sortino against the
    # derived daily T-bill rate; the fixture's ff3_monthly.csv and complete
    # first/last months (see _write_pipeline_and_factor_fixtures) make that
    # succeed instead of raising, and the derived rate is strictly positive.
    assert result.risk_free_rate > 0.0

    returns_by_name = {name: sr.returns for name, sr in result.strategy_results.items()}
    assert len(returns_by_name) == 3

    # All three strategies share the same backtest dates (see
    # BacktestResult/rolling_backtest docs); the factor files above cover
    # every price date, so the overlap equals the strategy return length.
    strategy_dates = next(iter(returns_by_name.values())).index
    for name, series in returns_by_name.items():
        assert series.index.equals(strategy_dates), name
    expected_n_obs = len(strategy_dates.intersection(price_dates))
    assert expected_n_obs == len(strategy_dates)

    # derive_daily_rf=False here: this part of the test exercises the config
    # -> loader -> attribution wiring with the published daily rf, not the
    # rf-derivation edge cases (already covered in tests/test_factors_data.py).
    factor_sets = {
        model: load_fama_french(app_config.factors_dir, model=model, derive_daily_rf=False)
        for model in ("ff3", "ff5")
    }

    table = attribution_table(returns_by_name, factor_sets)

    assert table.shape[0] == 6
    assert set(table.index.get_level_values("model")) == {"ff3", "ff5"}
    assert set(table.index.get_level_values("strategy")) == set(returns_by_name)
    assert np.isfinite(table["alpha_annual"]).all()
    assert np.isfinite(table["alpha_t"]).all()
    assert (table["n_obs"] == expected_n_obs).all()
