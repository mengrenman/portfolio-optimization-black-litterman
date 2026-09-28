from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml
from test_pipeline_smoke import _write_smoke_fixtures_with_factors_dir

from portfolio_bl.backtest.metrics import infer_periods_per_year, sharpe_ratio
from portfolio_bl.config import load_config
from portfolio_bl.data.factors import load_fama_french
from portfolio_bl.data.prices import load_prices_csv, to_return_matrix
from portfolio_bl.pipeline import run_case_study


def _import_performance_tables_script() -> ModuleType:
    """Import scripts/performance_tables.py without adding it to sys.path."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "performance_tables.py"
    spec = importlib.util.spec_from_file_location("performance_tables", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pt = _import_performance_tables_script()

_PEOPLE = {
    "buffett": ("Warren Buffett", ["AAPL", "MSFT", "XOM"]),
    "pelosi": ("Nancy Pelosi", ["AAPL", "MSFT", "XOM"]),
    "trump": ("Donald Trump", ["AAPL", "MSFT", "XOM"]),
}


def _write_shared_data(data_dir: Path) -> tuple[Path, Path, Path]:
    """Write disclosures (all of CASE_ORDER), prices (incl. SPY) and a factors_dir.

    All three people share one window so the SPY-benchmark row's
    window-equality check passes; ``factors_dir`` charges a non-zero daily
    risk-free rate.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    disclosure_rows = []
    for label, tickers in _PEOPLE.values():
        for ticker, val in zip(tickers, [100.0, 80.0, 20.0]):
            disclosure_rows.append(
                {"person": label, "as_of_date": "2025-03-31", "ticker": ticker, "value_usd": val}
            )
    disclosures = pd.DataFrame(disclosure_rows)

    # Daily prices, so the daily risk-free series matches the return frequency,
    # each a noisy random walk so SPY has a real volatility and its Sharpe
    # ratio is an ordinary number rather than noise divided by noise.
    dates = pd.bdate_range("2024-01-01", "2025-02-28")
    price_rng = np.random.default_rng(3)
    starts = {"AAPL": 100.0, "MSFT": 90.0, "XOM": 70.0, "SPY": 400.0}
    prices = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "close": start * np.cumprod(1.0 + price_rng.normal(0.0005, 0.01, len(dates))),
                }
            )
            for ticker, start in starts.items()
        ],
        ignore_index=True,
    )

    disclosures_path = data_dir / "disclosures.csv"
    prices_path = data_dir / "prices.csv"
    factors_dir = data_dir / "factors"
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
            "RF": np.full(n, 0.008),
        }
    ).to_csv(factors_dir / "ff3_daily.csv", index=False)

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

    return disclosures_path, prices_path, factors_dir


def _write_config(
    config_path: Path,
    disclosures_path: Path,
    prices_path: Path,
    factors_dir: Path | None,
) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data_block = {
        "disclosures_path": str(disclosures_path),
        "prices_path": str(prices_path),
    }
    if factors_dir is not None:
        data_block["factors_dir"] = str(factors_dir)
    config = {
        "data": data_block,
        "backtest": {
            "lookback_periods": 6,
            "rebalance_frequency": "ME",
            "risk_aversion": 2.5,
            "tau": 0.05,
            "view_confidence": 0.65,
        },
        "case_studies": {
            key: {"person_label": label, "disclosure_aliases": [label.lower(), key]}
            for key, (label, _) in _PEOPLE.items()
        },
    }
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)


def _write_fixture_with_factors_dir(tmp_path: Path) -> Path:
    """Config (with ``data.factors_dir``) plus its fixture files; for the
    Sharpe-charging and window-mismatch tests, which call the module's
    functions directly rather than through ``main()``.
    """
    disclosures_path, prices_path, factors_dir = _write_shared_data(tmp_path / "data")
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, disclosures_path, prices_path, factors_dir)
    return config_path


# ---------------------------------------------------------------------------
# (0) _rf_annual_fn reuses the pipeline's own per-period conversion
# ---------------------------------------------------------------------------


def test_rf_annual_fn_matches_pipeline_and_true_rate_on_monthly_prices(tmp_path):
    """``_rf_annual_fn`` must charge exactly the pipeline's own per-period
    risk-free series (``CaseStudyResult.risk_free_series``), not reconvert
    the daily series itself -- and certainly not fall back to reindexing the
    raw daily series directly onto monthly dates, which would silently keep
    only one day's (derived) rate per month, about 1/21 of the truth.

    Uses MONTHLY (BME) prices, exactly the frequency at which the old,
    reindex-based approach was wrong, with a constant synthetic 0.2%/month
    T-bill rate (``ff3_monthly.csv`` in
    :func:`_write_smoke_fixtures_with_factors_dir`) so the true annualized
    rate is known exactly: ``(1.002)**12 - 1``.
    """
    config_path = _write_smoke_fixtures_with_factors_dir(tmp_path)
    app_config = load_config(config_path)
    assert app_config.factors_dir is not None

    result = run_case_study(app_config, person_key="buffett")
    results = {"buffett": result}
    rf_annual_fn = pt._rf_annual_fn(results, zero_rf=False)

    disclosed_dates = result.strategy_results["disclosed"].returns.index
    periods_per_year = infer_periods_per_year(disclosed_dates)
    assert periods_per_year == 12

    script_rf = rf_annual_fn("buffett", disclosed_dates, periods_per_year)

    # Exactly the pipeline's own annualized rate (same series, same dates).
    assert result.risk_free_series is not None
    assert script_rf == pytest.approx(result.risk_free_rate)

    # Close to the true rate.
    true_annual_rate = 1.002**12 - 1.0
    assert script_rf == pytest.approx(true_annual_rate, rel=1e-6)

    # Far from the old, wrong one-day-per-month value: reindexing the raw
    # daily (derived) risk-free series directly onto the monthly dates
    # instead of compounding every day of each month into it.
    daily_risk_free = load_fama_french(app_config.factors_dir, "capm")["rf"]
    naive = daily_risk_free.reindex(disclosed_dates).ffill().bfill()
    naive_annual_rate = float((1.0 + naive).prod() ** (periods_per_year / len(naive)) - 1.0)
    assert script_rf > naive_annual_rate * 15  # true rate is ~21x the naive one


# ---------------------------------------------------------------------------
# (i) SPY benchmark row charges the risk-free rate
# ---------------------------------------------------------------------------


def test_spy_benchmark_sharpe_charges_the_risk_free_rate(tmp_path, capsys):
    config_path = _write_fixture_with_factors_dir(tmp_path)
    app_config = load_config(config_path)
    assert app_config.factors_dir is not None

    results = {key: run_case_study(app_config, key) for key in pt.CASE_ORDER}
    rf_annual_fn = pt._rf_annual_fn(results, zero_rf=False)

    pt.print_strategy_comparison(results, app_config, rf_annual_fn)
    out = capsys.readouterr().out

    disclosed_dates = results["buffett"].strategy_results["disclosed"].returns.index
    prices = load_prices_csv(app_config.prices_path)
    spy_returns = to_return_matrix(prices)["SPY"].reindex(disclosed_dates)
    periods_per_year = infer_periods_per_year(disclosed_dates)
    rf_annual = rf_annual_fn("buffett", disclosed_dates, periods_per_year)
    assert rf_annual != 0.0  # sanity: this fixture actually charges a nonzero rf

    expected_sharpe = sharpe_ratio(spy_returns, periods_per_year, risk_free_rate=rf_annual)
    naive_sharpe = sharpe_ratio(spy_returns, periods_per_year)
    assert expected_sharpe != pytest.approx(naive_sharpe)  # sanity: the rf charge moves the number

    bench_line = next(ln for ln in out.splitlines() if "benchmark" in ln)
    cells = [c.strip() for c in bench_line.strip().strip("|").split("|")]
    sharpe_cell = cells[4].strip("*")
    assert sharpe_cell == f"{expected_sharpe:.2f}"
    assert sharpe_cell != f"{naive_sharpe:.2f}"


# ---------------------------------------------------------------------------
# (ii) --zero-rf matches a config with no data.factors_dir
# ---------------------------------------------------------------------------


def test_zero_rf_flag_matches_a_config_without_factors_dir(tmp_path, monkeypatch, capsys):
    disclosures_path, prices_path, factors_dir = _write_shared_data(tmp_path / "data")

    root_with_factors_dir = tmp_path / "with_rf"
    root_without_factors_dir = tmp_path / "without_rf"
    _write_config(
        root_with_factors_dir / "configs" / "case_studies.yaml",
        disclosures_path,
        prices_path,
        factors_dir,
    )
    _write_config(
        root_without_factors_dir / "configs" / "case_studies.yaml",
        disclosures_path,
        prices_path,
        None,
    )

    monkeypatch.setattr(pt, "ROOT", root_with_factors_dir)
    monkeypatch.setattr(sys, "argv", ["performance_tables.py", "--zero-rf"])
    pt.main()
    zero_rf_out = capsys.readouterr().out

    monkeypatch.setattr(pt, "ROOT", root_without_factors_dir)
    monkeypatch.setattr(sys, "argv", ["performance_tables.py"])
    pt.main()
    no_factors_dir_out = capsys.readouterr().out

    assert zero_rf_out  # sanity: something was actually printed
    assert zero_rf_out == no_factors_dir_out


# ---------------------------------------------------------------------------
# (iii) mismatched return windows raise instead of printing a bogus SPY row
# ---------------------------------------------------------------------------


def test_print_strategy_comparison_raises_on_mismatched_windows(tmp_path):
    config_path = _write_fixture_with_factors_dir(tmp_path)
    app_config = load_config(config_path)
    results = {key: run_case_study(app_config, key) for key in pt.CASE_ORDER}
    rf_annual_fn = pt._rf_annual_fn(results, zero_rf=False)

    disclosed = results["pelosi"].strategy_results["disclosed"]
    shorter_disclosed = dataclasses.replace(disclosed, returns=disclosed.returns.iloc[:-1])
    tampered_pelosi = dataclasses.replace(
        results["pelosi"],
        strategy_results={**results["pelosi"].strategy_results, "disclosed": shorter_disclosed},
    )
    tampered_results = {**results, "pelosi": tampered_pelosi}

    with pytest.raises(ValueError, match="do not share one return window"):
        pt.print_strategy_comparison(tampered_results, app_config, rf_annual_fn)
