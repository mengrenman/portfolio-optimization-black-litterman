from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
import yaml
from test_pipeline_smoke import (
    _write_smoke_fixtures,
    _write_smoke_fixtures_with_factors_dir,
)

from portfolio_bl.config import load_config
from portfolio_bl.pipeline import run_case_study


def _import_run_case_study_script() -> ModuleType:
    """Import scripts/run_case_study.py without adding it to sys.path."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_case_study.py"
    spec = importlib.util.spec_from_file_location("run_case_study", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_case_study_script = _import_run_case_study_script()

_METADATA_ROWS = ["person_label", "as_of_date", "n_assets", "universe", "risk_free_rate"]


def _add_ff5_daily(factors_dir: Path, seed: int = 11) -> None:
    """Write an ff5_daily.csv derived from the fixture's ff3_daily.csv.

    ``run_case_study.py`` always attempts factor attribution (CAPM, FF3 and
    FF5) once ``data.factors_dir`` is configured, so the FF5 file has to
    exist alongside the FF3 files the smoke fixture already writes.
    """
    ff3 = pd.read_csv(factors_dir / "ff3_daily.csv")
    rng = np.random.default_rng(seed)
    n = len(ff3)
    pd.DataFrame(
        {
            "date": ff3["date"],
            "Mkt-RF": ff3["Mkt-RF"],
            "SMB": ff3["SMB"],
            "HML": ff3["HML"],
            "RMW": rng.normal(0.0, 0.3, n),
            "CMA": rng.normal(0.0, 0.3, n),
            "RF": ff3["RF"],
        }
    ).to_csv(factors_dir / "ff5_daily.csv", index=False)


def _write_daily_fixtures_with_factors_dir(tmp_path: Path) -> Path:
    """Same shape as ``_write_smoke_fixtures_with_factors_dir``, but with
    DAILY business-day prices instead of monthly (BME) ones.

    ``run_case_study.py`` always attempts factor attribution once
    ``data.factors_dir`` is configured, and
    :func:`~portfolio_bl.backtest.attribution.factor_regression` refuses to
    regress returns against a factor frame of a different inferred
    frequency -- the bundled factors are daily, so a test whose whole point
    is that ``factor_attribution.csv`` gets written needs daily prices too.

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

    dates = pd.bdate_range("2024-01-01", periods=380)
    rng = np.random.default_rng(5)
    base_prices = {"AAPL": 100.0, "MSFT": 90.0, "XOM": 70.0}
    rows = []
    for i, date in enumerate(dates):
        for ticker, base in base_prices.items():
            price = base * (1.0 + 0.001 * i + 0.01 * rng.standard_normal())
            rows.append({"date": date, "ticker": ticker, "close": price})
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


def test_main_with_factors_dir_writes_metadata_and_factor_attribution(tmp_path, monkeypatch):
    config_path = _write_daily_fixtures_with_factors_dir(tmp_path)
    _add_ff5_daily(config_path.parent / "factors")

    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_case_study.py",
            "--person", "buffett",
            "--config", str(config_path),
            "--output-dir", str(out_dir),
        ],
    )
    run_case_study_script.main()

    person_dir = out_dir / "buffett"
    written = pd.read_csv(person_dir / "metadata.csv", index_col=0)["value"]
    assert list(written.index) == _METADATA_ROWS

    app_config = load_config(config_path)
    expected = run_case_study(app_config, person_key="buffett").risk_free_rate
    written_rf = float(written["risk_free_rate"])
    assert written_rf == pytest.approx(expected)
    assert written_rf > 0.0

    assert (person_dir / "factor_attribution.csv").exists()


def test_main_without_factors_dir_zero_rf_and_no_factor_attribution(tmp_path, monkeypatch):
    _, _, config_path = _write_smoke_fixtures(tmp_path)

    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_case_study.py",
            "--person", "buffett",
            "--config", str(config_path),
            "--output-dir", str(out_dir),
        ],
    )
    run_case_study_script.main()

    person_dir = out_dir / "buffett"
    written = pd.read_csv(person_dir / "metadata.csv", index_col=0)["value"]
    assert list(written.index) == _METADATA_ROWS
    assert float(written["risk_free_rate"]) == 0.0

    assert not (person_dir / "factor_attribution.csv").exists()


def test_main_with_monthly_prices_writes_everything_but_skips_factor_attribution(
    tmp_path, monkeypatch, caplog
):
    """Monthly prices against the (daily) bundled factors is exactly the
    frequency mismatch :func:`~portfolio_bl.backtest.attribution.factor_regression`
    now refuses. ``run_case_study.py`` must still write every other output --
    the case study itself has nothing wrong with it -- and only skip
    ``factor_attribution.csv``, with a warning explaining why.
    """
    config_path = _write_smoke_fixtures_with_factors_dir(tmp_path)
    _add_ff5_daily(config_path.parent / "factors")

    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_case_study.py",
            "--person", "buffett",
            "--config", str(config_path),
            "--output-dir", str(out_dir),
        ],
    )
    with caplog.at_level(logging.WARNING, logger="run_case_study"):
        run_case_study_script.main()

    person_dir = out_dir / "buffett"
    assert (person_dir / "summary.csv").exists()
    assert (person_dir / "equity_curve.csv").exists()
    assert (person_dir / "strategy_returns.csv").exists()
    assert (person_dir / "weights_disclosed.csv").exists()
    assert (person_dir / "metadata.csv").exists()

    written = pd.read_csv(person_dir / "metadata.csv", index_col=0)["value"]
    assert list(written.index) == _METADATA_ROWS
    assert float(written["risk_free_rate"]) > 0.0

    assert not (person_dir / "factor_attribution.csv").exists()

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1
    assert "Skipping factor attribution" in warning_records[0].getMessage()


def test_main_propagates_non_frequency_attribution_errors(tmp_path, monkeypatch):
    """Only the frequency-mismatch case is caught and turned into a skipped-
    with-warning outcome. Any other error from the attribution table (too
    few aligned observations, a missing risk-free column, or anything else)
    must propagate exactly as it would have before factor attribution was
    added, not be swallowed as a warning.
    """
    config_path = _write_daily_fixtures_with_factors_dir(tmp_path)
    _add_ff5_daily(config_path.parent / "factors")

    def _boom(*args, **kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(run_case_study_script, "attribution_table", _boom)

    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_case_study.py",
            "--person", "buffett",
            "--config", str(config_path),
            "--output-dir", str(out_dir),
        ],
    )
    with pytest.raises(ValueError, match="boom"):
        run_case_study_script.main()
