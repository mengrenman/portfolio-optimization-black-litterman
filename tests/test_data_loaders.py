from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pytest

from portfolio_bl.data.disclosures import latest_portfolio_for_aliases, load_disclosures_csv
from portfolio_bl.data.prices import load_prices_csv, to_return_matrix


# ---------------------------------------------------------------------------
# load_disclosures_csv
# ---------------------------------------------------------------------------


def _minimal_disclosures() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person": ["Alice", "Alice", "Bob"],
            "as_of_date": ["2025-01-01", "2025-01-01", "2025-01-01"],
            "ticker": ["AAPL", "MSFT", "GOOG"],
            "value_usd": [100.0, 200.0, 50.0],
        }
    )


def test_load_disclosures_csv_happy_path(tmp_path: Path) -> None:
    path = tmp_path / "d.csv"
    _minimal_disclosures().to_csv(path, index=False)
    df = load_disclosures_csv(path)
    assert set(df.columns) >= {"person_norm", "ticker", "as_of_date", "value_usd"}
    assert (df["ticker"] == df["ticker"].str.upper()).all()
    assert (df["person_norm"] == df["person_norm"].str.lower()).all()


def test_load_disclosures_csv_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_disclosures_csv("/nonexistent/path/disclosures.csv")


def test_load_disclosures_csv_missing_columns_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"person": ["Alice"], "ticker": ["AAPL"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required disclosure columns"):
        load_disclosures_csv(path)


def test_load_disclosures_csv_drops_rows_with_zero_value(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    df = _minimal_disclosures()
    df.loc[0, "value_usd"] = 0.0
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)

    with caplog.at_level(logging.WARNING, logger="portfolio_bl.data.disclosures"):
        result = load_disclosures_csv(path)

    assert len(result) == 2
    assert "Dropped" in caplog.text


def test_load_disclosures_csv_drops_nan_rows(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    df = _minimal_disclosures()
    df.loc[1, "value_usd"] = float("nan")
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)

    with caplog.at_level(logging.WARNING, logger="portfolio_bl.data.disclosures"):
        result = load_disclosures_csv(path)

    assert len(result) == 2


def test_load_disclosures_csv_empty_after_cleaning_raises(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "person": ["Alice"],
            "as_of_date": ["2025-01-01"],
            "ticker": ["AAPL"],
            "value_usd": [0.0],  # will be filtered out
        }
    )
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="empty after cleaning"):
        load_disclosures_csv(path)


# ---------------------------------------------------------------------------
# latest_portfolio_for_aliases
# ---------------------------------------------------------------------------


def test_latest_portfolio_for_aliases_returns_most_recent(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "person": ["alice", "alice"],
            "as_of_date": ["2024-12-31", "2025-03-31"],
            "ticker": ["AAPL", "MSFT"],
            "value_usd": [100.0, 200.0],
        }
    )
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    disclosures = load_disclosures_csv(path)
    portfolio, as_of = latest_portfolio_for_aliases(disclosures, ["alice"])
    assert as_of == pd.Timestamp("2025-03-31")
    assert "MSFT" in portfolio["ticker"].values


def test_latest_portfolio_for_aliases_unknown_raises(tmp_path: Path) -> None:
    path = tmp_path / "d.csv"
    _minimal_disclosures().to_csv(path, index=False)
    disclosures = load_disclosures_csv(path)
    with pytest.raises(ValueError, match="No disclosure rows found"):
        latest_portfolio_for_aliases(disclosures, ["unknown_person_xyz"])


def test_latest_portfolio_weights_sum_to_one(tmp_path: Path) -> None:
    path = tmp_path / "d.csv"
    _minimal_disclosures().to_csv(path, index=False)
    disclosures = load_disclosures_csv(path)
    portfolio, _ = latest_portfolio_for_aliases(disclosures, ["alice"])
    import numpy as np
    assert pytest.approx(portfolio["weight"].sum()) == 1.0


# ---------------------------------------------------------------------------
# load_prices_csv
# ---------------------------------------------------------------------------


def _minimal_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2025-01-31", "2025-02-28", "2025-03-31"],
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "close": [150.0, 155.0, 160.0],
        }
    )


def test_load_prices_csv_happy_path(tmp_path: Path) -> None:
    path = tmp_path / "p.csv"
    _minimal_prices().to_csv(path, index=False)
    df = load_prices_csv(path)
    assert set(df.columns) >= {"date", "ticker", "close"}
    assert (df["close"] > 0).all()


def test_load_prices_csv_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_prices_csv("/nonexistent/path/prices.csv")


def test_load_prices_csv_missing_columns_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2025-01-31"], "ticker": ["AAPL"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required price columns"):
        load_prices_csv(path)


def test_load_prices_csv_drops_non_positive_close(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    df = _minimal_prices()
    df.loc[0, "close"] = -5.0
    path = tmp_path / "p.csv"
    df.to_csv(path, index=False)

    with caplog.at_level(logging.WARNING, logger="portfolio_bl.data.prices"):
        result = load_prices_csv(path)

    assert len(result) == 2
    assert "Dropped" in caplog.text


# ---------------------------------------------------------------------------
# to_return_matrix — duplicate warning
# ---------------------------------------------------------------------------


def test_to_return_matrix_warns_on_duplicates(caplog: pytest.LogCaptureFixture) -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-01-31", "2025-01-31", "2025-02-28", "2025-03-31"]
            ),
            "ticker": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "close": [150.0, 151.0, 155.0, 160.0],
        }
    )

    with caplog.at_level(logging.WARNING, logger="portfolio_bl.data.prices"):
        returns = to_return_matrix(prices)

    assert "duplicate" in caplog.text.lower()
    # After deduplication there should be 2 return rows (3 prices → 2 returns).
    assert len(returns) == 2


def test_to_return_matrix_no_warning_without_duplicates(caplog: pytest.LogCaptureFixture) -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"]),
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "close": [150.0, 155.0, 160.0],
        }
    )

    with caplog.at_level(logging.WARNING, logger="portfolio_bl.data.prices"):
        to_return_matrix(prices)

    assert "duplicate" not in caplog.text.lower()
