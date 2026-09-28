from __future__ import annotations

import csv
import importlib.util
import logging
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from portfolio_bl.data.factors import FACTOR_MODELS, load_factor_csv, load_fama_french


def _import_fetch_script() -> ModuleType:
    """Import scripts/fetch_fama_french.py without adding it to sys.path."""
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "fetch_fama_french.py"
    spec = importlib.util.spec_from_file_location("fetch_fama_french", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetch_fama_french = _import_fetch_script()


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _ff3_daily_rows(dates: list[str]) -> list[list[str]]:
    return [[d, "0.85", "0.10", "-0.20", "0.01"] for d in dates]


# ---------------------------------------------------------------------------
# load_factor_csv -- percent-to-decimal conversion and shape
# ---------------------------------------------------------------------------


def test_load_factor_csv_converts_percent_to_exact_decimal(tmp_path: Path) -> None:
    path = tmp_path / "ff3_daily.csv"
    _write_csv(
        path,
        ["date", "Mkt-RF", "SMB", "HML", "RF"],
        [["2020-01-02", "0.85", "0.10", "-0.20", "0.01"]],
    )
    df = load_factor_csv(path)
    assert df.loc[pd.Timestamp("2020-01-02"), "mkt_rf"] == 0.0085
    assert df.loc[pd.Timestamp("2020-01-02"), "smb"] == 0.001
    assert df.loc[pd.Timestamp("2020-01-02"), "hml"] == -0.002
    assert df.loc[pd.Timestamp("2020-01-02"), "rf"] == 0.0001
    assert list(df.columns) == ["mkt_rf", "smb", "hml", "rf"]
    assert df.index.name == "date"


def test_load_factor_csv_ff5_columns_lowercased_and_renamed(tmp_path: Path) -> None:
    path = tmp_path / "ff5_daily.csv"
    _write_csv(
        path,
        ["date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
        [["2020-01-02", "0.85", "0.10", "-0.20", "0.30", "0.40", "0.01"]],
    )
    df = load_factor_csv(path)
    assert list(df.columns) == ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    assert df.loc[pd.Timestamp("2020-01-02"), "rmw"] == 0.003
    assert df.loc[pd.Timestamp("2020-01-02"), "cma"] == 0.004


def test_load_factor_csv_monthly_dates_parse_to_month_start(tmp_path: Path) -> None:
    path = tmp_path / "ff3_monthly.csv"
    _write_csv(
        path,
        ["date", "Mkt-RF", "SMB", "HML", "RF"],
        [["2021-01", "1.00", "0.50", "-0.50", "0.10"], ["2021-02", "0.50", "0.25", "0.10", "0.11"]],
    )
    df = load_factor_csv(path)
    assert list(df.index) == [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-02-01")]


def test_load_factor_csv_unsorted_dates_are_sorted_not_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ff3_daily.csv"
    _write_csv(
        path,
        ["date", "Mkt-RF", "SMB", "HML", "RF"],
        [
            ["2020-01-03", "0.10", "0.10", "0.10", "0.01"],
            ["2020-01-02", "0.20", "0.10", "0.10", "0.01"],
        ],
    )
    df = load_factor_csv(path)
    assert list(df.index) == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]


# ---------------------------------------------------------------------------
# load_factor_csv -- error handling
# ---------------------------------------------------------------------------


def test_load_factor_csv_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_factor_csv("/nonexistent/path/ff3_daily.csv")


def test_load_factor_csv_missing_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    _write_csv(path, ["date", "Mkt-RF", "SMB", "HML"], [["2020-01-02", "0.10", "0.10", "0.10"]])
    with pytest.raises(ValueError, match="missing required column"):
        load_factor_csv(path)


def test_load_factor_csv_duplicate_date_raises(tmp_path: Path) -> None:
    path = tmp_path / "dup.csv"
    _write_csv(
        path,
        ["date", "Mkt-RF", "SMB", "HML", "RF"],
        [
            ["2020-01-02", "0.10", "0.10", "0.10", "0.01"],
            ["2020-01-02", "0.20", "0.10", "0.10", "0.01"],
        ],
    )
    with pytest.raises(ValueError, match="duplicate date"):
        load_factor_csv(path)


def test_load_factor_csv_nan_value_raises(tmp_path: Path) -> None:
    path = tmp_path / "nan.csv"
    _write_csv(
        path,
        ["date", "Mkt-RF", "SMB", "HML", "RF"],
        [["2020-01-02", "", "0.10", "0.10", "0.01"]],
    )
    with pytest.raises(ValueError, match="non-numeric or missing value"):
        load_factor_csv(path)


def test_load_factor_csv_unparseable_date_raises(tmp_path: Path) -> None:
    path = tmp_path / "baddate.csv"
    _write_csv(
        path,
        ["date", "Mkt-RF", "SMB", "HML", "RF"],
        [["2020-13-99", "0.10", "0.10", "0.10", "0.01"]],
    )
    with pytest.raises(ValueError, match="unparseable date"):
        load_factor_csv(path)


def test_load_factor_csv_decimal_values_trip_unit_guard(tmp_path: Path) -> None:
    # 25 rows, header says "percent" but values are already ~100x too small
    # (i.e. actually decimals), so dividing by 100 again lands far below the
    # ~1% daily market-vol range the guard expects.
    dates = pd.date_range("2020-01-02", periods=25, freq="D").strftime("%Y-%m-%d").tolist()
    rows = [[d, f"{0.01 * (-1 if i % 2 else 1):.4f}", "0.10", "0.10", "0.01"] for i, d in enumerate(dates)]
    path = tmp_path / "decimals.csv"
    _write_csv(path, ["date", "Mkt-RF", "SMB", "HML", "RF"], rows)
    with pytest.raises(ValueError, match="not in percent"):
        load_factor_csv(path)


# ---------------------------------------------------------------------------
# load_fama_french -- pinned interface and derived rf
# ---------------------------------------------------------------------------


def test_load_fama_french_ff3_pinned_columns_and_order(tmp_path: Path) -> None:
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    _write_csv(tmp_path / "ff3_daily.csv", ["date", "Mkt-RF", "SMB", "HML", "RF"], _ff3_daily_rows(dates))
    df = load_fama_french(tmp_path, "ff3", derive_daily_rf=False)
    assert list(df.columns) == [*FACTOR_MODELS["ff3"], "rf"]
    assert df.index.name == "date"


def test_load_fama_french_ff5_pinned_columns_and_order(tmp_path: Path) -> None:
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    rows = [[d, "0.85", "0.10", "-0.20", "0.30", "0.40", "0.01"] for d in dates]
    _write_csv(tmp_path / "ff5_daily.csv", ["date", "Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"], rows)
    df = load_fama_french(tmp_path, "ff5", derive_daily_rf=False)
    assert list(df.columns) == [*FACTOR_MODELS["ff5"], "rf"]
    assert df.index.name == "date"


def test_load_fama_french_derive_daily_rf_false_returns_published_rf(tmp_path: Path) -> None:
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    rows = [[d, "0.85", "0.10", "-0.20", rf] for d, rf in zip(dates, ["0.01", "0.02", "0.01"], strict=True)]
    _write_csv(tmp_path / "ff3_daily.csv", ["date", "Mkt-RF", "SMB", "HML", "RF"], rows)
    df = load_fama_french(tmp_path, "ff3", derive_daily_rf=False)
    assert df["rf"].tolist() == pytest.approx([0.0001, 0.0002, 0.0001])


def test_load_fama_french_derived_rf_compounds_exactly_to_monthly(tmp_path: Path) -> None:
    # Three months of different lengths: 3, 5, and 4 trading days. The last
    # month ends within 4 days of month end so it counts as complete.
    jan_dates = ["2021-01-04", "2021-01-05", "2021-01-06"]
    feb_dates = ["2021-02-01", "2021-02-02", "2021-02-03", "2021-02-04", "2021-02-05"]
    mar_dates = ["2021-03-01", "2021-03-02", "2021-03-03", "2021-03-31"]
    all_dates = jan_dates + feb_dates + mar_dates
    _write_csv(
        tmp_path / "ff3_daily.csv",
        ["date", "Mkt-RF", "SMB", "HML", "RF"],
        _ff3_daily_rows(all_dates),  # published RF placeholder; overwritten by derivation
    )
    _write_csv(
        tmp_path / "ff3_monthly.csv",
        ["date", "Mkt-RF", "SMB", "HML", "RF"],
        [["2021-01", "1.00", "0.10", "0.10", "0.20"], ["2021-02", "1.00", "0.10", "0.10", "0.15"],
         ["2021-03", "1.00", "0.10", "0.10", "0.25"]],
    )

    df = load_fama_french(tmp_path, "ff3", derive_daily_rf=True)
    assert len(df) == len(all_dates)

    month_key = df.index.to_period("M")
    expected = {"2021-01": 0.0020, "2021-02": 0.0015, "2021-03": 0.0025}
    for period_str, monthly_decimal in expected.items():
        mask = month_key == pd.Period(period_str, freq="M")
        compounded = (1.0 + df.loc[mask, "rf"]).prod() - 1.0
        assert compounded == pytest.approx(monthly_decimal, abs=1e-12)


def test_load_fama_french_incomplete_first_month_raises(tmp_path: Path) -> None:
    # First row on the 5th calendar day of its month: n_M would be wrong.
    dates = ["2021-01-05", "2021-01-06", "2021-01-07"]
    _write_csv(tmp_path / "ff3_daily.csv", ["date", "Mkt-RF", "SMB", "HML", "RF"], _ff3_daily_rows(dates))
    _write_csv(
        tmp_path / "ff3_monthly.csv",
        ["date", "Mkt-RF", "SMB", "HML", "RF"],
        [["2021-01", "1.00", "0.10", "0.10", "0.20"]],
    )
    with pytest.raises(ValueError, match="incomplete"):
        load_fama_french(tmp_path, "ff3", derive_daily_rf=True)


def test_load_fama_french_missing_monthly_month_drops_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    jan_dates = ["2021-01-04", "2021-01-05", "2021-01-29"]
    feb_dates = ["2021-02-01", "2021-02-02"]
    _write_csv(
        tmp_path / "ff3_daily.csv",
        ["date", "Mkt-RF", "SMB", "HML", "RF"],
        _ff3_daily_rows(jan_dates + feb_dates),
    )
    # Only January has a monthly RF; February is missing entirely.
    _write_csv(
        tmp_path / "ff3_monthly.csv",
        ["date", "Mkt-RF", "SMB", "HML", "RF"],
        [["2021-01", "1.00", "0.10", "0.10", "0.20"]],
    )

    with caplog.at_level(logging.WARNING, logger="portfolio_bl.data.factors"):
        df = load_fama_french(tmp_path, "ff3", derive_daily_rf=True)

    assert len(df) == 3
    assert "2" in caplog.text
    assert "2021-02" in caplog.text


def test_load_fama_french_incomplete_last_month_raises(tmp_path: Path) -> None:
    # Last row 26 days before the end of February: n_M would be too small and
    # the derived daily rate for that month too large.
    dates = ["2021-01-04", "2021-01-29", "2021-02-01", "2021-02-02"]
    _write_csv(tmp_path / "ff3_daily.csv", ["date", "Mkt-RF", "SMB", "HML", "RF"], _ff3_daily_rows(dates))
    _write_csv(
        tmp_path / "ff3_monthly.csv",
        ["date", "Mkt-RF", "SMB", "HML", "RF"],
        [["2021-01", "1.00", "0.10", "0.10", "0.20"], ["2021-02", "1.00", "0.10", "0.10", "0.20"]],
    )
    with pytest.raises(ValueError, match="incomplete"):
        load_fama_french(tmp_path, "ff3", derive_daily_rf=True)


def test_load_factor_csv_implausible_daily_rf_raises(tmp_path: Path) -> None:
    # A daily rf of 0.5% (5 bp would already be high) means the file is mis-scaled.
    path = tmp_path / "ff3_daily.csv"
    _write_csv(path, ["date", "Mkt-RF", "SMB", "HML", "RF"], [["2020-01-02", "0.85", "0.10", "-0.20", "0.50"]])
    with pytest.raises(ValueError, match="rf value"):
        load_factor_csv(path)


def test_load_factor_csv_implausible_monthly_rf_raises(tmp_path: Path) -> None:
    path = tmp_path / "ff3_monthly.csv"
    _write_csv(path, ["date", "Mkt-RF", "SMB", "HML", "RF"], [["2020-01", "1.00", "0.10", "0.10", "3.00"]])
    with pytest.raises(ValueError, match="rf value"):
        load_factor_csv(path)


def test_load_fama_french_capm_reads_market_factor_from_ff3_file(tmp_path: Path) -> None:
    dates = [f"2020-01-{d:02d}" for d in range(2, 7)]
    _write_csv(tmp_path / "ff3_daily.csv", ["date", "Mkt-RF", "SMB", "HML", "RF"], _ff3_daily_rows(dates))
    capm = load_fama_french(tmp_path, "capm", derive_daily_rf=False)
    ff3 = load_fama_french(tmp_path, "ff3", derive_daily_rf=False)
    assert list(capm.columns) == ["mkt_rf", "rf"]
    pd.testing.assert_frame_equal(capm, ff3[["mkt_rf", "rf"]])


def test_load_fama_french_unknown_model_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown factor model"):
        load_fama_french(tmp_path, "ff7")


# ---------------------------------------------------------------------------
# fetch_fama_french.parse_ff_block
# ---------------------------------------------------------------------------


def test_parse_ff_block_excludes_annual_block_and_keeps_values_verbatim() -> None:
    raw = (
        "This file was created by using the 202607 CRSP database.\r\n"
        "Some other preamble line explaining the T-bill methodology.\r\n"
        "\r\n"
        ",Mkt-RF,SMB,HML,RF\r\n"
        "20200102,    0.85,   0.10,  -0.20,    0.01\r\n"
        "20200103,    0.10,  -0.05,   0.30,    0.01\r\n"
        "\r\n"
        " Annual Factors: January-December \r\n"
        ",Mkt-RF,SMB,HML,RF\r\n"
        "  2020,  12.34,   1.23,  -4.56,   0.30\r\n"
        "\r\n"
        "Copyright 2026 Eugene F. Fama and Kenneth R. French\r\n"
    )
    columns, rows = fetch_fama_french.parse_ff_block(raw)
    assert columns == ["Mkt-RF", "SMB", "HML", "RF"]
    assert rows == [
        ["20200102", "0.85", "0.10", "-0.20", "0.01"],
        ["20200103", "0.10", "-0.05", "0.30", "0.01"],
    ]
    # The annual block's year rows must never appear.
    assert not any(row[0] == "2020" for row in rows)
