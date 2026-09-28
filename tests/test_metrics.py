from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from portfolio_bl.backtest.metrics import (
    annualized_return,
    annualized_risk_free,
    annualized_volatility,
    average_turnover,
    concentration_hhi,
    infer_periods_per_year,
    max_drawdown,
    risk_free_per_period,
    sharpe_ratio,
    sortino_ratio,
    summarize_strategy,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MONTHLY_DATES = pd.to_datetime(
    ["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30", "2025-05-31"]
)

WEIGHT_HISTORY = pd.DataFrame(
    {
        "AAPL": [0.5, 0.6, 0.4, 0.5, 0.5],
        "MSFT": [0.5, 0.4, 0.6, 0.5, 0.5],
    },
    index=MONTHLY_DATES,
)


# ---------------------------------------------------------------------------
# infer_periods_per_year
# ---------------------------------------------------------------------------


def test_infer_periods_per_year_monthly() -> None:
    idx = pd.date_range("2024-01-31", periods=12, freq="ME")
    assert infer_periods_per_year(idx) == 12


def test_infer_periods_per_year_daily() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    assert infer_periods_per_year(idx) == 252


def test_infer_periods_per_year_quarterly() -> None:
    idx = pd.date_range("2020-03-31", periods=8, freq="QE")
    assert infer_periods_per_year(idx) == 4


def test_infer_periods_per_year_warns_with_few_observations() -> None:
    idx = pd.DatetimeIndex(["2025-01-31", "2025-02-28"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = infer_periods_per_year(idx)
    assert result == 12
    assert len(w) == 1
    assert issubclass(w[0].category, UserWarning)
    assert "reliably infer" in str(w[0].message)


def test_infer_periods_per_year_empty_warns() -> None:
    idx = pd.DatetimeIndex([])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = infer_periods_per_year(idx)
    assert result == 12
    assert len(w) == 1


# ---------------------------------------------------------------------------
# annualized_return
# ---------------------------------------------------------------------------


def test_annualized_return_empty() -> None:
    assert np.isnan(annualized_return(pd.Series([], dtype=float), 12))


def test_annualized_return_positive_growth() -> None:
    returns = pd.Series([0.01] * 12)
    ann = annualized_return(returns, 12)
    assert ann > 0


# ---------------------------------------------------------------------------
# sortino_ratio
# ---------------------------------------------------------------------------


def test_sortino_ratio_is_nan_when_all_returns_positive() -> None:
    """Sortino should return NaN (not +Inf) when there are no downside returns."""
    returns = pd.Series([0.01, 0.02, 0.03, 0.01, 0.02])
    result = sortino_ratio(returns, periods_per_year=12)
    assert np.isnan(result), f"Expected NaN, got {result}"


def test_sortino_ratio_empty_series() -> None:
    assert np.isnan(sortino_ratio(pd.Series([], dtype=float), 12))


def test_sortino_ratio_with_downside() -> None:
    returns = pd.Series([0.02, -0.05, 0.01, -0.02, 0.03])
    result = sortino_ratio(returns, periods_per_year=12)
    assert np.isfinite(result)


# ---------------------------------------------------------------------------
# annualized_risk_free
# ---------------------------------------------------------------------------


def test_annualized_risk_free_matches_hand_computed_geometric_annualization() -> None:
    idx = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    rf = pd.Series([0.001, 0.0015, 0.0012], index=idx)
    result = annualized_risk_free(rf, idx, periods_per_year=12)
    expected = (1.001 * 1.0015 * 1.0012) ** (12 / 3) - 1.0
    assert result == pytest.approx(expected)


def test_annualized_risk_free_ignores_dates_outside_the_index() -> None:
    """A risk-free series covering more dates than ``index`` must use only ``index``'s dates."""
    idx = pd.to_datetime(["2024-01-31", "2024-02-29"])
    rf_full = pd.Series(
        [0.05, 0.001, 0.0015, 0.05],
        index=pd.to_datetime(["2023-12-31", "2024-01-31", "2024-02-29", "2024-03-31"]),
    )
    result = annualized_risk_free(rf_full, idx, periods_per_year=12)
    expected = (1.001 * 1.0015) ** (12 / 2) - 1.0
    assert result == pytest.approx(expected)


def test_annualized_risk_free_partial_coverage_warns_once_and_fills(
    caplog: pytest.LogCaptureFixture,
) -> None:
    idx = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31", "2024-04-30"])
    # Only the first two dates are covered; the last two must be filled via
    # ffill (both take the 2024-02-29 rate, since bfill has nothing after it).
    rf = pd.Series([0.001, 0.0012], index=pd.to_datetime(["2024-01-31", "2024-02-29"]))

    with caplog.at_level(logging.WARNING, logger="portfolio_bl.backtest.metrics"):
        result = annualized_risk_free(rf, idx, periods_per_year=12)

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1
    message = warning_records[0].getMessage()
    assert "2 of 4" in message
    assert "2024-03-31" in message
    assert "2024-04-30" in message

    expected = (1.001 * 1.0012 * 1.0012 * 1.0012) ** (12 / 4) - 1.0
    assert result == pytest.approx(expected)


def test_annualized_risk_free_zero_coverage_raises() -> None:
    idx = pd.to_datetime(["2024-01-31", "2024-02-29"])
    rf = pd.Series([0.001], index=pd.to_datetime(["2023-01-31"]))
    with pytest.raises(ValueError, match="None of the 2"):
        annualized_risk_free(rf, idx, periods_per_year=12)


def test_annualized_risk_free_empty_index_returns_zero() -> None:
    idx = pd.DatetimeIndex([])
    rf = pd.Series([0.001], index=pd.to_datetime(["2024-01-31"]))
    assert annualized_risk_free(rf, idx, periods_per_year=12) == 0.0


def test_annualized_risk_free_rejects_daily_series_against_monthly_index() -> None:
    """A daily risk-free series reindexed directly onto monthly return dates
    would silently keep only one day's rate per month; must be refused.
    """
    daily_idx = pd.bdate_range("2024-01-01", periods=252)
    rf = pd.Series(0.0001, index=daily_idx)
    monthly_index = pd.date_range("2024-01-31", periods=11, freq="ME")

    with pytest.raises(ValueError, match="finer than the return periods"):
        annualized_risk_free(rf, monthly_index, periods_per_year=12)


def test_annualized_risk_free_accepts_correctly_converted_per_period_series() -> None:
    """The frequency guard must not fire on a series already converted by
    :func:`risk_free_per_period` -- exactly one rate per return period.
    """
    daily_idx = pd.bdate_range("2024-01-01", "2024-12-31")
    rf = pd.Series(0.0001, index=daily_idx)
    calendar = pd.DatetimeIndex([daily_idx[0]]).append(
        pd.bdate_range("2024-01-01", "2024-12-31", freq="BME")
    )

    per_period = risk_free_per_period(rf, calendar)
    assert len(per_period) == 12

    result = annualized_risk_free(per_period, per_period.index, periods_per_year=12)
    assert np.isfinite(result)


# ---------------------------------------------------------------------------
# risk_free_per_period
# ---------------------------------------------------------------------------


def test_risk_free_per_period_daily_calendar_reproduces_daily_rates() -> None:
    """A calendar equal to the rate dates themselves is a no-op conversion."""
    idx = pd.bdate_range("2024-01-01", periods=10)
    rf = pd.Series(np.linspace(0.0001, 0.0002, 10), index=idx)

    result = risk_free_per_period(rf, idx)

    assert result.index.equals(idx[1:])
    np.testing.assert_allclose(result.to_numpy(), rf.iloc[1:].to_numpy(), atol=1e-15)


def test_risk_free_per_period_month_end_calendar_compounds_every_daily_rate() -> None:
    """Every daily rate in a month must be compounded, not just one sampled day.

    Includes rate dates that fall between two calendar dates but are not
    themselves calendar dates (ordinary business days within a month whose
    boundaries are the month-end dates), which is exactly the case the
    reindex-based approach silently mishandles.
    """
    daily_idx = pd.bdate_range("2024-01-01", "2024-03-29")
    rng = np.random.default_rng(1)
    rf = pd.Series(rng.normal(0.0002, 0.00005, len(daily_idx)), index=daily_idx)
    # calendar[0] is the opening boundary, exactly the first daily date, so no
    # extension is needed and this test isolates the within-coverage grouping.
    calendar = pd.to_datetime(["2024-01-01", "2024-01-31", "2024-02-29", "2024-03-29"])

    result = risk_free_per_period(rf, calendar)

    expected = []
    prev = calendar[0]
    for end in calendar[1:]:
        mask = (rf.index > prev) & (rf.index <= end)
        expected.append(float(np.prod(1.0 + rf[mask].to_numpy()) - 1.0))
        prev = end

    assert result.index.equals(calendar[1:])
    np.testing.assert_allclose(result.to_numpy(), expected, atol=1e-12)


def test_risk_free_per_period_is_frequency_invariant_when_annualized() -> None:
    """The whole point of this function: annualizing must not depend on how
    the same underlying daily rates happen to be grouped into periods.

    252 daily rates are grouped two ways over the exact same span: as 252
    daily periods (periods_per_year=252) and as 12 periods of exactly 21
    daily rates each (periods_per_year=12) -- so both annualizations reduce
    to the same total compounded growth over the same one "year". They must
    agree to floating-point precision.

    The same computation with the *old* approach -- reindexing the raw daily
    series directly onto the month-end dates, so only one arbitrary day's
    rate represents each ~21-day period -- is also attempted here, and must
    now be refused outright by :func:`annualized_risk_free`'s frequency
    guard, which is the bug this function (and that guard) fix.
    """
    bdays = pd.bdate_range("2024-01-01", periods=253)  # 252 periods + 1 opening boundary
    rng = np.random.default_rng(2)
    rf = pd.Series(rng.normal(0.0001, 0.00002, len(bdays)), index=bdays)

    daily_calendar = bdays
    month_end_calendar = bdays[::21]  # 13 points -> 12 periods of 21 days each
    assert len(month_end_calendar) == 13

    per_daily = risk_free_per_period(rf, daily_calendar)
    per_month_end = risk_free_per_period(rf, month_end_calendar)
    assert len(per_daily) == 252
    assert len(per_month_end) == 12

    ann_daily = annualized_risk_free(per_daily, per_daily.index, periods_per_year=252)
    ann_month_end = annualized_risk_free(per_month_end, per_month_end.index, periods_per_year=12)
    assert ann_daily == pytest.approx(ann_month_end, abs=1e-12)

    # The old, wrong approach: annualized_risk_free called directly on the raw
    # daily series over the month-end dates, which would reindex and so keep
    # only one day's rate per ~21-day period. Now refused outright.
    with pytest.raises(ValueError, match="finer than the return periods"):
        annualized_risk_free(rf, month_end_calendar[1:], periods_per_year=12)


def test_risk_free_per_period_extends_after_coverage_with_one_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    daily_idx = pd.bdate_range("2024-01-01", periods=5)  # Mon 1/1 .. Fri 1/5
    rf = pd.Series(0.0001, index=daily_idx)
    # 1/12 is 5 business days after the last covered date, 1/5.
    calendar = pd.to_datetime(["2024-01-01", "2024-01-12"])

    with caplog.at_level(logging.WARNING, logger="portfolio_bl.backtest.metrics"):
        result = risk_free_per_period(rf, calendar)

    n_bd_after = int(
        np.busday_count(
            np.datetime64("2024-01-05", "D") + np.timedelta64(1, "D"),
            np.datetime64("2024-01-12", "D") + np.timedelta64(1, "D"),
        )
    )
    # 4 real daily rates (1/2..1/5) plus the extension at the last rate.
    expected = (1.0001) ** (4 + n_bd_after) - 1.0
    assert result.iloc[0] == pytest.approx(expected)

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1
    message = warning_records[0].getMessage()
    assert "1 of 1" in message
    assert "2024-01-12" in message


def test_risk_free_per_period_extends_before_coverage_symmetrically(
    caplog: pytest.LogCaptureFixture,
) -> None:
    daily_idx = pd.bdate_range("2024-01-01", periods=5)  # Mon 1/1 .. Fri 1/5
    rf = pd.Series(0.0001, index=daily_idx)
    # 2023-12-22 is 6 business days before the first covered date, 1/1.
    calendar = pd.to_datetime(["2023-12-22", "2024-01-05"])

    with caplog.at_level(logging.WARNING, logger="portfolio_bl.backtest.metrics"):
        result = risk_free_per_period(rf, calendar)

    n_bd_before = int(
        np.busday_count(
            np.datetime64("2023-12-22", "D") + np.timedelta64(1, "D"),
            np.datetime64("2024-01-01", "D") + np.timedelta64(1, "D"),
        )
    )
    # n_bd_before counts (12/22, 1/1], i.e. the 5 extrapolated business days
    # plus 1/1's own real rate, and 1/2..1/5 are the remaining 4 real rates.
    expected = (1.0001) ** (n_bd_before + 4) - 1.0
    assert result.iloc[0] == pytest.approx(expected)

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1
    assert "1 of 1" in warning_records[0].getMessage()


def test_risk_free_per_period_no_overlap_raises() -> None:
    rf = pd.Series([0.0001], index=pd.to_datetime(["2020-01-01"]))
    calendar = pd.to_datetime(["2024-01-01", "2024-02-01"])
    with pytest.raises(ValueError, match="No period spanned by the calendar"):
        risk_free_per_period(rf, calendar)


def test_risk_free_per_period_calendar_starting_on_last_covered_day_raises() -> None:
    """Off-by-one regression test: periods are the half-open interval
    ``(cal[i-1], cal[i]]``, so a calendar whose first date equals the last
    day covered by ``daily_risk_free`` (``cal[0] == d_max``) has no period
    overlapping any real daily rate at all -- the only period it defines,
    ``(d_max, cal[1]]``, is entirely past coverage. The old
    ``cal[-1] < d_min or cal[0] > d_max`` check missed this (``cal[0] ==
    d_max`` is not ``> d_max``) and did not raise.
    """
    daily_idx = pd.bdate_range("2024-01-01", periods=5)  # Mon 1/1 .. Fri 1/5
    rf = pd.Series(0.0001, index=daily_idx)
    calendar = pd.to_datetime(["2024-01-05", "2024-01-12"])  # cal[0] == d_max

    with pytest.raises(ValueError, match="No period spanned by the calendar"):
        risk_free_per_period(rf, calendar)


def test_risk_free_per_period_cap_exceeded_before_coverage_raises() -> None:
    daily_idx = pd.bdate_range("2024-06-01", periods=5)
    rf = pd.Series(0.0001, index=daily_idx)
    # 99 business days before d_min -- well past the default 63-day cap.
    far_before = pd.bdate_range(end=daily_idx[0], periods=100)[0]
    calendar = pd.DatetimeIndex([far_before, daily_idx[-1]])

    with pytest.raises(ValueError, match=r"business day.*max_extension_days"):
        risk_free_per_period(rf, calendar)


def test_risk_free_per_period_cap_exceeded_after_coverage_raises() -> None:
    daily_idx = pd.bdate_range("2024-06-01", periods=5)
    rf = pd.Series(0.0001, index=daily_idx)
    # 99 business days after d_max -- well past the default 63-day cap.
    far_after = pd.bdate_range(start=daily_idx[-1], periods=100)[-1]
    calendar = pd.DatetimeIndex([daily_idx[0], far_after])

    with pytest.raises(ValueError, match=r"business day.*max_extension_days"):
        risk_free_per_period(rf, calendar)


def test_risk_free_per_period_max_extension_days_respected() -> None:
    """A calendar reaching further than the default cap raises, but succeeds
    once ``max_extension_days`` is raised enough to accommodate it.
    """
    daily_idx = pd.bdate_range("2024-06-01", periods=5)
    rf = pd.Series(0.0001, index=daily_idx)
    far_after = pd.bdate_range(start=daily_idx[-1], periods=100)[-1]
    calendar = pd.DatetimeIndex([daily_idx[0], far_after])

    with pytest.raises(ValueError, match="max_extension_days"):
        risk_free_per_period(rf, calendar)

    result = risk_free_per_period(rf, calendar, max_extension_days=200)
    assert len(result) == 1
    assert np.isfinite(result.iloc[0])


def test_risk_free_per_period_saturday_boundary_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A calendar boundary that falls outside coverage but on a non-business
    day (the Saturday right after the last covered Friday) contributes zero
    extrapolated business days and must not trigger a warning.
    """
    daily_idx = pd.bdate_range("2024-01-01", periods=5)  # Mon 1/1 .. Fri 1/5
    rf = pd.Series(0.0001, index=daily_idx)
    calendar = pd.to_datetime(["2024-01-01", "2024-01-06"])  # 2024-01-06 is a Saturday

    with caplog.at_level(logging.WARNING, logger="portfolio_bl.backtest.metrics"):
        result = risk_free_per_period(rf, calendar)

    expected = float(np.prod(1.0 + rf.iloc[1:].to_numpy()) - 1.0)  # rates for 1/2..1/5
    assert result.iloc[0] == pytest.approx(expected)

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 0


def test_risk_free_per_period_saturday_boundary_before_coverage_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Before-coverage twin of the Saturday test: a boundary on the Saturday
    before the first covered Monday extrapolates zero business days."""
    daily_idx = pd.bdate_range("2024-01-08", "2024-01-19")  # Mon 1/8 .. Fri 1/19
    rf = pd.Series(0.0001, index=daily_idx)
    calendar = pd.to_datetime(["2024-01-06", "2024-01-10"])  # Sat 1/6, Wed 1/10

    with caplog.at_level(logging.WARNING, logger="portfolio_bl.backtest.metrics"):
        result = risk_free_per_period(rf, calendar)

    assert result.iloc[0] == pytest.approx(1.0001**3 - 1.0)  # 1/8, 1/9, 1/10
    assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


def test_risk_free_per_period_cap_is_symmetric(caplog: pytest.LogCaptureFixture) -> None:
    """Exactly max_extension_days extrapolated business days pass on either
    side and are reported as such; one more raises on either side."""
    daily_idx = pd.bdate_range("2024-06-03", periods=20)
    rf = pd.Series(0.0001, index=daily_idx)
    d_min = np.datetime64(daily_idx[0].date(), "D")
    d_max = np.datetime64(daily_idx[-1].date(), "D")

    def before(n: int) -> pd.Timestamp:
        # n business days strictly between the returned date and d_min.
        return pd.Timestamp(np.busday_offset(d_min, -(n + 1)))

    def after(n: int) -> pd.Timestamp:
        return pd.Timestamp(np.busday_offset(d_max, n))

    for side_calendar in (
        pd.DatetimeIndex([before(63), daily_idx[5]]),
        pd.DatetimeIndex([daily_idx[5], after(63)]),
    ):
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="portfolio_bl.backtest.metrics"):
            risk_free_per_period(rf, side_calendar)
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert len(messages) == 1
        assert "63 business day" in messages[0]

    for side_calendar in (
        pd.DatetimeIndex([before(64), daily_idx[5]]),
        pd.DatetimeIndex([daily_idx[5], after(64)]),
    ):
        with pytest.raises(ValueError, match="64 business day"):
            risk_free_per_period(rf, side_calendar)


def test_risk_free_per_period_warning_reports_period_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    daily_idx = pd.bdate_range("2024-01-01", periods=5)  # Mon 1/1 .. Fri 1/5
    rf = pd.Series(0.0001, index=daily_idx)
    calendar = pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-08"])  # Mon 1/8 past coverage

    with caplog.at_level(logging.WARNING, logger="portfolio_bl.backtest.metrics"):
        risk_free_per_period(rf, calendar)

    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(messages) == 1
    assert "1 of 2 period(s)" in messages[0]
    assert "1 business day" in messages[0]


def test_risk_free_per_period_warning_reports_business_day_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    daily_idx = pd.bdate_range("2024-01-01", periods=5)  # Mon 1/1 .. Fri 1/5
    rf = pd.Series(0.0001, index=daily_idx)
    calendar = pd.to_datetime(["2024-01-01", "2024-01-12"])  # 5 business days after d_max

    with caplog.at_level(logging.WARNING, logger="portfolio_bl.backtest.metrics"):
        risk_free_per_period(rf, calendar)

    n_bd_after = int(
        np.busday_count(
            np.datetime64("2024-01-05", "D") + np.timedelta64(1, "D"),
            np.datetime64("2024-01-12", "D") + np.timedelta64(1, "D"),
        )
    )
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1
    assert f"{n_bd_after} business day" in warning_records[0].getMessage()


def test_risk_free_per_period_empty_daily_series_raises() -> None:
    rf = pd.Series([], dtype=float)
    calendar = pd.to_datetime(["2024-01-01", "2024-02-01"])
    with pytest.raises(ValueError, match="no non-null observations"):
        risk_free_per_period(rf, calendar)


def test_risk_free_per_period_short_calendar_returns_empty() -> None:
    rf = pd.Series([0.0001], index=pd.to_datetime(["2024-01-01"]))

    empty_calendar = pd.DatetimeIndex([])
    result_empty = risk_free_per_period(rf, empty_calendar)
    assert result_empty.empty

    single_date_calendar = pd.to_datetime(["2024-01-01"])
    result_single = risk_free_per_period(rf, single_date_calendar)
    assert result_single.empty


def test_risk_free_per_period_drops_nan_and_handles_unsorted_calendar() -> None:
    rf = pd.Series(
        [0.0001, np.nan, 0.0002, 0.0003],
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
    )
    # Deliberately unsorted; also exercises the opening-boundary date being
    # last in input order.
    unsorted_calendar = pd.to_datetime(["2024-01-04", "2024-01-01"])

    result = risk_free_per_period(rf, unsorted_calendar)

    assert list(result.index) == [pd.Timestamp("2024-01-04")]
    expected = (1.0002) * (1.0003) - 1.0
    assert result.iloc[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------


def test_sharpe_ratio_nan_when_zero_vol() -> None:
    returns = pd.Series([0.01, 0.01, 0.01])
    result = sharpe_ratio(returns, periods_per_year=12)
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------


def test_max_drawdown_is_negative_when_path_drops() -> None:
    returns = pd.Series([0.02, -0.05, 0.01, -0.02, 0.03])
    mdd = max_drawdown(returns)
    assert mdd < 0


def test_max_drawdown_zero_when_always_up() -> None:
    returns = pd.Series([0.01, 0.02, 0.03])
    mdd = max_drawdown(returns)
    assert np.isclose(mdd, 0.0)


def test_max_drawdown_empty() -> None:
    assert np.isnan(max_drawdown(pd.Series([], dtype=float)))


# ---------------------------------------------------------------------------
# concentration_hhi
# ---------------------------------------------------------------------------


def test_hhi_equal_weight_is_one_over_n() -> None:
    weights = pd.DataFrame({"A": [0.5], "B": [0.5]})
    hhi = concentration_hhi(weights)
    assert np.isclose(hhi, 0.5)


def test_hhi_full_concentration_is_one() -> None:
    weights = pd.DataFrame({"A": [1.0], "B": [0.0]})
    hhi = concentration_hhi(weights)
    assert np.isclose(hhi, 1.0)


# ---------------------------------------------------------------------------
# average_turnover
# ---------------------------------------------------------------------------


def test_average_turnover_no_change_is_zero() -> None:
    weights = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]})
    assert np.isclose(average_turnover(weights), 0.0)


def test_average_turnover_single_row_is_zero() -> None:
    weights = pd.DataFrame({"A": [0.6], "B": [0.4]})
    assert average_turnover(weights) == 0.0


# ---------------------------------------------------------------------------
# summarize_strategy
# ---------------------------------------------------------------------------


def test_summarize_strategy_contains_expected_keys() -> None:
    returns = pd.Series([0.01, 0.0, -0.01, 0.02, 0.01])
    summary = summarize_strategy(returns, WEIGHT_HISTORY, periods_per_year=12)

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


def test_summarize_strategy_all_positive_returns_sortino_is_nan() -> None:
    """Verify the sortino NaN fix propagates correctly through summarize_strategy."""
    returns = pd.Series([0.01, 0.02, 0.03, 0.01])
    summary = summarize_strategy(returns, WEIGHT_HISTORY.iloc[:4], periods_per_year=12)
    assert np.isnan(summary["sortino"])


def test_summarize_strategy_risk_free_none_matches_direct_metric_calls() -> None:
    """``risk_free=None`` must give exactly the same dict as calling the metrics directly."""
    returns = pd.Series([0.01, 0.0, -0.01, 0.02, 0.01])
    summary = summarize_strategy(returns, WEIGHT_HISTORY, periods_per_year=12, risk_free=None)
    expected = {
        "annual_return": annualized_return(returns, 12),
        "annual_volatility": annualized_volatility(returns, 12),
        "sharpe": sharpe_ratio(returns, 12),
        "sortino": sortino_ratio(returns, 12),
        "max_drawdown": max_drawdown(returns),
        "hhi": concentration_hhi(WEIGHT_HISTORY),
        "avg_turnover": average_turnover(WEIGHT_HISTORY),
    }
    assert summary == pytest.approx(expected, nan_ok=True)


def test_summarize_strategy_with_constant_risk_free_matches_manual_formula() -> None:
    # Two downside observations so the Sortino denominator is finite (not NaN),
    # letting this test actually exercise the risk-free-adjusted numerator.
    returns = pd.Series([0.02, -0.01, -0.02, 0.03, 0.01], index=MONTHLY_DATES)
    risk_free = pd.Series(0.001, index=MONTHLY_DATES)

    summary_rf = summarize_strategy(returns, WEIGHT_HISTORY, periods_per_year=12, risk_free=risk_free)
    summary_plain = summarize_strategy(returns, WEIGHT_HISTORY, periods_per_year=12, risk_free=None)

    rf_annual = annualized_risk_free(risk_free, returns.index, periods_per_year=12)
    ann_ret = annualized_return(returns, 12)
    ann_vol = annualized_volatility(returns, 12)
    assert summary_rf["sharpe"] == pytest.approx((ann_ret - rf_annual) / ann_vol)

    downside_vol = returns[returns < 0].std(ddof=1) * np.sqrt(12)
    assert summary_rf["sortino"] == pytest.approx((ann_ret - rf_annual) / downside_vol)

    # The other five metrics must be identical with and without risk_free.
    for key in ["annual_return", "annual_volatility", "max_drawdown", "hhi", "avg_turnover"]:
        assert summary_rf[key] == pytest.approx(summary_plain[key])
