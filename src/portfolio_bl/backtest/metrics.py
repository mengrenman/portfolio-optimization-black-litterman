from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    """Infer the number of return periods per year from a datetime index.

    Estimates the data frequency based on the median number of calendar days
    between consecutive dates:

    - ≤ 2 days  → 252 (daily)
    - ≤ 10 days → 52  (weekly)
    - ≤ 40 days → 12  (monthly)
    - > 40 days → 4   (quarterly)

    Args:
        index: DatetimeIndex of return dates.

    Returns:
        Estimated number of periods per year.

    Warns:
        UserWarning: If fewer than 3 observations are present; frequency cannot
            be reliably inferred and defaults to 12 (monthly). Annualized
            metrics may be inaccurate.
    """
    if len(index) < 3:
        warnings.warn(
            f"Only {len(index)} observation(s) in index; cannot reliably infer frequency. "
            "Defaulting to 12 (monthly). Annualized metrics may be inaccurate.",
            UserWarning,
            stacklevel=2,
        )
        return 12

    deltas = index.to_series().diff().dt.days.dropna()
    if deltas.empty:
        return 12

    median_days = deltas.median()
    if median_days <= 2:
        return 252
    if median_days <= 10:
        return 52
    if median_days <= 40:
        return 12
    return 4


def annualized_return(returns: pd.Series, periods_per_year: int) -> float:
    """Compute annualized geometric return.

    Args:
        returns: Period return series.
        periods_per_year: Number of return periods per calendar year.

    Returns:
        Annualized geometric return, or NaN if the series is empty.
    """
    if returns.empty:
        return float("nan")
    growth = float((1.0 + returns).prod())
    n = len(returns)
    return growth ** (periods_per_year / n) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int) -> float:
    """Compute annualized return volatility.

    Args:
        returns: Period return series.
        periods_per_year: Number of return periods per calendar year.

    Returns:
        Annualized volatility (standard deviation), or NaN if the series is
        empty.
    """
    if returns.empty:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def risk_free_per_period(
    daily_risk_free: pd.Series, calendar: pd.DatetimeIndex
) -> pd.Series:
    """Compound a daily risk-free series into one rate per period of ``calendar``.

    ``daily_risk_free`` (one rate per NYSE trading day, e.g. from
    ``load_fama_french(factors_dir, "capm")["rf"]``) cannot be reindexed
    directly onto a non-daily return-date index: reindexing samples a single
    day's rate per period instead of compounding every day in it, which
    charges roughly ``1/periods_per_year`` of the true period rate. This
    function does the compounding properly: for consecutive calendar dates
    ``calendar[i-1] < calendar[i]``, the returned rate for period ``i`` is
    ``prod(1 + r_t) - 1`` over every daily rate ``r_t`` dated in the
    half-open interval ``(calendar[i-1], calendar[i]]`` -- including days
    that fall between two calendar dates but are not calendar dates
    themselves (e.g. every business day of a month, when ``calendar`` holds
    only month-end dates).

    Internally this evaluates a cumulative log-growth index (``cumsum`` of
    ``log1p(daily_risk_free)``) "as of" each calendar date, so a period's
    rate is exact regardless of how many daily rates fall inside it. Days
    strictly inside ``daily_risk_free``'s date range that are missing from
    it are assumed to not exist (not to accrue zero return) -- the loader
    this is designed for never leaves such internal holes, one row per NYSE
    trading day, so this is a safe assumption; it is only the calendar's
    reach *outside* the daily series' coverage that is handled explicitly,
    described next.

    When a period's span extends before ``daily_risk_free``'s first date or
    after its last (for example because prices were refreshed past the
    bundled factor file's last month), the missing days are extended at the
    nearest available daily rate -- the first rate for days before coverage,
    the last for days after -- compounded over the number of *business* days
    (via :func:`numpy.busday_count`) in the missing span. Every period that
    needed such an extension is folded into a single warning.

    Args:
        daily_risk_free: Daily risk-free-rate series (decimal, not percent),
            indexed by date. NaN values are dropped and the index is sorted
            before use.
        calendar: Dates bounding the periods to compound onto, typically a
            strategy's price-date index (so ``calendar[0]`` is the opening
            boundary of the first return period). Sorted and de-duplicated
            before use.

    Returns:
        A Series indexed by ``calendar[1:]`` (after sorting/de-duplication)
        holding one compounded rate per period. Empty (with an empty
        ``DatetimeIndex``) when ``calendar`` has fewer than two distinct,
        non-null dates.

    Raises:
        ValueError: If ``daily_risk_free`` is empty (after dropping NaNs), or
            if no period spanned by ``calendar`` overlaps ``daily_risk_free``'s
            date coverage at all.
    """
    daily = daily_risk_free.dropna().sort_index()
    if daily.empty:
        raise ValueError("daily_risk_free has no non-null observations.")

    cal = pd.DatetimeIndex(calendar)
    cal = cal[~cal.isna()]
    cal = pd.DatetimeIndex(pd.unique(cal)).sort_values()

    if len(cal) < 2:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))

    d_min = daily.index[0]
    d_max = daily.index[-1]
    if cal[-1] < d_min or cal[0] > d_max:
        raise ValueError(
            f"No period spanned by the calendar ({cal[0].date()}..{cal[-1].date()}) "
            f"overlaps the daily risk-free series' coverage ({d_min.date()}..{d_max.date()})."
        )

    log_rate = np.log1p(daily.to_numpy())
    cum_log_growth = np.cumsum(log_rate)
    idx_values = daily.index.to_numpy()
    cal_values = cal.to_numpy()

    before_mask = cal_values < d_min.to_numpy()
    after_mask = cal_values > d_max.to_numpy()
    within_mask = ~before_mask & ~after_mask

    # Cumulative log growth "as of" each calendar date: for dates within the
    # daily series' coverage this is the plain asof sum of every daily rate
    # dated on or before that date -- taking the difference between two such
    # values therefore sums exactly the daily rates dated in between.
    s_values = np.empty(len(cal), dtype=float)
    if within_mask.any():
        positions = np.searchsorted(idx_values, cal_values[within_mask], side="right") - 1
        s_values[within_mask] = cum_log_growth[positions]

    # Outside coverage, extend at the nearest boundary rate over the number
    # of business days in the missing span. Shifting both ends of the
    # np.busday_count interval by one calendar day turns its half-open
    # [begin, end) convention into the (excluded start, included end] this
    # function wants, in both directions.
    one_day = np.timedelta64(1, "D")
    if before_mask.any():
        before_days = cal_values[before_mask].astype("datetime64[D]")
        d_min_day = d_min.to_numpy().astype("datetime64[D]")
        n_bd = np.busday_count(before_days + one_day, d_min_day + one_day)
        s_values[before_mask] = cum_log_growth[0] - log_rate[0] * n_bd

    if after_mask.any():
        after_days = cal_values[after_mask].astype("datetime64[D]")
        d_max_day = d_max.to_numpy().astype("datetime64[D]")
        n_bd = np.busday_count(d_max_day + one_day, after_days + one_day)
        s_values[after_mask] = cum_log_growth[-1] + log_rate[-1] * n_bd

    period_ends = cal[1:]
    period_rates = np.exp(s_values[1:] - s_values[:-1]) - 1.0

    extended = before_mask | after_mask
    period_extended = extended[1:] | extended[:-1]
    n_extended = int(period_extended.sum())
    if n_extended > 0:
        extended_ends = period_ends[period_extended]
        logger.warning(
            "%d of %d period(s) extend beyond the daily risk-free series' coverage "
            "(%s..%s); extended using the nearest boundary daily rate over the "
            "missing business days (first extended period ends %s, last %s).",
            n_extended,
            len(period_ends),
            d_min.date(),
            d_max.date(),
            extended_ends.min().date(),
            extended_ends.max().date(),
        )

    return pd.Series(period_rates, index=period_ends)


def annualized_risk_free(
    risk_free: pd.Series, index: pd.DatetimeIndex, periods_per_year: int
) -> float:
    """Compute the geometric annualized risk-free rate over a set of dates.

    Reindexes ``risk_free`` to exactly ``index`` and geometrically annualizes
    it with :func:`annualized_return`, so the result is directly comparable
    with a strategy's ``annualized_return`` computed over the same dates.

    ``risk_free`` must already hold exactly one rate per return period of
    ``index`` -- this function does no frequency conversion of its own. In
    particular, do **not** pass a *daily* risk-free series here when
    ``index`` is a monthly (or weekly, or any non-daily) return-date index:
    reindexing would silently keep only one day's rate per period, charging
    roughly ``1/periods_per_year_of_the_daily_series`` of the true period
    rate. Use :func:`risk_free_per_period` first to compound a daily series
    into one rate per period of ``index``, then pass that result here.

    Some of ``index`` may fall outside ``risk_free``'s coverage -- for
    example when prices have been refreshed past the bundled factor file's
    last month, which is normal because Ken French's data library publishes
    with a lag. Those dates are filled from the nearest available rate
    (forward-fill, then back-fill for any still missing at the start).

    Args:
        risk_free: Per-period risk-free-rate series (decimal, not percent),
            indexed by date, with one rate per period of ``index`` (see
            :func:`risk_free_per_period` to derive this from a daily series).
        index: Dates to annualize the risk-free rate over, typically a
            strategy's return-series index.
        periods_per_year: Number of return periods per calendar year.

    Returns:
        The geometric annualized risk-free rate over ``index``, or ``0.0``
        if ``index`` is empty.

    Raises:
        ValueError: If none of the dates in ``index`` are covered by
            ``risk_free``.
    """
    if len(index) == 0:
        return 0.0

    reindexed = risk_free.reindex(index)
    missing = reindexed.isna()
    n_missing = int(missing.sum())

    if n_missing == len(index):
        raise ValueError(
            f"None of the {len(index)} requested date(s) are covered by the risk-free series."
        )

    if n_missing > 0:
        missing_dates = index[missing]
        logger.warning(
            "%d of %d date(s) not covered by the risk-free series (first missing %s, "
            "last missing %s); filling from the nearest available rate.",
            n_missing,
            len(index),
            missing_dates.min().date(),
            missing_dates.max().date(),
        )
        reindexed = reindexed.ffill().bfill()

    return annualized_return(reindexed, periods_per_year)


def sharpe_ratio(
    returns: pd.Series, periods_per_year: int, risk_free_rate: float = 0.0
) -> float:
    """Compute annualized Sharpe ratio.

    The numerator is the geometric annualized return (see
    :func:`annualized_return`) minus the geometric annualized risk-free rate
    over the same dates (see :func:`annualized_risk_free`); the denominator
    is unchanged: the annualized volatility of the raw returns.

    Args:
        returns: Period return series.
        periods_per_year: Number of return periods per calendar year.
        risk_free_rate: Geometric annualized risk-free rate, over the same
            dates as ``returns``, used as the numerator's hurdle. Defaults to
            0.0 (no risk-free adjustment).

    Returns:
        Sharpe ratio, or NaN if volatility is zero or not finite.
    """
    ann_ret = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    if not np.isfinite(ann_vol) or ann_vol <= 0:
        return float("nan")
    return float((ann_ret - risk_free_rate) / ann_vol)


def sortino_ratio(
    returns: pd.Series, periods_per_year: int, risk_free_rate: float = 0.0
) -> float:
    """Compute annualized Sortino ratio.

    The numerator is the geometric annualized return (see
    :func:`annualized_return`) minus the geometric annualized risk-free rate
    over the same dates (see :func:`annualized_risk_free`), exactly as in
    :func:`sharpe_ratio`. The denominator is unchanged: downside deviation
    computed from returns below zero -- that threshold stays at zero, not
    the risk-free rate, unlike the numerator.

    Unlike the Sharpe ratio, Sortino penalizes only downside volatility
    (returns below zero). Returns NaN — rather than +Inf — when no negative
    returns exist. This edge case is common in short bull-market windows and
    returning NaN keeps downstream comparisons and CSV outputs well-defined.

    Args:
        returns: Period return series.
        periods_per_year: Number of return periods per calendar year.
        risk_free_rate: Geometric annualized risk-free rate, over the same
            dates as ``returns``, used as the numerator's hurdle. Defaults to
            0.0 (no risk-free adjustment).

    Returns:
        Sortino ratio, or NaN if the series is empty, has no downside returns,
        or if downside volatility is not finite.
    """
    if returns.empty:
        return float("nan")

    downside = returns[returns < 0]
    if downside.empty:
        return float("nan")

    downside_vol = downside.std(ddof=1) * np.sqrt(periods_per_year)
    if not np.isfinite(downside_vol) or downside_vol <= 0:
        return float("nan")

    ann_ret = annualized_return(returns, periods_per_year)
    return float((ann_ret - risk_free_rate) / downside_vol)


def max_drawdown(returns: pd.Series) -> float:
    """Compute maximum peak-to-trough drawdown.

    Args:
        returns: Period return series.

    Returns:
        Maximum drawdown as a negative fraction (e.g. -0.20 for a 20% drop),
        or NaN if the series is empty.
    """
    if returns.empty:
        return float("nan")

    nav = (1.0 + returns).cumprod()
    peak = nav.cummax()
    drawdown = nav / peak - 1.0
    return float(drawdown.min())


def concentration_hhi(weight_history: pd.DataFrame) -> float:
    """Compute the mean Herfindahl-Hirschman Index (HHI) of portfolio concentration.

    HHI is defined as the sum of squared portfolio weights. A value of 1/n
    represents an equal-weight portfolio; a value of 1.0 represents full
    concentration in a single asset.

    Args:
        weight_history: DataFrame of rebalance-date portfolio weights
            (dates × tickers).

    Returns:
        Mean HHI across all rebalance dates, or NaN if the history is empty.
    """
    if weight_history.empty:
        return float("nan")

    normalized = weight_history.div(weight_history.sum(axis=1), axis=0).fillna(0.0)
    hhi = (normalized**2).sum(axis=1).mean()
    return float(hhi)


def average_turnover(weight_history: pd.DataFrame) -> float:
    """Compute mean one-way portfolio turnover across rebalances.

    Turnover is defined as half the sum of absolute weight changes at each
    rebalance, averaged over all rebalance intervals. A value of 0.10 means
    10% of the portfolio is replaced on average per rebalance.

    Args:
        weight_history: DataFrame of rebalance-date portfolio weights.

    Returns:
        Mean one-way turnover, or 0.0 if fewer than two rebalances exist.
    """
    if weight_history.shape[0] <= 1:
        return 0.0

    delta = weight_history.diff().abs().sum(axis=1) / 2.0
    return float(delta.iloc[1:].mean())


def summarize_strategy(
    returns: pd.Series,
    weight_history: pd.DataFrame,
    periods_per_year: int,
    risk_free: pd.Series | None = None,
) -> dict[str, float]:
    """Compute a standard set of performance metrics for a strategy.

    Args:
        returns: Portfolio period return series.
        weight_history: Rebalance-date weight history DataFrame.
        periods_per_year: Number of return periods per calendar year.
        risk_free: Optional per-period risk-free-rate series (decimal, not
            percent), such as ``load_fama_french(directory, "capm")["rf"]``.
            When given, it is reindexed to ``returns.index`` and
            geometrically annualized via :func:`annualized_risk_free` (0.0
            when ``returns`` is empty), and that rate is charged as the
            numerator's hurdle in both ``sharpe`` and ``sortino`` — their
            denominators are unaffected; see :func:`sharpe_ratio` and
            :func:`sortino_ratio`. When ``None`` (default), both ratios use a
            zero risk-free rate and behavior is unchanged from before this
            parameter existed.

    Returns:
        A dictionary with keys: ``annual_return``, ``annual_volatility``,
        ``sharpe``, ``sortino``, ``max_drawdown``, ``hhi``, ``avg_turnover``.
    """
    rf_annual = (
        annualized_risk_free(risk_free, returns.index, periods_per_year)
        if risk_free is not None
        else 0.0
    )
    return {
        "annual_return": annualized_return(returns, periods_per_year),
        "annual_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, periods_per_year, risk_free_rate=rf_annual),
        "sortino": sortino_ratio(returns, periods_per_year, risk_free_rate=rf_annual),
        "max_drawdown": max_drawdown(returns),
        "hhi": concentration_hhi(weight_history),
        "avg_turnover": average_turnover(weight_history),
    }
