from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from portfolio_bl.backtest.attribution import (
    FrequencyMismatchError,
    attribution_table,
    factor_regression,
    newey_west_lags,
    ols_newey_west,
)

# ---------------------------------------------------------------------------
# Synthetic factor-frame helpers
# ---------------------------------------------------------------------------


def _make_ff3(rng: np.random.Generator, n: int, rf_value: float = 0.00005) -> pd.DataFrame:
    """Build a synthetic FF3-style factor frame (decimals, DatetimeIndex "date")."""
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    dates.name = "date"
    return pd.DataFrame(
        {
            "mkt_rf": rng.normal(0.0004, 0.01, n),
            "smb": rng.normal(0.0, 0.005, n),
            "hml": rng.normal(0.0, 0.005, n),
            "rf": np.full(n, rf_value),
        },
        index=dates,
    )


def _make_ff5(rng: np.random.Generator, n: int, rf_value: float = 0.00005) -> pd.DataFrame:
    """Build a synthetic FF5-style factor frame sharing FF3's mkt_rf/smb/hml/rf."""
    ff3 = _make_ff3(rng, n, rf_value=rf_value)
    ff3["rmw"] = rng.normal(0.0, 0.004, n)
    ff3["cma"] = rng.normal(0.0, 0.004, n)
    return ff3[["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]]


# ---------------------------------------------------------------------------
# newey_west_lags
# ---------------------------------------------------------------------------


def test_newey_west_lags_matches_published_examples() -> None:
    assert newey_west_lags(1864) == 7
    assert newey_west_lags(100) == 4


def test_newey_west_lags_at_n_equals_one() -> None:
    # floor(4 * (1 / 100) ** (2 / 9)) = floor(1.4375) = 1. n_obs=1 is a valid,
    # non-raising input under the "n_obs < 1 raises" rule, so the value used
    # here is the formula's own output (verified independently with
    # math.floor), not an assumed round number.
    assert newey_west_lags(1) == 1


def test_newey_west_lags_zero_raises() -> None:
    with pytest.raises(ValueError):
        newey_west_lags(0)


def test_newey_west_lags_negative_raises() -> None:
    with pytest.raises(ValueError):
        newey_west_lags(-3)


# ---------------------------------------------------------------------------
# ols_newey_west: recovers known coefficients
# ---------------------------------------------------------------------------


def test_ols_recovers_known_coefficients_with_small_noise() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    x = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    true_beta = np.array([0.001, 1.2, 0.3])
    y = x @ true_beta + rng.normal(scale=1e-4, size=n)

    result = ols_newey_west(y, x, lags=5)

    assert result.coefficients == pytest.approx(true_beta, abs=5e-4)
    assert result.n_obs == n
    assert result.lags == 5


def test_ols_matches_plain_lstsq() -> None:
    rng = np.random.default_rng(1)
    n = 500
    x = np.column_stack([np.ones(n), rng.normal(size=(n, 3))])
    y = rng.normal(size=n)

    result = ols_newey_west(y, x, lags=3)
    expected_beta, *_ = np.linalg.lstsq(x, y, rcond=None)

    assert result.coefficients == pytest.approx(expected_beta)


# ---------------------------------------------------------------------------
# ols_newey_west: lags=0 equals White's HC0 sandwich
# ---------------------------------------------------------------------------


def test_lags_zero_matches_white_hc0_sandwich_computed_with_explicit_loop() -> None:
    rng = np.random.default_rng(2)
    n = 300
    x = np.column_stack([np.ones(n), rng.normal(size=n)])
    beta = np.array([0.5, -0.3])
    y = x @ beta + rng.normal(size=n)

    result = ols_newey_west(y, x, lags=0)

    beta_hat, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta_hat
    xtx_inv = np.linalg.inv(x.T @ x)

    k = x.shape[1]
    s_matrix = np.zeros((k, k))
    for t in range(n):
        xt = x[t, :].reshape(-1, 1)
        s_matrix += resid[t] ** 2 * (xt @ xt.T)
    cov = xtx_inv @ s_matrix @ xtx_inv
    expected_se = np.sqrt(np.diag(cov))

    assert result.std_errors == pytest.approx(expected_se)


# ---------------------------------------------------------------------------
# ols_newey_west: intercept-only Bartlett long-run variance
# ---------------------------------------------------------------------------


def test_intercept_only_hac_variance_matches_bartlett_long_run_variance() -> None:
    rng = np.random.default_rng(3)
    n = 800
    lags = 8
    y = 0.0003 + rng.standard_t(df=5, size=n) * 0.01
    x = np.ones((n, 1))

    result = ols_newey_west(y, x, lags=lags)

    demeaned = y - y.mean()
    gamma0 = np.sum(demeaned * demeaned) / n
    long_run_var = gamma0
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1)
        gamma_l = np.sum(demeaned[lag:] * demeaned[:-lag]) / n
        long_run_var += 2.0 * weight * gamma_l
    expected_variance = long_run_var / n

    assert result.std_errors[0] ** 2 == pytest.approx(expected_variance)


# ---------------------------------------------------------------------------
# ols_newey_west: AR(1) errors inflate SE relative to iid
# ---------------------------------------------------------------------------


def test_ar1_errors_inflate_intercept_se_at_lags_ten() -> None:
    rng = np.random.default_rng(4)
    n = 2000
    phi = 0.6
    innovations = rng.normal(size=n)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = phi * u[t - 1] + innovations[t]
    x = np.ones((n, 1))
    y = 0.02 + u

    se_lags0 = ols_newey_west(y, x, lags=0).std_errors[0]
    se_lags10 = ols_newey_west(y, x, lags=10).std_errors[0]

    assert se_lags10 > se_lags0


def test_iid_errors_have_close_se_at_lags_zero_and_ten() -> None:
    rng = np.random.default_rng(5)
    n = 2000
    x = np.ones((n, 1))
    y = 0.02 + rng.normal(size=n)

    se_lags0 = ols_newey_west(y, x, lags=0).std_errors[0]
    se_lags10 = ols_newey_west(y, x, lags=10).std_errors[0]

    assert se_lags10 == pytest.approx(se_lags0, rel=0.15)


# ---------------------------------------------------------------------------
# ols_newey_west: scale invariance
# ---------------------------------------------------------------------------


def test_scale_invariance_of_t_stats_and_coefficients() -> None:
    rng = np.random.default_rng(6)
    n = 400
    x = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    y = 0.5 + 1.5 * x[:, 1] - 0.7 * x[:, 2] + rng.normal(scale=0.5, size=n)

    base = ols_newey_west(y, x, lags=4)
    scaled = ols_newey_west(y * 100.0, x, lags=4)

    assert scaled.t_stats == pytest.approx(base.t_stats)
    assert scaled.coefficients == pytest.approx(base.coefficients * 100.0)


# ---------------------------------------------------------------------------
# ols_newey_west: input validation
# ---------------------------------------------------------------------------


def test_ols_rejects_2d_y() -> None:
    with pytest.raises(ValueError):
        ols_newey_west(np.ones((10, 1)), np.ones((10, 2)), lags=0)


def test_ols_rejects_1d_x() -> None:
    with pytest.raises(ValueError):
        ols_newey_west(np.ones(10), np.ones(10), lags=0)


def test_ols_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        ols_newey_west(np.ones(10), np.ones((9, 2)), lags=0)


def test_ols_rejects_non_finite_values() -> None:
    y = np.array([1.0, np.nan, 3.0])
    x = np.ones((3, 1))
    with pytest.raises(ValueError):
        ols_newey_west(y, x, lags=0)


def test_ols_rejects_negative_lags() -> None:
    x = np.column_stack([np.ones(10), np.arange(10.0)])
    with pytest.raises(ValueError):
        ols_newey_west(np.ones(10), x, lags=-1)


def test_ols_rejects_too_few_observations() -> None:
    x = np.column_stack([np.ones(3), np.arange(3.0), np.arange(3.0) ** 2])
    with pytest.raises(ValueError):
        ols_newey_west(np.ones(3), x, lags=0)


# ---------------------------------------------------------------------------
# factor_regression
# ---------------------------------------------------------------------------


def test_factor_regression_recovers_known_alpha_and_loadings() -> None:
    rng = np.random.default_rng(7)
    n = 1500
    factors = _make_ff3(rng, n)
    alpha_true = 0.001
    excess_true = 1.2 * factors["mkt_rf"].to_numpy() + 0.3 * factors["smb"].to_numpy()
    tiny_noise = rng.normal(scale=1e-7, size=n)
    returns = pd.Series(
        factors["rf"].to_numpy() + alpha_true + excess_true + tiny_noise,
        index=factors.index,
        name="strategy",
    )

    reg = factor_regression(returns, factors, periods_per_year=252)

    assert reg.factors == ("mkt_rf", "smb", "hml")
    assert reg.alpha == pytest.approx(alpha_true, abs=1e-5)
    assert reg.alpha_annual == pytest.approx(alpha_true * 252, abs=1e-3)
    assert reg.loadings["mkt_rf"] == pytest.approx(1.2, abs=1e-3)
    assert reg.loadings["smb"] == pytest.approx(0.3, abs=1e-3)
    assert np.isfinite(reg.alpha_t)
    assert np.isfinite(reg.loading_t["mkt_rf"])
    assert np.isfinite(reg.loading_t["smb"])
    assert np.isfinite(reg.loading_t["hml"])


def test_rf_subtracted_from_returns_but_not_from_factors() -> None:
    """Fails if rf were (incorrectly) also subtracted from the factor columns.

    ``mkt_rf`` is set identical to ``rf`` here. If the implementation wrongly
    subtracted rf from the factors as well, that regressor column would
    collapse to all zeros and the estimated loading would collapse toward 0
    instead of recovering ``beta_true``.
    """
    rng = np.random.default_rng(8)
    n = 1000
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    dates.name = "date"
    rf = rng.normal(0.0001, 0.00002, n)
    mkt_rf = rf.copy()
    alpha_true = 0.0002
    beta_true = 1.5

    factors = pd.DataFrame({"mkt_rf": mkt_rf, "rf": rf}, index=dates)
    returns = pd.Series(rf + alpha_true + beta_true * mkt_rf, index=dates, name="strategy")

    reg = factor_regression(returns, factors)

    assert reg.loadings["mkt_rf"] == pytest.approx(beta_true, abs=1e-6)
    assert reg.alpha == pytest.approx(alpha_true, abs=1e-6)


def test_missing_dates_are_dropped_and_logged_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rng = np.random.default_rng(9)
    n = 200
    factors = _make_ff3(rng, n)
    returns = pd.Series(
        factors["rf"].to_numpy() + rng.normal(0.0, 0.001, n),
        index=factors.index,
        name="strategy",
    )

    dropped_dates = factors.index[10:13]
    factors_missing = factors.drop(index=dropped_dates)

    with caplog.at_level(logging.WARNING, logger="portfolio_bl.backtest.attribution"):
        reg = factor_regression(returns, factors_missing)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "3" in warnings[0].getMessage()
    assert reg.n_obs == n - 3


def test_nan_returns_on_factor_dates_are_dropped_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rng = np.random.default_rng(11)
    n = 200
    factors = _make_ff3(rng, n)
    returns = pd.Series(rng.normal(0.0, 0.01, n), index=factors.index, name="strategy")
    returns.iloc[[20, 40]] = np.nan

    with caplog.at_level(logging.WARNING, logger="portfolio_bl.backtest.attribution"):
        reg = factor_regression(returns, factors)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "2 row(s)" in warnings[0].getMessage()
    assert reg.n_obs == n - 2


def test_r_squared_is_centered_around_the_mean() -> None:
    # A large mean in y makes centered and uncentered R-squared differ sharply.
    rng = np.random.default_rng(12)
    n = 200
    x = np.column_stack([np.ones(n), rng.normal(0.0, 1.0, n)])
    y = 5.0 + 0.3 * x[:, 1] + rng.normal(0.0, 1.0, n)

    fit = ols_newey_west(y, x, lags=0)

    ssr = float(np.sum(fit.residuals**2))
    centered = 1.0 - ssr / float(np.sum((y - y.mean()) ** 2))
    uncentered = 1.0 - ssr / float(np.sum(y**2))
    assert fit.r_squared == pytest.approx(centered, rel=1e-12)
    assert abs(centered - uncentered) > 0.5
    k = x.shape[1]
    assert fit.adj_r_squared == pytest.approx(1.0 - (1.0 - centered) * (n - 1) / (n - k), rel=1e-12)


def test_difference_regression_without_rf_equals_difference_of_alphas() -> None:
    # A time-varying rf makes the test sensitive to whether rf is subtracted
    # from the difference: with subtract_rf=True the identity fails by mean(rf).
    rng = np.random.default_rng(13)
    n = 300
    factors = _make_ff5(rng, n)
    factors["rf"] = rng.uniform(0.0, 0.0004, n)
    a = pd.Series(factors["rf"] + 0.0003 + 1.1 * factors["mkt_rf"] + rng.normal(0.0, 0.002, n))
    b = pd.Series(factors["rf"] + 0.0001 + 0.9 * factors["mkt_rf"] + 0.4 * factors["hml"]
                  + rng.normal(0.0, 0.002, n))

    reg_a = factor_regression(a, factors)
    reg_b = factor_regression(b, factors)
    reg_diff = factor_regression(a - b, factors, subtract_rf=False)
    reg_wrong = factor_regression(a - b, factors, subtract_rf=True)

    assert reg_diff.alpha == pytest.approx(reg_a.alpha - reg_b.alpha, abs=1e-14)
    pd.testing.assert_series_equal(reg_diff.loadings, reg_a.loadings - reg_b.loadings, atol=1e-12)
    assert abs(reg_wrong.alpha - (reg_a.alpha - reg_b.alpha)) > 1e-4


def test_attribution_table_passes_subtract_rf_through() -> None:
    rng = np.random.default_rng(14)
    n = 150
    factors = _make_ff3(rng, n, rf_value=0.0002)
    returns = pd.Series(0.0005 + factors["mkt_rf"] + rng.normal(0.0, 0.001, n))

    with_rf = attribution_table({"s": returns}, {"ff3": factors})
    without_rf = attribution_table({"s": returns}, {"ff3": factors}, subtract_rf=False)

    shift = without_rf.loc[("s", "ff3"), "alpha_annual"] - with_rf.loc[("s", "ff3"), "alpha_annual"]
    assert shift == pytest.approx(0.0002 * 252, rel=1e-6)


def test_too_few_observations_raises() -> None:
    rng = np.random.default_rng(10)
    factors = _make_ff3(rng, 3)
    returns = pd.Series(rng.normal(size=3), index=factors.index, name="strategy")
    with pytest.raises(ValueError):
        factor_regression(returns, factors)


def test_missing_rf_column_raises() -> None:
    rng = np.random.default_rng(11)
    factors = _make_ff3(rng, 100).drop(columns=["rf"])
    returns = pd.Series(rng.normal(size=100), index=factors.index, name="strategy")
    with pytest.raises(ValueError):
        factor_regression(returns, factors)


def test_monthly_returns_against_daily_factors_raises() -> None:
    """Regressing a non-daily return series against the daily bundled
    factors would otherwise join silently: the inner join matches returns
    only to the factor rows dated on exactly the same day, so a monthly
    return ends up regressed on a single day's factor values per month.
    That must be refused before it produces a meaningless-but-plausible-
    looking result.
    """
    rng = np.random.default_rng(15)
    n_daily = 400
    factors = _make_ff3(rng, n_daily)  # daily, freq="B" -> 252 periods/year

    monthly_dates = pd.date_range("2020-01-31", periods=18, freq="ME")
    returns = pd.Series(rng.normal(0.01, 0.02, len(monthly_dates)), index=monthly_dates)

    with pytest.raises(ValueError, match="12 period.*252|252 period.*12"):
        factor_regression(returns, factors)


def test_frequency_mismatch_error_is_a_value_error_subclass() -> None:
    assert issubclass(FrequencyMismatchError, ValueError)


def test_monthly_returns_against_daily_factors_raises_frequency_mismatch_error() -> None:
    """The frequency guard must raise the specific
    :class:`FrequencyMismatchError` subclass, not a bare :class:`ValueError`,
    so callers can distinguish it from every other way the regression fails.
    """
    rng = np.random.default_rng(17)
    n_daily = 400
    factors = _make_ff3(rng, n_daily)

    monthly_dates = pd.date_range("2020-01-31", periods=18, freq="ME")
    returns = pd.Series(rng.normal(0.01, 0.02, len(monthly_dates)), index=monthly_dates)

    with pytest.raises(FrequencyMismatchError):
        factor_regression(returns, factors)


def test_matching_daily_frequencies_do_not_raise() -> None:
    """Sanity check for the frequency guard: identical (daily) frequencies
    must not be refused, even though the actual dates barely overlap.
    """
    rng = np.random.default_rng(16)
    factors = _make_ff3(rng, 400)
    other_daily_dates = pd.bdate_range(factors.index[-2], periods=400)
    returns = pd.Series(rng.normal(0.0, 0.01, 400), index=other_daily_dates)

    # Only the last 2 dates of `factors` and the first 2 of `returns`
    # overlap; this must fail on "too few observations", not the frequency
    # guard, proving the guard did not fire for two same-frequency series.
    with pytest.raises(ValueError, match="aligned observation"):
        factor_regression(returns, factors)


def test_reordering_factor_columns_preserves_loadings_by_name() -> None:
    rng = np.random.default_rng(12)
    n = 800
    factors = _make_ff3(rng, n)
    returns = pd.Series(
        factors["rf"].to_numpy()
        + 0.0005
        + 0.8 * factors["mkt_rf"].to_numpy()
        - 0.4 * factors["hml"].to_numpy()
        + rng.normal(scale=1e-4, size=n),
        index=factors.index,
        name="strategy",
    )

    reg_original = factor_regression(returns, factors)
    reg_reordered = factor_regression(returns, factors[["hml", "rf", "mkt_rf", "smb"]])

    assert reg_reordered.factors == ("hml", "mkt_rf", "smb")
    for name in ("mkt_rf", "smb", "hml"):
        assert reg_reordered.loadings[name] == pytest.approx(reg_original.loadings[name])
    assert reg_reordered.alpha == pytest.approx(reg_original.alpha)


# ---------------------------------------------------------------------------
# attribution_table
# ---------------------------------------------------------------------------


def test_attribution_table_shape_index_and_nan_pattern() -> None:
    rng = np.random.default_rng(13)
    n = 600
    ff3 = _make_ff3(rng, n)
    ff5 = _make_ff5(rng, n)
    returns_by_name = {
        "alpha_strategy": pd.Series(
            ff3["rf"].to_numpy() + 0.0004 + 1.1 * ff3["mkt_rf"].to_numpy(),
            index=ff3.index,
            name="alpha_strategy",
        ),
        "beta_strategy": pd.Series(
            ff3["rf"].to_numpy()
            + 0.9 * ff3["mkt_rf"].to_numpy()
            + rng.normal(scale=1e-4, size=n),
            index=ff3.index,
            name="beta_strategy",
        ),
    }
    factor_sets = {"ff3": ff3, "ff5": ff5}

    table = attribution_table(returns_by_name, factor_sets)

    expected_columns = [
        "alpha_annual",
        "alpha_t",
        "mkt_rf",
        "mkt_rf_t",
        "smb",
        "smb_t",
        "hml",
        "hml_t",
        "rmw",
        "rmw_t",
        "cma",
        "cma_t",
        "r_squared",
        "adj_r_squared",
        "n_obs",
        "lags",
        "start",
        "end",
    ]

    assert table.shape == (4, len(expected_columns))
    assert table.index.names == ["strategy", "model"]
    assert list(table.index) == [
        ("alpha_strategy", "ff3"),
        ("alpha_strategy", "ff5"),
        ("beta_strategy", "ff3"),
        ("beta_strategy", "ff5"),
    ]
    assert table.columns.tolist() == expected_columns
    assert pd.isna(table.loc[("alpha_strategy", "ff3"), "rmw"])
    assert pd.isna(table.loc[("alpha_strategy", "ff3"), "cma"])
    assert pd.isna(table.loc[("beta_strategy", "ff3"), "rmw"])
    assert not pd.isna(table.loc[("alpha_strategy", "ff5"), "rmw"])
    assert not pd.isna(table.loc[("alpha_strategy", "ff5"), "cma"])
