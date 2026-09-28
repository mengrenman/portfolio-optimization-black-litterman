"""Factor-model performance attribution with Newey-West (HAC) standard errors.

Strategy returns are regressed on a set of pricing factors (e.g. the
Fama-French three- or five-factor models) to estimate alpha and factor
loadings. Standard errors use the Newey-West (1987, 1994) heteroskedasticity-
and autocorrelation-consistent (HAC) sandwich estimator with a Bartlett
kernel, so daily residual autocorrelation does not bias inference.

statsmodels is intentionally not a dependency of this package; the OLS fit
and the HAC sandwich covariance are implemented directly with numpy.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_bl.backtest.metrics import infer_periods_per_year

logger = logging.getLogger(__name__)


class FrequencyMismatchError(ValueError):
    """Raised when returns and factors have different inferred frequencies.

    :func:`factor_regression` refuses to regress a return series against a
    factor frame whose own inferred frequency (see
    :func:`~portfolio_bl.backtest.metrics.infer_periods_per_year`) differs
    from the returns': the inner join it otherwise performs would match each
    return only to the factor row dated on exactly the same day, so a
    non-daily return (e.g. monthly) regressed against daily factors would
    silently end up fit on a single day's factor values per period --
    meaningless, and not something the join's row count alone reveals. This
    is a :class:`ValueError` subclass so existing ``except ValueError``
    handlers keep working; catch this type specifically to distinguish a
    frequency mismatch from every other way the regression can fail (too few
    aligned observations, a missing risk-free column, and so on).
    """


def newey_west_lags(n_obs: int) -> int:
    """Compute the Newey-West (1994) rule-of-thumb truncation lag.

    Uses the standard automatic bandwidth ``floor(4 * (n_obs / 100) ** (2 / 9))``
    for Bartlett-kernel HAC standard errors.

    Args:
        n_obs: Number of observations the regression will be run on.

    Returns:
        The suggested number of autocovariance lags (a non-negative integer).

    Raises:
        ValueError: If ``n_obs`` is less than 1.
    """
    if n_obs < 1:
        raise ValueError(f"n_obs must be >= 1, got {n_obs}.")
    return int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))


@dataclass(frozen=True)
class RegressionResult:
    """Result of an OLS fit with a Newey-West (HAC) covariance estimate.

    Attributes:
        coefficients: Estimated coefficients, in the column order of the
            design matrix ``x`` passed to :func:`ols_newey_west`.
        std_errors: Newey-West (HAC) standard errors, in the same order as
            ``coefficients``.
        t_stats: ``coefficients / std_errors``, element-wise.
        r_squared: Centered R-squared (see :func:`ols_newey_west`).
        adj_r_squared: R-squared adjusted for the number of parameters.
        residuals: Fitted residuals, ``y - x @ coefficients``.
        n_obs: Number of observations used in the fit.
        lags: Number of Bartlett-kernel lags used for the HAC covariance.
    """

    coefficients: np.ndarray
    std_errors: np.ndarray
    t_stats: np.ndarray
    r_squared: float
    adj_r_squared: float
    residuals: np.ndarray
    n_obs: int
    lags: int


def ols_newey_west(y: np.ndarray, x: np.ndarray, lags: int) -> RegressionResult:
    """Fit OLS and estimate a Newey-West (HAC) sandwich covariance.

    Fits ``y = x @ beta + e`` by plain OLS (``beta = lstsq(x, y)``) and
    estimates the coefficient covariance with the Newey-West (1987, 1994)
    heteroskedasticity- and autocorrelation-consistent sandwich estimator
    using a Bartlett kernel::

        V = (X'X)^-1 S (X'X)^-1
        S = sum_t u_t^2 x_t x_t'
            + sum_{l=1}^{L} w_l * sum_{t=l}^{T-1} u_t u_{t-l} (x_t x_{t-l}' + x_{t-l} x_t')
        w_l = 1 - l / (L + 1)

    where ``u`` are the OLS residuals and ``L`` is ``lags``. ``lags=0``
    collapses ``S`` to its first term, i.e. White's (1980) HC0
    heteroskedasticity-consistent sandwich.

    No small-sample degrees-of-freedom correction is applied to ``V`` — this
    is the plain asymptotic Newey-West estimator, not a finite-sample-
    corrected variant, so standard errors can be a little optimistic when
    ``lags`` is large relative to ``n_obs``.

    Args:
        y: 1-D array of the dependent variable, length ``n_obs``.
        x: 2-D design matrix, shape ``(n_obs, n_params)``. The caller adds a
            constant column itself if an intercept is wanted.
        lags: Number of Bartlett-kernel autocovariance lags (``L`` above).
            Must be non-negative.

    Returns:
        A :class:`RegressionResult`.

    Raises:
        ValueError: If ``y`` is not 1-D, ``x`` is not 2-D, their lengths
            differ, either contains a non-finite value, ``lags`` is
            negative, or there are not more observations than parameters.

    Note:
        R-squared is computed as the *centered* value, ``1 - SSR / SST``
        with SST measured around the sample mean of ``y``. That decomposition
        is only meaningful when ``x`` includes an intercept column, which is
        the only way this module ever calls ``ols_newey_west`` (see
        :func:`factor_regression`); no uncentered-R-squared branch is
        implemented.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}.")
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D, got shape {x.shape}.")
    if y.shape[0] != x.shape[0]:
        raise ValueError(
            f"y and x must have the same length, got {y.shape[0]} and {x.shape[0]}."
        )
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
        raise ValueError("y and x must not contain NaN or infinite values.")
    if lags < 0:
        raise ValueError(f"lags must be >= 0, got {lags}.")

    n_obs, n_params = x.shape
    if n_obs <= n_params:
        raise ValueError(
            f"n_obs ({n_obs}) must exceed n_params ({n_params}) for the fit to be identified."
        )

    coefficients, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ coefficients

    xtx_inv = np.linalg.inv(x.T @ x)

    # xu[t] = x_t * u_t, so xu[a:].T @ xu[:len(xu) - a] sums u_t * u_{t-a} * x_t
    # x_{t-a}' over the valid range of t in one shot, for any lag a >= 0.
    xu = x * residuals[:, None]
    s_matrix = xu.T @ xu
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1)
        cross = xu[lag:].T @ xu[:-lag]
        s_matrix = s_matrix + weight * (cross + cross.T)

    cov = xtx_inv @ s_matrix @ xtx_inv
    std_errors = np.sqrt(np.diag(cov))
    t_stats = coefficients / std_errors

    y_mean = y.mean()
    sst = float(np.sum((y - y_mean) ** 2))
    ssr = float(np.sum(residuals**2))
    r_squared = 1.0 - ssr / sst
    adj_r_squared = 1.0 - (1.0 - r_squared) * (n_obs - 1) / (n_obs - n_params)

    return RegressionResult(
        coefficients=coefficients,
        std_errors=std_errors,
        t_stats=t_stats,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        residuals=residuals,
        n_obs=n_obs,
        lags=lags,
    )


@dataclass(frozen=True)
class FactorRegression:
    """Time-series factor regression of excess returns on a set of factors.

    Attributes:
        factors: Regressor names, in the order they entered the design
            matrix (matches the column order of the factor frame, excluding
            the risk-free column).
        alpha: Per-period intercept (excess return unexplained by the
            factors).
        alpha_annual: ``alpha * periods_per_year``. This is an *arithmetic*
            annualization (simple scaling), not a compounded one.
        alpha_t: Newey-West t-statistic for ``alpha``.
        loadings: Factor loadings (betas), indexed by factor name.
        loading_t: Newey-West t-statistics for ``loadings``, indexed by
            factor name.
        std_errors: Newey-West per-period standard errors, indexed by
            ``("alpha", *factors)``.
        r_squared: Centered R-squared of the regression.
        adj_r_squared: R-squared adjusted for the number of parameters.
        n_obs: Number of aligned observations used in the fit.
        lags: Number of Bartlett-kernel lags used for the HAC covariance.
        periods_per_year: Periods-per-year assumption used for
            ``alpha_annual``.
        start: First date in the aligned sample.
        end: Last date in the aligned sample.
    """

    factors: tuple[str, ...]
    alpha: float
    alpha_annual: float
    alpha_t: float
    loadings: pd.Series
    loading_t: pd.Series
    std_errors: pd.Series
    r_squared: float
    adj_r_squared: float
    n_obs: int
    lags: int
    periods_per_year: int
    start: pd.Timestamp
    end: pd.Timestamp

    def as_row(self) -> dict[str, float | int | str]:
        """Flatten this result into a single dict, suitable for a CSV row.

        Returns:
            A flat mapping with keys ``alpha_annual``, ``alpha_t``, then for
            each factor ``"<factor>"`` and ``"<factor>_t"`` (in
            :attr:`factors` order), then ``r_squared``, ``adj_r_squared``,
            ``n_obs``, ``lags``, ``start`` and ``end`` (ISO date strings).
        """
        row: dict[str, float | int | str] = {
            "alpha_annual": self.alpha_annual,
            "alpha_t": self.alpha_t,
        }
        for name in self.factors:
            row[name] = float(self.loadings[name])
            row[f"{name}_t"] = float(self.loading_t[name])
        row["r_squared"] = self.r_squared
        row["adj_r_squared"] = self.adj_r_squared
        row["n_obs"] = self.n_obs
        row["lags"] = self.lags
        row["start"] = self.start.date().isoformat()
        row["end"] = self.end.date().isoformat()
        return row


def factor_regression(
    returns: pd.Series,
    factors: pd.DataFrame,
    periods_per_year: int = 252,
    lags: int | None = None,
    rf_column: str = "rf",
    subtract_rf: bool = True,
) -> FactorRegression:
    """Regress excess returns on a set of factors.

    Fits the time-series regression ``(r_t - rf_t) = alpha + sum_j b_j f_jt +
    e_t``, where ``r_t`` is ``returns`` and ``f_jt`` are the columns of
    ``factors`` other than ``rf_column``. The factor columns are used as-is:
    Mkt-RF is already an excess return and SMB/HML/RMW/CMA are long-short
    spreads, so ``rf_column`` is never subtracted from them.

    Set ``subtract_rf=False`` when ``returns`` is already an excess return,
    such as the difference between two strategies' returns: that is a
    zero-cost long-short position, and subtracting the risk-free rate again
    would bias alpha by its mean. The regression then gives the difference's
    alpha, which equals the difference of the two strategies' alphas over the
    same sample, together with its own standard error.

    Args:
        returns: Period return series to attribute, indexed by date.
        factors: A "factor frame": a DataFrame indexed by date with float
            columns in decimals, one of which is ``rf_column`` (the daily
            risk-free return) and the rest of which are regressors, used in
            column order.
        periods_per_year: Number of return periods per year, used to
            annualize alpha.
        lags: Number of Newey-West lags. If ``None``, uses
            :func:`newey_west_lags` on the number of aligned observations.
        rf_column: Name of the risk-free column in ``factors``.
        subtract_rf: Whether to subtract ``rf_column`` from ``returns``.
            ``False`` for a series that is already an excess return.

    Returns:
        A :class:`FactorRegression`.

    Raises:
        FrequencyMismatchError: If ``returns``' own inferred frequency (see
            :func:`~portfolio_bl.backtest.metrics.infer_periods_per_year`,
            applied to its non-null dates) differs from ``factors``' own
            inferred frequency. A :class:`ValueError` subclass.
        ValueError: If ``rf_column`` is not a column of ``factors``, or if
            fewer than ``len(factors columns) - 1 + 2`` observations remain
            after aligning dates and dropping missing values.
    """
    if rf_column not in factors.columns:
        raise ValueError(
            f"factors is missing the risk-free column {rf_column!r}; "
            f"available columns are {list(factors.columns)}."
        )
    factor_names = tuple(c for c in factors.columns if c != rf_column)

    returns_valid = returns.dropna()

    # A frequency mismatch (e.g. monthly returns against daily factors) would
    # otherwise join silently: the inner join below matches returns only to
    # the factor rows dated on exactly the same day, so a monthly return
    # would end up regressed on a single day's factor values per month --
    # meaningless, and not something the join's row count alone reveals.
    # Catch it here, before any joining happens.
    return_ppy = infer_periods_per_year(returns_valid.index)
    factor_ppy = infer_periods_per_year(factors.index)
    if return_ppy != factor_ppy:
        raise FrequencyMismatchError(
            f"returns has an inferred frequency of {return_ppy} period(s)/year but factors "
            f"has {factor_ppy}; they must match to regress meaningfully (the bundled "
            "Fama-French factors are daily, periods_per_year=252)."
        )

    missing_dates = returns_valid.index.difference(factors.index)
    if len(missing_dates) > 0:
        missing_sorted = missing_dates.sort_values()
        logger.warning(
            "%d return date(s) are absent from the factor frame and were dropped "
            "(first missing=%s, last missing=%s).",
            len(missing_sorted),
            missing_sorted[0].date(),
            missing_sorted[-1].date(),
        )

    joined = factors.join(returns.rename("__return__"), how="inner")
    aligned = joined.dropna(how="any").sort_index()
    if len(aligned) < len(joined):
        logger.warning(
            "%d row(s) with a NaN return or factor value were dropped.",
            len(joined) - len(aligned),
        )

    n_obs = len(aligned)
    min_obs = len(factor_names) + 2
    if n_obs < min_obs:
        raise ValueError(
            f"Only {n_obs} aligned observation(s) available; need at least "
            f"{min_obs} for {len(factor_names)} factor(s)."
        )

    excess = aligned["__return__"].to_numpy()
    if subtract_rf:
        excess = excess - aligned[rf_column].to_numpy()
    x_factors = aligned[list(factor_names)].to_numpy()
    x = np.column_stack([np.ones(n_obs), x_factors])

    if lags is None:
        lags = newey_west_lags(n_obs)

    fit = ols_newey_west(excess, x, lags)

    loadings = pd.Series(fit.coefficients[1:], index=factor_names)
    loading_t = pd.Series(fit.t_stats[1:], index=factor_names)
    std_errors = pd.Series(fit.std_errors, index=("alpha", *factor_names))

    return FactorRegression(
        factors=factor_names,
        alpha=float(fit.coefficients[0]),
        alpha_annual=float(fit.coefficients[0]) * periods_per_year,
        alpha_t=float(fit.t_stats[0]),
        loadings=loadings,
        loading_t=loading_t,
        std_errors=std_errors,
        r_squared=fit.r_squared,
        adj_r_squared=fit.adj_r_squared,
        n_obs=fit.n_obs,
        lags=fit.lags,
        periods_per_year=periods_per_year,
        start=aligned.index[0],
        end=aligned.index[-1],
    )


def _merge_ordered_keys(existing: list[str], new_keys: list[str]) -> list[str]:
    """Merge ``new_keys`` into ``existing``, preserving relative order.

    Keys already present in ``existing`` keep their position. A key seen for
    the first time is inserted immediately after the nearest earlier key from
    its own sequence that is already present (or at the front, if none are).
    Folding this over every row of a table therefore yields a "first
    appearance" column order even when later rows introduce columns (such as
    extra factors) that belong logically in the middle of the earlier order.

    Args:
        existing: Column order accumulated so far.
        new_keys: Keys from the next row, in that row's own order.

    Returns:
        The merged column order.
    """
    result = list(existing)
    anchor = -1
    for key in new_keys:
        if key in result:
            anchor = result.index(key)
        else:
            anchor += 1
            result.insert(anchor, key)
    return result


def attribution_table(
    returns_by_name: Mapping[str, pd.Series],
    factor_sets: Mapping[str, pd.DataFrame],
    periods_per_year: int = 252,
    lags: int | None = None,
    subtract_rf: bool = True,
) -> pd.DataFrame:
    """Build a factor-attribution table for several strategies and models.

    Runs :func:`factor_regression` for every (strategy, model) pair, in the
    input order of ``returns_by_name`` and ``factor_sets``, and stacks the
    results into a single table indexed by ``("strategy", "model")``.

    Args:
        returns_by_name: Mapping from strategy name to its return series.
        factor_sets: Mapping from model name (e.g. ``"ff3"``, ``"ff5"``) to
            its factor frame.
        periods_per_year: Passed through to :func:`factor_regression`.
        lags: Passed through to :func:`factor_regression`.
        subtract_rf: Passed through to :func:`factor_regression`; ``False``
            when every series is already an excess return.

    Returns:
        A DataFrame with a ``("strategy", "model")`` MultiIndex and one
        column per :meth:`FactorRegression.as_row` key. Columns are ordered
        by first appearance across all rows; a factor absent from a given
        model (e.g. RMW/CMA under an FF3 row when another row uses FF5) is
        ``NaN`` in that row.
    """
    row_data: dict[tuple[str, str], dict[str, float | int | str]] = {}
    column_order: list[str] = []
    for name, returns in returns_by_name.items():
        for model_name, factors in factor_sets.items():
            fit = factor_regression(
                returns,
                factors,
                periods_per_year=periods_per_year,
                lags=lags,
                subtract_rf=subtract_rf,
            )
            row = fit.as_row()
            row_data[(name, model_name)] = row
            column_order = _merge_ordered_keys(column_order, list(row.keys()))

    table = pd.DataFrame.from_dict(row_data, orient="index")
    table = table.reindex(columns=column_order)
    table.index = pd.MultiIndex.from_tuples(table.index, names=["strategy", "model"])
    return table
