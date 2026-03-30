from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from portfolio_bl.backtest.engine import BacktestResult, rolling_backtest
from portfolio_bl.backtest.metrics import infer_periods_per_year, summarize_strategy
from portfolio_bl.config import AppConfig
from portfolio_bl.data.disclosures import latest_portfolio_for_aliases, load_disclosures_csv
from portfolio_bl.data.prices import load_prices_csv, monthly_rebalance_dates, to_return_matrix
from portfolio_bl.models.black_litterman import (
    black_litterman_posterior,
    diagonal_omega_from_confidence,
    implied_equilibrium_returns,
)
from portfolio_bl.models.mean_variance import estimate_mean_cov, long_only_markowitz_weights

logger = logging.getLogger(__name__)


@dataclass
class CaseStudyResult:
    """Aggregated outputs from a single case-study run.

    Attributes:
        person_label: Human-readable label for the subject (e.g.
            ``'Warren Buffett'``).
        as_of_date: Disclosure snapshot date used to construct the universe.
        universe: Sorted list of tickers included in the backtest.
        strategy_results: Mapping from strategy name to its
            :class:`~portfolio_bl.backtest.engine.BacktestResult`.
        summary: DataFrame of performance metrics (strategies × metrics).
    """

    person_label: str
    as_of_date: pd.Timestamp
    universe: list[str]
    strategy_results: dict[str, BacktestResult]
    summary: pd.DataFrame


def _constant_weight_fn(weights: pd.Series):
    """Return a weight function that always returns the given fixed weights."""

    def _fn(_train: pd.DataFrame, _date: pd.Timestamp) -> pd.Series:
        return weights

    return _fn


def run_case_study(
    app_config: AppConfig,
    person_key: str,
    view_confidence: float | None = None,
) -> CaseStudyResult:
    """Run a three-strategy portfolio backtest for a disclosed portfolio.

    Loads disclosures and prices, builds the asset universe (intersection of
    disclosed tickers and available price history), then runs three rolling
    backtests:

    - **disclosed**: static weights held at the disclosed portfolio fractions.
    - **mean_variance**: rolling Markowitz MVO weights estimated from the
      lookback window.
    - **black_litterman**: rolling BL posterior weights combining equilibrium
      returns with sample-mean views.

    The ``view_confidence`` argument controls how strongly analyst views
    override the BL equilibrium prior. When not provided here, the value from
    ``app_config.backtest.view_confidence`` is used (set via YAML or the
    :class:`~portfolio_bl.config.BacktestConfig` default of 0.65).

    Args:
        app_config: Application configuration (paths, backtest hyper-params,
            case-study definitions).
        person_key: Key identifying the case study in
            ``app_config.case_studies``.
        view_confidence: Optional override for the BL view-confidence scalar.
            Falls back to ``app_config.backtest.view_confidence`` when
            ``None``.

    Returns:
        A :class:`CaseStudyResult` with the backtest outcomes and summary
        metrics for all three strategies.

    Raises:
        ValueError: If ``person_key`` is not found in the config, the universe
            intersection has fewer than 2 assets, or other data validation
            errors occur in upstream loaders.
    """
    if person_key not in app_config.case_studies:
        keys = ", ".join(sorted(app_config.case_studies))
        raise ValueError(f"Unknown person key '{person_key}'. Available: {keys}")

    confidence = (
        view_confidence if view_confidence is not None else app_config.backtest.view_confidence
    )
    case_cfg = app_config.case_studies[person_key]
    logger.info("Running case study: %s (view_confidence=%.3f)", case_cfg.person_label, confidence)

    disclosures = load_disclosures_csv(app_config.disclosures_path)
    latest_disclosed, as_of_date = latest_portfolio_for_aliases(
        disclosures, case_cfg.disclosure_aliases
    )

    prices = load_prices_csv(app_config.prices_path)
    returns = to_return_matrix(prices)

    universe = sorted(set(latest_disclosed["ticker"]).intersection(returns.columns))
    if len(universe) < 2:
        raise ValueError("Universe intersection has fewer than 2 assets.")

    logger.info("Universe: %d asset(s) — %s", len(universe), ", ".join(universe))

    returns = returns[universe].dropna(how="all")
    market_weights = latest_disclosed.set_index("ticker")["weight"].reindex(universe).fillna(0.0)
    market_weights = market_weights / market_weights.sum()

    rebalance_dates = monthly_rebalance_dates(
        returns.index, frequency=app_config.backtest.rebalance_frequency
    )
    lookback = app_config.backtest.lookback_periods

    def _drop_nan_tickers(train_returns: pd.DataFrame) -> pd.DataFrame:
        """Drop tickers that have no data yet (pre-IPO NaN columns)."""
        return train_returns.dropna(axis=1)

    def mvo_fn(train_returns: pd.DataFrame, _date: pd.Timestamp) -> pd.Series:
        valid = _drop_nan_tickers(train_returns)
        if valid.shape[1] < 2:
            return pd.Series(dtype=float)
        mu, cov = estimate_mean_cov(valid)
        return long_only_markowitz_weights(mu, cov)

    def bl_fn(train_returns: pd.DataFrame, _date: pd.Timestamp) -> pd.Series:
        valid = _drop_nan_tickers(train_returns)
        if valid.shape[1] < 2:
            return pd.Series(dtype=float)
        valid_tickers = list(valid.columns)
        mu, cov = estimate_mean_cov(valid)

        valid_weights = market_weights.reindex(valid_tickers).fillna(0.0)
        if valid_weights.sum() > 0:
            valid_weights = valid_weights / valid_weights.sum()
        else:
            valid_weights = pd.Series(1.0 / len(valid_tickers), index=valid_tickers)

        pi = implied_equilibrium_returns(
            covariance=cov,
            market_weights=valid_weights,
            risk_aversion=app_config.backtest.risk_aversion,
        )

        p = np.eye(len(valid_tickers), dtype=float)
        q = mu.to_numpy(dtype=float)
        omega = diagonal_omega_from_confidence(
            covariance=cov.to_numpy(dtype=float),
            p_matrix=p,
            tau=app_config.backtest.tau,
            confidence=confidence,
        )

        posterior_mu, posterior_cov = black_litterman_posterior(
            pi=pi,
            covariance=cov.to_numpy(dtype=float),
            p_matrix=p,
            q_views=q,
            tau=app_config.backtest.tau,
            omega=omega,
        )

        posterior_mu_s = pd.Series(posterior_mu, index=valid_tickers)
        posterior_cov_df = pd.DataFrame(posterior_cov, index=valid_tickers, columns=valid_tickers)
        return long_only_markowitz_weights(posterior_mu_s, posterior_cov_df)

    disclosed_fn = _constant_weight_fn(market_weights)

    logger.info("Running backtests for all three strategies.")
    strategy_results: dict[str, BacktestResult] = {
        "disclosed": rolling_backtest(returns, rebalance_dates, lookback, disclosed_fn),
        "mean_variance": rolling_backtest(returns, rebalance_dates, lookback, mvo_fn),
        "black_litterman": rolling_backtest(returns, rebalance_dates, lookback, bl_fn),
    }

    periods_per_year = infer_periods_per_year(returns.index)
    summary = pd.DataFrame(
        {
            name: summarize_strategy(
                result.returns,
                result.weight_history,
                periods_per_year=periods_per_year,
            )
            for name, result in strategy_results.items()
        }
    ).T

    summary.index.name = "strategy"

    logger.info("Case study complete for %s.", case_cfg.person_label)
    return CaseStudyResult(
        person_label=case_cfg.person_label,
        as_of_date=as_of_date,
        universe=universe,
        strategy_results=strategy_results,
        summary=summary,
    )
