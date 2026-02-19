from __future__ import annotations

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


@dataclass
class CaseStudyResult:
    person_label: str
    as_of_date: pd.Timestamp
    universe: list[str]
    strategy_results: dict[str, BacktestResult]
    summary: pd.DataFrame



def _constant_weight_fn(weights: pd.Series):
    def _fn(_train: pd.DataFrame, _date: pd.Timestamp) -> pd.Series:
        return weights

    return _fn



def run_case_study(app_config: AppConfig, person_key: str, view_confidence: float = 0.65) -> CaseStudyResult:
    if person_key not in app_config.case_studies:
        keys = ", ".join(sorted(app_config.case_studies))
        raise ValueError(f"Unknown person key '{person_key}'. Available: {keys}")

    case_cfg = app_config.case_studies[person_key]

    disclosures = load_disclosures_csv(app_config.disclosures_path)
    latest_disclosed, as_of_date = latest_portfolio_for_aliases(
        disclosures, case_cfg.disclosure_aliases
    )

    prices = load_prices_csv(app_config.prices_path)
    returns = to_return_matrix(prices)

    universe = sorted(set(latest_disclosed["ticker"]).intersection(returns.columns))
    if len(universe) < 2:
        raise ValueError("Universe intersection has fewer than 2 assets.")

    returns = returns[universe].dropna(how="all")
    market_weights = latest_disclosed.set_index("ticker")["weight"].reindex(universe).fillna(0.0)
    market_weights = market_weights / market_weights.sum()

    rebalance_dates = monthly_rebalance_dates(
        returns.index, frequency=app_config.backtest.rebalance_frequency
    )
    lookback = app_config.backtest.lookback_periods

    def mvo_fn(train_returns: pd.DataFrame, _date: pd.Timestamp) -> pd.Series:
        mu, cov = estimate_mean_cov(train_returns)
        return long_only_markowitz_weights(mu, cov)

    def bl_fn(train_returns: pd.DataFrame, _date: pd.Timestamp) -> pd.Series:
        mu, cov = estimate_mean_cov(train_returns)

        pi = implied_equilibrium_returns(
            covariance=cov,
            market_weights=market_weights,
            risk_aversion=app_config.backtest.risk_aversion,
        )

        p = np.eye(len(universe), dtype=float)
        q = mu.to_numpy(dtype=float)
        omega = diagonal_omega_from_confidence(
            covariance=cov.to_numpy(dtype=float),
            p_matrix=p,
            tau=app_config.backtest.tau,
            confidence=view_confidence,
        )

        posterior_mu, posterior_cov = black_litterman_posterior(
            pi=pi,
            covariance=cov.to_numpy(dtype=float),
            p_matrix=p,
            q_views=q,
            tau=app_config.backtest.tau,
            omega=omega,
        )

        posterior_mu_s = pd.Series(posterior_mu, index=universe)
        posterior_cov_df = pd.DataFrame(posterior_cov, index=universe, columns=universe)
        return long_only_markowitz_weights(posterior_mu_s, posterior_cov_df)

    disclosed_fn = _constant_weight_fn(market_weights)

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

    return CaseStudyResult(
        person_label=case_cfg.person_label,
        as_of_date=as_of_date,
        universe=universe,
        strategy_results=strategy_results,
        summary=summary,
    )
