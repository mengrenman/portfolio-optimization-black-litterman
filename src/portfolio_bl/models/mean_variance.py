from __future__ import annotations

import numpy as np
import pandas as pd



def estimate_mean_cov(returns: pd.DataFrame, min_observations: int = 6) -> tuple[pd.Series, pd.DataFrame]:
    if returns.shape[0] < min_observations:
        raise ValueError(
            f"Need at least {min_observations} observations, got {returns.shape[0]}."
        )

    mu = returns.mean()
    cov = returns.cov()

    if cov.isna().all().all():
        raise ValueError("Covariance matrix is invalid (all NaN).")

    cov = cov.fillna(0.0)
    return mu, cov



def long_only_markowitz_weights(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    ridge: float = 1e-6,
) -> pd.Series:
    tickers = list(expected_returns.index)
    mu = expected_returns.to_numpy(dtype=float)
    cov = covariance.loc[tickers, tickers].to_numpy(dtype=float)

    cov_reg = cov + np.eye(len(tickers)) * ridge
    raw = np.linalg.solve(cov_reg, mu)
    raw = np.clip(raw, 0.0, None)

    if raw.sum() <= 0:
        weights = np.ones(len(tickers), dtype=float) / len(tickers)
    else:
        weights = raw / raw.sum()

    return pd.Series(weights, index=tickers)
