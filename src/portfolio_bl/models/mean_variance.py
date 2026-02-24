from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def estimate_mean_cov(
    returns: pd.DataFrame, min_observations: int = 6
) -> tuple[pd.Series, pd.DataFrame]:
    """Estimate sample mean returns and covariance from a returns matrix.

    Args:
        returns: Period return matrix with tickers as columns and observations
            as rows.
        min_observations: Minimum number of rows required; raises
            :class:`ValueError` if fewer are present.

    Returns:
        A tuple ``(mu, cov)`` where ``mu`` is a Series of per-ticker mean
        returns and ``cov`` is the sample covariance DataFrame.

    Raises:
        ValueError: If fewer than ``min_observations`` rows are present, or if
            the covariance matrix is entirely NaN.
    """
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
    """Compute long-only Markowitz portfolio weights.

    Solves the unconstrained mean-variance system ``cov_reg w = mu`` and clips
    negative weights to zero (long-only projection). If all resulting weights
    are non-positive the solver has returned a degenerate solution and the
    portfolio falls back to equal weights; a warning is emitted in this case.

    A small ridge term is added to the covariance matrix before solving to
    improve numerical stability for near-singular matrices.

    Args:
        expected_returns: Per-ticker expected returns.
        covariance: Asset covariance matrix whose index and columns match
            ``expected_returns.index``.
        ridge: Ridge regularisation added to the diagonal of the covariance
            matrix.

    Returns:
        A Series of portfolio weights that sum to 1 and are non-negative.
    """
    tickers = list(expected_returns.index)
    mu = expected_returns.to_numpy(dtype=float)
    cov = covariance.loc[tickers, tickers].to_numpy(dtype=float)

    cov_reg = cov + np.eye(len(tickers)) * ridge
    raw = np.linalg.solve(cov_reg, mu)
    raw = np.clip(raw, 0.0, None)

    if raw.sum() <= 0:
        logger.warning(
            "Markowitz solver returned all non-positive weights; falling back to equal-weight."
        )
        weights = np.ones(len(tickers), dtype=float) / len(tickers)
    else:
        weights = raw / raw.sum()

    return pd.Series(weights, index=tickers)
