from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_bl.models.black_litterman import (
    black_litterman_posterior,
    diagonal_omega_from_confidence,
    implied_equilibrium_returns,
)
from portfolio_bl.models.mean_variance import long_only_markowitz_weights



def test_black_litterman_shapes_and_finite_values() -> None:
    tickers = ["AAPL", "MSFT", "XOM"]
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.00], [0.01, 0.05, 0.01], [0.00, 0.01, 0.03]],
        index=tickers,
        columns=tickers,
    )
    w_mkt = pd.Series([0.5, 0.3, 0.2], index=tickers)

    pi = implied_equilibrium_returns(covariance=cov, market_weights=w_mkt, risk_aversion=2.5)
    p = np.eye(len(tickers))
    q = np.array([0.11, 0.09, 0.05])
    omega = diagonal_omega_from_confidence(cov.to_numpy(), p, tau=0.05, confidence=0.7)

    mu_post, cov_post = black_litterman_posterior(
        pi=pi,
        covariance=cov.to_numpy(),
        p_matrix=p,
        q_views=q,
        tau=0.05,
        omega=omega,
    )

    assert mu_post.shape == (3,)
    assert cov_post.shape == (3, 3)
    assert np.isfinite(mu_post).all()
    assert np.isfinite(cov_post).all()



def test_long_only_markowitz_weights_are_valid() -> None:
    tickers = ["AAPL", "MSFT", "XOM"]
    expected = pd.Series([0.10, 0.08, 0.04], index=tickers)
    cov = pd.DataFrame(
        [[0.04, 0.01, 0.00], [0.01, 0.05, 0.01], [0.00, 0.01, 0.03]],
        index=tickers,
        columns=tickers,
    )

    w = long_only_markowitz_weights(expected_returns=expected, covariance=cov)

    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()
