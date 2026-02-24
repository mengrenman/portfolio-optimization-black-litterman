from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_bl.models.black_litterman import (
    black_litterman_posterior,
    diagonal_omega_from_confidence,
    implied_equilibrium_returns,
)
from portfolio_bl.models.mean_variance import long_only_markowitz_weights


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "MSFT", "XOM"]

COV_DF = pd.DataFrame(
    [[0.04, 0.01, 0.00], [0.01, 0.05, 0.01], [0.00, 0.01, 0.03]],
    index=TICKERS,
    columns=TICKERS,
)

W_MKT = pd.Series([0.5, 0.3, 0.2], index=TICKERS)


# ---------------------------------------------------------------------------
# implied_equilibrium_returns
# ---------------------------------------------------------------------------


def test_implied_equilibrium_returns_shape_and_finite() -> None:
    pi = implied_equilibrium_returns(COV_DF, W_MKT, risk_aversion=2.5)
    assert pi.shape == (3,)
    assert np.isfinite(pi).all()


def test_implied_equilibrium_returns_scales_with_risk_aversion() -> None:
    pi_low = implied_equilibrium_returns(COV_DF, W_MKT, risk_aversion=1.0)
    pi_high = implied_equilibrium_returns(COV_DF, W_MKT, risk_aversion=4.0)
    np.testing.assert_allclose(pi_high, pi_low * 4.0)


def test_implied_equilibrium_returns_missing_ticker_filled_zero() -> None:
    """Tickers not in market_weights should contribute zero to the result."""
    partial_weights = pd.Series([1.0], index=["AAPL"])
    pi_partial = implied_equilibrium_returns(COV_DF, partial_weights, risk_aversion=2.5)
    pi_full = implied_equilibrium_returns(
        COV_DF, pd.Series([1.0, 0.0, 0.0], index=TICKERS), risk_aversion=2.5
    )
    np.testing.assert_allclose(pi_partial, pi_full)


# ---------------------------------------------------------------------------
# diagonal_omega_from_confidence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("confidence", [0.1, 0.5, 0.9, 0.99])
def test_omega_is_diagonal_positive_definite(confidence: float) -> None:
    p = np.eye(len(TICKERS))
    omega = diagonal_omega_from_confidence(COV_DF.to_numpy(), p, tau=0.05, confidence=confidence)
    assert omega.shape == (3, 3)
    assert (np.diag(omega) > 0).all()
    # Off-diagonals must be zero for a diagonal matrix.
    off_diag = omega - np.diag(np.diag(omega))
    np.testing.assert_array_equal(off_diag, 0.0)


def test_higher_confidence_produces_smaller_omega() -> None:
    p = np.eye(len(TICKERS))
    omega_low = diagonal_omega_from_confidence(COV_DF.to_numpy(), p, tau=0.05, confidence=0.3)
    omega_high = diagonal_omega_from_confidence(COV_DF.to_numpy(), p, tau=0.05, confidence=0.9)
    assert np.diag(omega_high).sum() < np.diag(omega_low).sum()


# ---------------------------------------------------------------------------
# black_litterman_posterior
# ---------------------------------------------------------------------------


def test_black_litterman_shapes_and_finite_values() -> None:
    pi = implied_equilibrium_returns(COV_DF, W_MKT, risk_aversion=2.5)
    p = np.eye(len(TICKERS))
    q = np.array([0.11, 0.09, 0.05])
    omega = diagonal_omega_from_confidence(COV_DF.to_numpy(), p, tau=0.05, confidence=0.7)

    mu_post, cov_post = black_litterman_posterior(
        pi=pi, covariance=COV_DF.to_numpy(), p_matrix=p, q_views=q, tau=0.05, omega=omega
    )

    assert mu_post.shape == (3,)
    assert cov_post.shape == (3, 3)
    assert np.isfinite(mu_post).all()
    assert np.isfinite(cov_post).all()


@pytest.mark.parametrize("tau", [0.01, 0.05, 0.25])
def test_posterior_finite_across_tau_values(tau: float) -> None:
    pi = implied_equilibrium_returns(COV_DF, W_MKT, risk_aversion=2.5)
    p = np.eye(len(TICKERS))
    q = np.array([0.10, 0.08, 0.04])

    mu_post, cov_post = black_litterman_posterior(
        pi=pi, covariance=COV_DF.to_numpy(), p_matrix=p, q_views=q, tau=tau
    )

    assert np.isfinite(mu_post).all(), f"Non-finite posterior mean at tau={tau}"
    assert np.isfinite(cov_post).all(), f"Non-finite posterior cov at tau={tau}"


def test_posterior_blends_prior_and_views() -> None:
    """With very low confidence the posterior should be close to the prior."""
    pi = implied_equilibrium_returns(COV_DF, W_MKT, risk_aversion=2.5)
    p = np.eye(len(TICKERS))
    # Views far from the prior.
    q = np.array([0.50, 0.50, 0.50])

    omega_low_conf = diagonal_omega_from_confidence(
        COV_DF.to_numpy(), p, tau=0.05, confidence=0.001
    )
    omega_high_conf = diagonal_omega_from_confidence(
        COV_DF.to_numpy(), p, tau=0.05, confidence=0.999
    )

    mu_low, _ = black_litterman_posterior(
        pi=pi, covariance=COV_DF.to_numpy(), p_matrix=p, q_views=q, tau=0.05, omega=omega_low_conf
    )
    mu_high, _ = black_litterman_posterior(
        pi=pi, covariance=COV_DF.to_numpy(), p_matrix=p, q_views=q, tau=0.05, omega=omega_high_conf
    )

    # High-confidence posterior should be closer to q (the views).
    dist_low_to_q = np.linalg.norm(mu_low - q)
    dist_high_to_q = np.linalg.norm(mu_high - q)
    assert dist_high_to_q < dist_low_to_q


def test_posterior_with_none_omega_defaults_gracefully() -> None:
    pi = implied_equilibrium_returns(COV_DF, W_MKT, risk_aversion=2.5)
    p = np.eye(len(TICKERS))
    q = np.array([0.10, 0.08, 0.04])

    mu_post, cov_post = black_litterman_posterior(
        pi=pi, covariance=COV_DF.to_numpy(), p_matrix=p, q_views=q, tau=0.05, omega=None
    )

    assert np.isfinite(mu_post).all()
    assert np.isfinite(cov_post).all()


# ---------------------------------------------------------------------------
# long_only_markowitz_weights
# ---------------------------------------------------------------------------


def test_long_only_markowitz_weights_are_valid() -> None:
    expected = pd.Series([0.10, 0.08, 0.04], index=TICKERS)
    w = long_only_markowitz_weights(expected_returns=expected, covariance=COV_DF)

    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()


def test_markowitz_fallback_to_equal_weight_when_degenerate() -> None:
    """Negative expected returns should trigger the equal-weight fallback."""
    expected = pd.Series([-0.10, -0.08, -0.04], index=TICKERS)
    w = long_only_markowitz_weights(expected_returns=expected, covariance=COV_DF)

    expected_ew = 1.0 / len(TICKERS)
    np.testing.assert_allclose(w.values, expected_ew, atol=1e-9)


@pytest.mark.parametrize("risk_aversion", [1.0, 2.5, 5.0])
def test_bl_weights_valid_across_risk_aversion(risk_aversion: float) -> None:
    pi = implied_equilibrium_returns(COV_DF, W_MKT, risk_aversion=risk_aversion)
    p = np.eye(len(TICKERS))
    q = np.array([0.08, 0.07, 0.04])
    omega = diagonal_omega_from_confidence(COV_DF.to_numpy(), p, tau=0.05, confidence=0.65)

    mu_post, cov_post = black_litterman_posterior(
        pi=pi, covariance=COV_DF.to_numpy(), p_matrix=p, q_views=q, tau=0.05, omega=omega
    )

    mu_s = pd.Series(mu_post, index=TICKERS)
    cov_df = pd.DataFrame(cov_post, index=TICKERS, columns=TICKERS)
    w = long_only_markowitz_weights(mu_s, cov_df)

    assert np.isclose(w.sum(), 1.0), f"Weights don't sum to 1 at risk_aversion={risk_aversion}"
    assert (w >= 0).all()
