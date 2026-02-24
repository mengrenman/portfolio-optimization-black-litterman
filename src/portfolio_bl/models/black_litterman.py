from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def implied_equilibrium_returns(
    covariance: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float,
) -> np.ndarray:
    """Compute Black-Litterman implied equilibrium excess returns.

    Implements π = λ Σ w_mkt, where λ is the risk-aversion coefficient, Σ is
    the covariance matrix, and w_mkt are the market-cap weights. The result
    represents the expected returns that are consistent with the market
    portfolio being mean-variance efficient.

    Args:
        covariance: Asset covariance matrix (n × n) as a DataFrame whose
            index and columns are ticker strings.
        market_weights: Market-cap weights indexed by ticker. Tickers not
            present in ``covariance`` are filled with zero.
        risk_aversion: Risk-aversion coefficient λ (typically 2–4).

    Returns:
        A 1-D array of equilibrium excess returns, shape (n,).
    """
    tickers = list(covariance.columns)
    sigma = covariance.loc[tickers, tickers].to_numpy(dtype=float)
    w_mkt = market_weights.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
    return risk_aversion * sigma @ w_mkt


def diagonal_omega_from_confidence(
    covariance: np.ndarray,
    p_matrix: np.ndarray,
    tau: float,
    confidence: float,
) -> np.ndarray:
    """Construct a diagonal view-uncertainty matrix Ω from a confidence scalar.

    Scales the projected prior covariance P(τΣ)Pᵀ by (1−c)/c, where c is the
    analyst's confidence level. Higher confidence (c → 1) shrinks Ω, placing
    more weight on the views relative to the equilibrium prior.

    Args:
        covariance: Asset covariance matrix Σ as a 2-D numpy array (n × n).
        p_matrix: View-picking matrix P (k × n) mapping assets to views.
        tau: Prior uncertainty scalar (typically 0.01–0.10).
        confidence: Analyst confidence in [1e-3, 1.0]. Clipped to this range.

    Returns:
        A diagonal matrix Ω of shape (k × k).
    """
    confidence = float(np.clip(confidence, 1e-3, 1.0))
    projected = p_matrix @ (tau * covariance) @ p_matrix.T
    diag = np.diag(projected)
    diag = np.where(diag <= 0, 1e-8, diag)

    # Higher confidence -> lower view uncertainty.
    scale = (1.0 - confidence) / confidence
    return np.diag(diag * scale)


def black_litterman_posterior(
    pi: np.ndarray,
    covariance: np.ndarray,
    p_matrix: np.ndarray,
    q_views: np.ndarray,
    tau: float,
    omega: np.ndarray | None = None,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Black-Litterman posterior mean and covariance.

    Combines the equilibrium prior π with analyst views (P, q, Ω) using
    Bayes' theorem. The posterior formula is::

        M    = (τΣ)⁻¹ + PᵀΩ⁻¹P
        μ_BL = M⁻¹ [(τΣ)⁻¹ π + PᵀΩ⁻¹ q]
        Σ_BL = Σ + M⁻¹

    Pseudo-inverses are used throughout for numerical stability. A small ridge
    term is added to Σ and Ω to guard against singularity.

    Args:
        pi: Equilibrium excess returns, shape (n,).
        covariance: Asset covariance matrix Σ (n × n).
        p_matrix: View-picking matrix P (k × n). For per-asset absolute views
            use P = I (identity). Relative views can be expressed by setting
            rows to ±1 for long/short pairs.
        q_views: View return vector q, shape (k,).
        tau: Prior uncertainty scalar. Smaller values trust the equilibrium
            prior more strongly.
        omega: View-uncertainty matrix Ω (k × k). If ``None``, defaults to the
            diagonal of P(τΣ)Pᵀ.
        ridge: Ridge regularisation added to Σ and Ω to prevent singularity.

    Returns:
        A tuple ``(posterior_mean, posterior_covariance)`` where
        ``posterior_mean`` has shape (n,) and ``posterior_covariance`` has
        shape (n × n).
    """
    sigma = np.asarray(covariance, dtype=float)
    p = np.asarray(p_matrix, dtype=float)
    q = np.asarray(q_views, dtype=float)

    sigma = sigma + np.eye(sigma.shape[0]) * ridge

    if omega is None:
        omega = np.diag(np.diag(p @ (tau * sigma) @ p.T))
    omega = np.asarray(omega, dtype=float) + np.eye(omega.shape[0]) * ridge

    tau_sigma_inv = np.linalg.pinv(tau * sigma)
    omega_inv = np.linalg.pinv(omega)

    middle = tau_sigma_inv + p.T @ omega_inv @ p
    middle_inv = np.linalg.pinv(middle)

    posterior_mean = middle_inv @ (tau_sigma_inv @ pi + p.T @ omega_inv @ q)
    posterior_covariance = sigma + middle_inv

    return posterior_mean, posterior_covariance
