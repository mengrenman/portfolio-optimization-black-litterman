from __future__ import annotations

import numpy as np
import pandas as pd



def implied_equilibrium_returns(
    covariance: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float,
) -> np.ndarray:
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
