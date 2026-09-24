from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from portfolio_bl.models._numeric import mean_diagonal, relative_ridge

logger = logging.getLogger(__name__)

# Floor on each view's uncertainty, as a fraction of that view's own projected
# prior variance. It keeps Omega invertible at confidence == 1 without letting
# an absolute constant decide how far a fully-trusted view moves the posterior.
OMEGA_RELATIVE_FLOOR = 1e-10


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
    confidence: float | Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Construct a diagonal view-uncertainty matrix Ω from view confidences.

    Scales each view's projected prior variance, the diagonal of P(τΣ)Pᵀ, by
    (1−c)/c, where c is the confidence in that view. Higher confidence
    (c → 1) shrinks Ω, placing more weight on the view relative to the
    equilibrium prior.

    Args:
        covariance: Asset covariance matrix Σ as a 2-D numpy array (n × n).
        p_matrix: View-picking matrix P (k × n) mapping assets to views.
        tau: Prior uncertainty scalar (typically 0.01–0.10).
        confidence: Either one confidence applied to every view, or a
            sequence of length k with one confidence per view. Values are
            clipped to ``[1e-3, 1.0]``; a confidence at or below zero is
            therefore treated as 1e-3 rather than rejected, and the posterior
            still moves slightly toward such a view.

    Returns:
        A diagonal matrix Ω of shape (k × k).

    Notes:
        A view whose projected prior variance ``diag(P(τΣ)Pᵀ)`` is zero or
        negative carries no scale of its own. Its entry is replaced with the
        mean of the usable projected variances, so the substitute still tracks
        the data's units, and a warning is logged. This happens for an absolute
        view on a constant-price series, or a relative view between two
        perfectly correlated ones.

        Such a view does **not** realize its configured fraction ``c``: the
        posterior mean moves only ``ridge * c/(1-c)`` of the way toward it.
        That is not the same as the view being ignored. The asset's posterior
        variance is ridge-sized too, and a mean-variance optimizer takes the
        ratio, so the two cancel and ``c`` remains a powerful dial on the
        allocation. A zero-variance asset is risk-free, and asserting a
        positive return for it will pull most of the portfolio into it.

    Raises:
        ValueError: If a per-view confidence sequence does not have exactly
            one entry per row of ``p_matrix``.
    """
    p_matrix = np.asarray(p_matrix, dtype=float)
    n_views = p_matrix.shape[0]

    conf = np.atleast_1d(np.asarray(confidence, dtype=float))
    if conf.size == 1:
        conf = np.full(n_views, conf.item(), dtype=float)
    elif conf.shape != (n_views,):
        raise ValueError(
            f"Expected one confidence per view ({n_views}), got shape {conf.shape}."
        )
    conf = np.clip(conf, 1e-3, 1.0)

    projected = p_matrix @ (tau * np.asarray(covariance, dtype=float)) @ p_matrix.T
    diag = np.diag(projected) if n_views > 0 else np.zeros(0, dtype=float)

    # A view with no prior variance of its own has no scale to derive Omega
    # from. Substitute the mean of the usable projected variances rather than
    # an absolute constant, so the guard carries the data's units and the
    # weights stay invariant to whether returns are daily, monthly or annual.
    degenerate = ~(np.isfinite(diag) & (diag > 0.0))
    if degenerate.any():
        usable = diag[~degenerate]
        if usable.size:
            substitute = float(np.mean(usable))
        else:
            # No view carries a usable scale, so take one from the prior itself.
            substitute = tau * mean_diagonal(np.asarray(covariance, dtype=float))
        logger.warning(
            "%d view(s) have a non-positive projected prior variance; substituting %.3e. "
            "Their confidence no longer maps to the documented fraction, and because such "
            "an asset is risk-free the view may pull most of the portfolio into it.",
            int(degenerate.sum()),
            substitute,
        )
        diag = np.where(degenerate, substitute, diag)

    # Higher confidence -> lower view uncertainty.
    scale = (1.0 - conf) / conf
    omega_diag = diag * scale

    # At confidence == 1 the scale is exactly 0, which would make Omega
    # singular. Floor it relative to each view's own projected variance so the
    # limit stays well defined and calibration is preserved.
    return np.diag(np.maximum(omega_diag, diag * OMEGA_RELATIVE_FLOOR))


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
    term is added to Σ and Ω to guard against singularity. When ``p_matrix``
    has no rows (no views) the posterior mean equals ``pi`` and the posterior
    covariance equals (1 + τ)Σ, which is the same formula with the view term
    removed.

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
        ridge: Relative ridge regularization guarding against singularity.
            It is a dimensionless fraction: the amount added to the diagonal
            added to each diagonal entry of Σ is ``ridge`` times that asset's
            own variance, and the amount added to Ω's k-th entry is ``ridge``
            times that view's projected prior variance ``diag(P(τΣ)Pᵀ)``.
            Scaling it this way keeps the model's behavior identical whether
            returns are daily, monthly or annual, and keeps it proportionate
            when variances span orders of magnitude.

    Returns:
        A tuple ``(posterior_mean, posterior_covariance)`` where
        ``posterior_mean`` has shape (n,) and ``posterior_covariance`` has
        shape (n × n).
    """
    pi = np.asarray(pi, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    p = np.asarray(p_matrix, dtype=float)
    q = np.asarray(q_views, dtype=float)

    sigma = sigma + np.diag(relative_ridge(sigma, ridge))

    if p.ndim != 2 or p.shape[1] != sigma.shape[0]:
        raise ValueError(f"p_matrix must have shape (k, {sigma.shape[0]}), got {p.shape}.")
    if p.shape[0] == 0:
        # No views: the posterior collapses to the prior and M⁻¹ = τΣ.
        return pi.copy(), sigma * (1.0 + tau)

    projected = p @ (tau * sigma) @ p.T
    if omega is None:
        omega = np.diag(np.diag(projected))
    omega = np.asarray(omega, dtype=float)
    omega = omega + np.diag(relative_ridge(projected, ridge))

    tau_sigma_inv = np.linalg.pinv(tau * sigma)
    omega_inv = np.linalg.pinv(omega)

    middle = tau_sigma_inv + p.T @ omega_inv @ p
    middle_inv = np.linalg.pinv(middle)

    posterior_mean = middle_inv @ (tau_sigma_inv @ pi + p.T @ omega_inv @ q)
    posterior_covariance = sigma + middle_inv

    return posterior_mean, posterior_covariance
