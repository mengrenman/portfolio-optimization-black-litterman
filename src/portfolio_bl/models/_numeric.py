"""Small numerical helpers shared by the model modules."""

from __future__ import annotations

import numpy as np


def mean_diagonal(matrix: np.ndarray, fallback: float = 1.0) -> float:
    """Return the mean absolute diagonal entry, the overall scale of a matrix.

    Args:
        matrix: A square matrix, or an empty array.
        fallback: Value returned when the matrix is empty or its diagonal is
            unusable (all zero, NaN or infinite).

    Returns:
        The mean of the absolute diagonal entries, or ``fallback``.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0 or matrix.ndim != 2:
        return fallback
    scale = float(np.mean(np.abs(np.diag(matrix))))
    return scale if np.isfinite(scale) and scale > 0.0 else fallback


def relative_ridge(matrix: np.ndarray, ridge: float) -> np.ndarray:
    """Per-entry ridge sized to each diagonal element of ``matrix``.

    Regularisation added to a covariance-like matrix must be expressed
    relative to that matrix's own scale, otherwise an absolute constant means
    something different for daily returns (variances near 1e-4) than for
    monthly or annual ones, and it silently changes the model's behaviour with
    the data frequency.

    Scaling by each diagonal entry rather than by one matrix-wide average also
    keeps the perturbation proportionate when variances span orders of
    magnitude, as they do when bond funds sit alongside a volatile equity.
    Entries whose variance is zero or unusable fall back to the matrix's mean
    diagonal, so an exactly singular or all-zero matrix is still regularised.

    Args:
        matrix: Square covariance-like matrix (n x n).
        ridge: Dimensionless fraction, for example 1e-6.

    Returns:
        A 1-D array of length n to add to the matrix's diagonal.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0 or matrix.ndim != 2:
        return np.zeros(0, dtype=float)
    diag = np.abs(np.diag(matrix))
    usable = np.isfinite(diag) & (diag > 0.0)
    return np.where(usable, diag, mean_diagonal(matrix)) * ridge
