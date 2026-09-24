"""Analyst views expressed as rows of a Black-Litterman pick matrix."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class View:
    """One analyst view on a linear combination of asset returns.

    A view is a single row of the Black-Litterman pick matrix P together with
    its target return q. Two conventions are supported:

    - **Absolute view**: one ticker with coefficient ``1.0``. "AAPL will
      return 8% per year" is ``View({"AAPL": 1.0}, 0.08)``.
    - **Relative view**: positive coefficients on the outperformers and
      negative coefficients on the underperformers. "CVX will beat OXY by
      2% per year" is ``View({"CVX": 1.0, "OXY": -1.0}, 0.02)``. By
      convention the positive and negative sides each sum to one in absolute
      value, but this is not enforced.

    Attributes:
        assets: Mapping from ticker to its coefficient in the pick-matrix
            row. Tickers are upper-cased and stripped on construction.
        annual_return: Expected annualized arithmetic return of the
            combination, as a fraction (``0.05`` for 5%). It is divided by
            the number of return periods per year before entering the
            posterior so that it lives on the same scale as the data.
        confidence: Optional confidence in this view, in ``(0, 1]``. ``None``
            means "use the global ``view_confidence`` setting".
        label: Optional human-readable name used in log messages.
    """

    assets: Mapping[str, float]
    annual_return: float
    confidence: float | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.assets, Mapping) or len(self.assets) == 0:
            raise ValueError(
                "View 'assets' must be a non-empty mapping of ticker to coefficient; "
                "a view must reference at least one asset."
            )

        normalized: dict[str, float] = {}
        for raw_ticker, raw_coef in self.assets.items():
            if not isinstance(raw_ticker, str):
                # YAML 1.1 resolves bare ON/OFF/YES/NO/TRUE/FALSE keys to booleans, which
                # would silently become a view on the ticker "TRUE" or "FALSE".
                raise ValueError(  # noqa: TRY004 - View validation raises ValueError
                    f"View asset ticker must be a string, got {raw_ticker!r}. Quote tickers "
                    "that YAML reads as booleans, for example 'ON', 'OFF', 'YES' or 'NO'."
                )
            ticker = raw_ticker.strip().upper()
            if not ticker:
                raise ValueError("View asset tickers must be non-empty strings.")
            try:
                coef = float(raw_coef)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"View coefficient for {ticker!r} must be numeric, got {raw_coef!r}."
                ) from exc
            if not math.isfinite(coef):
                raise ValueError(f"View coefficient for {ticker!r} must be finite.")
            if ticker in normalized:
                raise ValueError(f"Ticker {ticker!r} appears more than once in a view.")
            normalized[ticker] = coef
        if all(c == 0.0 for c in normalized.values()):
            raise ValueError("View coefficients cannot all be zero.")
        object.__setattr__(self, "assets", normalized)

        try:
            annual_return = float(self.annual_return)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"View annual_return must be numeric, got {self.annual_return!r}."
            ) from exc
        if not math.isfinite(annual_return):
            raise ValueError("View annual_return must be finite.")
        object.__setattr__(self, "annual_return", annual_return)

        if self.confidence is not None:
            try:
                confidence = float(self.confidence)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"View confidence must be numeric, got {self.confidence!r}."
                ) from exc
            if not (0.0 < confidence <= 1.0):
                raise ValueError(f"View confidence must be in (0, 1], got {confidence}.")
            object.__setattr__(self, "confidence", confidence)

        if self.label is not None:
            object.__setattr__(self, "label", str(self.label))

    @property
    def tickers(self) -> tuple[str, ...]:
        """Tickers referenced by this view, in declaration order."""
        return tuple(self.assets)

    @property
    def is_relative(self) -> bool:
        """``True`` when the view has at least one negative coefficient."""
        return any(coef < 0.0 for coef in self.assets.values())

    def describe(self) -> str:
        """Return the label if set, otherwise a compact textual form."""
        if self.label:
            return self.label
        parts = ", ".join(f"{ticker}:{coef:+g}" for ticker, coef in self.assets.items())
        return f"[{parts}] @ {self.annual_return:.2%}/yr"


@dataclass(frozen=True)
class ViewMatrices:
    """Pick matrix, view returns and confidences for one estimation window.

    Attributes:
        p_matrix: Pick matrix P of shape (k × n), one row per applied view.
        q_views: Per-period view returns q of shape (k,).
        confidences: Confidence per applied view, shape (k,).
        kept: Indices into the input view sequence that were applied.
        dropped: Indices of views skipped because at least one referenced
            ticker is not among the available tickers.
    """

    p_matrix: np.ndarray
    q_views: np.ndarray
    confidences: np.ndarray
    kept: tuple[int, ...]
    dropped: tuple[int, ...]

    @property
    def n_views(self) -> int:
        """Number of applied views (rows of ``p_matrix``)."""
        return int(self.p_matrix.shape[0])


def build_view_matrices(
    views: Sequence[View],
    tickers: Sequence[str],
    periods_per_year: int,
    default_confidence: float,
) -> ViewMatrices:
    """Build (P, q, confidence) for the views expressible on ``tickers``.

    Each view whose tickers are all present in ``tickers`` becomes one row of
    P, with coefficients placed in the matching columns and zeros elsewhere.
    Its annual target return is converted to a per-period return by dividing
    by ``periods_per_year``. Views referencing a ticker that is absent (for
    example a stock that has not started trading yet) are skipped and
    reported through :attr:`ViewMatrices.dropped`.

    Args:
        views: Views to express.
        tickers: Column order of the covariance matrix and pick matrix, i.e.
            the tickers with data in the current estimation window.
        periods_per_year: Return periods per year of the data (252 for daily,
            12 for monthly). Used to scale ``annual_return`` to one period.
        default_confidence: Confidence assigned to views whose own
            ``confidence`` is ``None``.

    Returns:
        A :class:`ViewMatrices`. With no applicable views, ``p_matrix`` has
        shape (0 × n) and the vectors are empty.

    Raises:
        ValueError: If ``periods_per_year`` is not positive.
    """
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}.")

    columns = {str(ticker): position for position, ticker in enumerate(tickers)}
    n_assets = len(columns)

    rows: list[np.ndarray] = []
    q: list[float] = []
    confidences: list[float] = []
    kept: list[int] = []
    dropped: list[int] = []

    for index, view in enumerate(views):
        missing = [ticker for ticker in view.assets if ticker not in columns]
        if missing:
            dropped.append(index)
            logger.debug(
                "Skipping view %s: no data for %s.", view.describe(), ", ".join(missing)
            )
            continue

        row = np.zeros(n_assets, dtype=float)
        for ticker, coef in view.assets.items():
            row[columns[ticker]] = coef
        rows.append(row)
        q.append(view.annual_return / periods_per_year)
        confidences.append(
            default_confidence if view.confidence is None else view.confidence
        )
        kept.append(index)

    p_matrix = np.vstack(rows) if rows else np.zeros((0, n_assets), dtype=float)
    return ViewMatrices(
        p_matrix=p_matrix,
        q_views=np.asarray(q, dtype=float),
        confidences=np.asarray(confidences, dtype=float),
        kept=tuple(kept),
        dropped=tuple(dropped),
    )
