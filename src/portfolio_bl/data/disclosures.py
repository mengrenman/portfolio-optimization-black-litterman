from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_DISCLOSURE_COLUMNS = {"person", "as_of_date", "ticker", "value_usd"}


def load_disclosures_csv(path: str | Path) -> pd.DataFrame:
    """Load and clean a disclosures CSV file.

    Parses and normalises holdings data. Person names are lowercased and
    stripped; tickers are uppercased; values are coerced to float. Rows with
    missing or non-positive values are dropped and a warning is emitted with
    the count of affected rows.

    Args:
        path: Path to the disclosures CSV file.

    Returns:
        A cleaned DataFrame sorted by ``as_of_date``, ``person_norm``, and
        ``ticker``, with a new ``person_norm`` column containing the normalised
        person name.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be read for any other reason.
        ValueError: If required columns are missing or the cleaned dataset is
            empty after filtering.
    """
    path = Path(path)
    try:
        disclosures = pd.read_csv(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Disclosures file not found: {path}") from None
    except Exception as exc:
        raise RuntimeError(f"Failed to load disclosures from {path}: {exc}") from exc

    missing = REQUIRED_DISCLOSURE_COLUMNS.difference(disclosures.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required disclosure columns: {missing_str}")

    out = disclosures.copy()
    out["person_norm"] = out["person"].astype(str).str.strip().str.lower()
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out["as_of_date"] = pd.to_datetime(out["as_of_date"], errors="coerce")
    out["value_usd"] = pd.to_numeric(out["value_usd"], errors="coerce")

    n_before = len(out)
    out = out.dropna(subset=["person_norm", "ticker", "as_of_date", "value_usd"])
    out = out[out["value_usd"] > 0].copy()
    n_dropped = n_before - len(out)
    if n_dropped > 0:
        logger.warning(
            "Dropped %d disclosure row(s) due to missing or non-positive values.", n_dropped
        )

    if out.empty:
        raise ValueError("Disclosure dataset is empty after cleaning.")

    logger.debug("Loaded %d disclosure rows from %s.", len(out), path)
    return out.sort_values(["as_of_date", "person_norm", "ticker"]).reset_index(drop=True)


def latest_portfolio_for_aliases(
    disclosures: pd.DataFrame,
    aliases: tuple[str, ...] | list[str],
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Extract the latest disclosed portfolio for a person identified by name aliases.

    Filters the disclosure dataset to rows matching any of the supplied aliases
    (case-insensitive) and selects those from the most recent ``as_of_date``.
    Weights are normalised to sum to 1.

    Args:
        disclosures: Cleaned disclosure DataFrame as returned by
            :func:`load_disclosures_csv`.
        aliases: Case-insensitive person-name variants to match against the
            ``person_norm`` column.

    Returns:
        A tuple ``(portfolio_df, as_of_date)`` where ``portfolio_df`` contains
        columns ``ticker``, ``value_usd``, and ``weight``.

    Raises:
        ValueError: If no rows match the supplied aliases.
    """
    alias_set = {a.strip().lower() for a in aliases}
    filtered = disclosures[disclosures["person_norm"].isin(alias_set)].copy()

    if filtered.empty:
        alias_str = ", ".join(sorted(alias_set))
        raise ValueError(f"No disclosure rows found for aliases: {alias_str}")

    as_of_date = filtered["as_of_date"].max()
    latest = filtered[filtered["as_of_date"] == as_of_date].copy()

    latest["weight"] = latest["value_usd"] / latest["value_usd"].sum()
    latest = latest[["ticker", "value_usd", "weight"]].sort_values("ticker").reset_index(drop=True)

    logger.debug(
        "Latest portfolio: %d positions as of %s.", len(latest), as_of_date.date()
    )
    return latest, as_of_date
