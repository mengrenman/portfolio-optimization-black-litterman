from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_DISCLOSURE_COLUMNS = {"person", "as_of_date", "ticker", "value_usd"}



def load_disclosures_csv(path: str | Path) -> pd.DataFrame:
    disclosures = pd.read_csv(path)
    missing = REQUIRED_DISCLOSURE_COLUMNS.difference(disclosures.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required disclosure columns: {missing_str}")

    out = disclosures.copy()
    out["person_norm"] = out["person"].astype(str).str.strip().str.lower()
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out["as_of_date"] = pd.to_datetime(out["as_of_date"], errors="coerce")
    out["value_usd"] = pd.to_numeric(out["value_usd"], errors="coerce")

    out = out.dropna(subset=["person_norm", "ticker", "as_of_date", "value_usd"])
    out = out[out["value_usd"] > 0].copy()

    if out.empty:
        raise ValueError("Disclosure dataset is empty after cleaning.")

    return out.sort_values(["as_of_date", "person_norm", "ticker"]).reset_index(drop=True)



def latest_portfolio_for_aliases(
    disclosures: pd.DataFrame,
    aliases: tuple[str, ...] | list[str],
) -> tuple[pd.DataFrame, pd.Timestamp]:
    alias_set = {a.strip().lower() for a in aliases}
    filtered = disclosures[disclosures["person_norm"].isin(alias_set)].copy()

    if filtered.empty:
        alias_str = ", ".join(sorted(alias_set))
        raise ValueError(f"No disclosure rows found for aliases: {alias_str}")

    as_of_date = filtered["as_of_date"].max()
    latest = filtered[filtered["as_of_date"] == as_of_date].copy()

    latest["weight"] = latest["value_usd"] / latest["value_usd"].sum()
    latest = latest[["ticker", "value_usd", "weight"]].sort_values("ticker").reset_index(drop=True)

    return latest, as_of_date
