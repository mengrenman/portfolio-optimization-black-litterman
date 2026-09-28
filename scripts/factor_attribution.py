#!/usr/bin/env python3
"""Cross-person Fama-French factor-attribution report.

Runs all three case studies configured in ``configs/case_studies.yaml``,
regresses each of their three strategies' (disclosed, mean-variance,
Black-Litterman) daily returns on the market factor alone (CAPM) and on the
FF3 and FF5 Fama-French factor models,
and adds a "SPY (check)" row that regresses SPY's own daily return the same
way, as a sanity check: a well-behaved regression should recover
approximately zero alpha, a market loading near 1, and an R-squared near 1
for the market against itself.

Writes the combined long table to ``reports/output/factor_attribution.csv``
and prints GitHub-flavored Markdown tables to stdout: alpha under each of the
three models side by side, then the full loadings for FF3 and for FF5, then
the alpha of three return differences per person: each overlay minus the
disclosed book, and Black-Litterman minus mean-variance.

The difference rows test the project's two questions directly. The difference
of two alphas on the same factors and the same dates equals the alpha of the
return difference, but only the regression on the difference gives that alpha
a standard error. A return difference is a zero-cost long-short position, so it
is already an excess return: those rows are regressed without subtracting the
risk-free rate (``subtract_rf=False``). Every series is restricted to the
strategies' common dates first, so the identity holds even if a strategy's
dates ever differ.

Usage:
    python scripts/factor_attribution.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from portfolio_bl.backtest.attribution import attribution_table
from portfolio_bl.backtest.metrics import infer_periods_per_year
from portfolio_bl.config import load_config
from portfolio_bl.data.factors import load_fama_french
from portfolio_bl.data.prices import load_prices_csv, to_return_matrix
from portfolio_bl.pipeline import run_case_study

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

STRATEGY_LABEL = {
    "disclosed": "Disclosed",
    "mean_variance": "Mean-variance",
    "black_litterman": "Black-Litterman",
}

SPY_ROW_KEY = "SPY (check)"

FF3_FACTOR_HEADER = [("mkt_rf", "Mkt"), ("smb", "SMB"), ("hml", "HML")]
FF5_FACTOR_HEADER = [*FF3_FACTOR_HEADER, ("rmw", "RMW"), ("cma", "CMA")]
MODEL_FACTOR_HEADER = {"ff3": FF3_FACTOR_HEADER, "ff5": FF5_FACTOR_HEADER}
MODELS = ("capm", "ff3", "ff5")
MODEL_LABEL = {"capm": "CAPM", "ff3": "FF3", "ff5": "FF5"}


def _fmt(value: float, spec: str) -> str:
    """Format a number, printing a value that rounds to zero without a sign."""
    text = format(value, spec)
    stripped = text.lstrip("-")
    return stripped if stripped.strip("0.%") == "" else text


def _verify_common_dates(returns_by_name: dict[str, pd.Series]) -> pd.DatetimeIndex:
    """Verify every strategy return series shares the same date index.

    This is the "like-for-like" precondition for comparing strategies (and
    later SPY) on the same regression sample. A mismatch is not silently
    ignored: it is reported to stdout and the intersection of all series is
    used instead so the report can still complete.

    Args:
        returns_by_name: Mapping from display name to daily return series.

    Returns:
        The sorted, common ``DatetimeIndex`` across all series.
    """
    indices = {name: pd.DatetimeIndex(s.dropna().index) for name, s in returns_by_name.items()}
    names = list(indices)
    reference = indices[names[0]]
    mismatched = [n for n in names[1:] if not indices[n].equals(reference)]

    if not mismatched:
        print(f"Verified: all {len(names)} strategy return series share the same {len(reference)} dates.")
        return reference.sort_values()

    print(
        f"WARNING: return date sets do NOT match across strategies. "
        f"{len(mismatched)} of {len(names)} series differ from {names[0]!r}: {mismatched}."
    )
    common = reference
    for name in mismatched:
        common = common.intersection(indices[name])
    print(f"  Restricting every series to the {len(common)}-date intersection for comparability.")
    return common.sort_values()


def _restrict_spy_to_strategy_dates(spy_returns: pd.Series, strategy_dates: pd.DatetimeIndex) -> pd.Series:
    """Restrict SPY's return series to the strategies' common date set.

    Args:
        spy_returns: SPY's full daily return series.
        strategy_dates: The common date index verified across the strategy
            return series (see :func:`_verify_common_dates`).

    Returns:
        SPY's return series indexed by exactly ``strategy_dates`` (dropping
        any date SPY has no return for, with a printed warning if so).
    """
    restricted = spy_returns.reindex(strategy_dates)
    missing = restricted[restricted.isna()].index
    if len(missing) > 0:
        print(
            f"WARNING: SPY has no return on {len(missing)} of the {len(strategy_dates)} strategy "
            f"date(s) (first missing={missing[0].date()}); dropping them from the {SPY_ROW_KEY} row."
        )
        restricted = restricted.dropna()
    else:
        print(f"Verified: SPY has a return on all {len(strategy_dates)} strategy dates.")
    return restricted


def _format_markdown_table(model: str, table: pd.DataFrame, rows_meta: list[tuple[str, str, str]]) -> str:
    """Format one factor model's rows of ``table`` as a GitHub Markdown table.

    Args:
        model: Factor-model key (``"ff3"`` or ``"ff5"``), selecting which
            factor columns to print.
        table: The full attribution table (``("strategy", "model")``
            MultiIndex), as returned by :func:`attribution_table`.
        rows_meta: ``(row_key, person_display, strategy_display)`` tuples, in
            the display order the report wants the rows printed in.

    Returns:
        The rendered Markdown table, followed by a summary line of
        ``n_obs``, the Newey-West ``lags``, and the sample ``start``/``end``.
    """
    factor_header = MODEL_FACTOR_HEADER[model]
    header = ["Person", "Strategy", "Alpha (ann.)", "t", *(label for _, label in factor_header), "R²"]

    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]

    stats: list[tuple[int, int, pd.Timestamp, pd.Timestamp]] = []
    for row_key, person_display, strategy_display in rows_meta:
        row = table.loc[(row_key, model)]
        cells = [
            person_display,
            strategy_display,
            _fmt(row["alpha_annual"], ".1%"),
            _fmt(row["alpha_t"], ".1f"),
            *(_fmt(row[col], ".2f") for col, _ in factor_header),
            _fmt(row["r_squared"], ".2f"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
        stats.append((int(row["n_obs"]), int(row["lags"]), row["start"], row["end"]))

    unique_stats = set(stats)
    if len(unique_stats) == 1:
        n_obs, lags, start, end = stats[0]
        lines.append(f"\nn_obs={n_obs}, Newey-West lags={lags}, start={start}, end={end}")
    else:
        lines.append(f"\nWARNING: n_obs/lags/start/end are not identical across rows: {sorted(unique_stats)}")

    return "\n".join(lines)


def _format_alpha_table(table: pd.DataFrame, rows_meta: list[tuple[str, str, str]]) -> str:
    """Format annualized alpha and its t-statistic under every model side by side.

    Args:
        table: The full attribution table (``("strategy", "model")``
            MultiIndex), as returned by :func:`attribution_table`.
        rows_meta: ``(row_key, person_display, strategy_display)`` tuples, in
            display order.

    Returns:
        The rendered Markdown table.
    """
    header = ["Person", "Strategy"]
    for model in MODELS:
        header += [f"{MODEL_LABEL[model]} alpha", "t"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row_key, person_display, strategy_display in rows_meta:
        cells = [person_display, strategy_display]
        for model in MODELS:
            row = table.loc[(row_key, model)]
            cells += [_fmt(row["alpha_annual"], ".1%"), _fmt(row["alpha_t"], ".1f")]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format_difference_table(table: pd.DataFrame, rows_meta: list[tuple[str, str, str]]) -> str:
    """Format the return-difference alphas under every model side by side.

    Args:
        table: Attribution table for the return-difference series.
        rows_meta: ``(row_key, person_display, strategy_display)`` tuples, in
            display order.

    Returns:
        The rendered Markdown table.
    """
    return _format_alpha_table(table, rows_meta)


def main() -> None:
    logging.disable(logging.CRITICAL)

    config_path = ROOT / "configs" / "case_studies.yaml"
    app_config = load_config(config_path)
    if app_config.factors_dir is None:
        raise ValueError(
            f"{config_path} has no data.factors_dir configured; cannot run factor attribution."
        )

    returns_by_name: dict[str, pd.Series] = {}
    rows_meta: list[tuple[str, str, str]] = []

    print("running case studies ...")
    for person_key, case_cfg in app_config.case_studies.items():
        result = run_case_study(app_config, person_key=person_key)
        for strategy_name, strategy_result in result.strategy_results.items():
            row_key = f"{case_cfg.person_label} | {strategy_name}"
            returns_by_name[row_key] = strategy_result.returns
            rows_meta.append((row_key, case_cfg.person_label, STRATEGY_LABEL[strategy_name]))

    strategy_dates = _verify_common_dates(returns_by_name)
    returns_by_name = {name: series.reindex(strategy_dates) for name, series in returns_by_name.items()}

    spy_returns = to_return_matrix(load_prices_csv(app_config.prices_path))["SPY"]
    spy_restricted = _restrict_spy_to_strategy_dates(spy_returns, strategy_dates)
    returns_by_name[SPY_ROW_KEY] = spy_restricted
    rows_meta.append((SPY_ROW_KEY, SPY_ROW_KEY, ""))

    periods_per_year = infer_periods_per_year(strategy_dates)
    factor_sets = {
        model: load_fama_french(app_config.factors_dir, model=model, derive_daily_rf=True)
        for model in MODELS
    }

    print("running factor regressions (CAPM, FF3, FF5) ...")
    table = attribution_table(returns_by_name, factor_sets, periods_per_year=periods_per_year)

    # Return differences per person. Each is already an excess return.
    difference_pairs = (
        ("mean_variance", "disclosed"),
        ("black_litterman", "disclosed"),
        ("black_litterman", "mean_variance"),
    )
    difference_by_name: dict[str, pd.Series] = {}
    difference_meta: list[tuple[str, str, str]] = []
    for person_label in dict.fromkeys(person for _, person, _ in rows_meta if person != SPY_ROW_KEY):
        for long_leg, short_leg in difference_pairs:
            row_key = f"{person_label} | {long_leg} - {short_leg}"
            difference_by_name[row_key] = (
                returns_by_name[f"{person_label} | {long_leg}"]
                - returns_by_name[f"{person_label} | {short_leg}"]
            )
            label = f"{STRATEGY_LABEL[long_leg]} minus {STRATEGY_LABEL[short_leg].lower()}"
            difference_meta.append((row_key, person_label, label))
    difference_table = attribution_table(
        difference_by_name, factor_sets, periods_per_year=periods_per_year, subtract_rf=False
    )

    rows_meta = rows_meta + difference_meta
    table = pd.concat([table, difference_table])

    key_to_meta = {row_key: (person, strategy) for row_key, person, strategy in rows_meta}
    long_table = table.reset_index()
    long_table.insert(0, "person", long_table["strategy"].map(lambda k: key_to_meta[k][0]))
    long_table.insert(1, "strategy_label", long_table["strategy"].map(lambda k: key_to_meta[k][1]))
    long_table = long_table.drop(columns=["strategy"]).rename(columns={"strategy_label": "strategy"})

    output_dir = ROOT / "reports" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "factor_attribution.csv"
    long_table.to_csv(output_path, index=False)
    print(f"\nSaved combined table to: {output_path}")

    level_rows = [meta for meta in rows_meta if meta not in difference_meta]
    print("\n### Alpha by model\n")
    print(_format_alpha_table(table, level_rows))
    for model in ("ff3", "ff5"):
        print(f"\n### {MODEL_LABEL[model]}\n")
        print(_format_markdown_table(model, table, level_rows))
    print("\n### Return differences (alpha of the difference)\n")
    print(_format_difference_table(table, difference_meta))


if __name__ == "__main__":
    main()
