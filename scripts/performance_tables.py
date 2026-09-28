#!/usr/bin/env python3
"""Regenerate every Sharpe-bearing table in README.md.

Prints GitHub-flavored Markdown tables using exactly the methods that
produced the current README numbers:

- **Strategy comparison** (README "Strategy comparison"): per case study and
  strategy, ``result.summary`` metrics, plus a benchmark row for SPY held
  over the same window.
- **Constant mix vs. buy-and-hold** (README "Between rebalances the book is
  re-set every day"): the ``disclosed`` strategy as actually run (a daily
  constant mix) against a buy-and-hold of the same initial weights.
- **Transaction costs** (README "Nothing is charged for trading."): Sharpe
  after charging a flat per-rebalance cost proportional to measured
  turnover, at 0/10/25/50 bps.
- **Confidence sweep** (README "View confidence interpolates between the two
  baselines"): for Buffett, how the Black-Litterman posterior moves between
  the disclosed and mean-variance baselines as view confidence rises.

By default every table charges the daily T-bill rate (derived from the
bundled Fama-French factor snapshot) as the Sharpe/Sortino numerator's
risk-free hurdle, exactly as ``run_case_study`` now does when
``app_config.factors_dir`` is configured. Pass ``--zero-rf`` to force a zero
risk-free rate everywhere instead, which reproduces exactly the zero-rate
tables README.md carried before the risk-free charge was added (commit
b954531) -- the regression check for the risk-free-rate feature. A config
without ``data.factors_dir`` also gives the zero-rate tables.

Usage:
    python scripts/performance_tables.py
    python scripts/performance_tables.py --zero-rf
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_bl.backtest.metrics import (
    annualized_return,
    annualized_risk_free,
    annualized_volatility,
    infer_periods_per_year,
    max_drawdown,
    risk_free_per_period,
    sharpe_ratio,
)
from portfolio_bl.config import AppConfig, load_config
from portfolio_bl.data.disclosures import (
    latest_portfolio_for_aliases,
    load_disclosures_csv,
)
from portfolio_bl.data.factors import load_fama_french
from portfolio_bl.data.prices import load_prices_csv, to_return_matrix
from portfolio_bl.pipeline import CaseStudyResult, run_case_study

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

CASE_ORDER = ("buffett", "pelosi", "trump")
PERSON_LABEL = {"buffett": "Buffett", "pelosi": "Pelosi", "trump": "Trump"}
STRATEGY_ORDER = ("disclosed", "mean_variance", "black_litterman")
STRATEGY_LABEL = {
    "disclosed": "disclosed",
    "mean_variance": "mean-variance",
    "black_litterman": "Black-Litterman",
}

# (case_key, strategy_key) rows for the transaction-cost table, in the order
# the README prints them.
COST_ROWS = (
    ("buffett", "mean_variance"),
    ("buffett", "black_litterman"),
    ("pelosi", "black_litterman"),
    ("trump", "mean_variance"),
)

# Confidence grid for the sweep table (distinct from the finer grid plotted
# in scripts/make_figures.py), paired with the display string README uses
# for each value (its own decimal width, not a fixed format spec).
CONFIDENCE_GRID = (
    (0.01, "0.01"),
    (0.20, "0.20"),
    (0.40, "0.40"),
    (0.65, "0.65"),
    (0.80, "0.80"),
    (0.95, "0.95"),
    (0.999, "0.999"),
)

TRANSACTION_COST_BPS = (0, 10, 25, 50)


def _pct(value: float) -> str:
    """Format a fraction as a percent with one decimal place."""
    return f"{value:.1%}"


def _sharpe(value: float) -> str:
    """Format a Sharpe ratio with two decimal places."""
    return f"{value:.2f}"


def _markdown_table(header: list[str], aligns: str, rows: list[list[str]]) -> str:
    """Render a GitHub-flavored Markdown table matching README.md's column alignment.

    Args:
        header: Column headers.
        aligns: One alignment character per column, ``"l"`` (left) or ``"r"``
            (right), matching README.md's own tables cell for cell.
        rows: Row cells, already formatted as strings.

    Returns:
        The rendered table as a single string.
    """
    separators = ["---:" if a == "r" else "---" for a in aligns]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(separators) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _rf_annual_fn(app_config: AppConfig, zero_rf: bool):
    """Build a function computing the annualized risk-free rate over a date index.

    The daily T-bill series is loaded once and converted, once, into one
    compounded rate per date of the price calendar (every date in
    ``app_config.prices_path``, via :func:`~portfolio_bl.backtest.metrics.risk_free_per_period`)
    -- exactly as :func:`~portfolio_bl.pipeline.run_case_study` now does.
    Reindexing the raw daily series directly onto a non-daily ``index``
    (e.g. monthly return dates) would silently keep only one day's rate per
    period instead of compounding every day in it; converting once over the
    price calendar avoids that regardless of what ``index`` the returned
    function is later called with.

    Args:
        app_config: Application configuration; ``app_config.factors_dir``
            and ``app_config.prices_path`` are used when ``zero_rf`` is
            ``False``.
        zero_rf: When ``True``, the returned function always returns 0.0
            without touching ``app_config.factors_dir``.

    Returns:
        A callable ``(index, periods_per_year) -> float`` matching
        :func:`~portfolio_bl.backtest.metrics.annualized_risk_free`.
    """
    if zero_rf or app_config.factors_dir is None:
        return lambda index, periods_per_year: 0.0

    daily_risk_free = load_fama_french(app_config.factors_dir, "capm")["rf"]
    prices = load_prices_csv(app_config.prices_path)
    price_dates = pd.DatetimeIndex(sorted(prices["date"].unique()))
    risk_free_series = risk_free_per_period(daily_risk_free, price_dates)

    def _fn(index: pd.DatetimeIndex, periods_per_year: int) -> float:
        return annualized_risk_free(risk_free_series, index, periods_per_year)

    return _fn


def print_risk_free_rates(results: dict[str, CaseStudyResult]) -> None:
    """Print each case study's annualized risk-free rate (``CaseStudyResult.risk_free_rate``).

    Args:
        results: Mapping from case-study key to its :class:`CaseStudyResult`.
    """
    print("Annualized risk-free rate charged (per case study):")
    for key in CASE_ORDER:
        print(f"  {PERSON_LABEL[key]}: {_pct(results[key].risk_free_rate)}")
    print()


# ---------------------------------------------------------------------------
# (a) Strategy comparison
# ---------------------------------------------------------------------------


def print_strategy_comparison(
    results: dict[str, CaseStudyResult],
    app_config: AppConfig,
    rf_annual_fn,
) -> None:
    """Print the README "Strategy comparison" table, plus an SPY benchmark row.

    Args:
        results: Mapping from case-study key to its :class:`CaseStudyResult`,
            run with the same risk-free treatment as ``rf_annual_fn``.
        app_config: Application configuration (for the prices path).
        rf_annual_fn: Callable ``(index, periods_per_year) -> float``
            returning the annualized risk-free rate to charge SPY, matching
            the treatment used for ``results``.
    """
    header = ["Person", "Strategy", "Annual return", "Annual vol", "Sharpe", "Max drawdown", "Turnover"]
    aligns = "llrrrrr"
    rows: list[list[str]] = []

    for case_key in CASE_ORDER:
        summary = results[case_key].summary
        for strategy_key in STRATEGY_ORDER:
            row = summary.loc[strategy_key]
            rows.append(
                [
                    PERSON_LABEL[case_key],
                    STRATEGY_LABEL[strategy_key],
                    _pct(row["annual_return"]),
                    _pct(row["annual_volatility"]),
                    _sharpe(row["sharpe"]),
                    _pct(row["max_drawdown"]),
                    _pct(row["avg_turnover"]),
                ]
            )

    # Benchmark row: SPY over the disclosed strategy's return dates. The row
    # needs one window, so check that all three case studies share it.
    disclosed_dates = results["buffett"].strategy_results["disclosed"].returns.index
    if not all(
        results[key].strategy_results["disclosed"].returns.index.equals(disclosed_dates)
        for key in CASE_ORDER
    ):
        raise ValueError("Case studies do not share one return window; the SPY row needs one.")
    prices = load_prices_csv(app_config.prices_path)
    spy_returns = to_return_matrix(prices)["SPY"].reindex(disclosed_dates)
    periods_per_year = infer_periods_per_year(disclosed_dates)
    rf_annual = rf_annual_fn(disclosed_dates, periods_per_year)

    bench_cells = [
        "benchmark",
        "SPY, same window",
        _pct(annualized_return(spy_returns, periods_per_year)),
        _pct(annualized_volatility(spy_returns, periods_per_year)),
        _sharpe(sharpe_ratio(spy_returns, periods_per_year, risk_free_rate=rf_annual)),
        _pct(max_drawdown(spy_returns)),
        "n/a",
    ]
    rows.append([f"*{c}*" for c in bench_cells])

    print("### Strategy comparison\n")
    print(_markdown_table(header, aligns, rows))
    print()


# ---------------------------------------------------------------------------
# (b) Constant mix vs. buy-and-hold
# ---------------------------------------------------------------------------


def _buy_and_hold_returns(
    app_config: AppConfig,
    case_key: str,
    result: CaseStudyResult,
    universe_returns: pd.DataFrame,
) -> pd.Series:
    """Buy-and-hold return series for one case study's disclosed weights.

    Args:
        app_config: Application configuration (for the disclosures path and
            this case study's alias list).
        case_key: Case-study key (``"buffett"``, ``"pelosi"`` or
            ``"trump"``).
        result: That case study's :class:`CaseStudyResult`.
        universe_returns: Full daily return matrix restricted to
            ``result.universe``.

    Returns:
        The buy-and-hold daily return series, indexed by the disclosed
        strategy's return dates.
    """
    disclosures = load_disclosures_csv(app_config.disclosures_path)
    aliases = app_config.case_studies[case_key].disclosure_aliases
    latest, _ = latest_portfolio_for_aliases(disclosures, aliases)
    w0 = latest.set_index("ticker")["weight"].reindex(result.universe).fillna(0.0)
    w0 = w0 / w0.sum()

    disclosed_dates = result.strategy_results["disclosed"].returns.index
    r = universe_returns.reindex(index=disclosed_dates, columns=result.universe).fillna(0.0)
    nav = (1.0 + r).cumprod()
    port = nav.mul(w0, axis=1).sum(axis=1)
    bh = port.pct_change()
    bh.iloc[0] = port.iloc[0] - 1.0
    return bh


def print_constant_mix_vs_buy_and_hold(
    results: dict[str, CaseStudyResult],
    app_config: AppConfig,
    universe_returns_by_case: dict[str, pd.DataFrame],
    rf_annual_fn,
) -> None:
    """Print the README "Between rebalances the book is re-set every day" table.

    Args:
        results: Mapping from case-study key to its :class:`CaseStudyResult`.
        app_config: Application configuration (for the disclosures path).
        universe_returns_by_case: Mapping from case-study key to the full
            daily return matrix restricted to that case's universe.
        rf_annual_fn: Callable ``(index, periods_per_year) -> float``
            returning the annualized risk-free rate for both the as-run and
            buy-and-hold Sharpe ratios.
    """
    header = ["Person", "`disclosed` as run (daily constant mix)", "Same weights, bought and held"]
    aligns = "lrr"
    rows: list[list[str]] = []

    for i, case_key in enumerate(CASE_ORDER):
        result = results[case_key]
        disclosed = result.strategy_results["disclosed"]
        periods_per_year = infer_periods_per_year(disclosed.returns.index)
        rf_annual = rf_annual_fn(disclosed.returns.index, periods_per_year)

        run_ret = annualized_return(disclosed.returns, periods_per_year)
        run_sharpe = sharpe_ratio(disclosed.returns, periods_per_year, risk_free_rate=rf_annual)

        bh = _buy_and_hold_returns(app_config, case_key, result, universe_returns_by_case[case_key])
        bh_ret = annualized_return(bh, periods_per_year)
        bh_sharpe = sharpe_ratio(bh, periods_per_year, risk_free_rate=rf_annual)

        if i == 0:
            run_cell = f"{run_ret:.1%} return, {run_sharpe:.2f} Sharpe"
        else:
            run_cell = f"{run_ret:.1%}, {run_sharpe:.2f}"
        bh_cell = f"{bh_ret:.1%}, {bh_sharpe:.2f}"

        rows.append([PERSON_LABEL[case_key], run_cell, bh_cell])

    print("### Between rebalances the book is re-set every day\n")
    print(_markdown_table(header, aligns, rows))
    print()


# ---------------------------------------------------------------------------
# (c) Transaction costs
# ---------------------------------------------------------------------------


def print_transaction_costs(results: dict[str, CaseStudyResult], rf_annual_fn) -> None:
    """Print the README "Nothing is charged for trading." transaction-cost table.

    Args:
        results: Mapping from case-study key to its :class:`CaseStudyResult`.
        rf_annual_fn: Callable ``(index, periods_per_year) -> float``
            returning the annualized risk-free rate to use for the
            cost-adjusted Sharpe ratios, matching the uncharged series.
    """
    header = ["Case", "Strategy", *(f"{bps} bp" for bps in TRANSACTION_COST_BPS)]
    aligns = "ll" + "r" * len(TRANSACTION_COST_BPS)
    rows: list[list[str]] = []

    for case_key, strategy_key in COST_ROWS:
        result = results[case_key]
        strategy = result.strategy_results[strategy_key]
        avg_turnover = result.summary.loc[strategy_key, "avg_turnover"]
        periods_per_year = infer_periods_per_year(strategy.returns.index)
        rf_annual = rf_annual_fn(strategy.returns.index, periods_per_year)

        charge_dates = strategy.weight_history.index.intersection(strategy.returns.index)
        cells = [PERSON_LABEL[case_key], STRATEGY_LABEL[strategy_key]]
        for bps in TRANSACTION_COST_BPS:
            cost = avg_turnover * bps / 1e4
            costed = strategy.returns.copy()
            costed.loc[charge_dates] = costed.loc[charge_dates] - cost
            cells.append(_sharpe(sharpe_ratio(costed, periods_per_year, risk_free_rate=rf_annual)))
        rows.append(cells)

    print("### Transaction costs\n")
    print(_markdown_table(header, aligns, rows))
    print()


# ---------------------------------------------------------------------------
# (d) Confidence sweep
# ---------------------------------------------------------------------------


def print_confidence_sweep(base_results: dict[str, CaseStudyResult], run_config: AppConfig) -> None:
    """Print the README "View confidence interpolates between the two baselines" table.

    Sweeps Buffett's view confidence over :data:`CONFIDENCE_GRID`, comparing
    each run's Black-Litterman weight history against the disclosed and
    mean-variance weight histories from ``base_results`` (the default,
    shipped-configuration run).

    Args:
        base_results: Mapping from case-study key to its
            :class:`CaseStudyResult` from the default-confidence run; only
            ``base_results["buffett"]`` is used.
        run_config: Application configuration to run the sweep with (already
            reflecting ``--zero-rf`` if set).
    """
    base = base_results["buffett"]
    disclosed_wh = base.strategy_results["disclosed"].weight_history
    mvo_wh = base.strategy_results["mean_variance"].weight_history

    header = ["Confidence", "Distance to disclosed", "Distance to mean-variance", "Sharpe", "Turnover"]
    aligns = "rrrrr"
    rows: list[list[str]] = []

    for confidence, confidence_display in CONFIDENCE_GRID:
        r = run_case_study(run_config, "buffett", view_confidence=confidence)
        bl_wh = r.strategy_results["black_litterman"].weight_history
        to_disclosed = float(np.linalg.norm(bl_wh.values - disclosed_wh.values, axis=1).mean())
        to_mvo = float(np.linalg.norm(bl_wh.values - mvo_wh.values, axis=1).mean())
        sharpe = float(r.summary.loc["black_litterman", "sharpe"])
        turnover = float(r.summary.loc["black_litterman", "avg_turnover"])

        rows.append(
            [
                confidence_display,
                f"{to_disclosed:.3f}",
                f"{to_mvo:.3f}",
                f"{sharpe:.2f}",
                _pct(turnover),
            ]
        )

    print("### View confidence interpolates between the two baselines\n")
    print(_markdown_table(header, aligns, rows))
    print()


def main() -> None:
    # Keep warnings (for example a risk-free gap after a price refresh) visible,
    # but not the pipeline's progress messages.
    logging.disable(logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zero-rf",
        action="store_true",
        help="Force a zero risk-free rate everywhere (reproduces the pre-risk-free README tables of commit b954531).",
    )
    args = parser.parse_args()

    app_config = load_config(ROOT / "configs" / "case_studies.yaml")
    run_config = dataclasses.replace(app_config, factors_dir=None) if args.zero_rf else app_config
    rf_annual_fn = _rf_annual_fn(app_config, args.zero_rf)

    print("running case studies ...")
    results = {key: run_case_study(run_config, key) for key in CASE_ORDER}

    prices = load_prices_csv(app_config.prices_path)
    all_returns = to_return_matrix(prices)
    universe_returns_by_case = {key: all_returns[results[key].universe] for key in CASE_ORDER}

    print()
    print_risk_free_rates(results)
    print_strategy_comparison(results, app_config, rf_annual_fn)
    print_constant_mix_vs_buy_and_hold(results, app_config, universe_returns_by_case, rf_annual_fn)
    print_transaction_costs(results, rf_annual_fn)
    print_confidence_sweep(results, run_config)


if __name__ == "__main__":
    main()
