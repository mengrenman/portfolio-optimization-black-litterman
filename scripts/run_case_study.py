#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from portfolio_bl.backtest.attribution import attribution_table
from portfolio_bl.backtest.metrics import infer_periods_per_year
from portfolio_bl.config import AppConfig, load_config
from portfolio_bl.data.factors import load_fama_french
from portfolio_bl.pipeline import CaseStudyResult, run_case_study

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def _format_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()

    pct_cols = ["annual_return", "annual_volatility", "max_drawdown", "avg_turnover"]
    for col in pct_cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{x:.2%}" if pd.notna(x) else "nan")

    for col in ["sharpe", "sortino", "hhi"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "nan")

    return out


def _write_factor_attribution(
    app_config: AppConfig, result: CaseStudyResult, output_dir: Path
) -> None:
    """Write a factor-attribution table for one case study's three strategies.

    Regresses each strategy's returns on the market factor alone (CAPM) and
    on the FF3 and FF5 Fama-French factor frames (loaded from
    ``app_config.factors_dir`` with
    the default ``derive_daily_rf=True``) and writes the combined table to
    ``<output_dir>/factor_attribution.csv``. Skipped silently (with a debug
    log line) when ``app_config.factors_dir`` is not configured.

    :func:`~portfolio_bl.backtest.attribution.factor_regression` refuses to
    regress returns whose inferred frequency differs from the (daily)
    bundled factors' -- meaningless otherwise, since a non-daily return
    would then be regressed against a single day's factor values per
    period. When that happens (e.g. the case study's prices are monthly),
    this function logs a warning and skips writing
    ``factor_attribution.csv`` instead of raising, since the case study's
    other outputs are already written by the caller by that point.

    Args:
        app_config: Application configuration; only used when
            ``factors_dir`` is set.
        result: The case study's outputs, supplying the three strategies'
            return series.
        output_dir: Directory to write ``factor_attribution.csv`` into.
    """
    if app_config.factors_dir is None:
        logger.debug("No data.factors_dir configured; skipping factor attribution.")
        return

    returns_by_name = {name: strategy.returns for name, strategy in result.strategy_results.items()}
    periods_per_year = infer_periods_per_year(next(iter(returns_by_name.values())).index)
    factor_sets = {
        model: load_fama_french(app_config.factors_dir, model=model, derive_daily_rf=True)
        for model in ("capm", "ff3", "ff5")
    }

    try:
        table = attribution_table(returns_by_name, factor_sets, periods_per_year=periods_per_year)
    except ValueError as exc:
        logger.warning("Skipping factor attribution: %s", exc)
        return

    table.to_csv(output_dir / "factor_attribution.csv")
    logger.info("Saved factor attribution to: %s", output_dir / "factor_attribution.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Black-Litterman public portfolio case study.")
    parser.add_argument("--person", required=True, help="Case-study key from configs/case_studies.yaml")
    parser.add_argument(
        "--config",
        default="configs/case_studies.yaml",
        help="Path to configuration YAML",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/output",
        help="Directory for generated outputs",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    args = parser.parse_args()

    _setup_logging(args.verbose)

    root = Path(__file__).resolve().parents[1]
    config_path = (root / args.config).resolve()

    app_config = load_config(config_path)
    logger.debug("Config loaded from %s.", config_path)

    result = run_case_study(app_config, person_key=args.person)

    output_dir = (root / args.output_dir / args.person).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    result.summary.to_csv(output_dir / "summary.csv")

    nav_df = pd.DataFrame(
        {
            name: strategy.nav
            for name, strategy in result.strategy_results.items()
        }
    ).sort_index()
    nav_df.to_csv(output_dir / "equity_curve.csv")

    ret_df = pd.DataFrame(
        {
            name: strategy.returns
            for name, strategy in result.strategy_results.items()
        }
    ).sort_index()
    ret_df.to_csv(output_dir / "strategy_returns.csv")

    for name, strategy in result.strategy_results.items():
        strategy.weight_history.to_csv(output_dir / f"weights_{name}.csv")

    metadata = pd.Series(
        {
            "person_label": result.person_label,
            "as_of_date": result.as_of_date.strftime("%Y-%m-%d"),
            "n_assets": len(result.universe),
            "universe": ",".join(result.universe),
            "risk_free_rate": result.risk_free_rate,
        }
    )
    metadata.to_csv(output_dir / "metadata.csv", header=["value"])

    _write_factor_attribution(app_config, result, output_dir)

    logger.info("Saved outputs to: %s", output_dir)
    print(f"\nSaved outputs to: {output_dir}")
    print()
    print(_format_summary(result.summary).to_string())


if __name__ == "__main__":
    main()
