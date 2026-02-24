#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from portfolio_bl.config import load_config
from portfolio_bl.pipeline import run_case_study


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
    logger = logging.getLogger(__name__)

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
        }
    )
    metadata.to_csv(output_dir / "metadata.csv", header=["value"])

    logger.info("Saved outputs to: %s", output_dir)
    print(f"\nSaved outputs to: {output_dir}")
    print()
    print(_format_summary(result.summary).to_string())


if __name__ == "__main__":
    main()
