# portfolio-optimization-black-litterman

Black-Litterman case-study framework for evaluating publicly disclosed portfolios from notable figures.

## Project Question
Can a Black-Litterman overlay improve portfolio quality relative to:
1. The disclosed portfolio itself
2. A mean-variance (sample-estimated) baseline

## Implemented Capabilities
- Disclosure and price ingestion with schema validation.
- Latest disclosed portfolio reconstruction by person aliases.
- Rolling rebalancing backtest engine (lookback-based, no look-ahead application).
- Strategy comparison:
  - `disclosed` (static disclosed weights),
  - `mean_variance` (sample-estimated Markowitz),
  - `black_litterman` (equilibrium + views posterior).
- Metrics: annual return/volatility, Sharpe, Sortino, max drawdown, HHI concentration, turnover.
- CLI pipeline that writes per-case outputs to `reports/output/<person>/`.
- Four notebooks with visual diagnostics, strategy comparison, sensitivity analysis, and benchmark attribution.

## Important Caveats
- Public disclosures are delayed, incomplete, and sometimes approximate.
- Disclosure quality is source-dependent; conclusions are only as good as the input coverage.
- This repo is for research only, not investment advice.

## Repository Layout
```text
portfolio-optimization-black-litterman/
  configs/                   # Case-study and backtest parameters
  data/
    raw/
      disclosures/           # Input holdings disclosures CSV
      prices/                # Input price history CSV
  notebooks/                 # Visual walkthrough notebooks
  reports/
    templates/               # Markdown report templates
    output/                  # Generated case-study artifacts
  scripts/
    run_case_study.py        # CLI entrypoint
  src/portfolio_bl/
    backtest/                # Rolling backtest and metrics
    data/                    # Disclosure + price loaders
    models/                  # BL + mean-variance logic
    pipeline.py              # End-to-end experiment runner
  tests/                     # Unit tests for core math/metrics
```

## Input Data Schemas
### Disclosures CSV
Required columns:
- `person`
- `as_of_date` (YYYY-MM-DD)
- `ticker`
- `value_usd`

Optional column:
- `source`

### Prices CSV
Required columns:
- `date` (YYYY-MM-DD)
- `ticker`
- `close`

## Quick Start
```bash
cd /Users/mengren/Documents/new_projects/portfolio-optimization-black-litterman
python -m pip install -e '.[dev,notebooks]'
pytest -q
python scripts/run_case_study.py --person buffett
python scripts/run_case_study.py --person pelosi
python scripts/run_case_study.py --person trump
```

Generated outputs include:
- `summary.csv`
- `equity_curve.csv`
- `strategy_returns.csv`
- `weights_<strategy>.csv`
- `metadata.csv`

All written under `reports/output/<person>/`.

## Notebook Demos
Launch:
```bash
cd /Users/mengren/Documents/new_projects/portfolio-optimization-black-litterman
jupyter notebook
```

Open:
- `notebooks/01_data_quality_and_universe_overview.ipynb` (schema checks, holdings coverage, portfolio composition)
- `notebooks/02_strategy_comparison_case_study.ipynb` (equity curves, drawdowns, BL weight evolution)
- `notebooks/03_black_litterman_sensitivity.ipynb` (view-confidence sweep and metric response)
- `notebooks/04_benchmark_attribution_alpha_decomposition.ipynb` (benchmark betas, alpha decomposition, rolling alpha/beta)

Note: Notebook 4 uses ETF benchmarks when present (for example `SPY`, `XLK`, `XLE`, `XLF`) and falls back to transparent ticker proxies when benchmark tickers are missing.

## Current Status and Next Steps
- Current sample data under `data/raw/` is synthetic so the project runs out-of-the-box.
- Replace synthetic data with real disclosure/price histories for production-quality conclusions.
- Extend benchmark/factor set and add transaction-cost/slippage assumptions.
- Add automated figure export from notebooks to `reports/output/figures/`.
