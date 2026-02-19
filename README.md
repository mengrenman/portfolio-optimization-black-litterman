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

## Data Source Snapshot
- Buffett holdings come from Berkshire Hathaway's latest SEC 13F filing (as of `2025-12-31`) with a major-position ticker-mapped subset in this starter dataset.
- Pelosi holdings come from U.S. House financial disclosure report `10066169` (range-based values converted to midpoints).
- Trump holdings come from OGE 278e annual disclosure (range-based values converted to midpoints/lower bounds).
- Bundled `prices.csv` is synthetic to keep examples runnable; replace with real prices for production analysis.

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

## Notebook Analysis Data Status (Updated February 19, 2026)
- The notebooks now ingest filing-derived `data/raw/disclosures/disclosures.csv` by default (Buffett, Pelosi, Trump).
- Portfolio composition, cross-person coverage, and strategy weights reflect these filing-derived holdings when you run the notebooks.
- Performance/backtest results still depend on `data/raw/prices/prices.csv`, which is currently synthetic.
- For research-grade conclusions, replace `data/raw/prices/prices.csv` with real historical market prices.

## Current Status and Next Steps
- `data/raw/disclosures/disclosures.csv` now uses real public-filing-derived holdings for Buffett, Pelosi, and Trump.
- `data/raw/prices/prices.csv` remains synthetic and should be replaced with real price histories.
- Extend benchmark/factor set and add transaction-cost/slippage assumptions.
- Add automated figure export from notebooks to `reports/output/figures/`.
