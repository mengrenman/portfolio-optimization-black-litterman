# Notebooks

This folder contains visual walkthroughs for the full Black-Litterman case-study workflow.

## Prerequisites
From the repository root:

```bash
python -m pip install -e '.[dev,notebooks]'
```

## Launch
From the repository root:

```bash
jupyter notebook
```

## Notebook Guide
1. `01_data_quality_and_universe_overview.ipynb`
   - Loads disclosure and price datasets.
   - Validates schema and basic coverage.
   - Visualizes latest holdings composition for a selected person.

2. `02_strategy_comparison_case_study.ipynb`
   - Runs end-to-end case study for one person.
   - Compares `disclosed`, `mean_variance`, and `black_litterman`.
   - Visualizes equity curves, drawdowns, and BL rebalance weights.

3. `03_black_litterman_sensitivity.ipynb`
   - Sweeps Black-Litterman view confidence.
   - Charts sensitivity of return, volatility, Sharpe, HHI, and turnover.
   - Helps choose stable confidence ranges before reporting.

4. `04_benchmark_attribution_alpha_decomposition.ipynb`
   - Performs benchmark exposure regression and alpha decomposition.
   - Uses ETF benchmarks when available (for example `SPY`, `XLK`, `XLE`, `XLF`).
   - Falls back to transparent ticker proxies when ETF benchmark tickers are missing.
   - Visualizes betas, contribution decomposition, and rolling alpha/beta.

## Data Notes
- `data/raw/disclosures/disclosures.csv` uses public-filing-derived holdings for Buffett, Pelosi, and Trump.
- `data/raw/prices/prices.csv` is synthetic for runnable demos; replace with real market prices for research-quality conclusions.

## Related Outputs
- Batch script outputs are written to `reports/output/<person>/`.
- You can use notebook results alongside:
  - `summary.csv`
  - `equity_curve.csv`
  - `strategy_returns.csv`
  - `weights_<strategy>.csv`
  - `metadata.csv`
