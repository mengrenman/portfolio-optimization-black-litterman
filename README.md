# portfolio-optimization-black-litterman

Black-Litterman case-study framework for evaluating publicly disclosed portfolios from notable figures.

## Project Question
Can a Black-Litterman overlay improve portfolio quality relative to:
1. The disclosed portfolio itself
2. A mean-variance (sample-estimated) baseline

## Implemented Capabilities
- Disclosure and price ingestion with schema validation and descriptive error messages.
- Latest disclosed portfolio reconstruction by person aliases.
- Rolling rebalancing backtest engine (lookback-based, no look-ahead application).
- Strategy comparison:
  - `disclosed` (static disclosed weights),
  - `mean_variance` (sample-estimated Markowitz),
  - `black_litterman` (equilibrium + views posterior).
- Metrics: annual return/volatility, Sharpe, Sortino, max drawdown, HHI concentration, turnover.
- CLI pipeline with structured logging (`--verbose` flag) that writes per-case outputs to `reports/output/<person>/`.
- Four notebooks with visual diagnostics, strategy comparison, sensitivity analysis, and benchmark attribution.
- Configurable `view_confidence` parameter exposed via YAML and `BacktestConfig`.

## Important Caveats
- Public disclosures are delayed, incomplete, and sometimes approximate.
- Disclosure quality is source-dependent; conclusions are only as good as the input coverage.
- This repo is for research only, not investment advice.

## Known Model Limitation

The BL posterior assumes multivariate Gaussianity in both returns and views. Equity
return distributions exhibit significant excess kurtosis and left-tail asymmetry,
particularly during stress periods directly visible in this backtest — the 2020 COVID
drawdown and the 2022 rate shock. A production-grade system would replace the Gaussian
prior with Meucci's Entropy Pooling / Copula Opinion Pooling framework, which separates
the marginal distributions from the dependence structure and allows views to be expressed
on arbitrary statistics including implied volatility.

## Repository Layout
```text
portfolio-optimization-black-litterman/
  configs/                   # Case-study and backtest parameters (incl. view_confidence)
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
  tests/                     # Unit + integration tests (63 tests)
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
cd portfolio-optimization-black-litterman
python -m pip install -e '.[dev,notebooks]'
pytest -q
python scripts/run_case_study.py --person buffett
python scripts/run_case_study.py --person pelosi
python scripts/run_case_study.py --person trump
```

Pass `--verbose` to enable debug-level logging:
```bash
python scripts/run_case_study.py --person buffett --verbose
```

Generated outputs include:
- `summary.csv`
- `equity_curve.csv`
- `strategy_returns.csv`
- `weights_<strategy>.csv`
- `metadata.csv`

All written under `reports/output/<person>/`.

## Configuration

All backtest and model hyper-parameters live in `configs/case_studies.yaml`:

```yaml
backtest:
  lookback_periods: 6        # Historical periods used for estimation
  rebalance_frequency: ME    # Month-end rebalancing
  risk_aversion: 2.5         # λ in π = λΣw_mkt
  tau: 0.05                  # Prior uncertainty scalar
  view_confidence: 0.65      # BL analyst confidence (0, 1]
```

`view_confidence` controls how strongly the analyst's sample-mean views override the
Black-Litterman equilibrium prior. Higher values reduce view uncertainty (Ω) and pull
the posterior mean toward the views. It can also be overridden programmatically:

```python
from portfolio_bl.config import load_config
from portfolio_bl.pipeline import run_case_study

cfg = load_config("configs/case_studies.yaml")
result = run_case_study(cfg, person_key="buffett", view_confidence=0.80)
```

## Data Source Snapshot
- Buffett holdings come from Berkshire Hathaway's latest SEC 13F filing (as of `2025-12-31`) with a major-position ticker-mapped subset in this starter dataset.
- Pelosi holdings come from U.S. House financial disclosure report `10066169` (range-based values converted to midpoints).
- Trump holdings come from OGE 278e annual disclosure (range-based values converted to midpoints/lower bounds).
- Bundled `prices.csv` contains real adjusted daily closes sourced from Yahoo Finance (2018–2025); see `data/raw/prices/README.md` for the refresh script.

## Notebooks

Launch Jupyter with:
```bash
jupyter notebook
```

| # | Notebook | Description |
|---|----------|-------------|
| 1 | [Data Quality & Universe Overview](notebooks/01_data_quality_and_universe_overview.ipynb) | Schema checks, holdings coverage, portfolio composition |
| 2 | [Strategy Comparison Case Study](notebooks/02_strategy_comparison_case_study.ipynb) | Equity curves, drawdowns, BL weight evolution |
| 3 | [Black-Litterman Sensitivity](notebooks/03_black_litterman_sensitivity.ipynb) | View-confidence sweep and metric response |
| 4 | [Benchmark Attribution & Alpha Decomposition](notebooks/04_benchmark_attribution_alpha_decomposition.ipynb) | Benchmark betas, alpha decomposition, rolling alpha/beta |

Note: Notebook 4 uses ETF benchmarks when present (e.g. `SPY`, `XLK`, `XLE`, `XLF`) and falls back to transparent ticker proxies when benchmark tickers are missing.

## Data Status

Both input datasets now use real data.

| Dataset | Source | Coverage |
|---|---|---|
| `data/raw/disclosures/disclosures.csv` | SEC 13F, House FD, OGE 278e (public filings) | Buffett (2025-12-31), Pelosi (2024-12-31), Trump (2024-12-31) |
| `data/raw/prices/prices.csv` | Yahoo Finance via yfinance (`auto_adjust=True`) | 46 tickers, 2018-01-02 → 2025-12-30, ~90k rows |

To refresh prices with the latest data, see `data/raw/prices/README.md`.

## Current Status and Next Steps
- Extend benchmark/factor set (e.g. factor-model attribution against Fama-French).
- Add transaction-cost and slippage assumptions to the backtest engine.
- Add automated figure export from notebooks to `reports/output/figures/`.
