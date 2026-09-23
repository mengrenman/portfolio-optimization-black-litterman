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
  - `black_litterman` (equilibrium + views posterior; explicit pick-matrix views per case study).
- Metrics: annual return/volatility, Sharpe, Sortino, max drawdown, HHI concentration, turnover.
- CLI pipeline with structured logging (`--verbose` flag) that writes per-case outputs to `reports/output/<person>/`.
- Four notebooks with visual diagnostics, strategy comparison, sensitivity analysis, and benchmark attribution.
- Configurable `view_confidence` parameter exposed via YAML and `BacktestConfig`.
- Explicit absolute and relative views with per-view confidence, defined in YAML per case study.

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
  configs/                   # Case-study and backtest parameters (incl. view_confidence, views)
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
    models/                  # BL posterior, pick-matrix views, mean-variance logic
    pipeline.py              # End-to-end experiment runner
  tests/                     # Unit + integration tests (110 tests)
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

## Selected Results

Every figure below comes from the bundled dataset and the shipped configuration (6-period
lookback, month-end rebalancing, `view_confidence: 0.65`). Regenerate them with
`python scripts/run_case_study.py --person <key>`.

### Dataset at a glance

| | |
|---|---:|
| Adjusted daily closes | 90,298 rows |
| Tickers | 46 |
| Price coverage | 2018-01-02 to 2025-12-30 (2,009 trading days) |
| Disclosure rows | 51 across 3 people |
| Backtest window | 2018-08-01 to 2025-12-30 (1,864 days after lookback warm-up) |

### The disclosed portfolios

| Person | Snapshot | Holdings | Largest position | HHI |
|---|---|---:|---|---:|
| Warren Buffett | 2025-12-31 | 14 | AXP at 24.6% | 0.136 |
| Nancy Pelosi | 2024-12-31 | 22 | AAPL at 28.1% | 0.145 |
| Donald Trump | 2024-12-31 | 15 | DJT at 91.0% | 0.828 |

Every disclosed ticker has price history, so no holding is dropped from any backtest.

### Strategy comparison

| Person | Strategy | Annual return | Annual vol | Sharpe | Max drawdown | Turnover |
|---|---|---:|---:|---:|---:|---:|
| Buffett | disclosed | 17.5% | 23.5% | 0.75 | -43.4% | 0.0% |
| Buffett | mean-variance | 10.7% | 21.4% | 0.50 | -34.9% | 36.0% |
| Buffett | Black-Litterman | 14.5% | 21.3% | 0.68 | -32.3% | 29.4% |
| Pelosi | disclosed | 27.9% | 27.6% | 1.01 | -38.9% | 0.0% |
| Pelosi | mean-variance | 22.7% | 25.1% | 0.90 | -36.7% | 32.0% |
| Pelosi | Black-Litterman | 23.5% | 25.2% | 0.93 | -37.2% | 29.0% |
| Trump | disclosed | 7.5% | 147.2% | 0.05 | -84.6% | 0.0% |
| Trump | mean-variance | 10.2% | 12.0% | 0.85 | -21.2% | 27.4% |
| Trump | Black-Litterman | 6.9% | 15.1% | 0.46 | -24.0% | 25.0% |
| *benchmark* | *SPY, same window* | *14.6%* | *19.7%* | *0.74* | *-33.7%* | *n/a* |

### Reading the table

- **The overlay does not beat the disclosed book for Buffett or Pelosi**, and the honest
  explanation is hindsight. A single snapshot dated 2025-12-31 (Buffett) or 2024-12-31 (the
  others) is held all the way back to 2018, so the names are selected by having survived to
  the snapshot date. Treat the `disclosed` column as an upper bound contaminated by
  look-ahead, not as a fair competitor.
- **Pelosi's disclosed book returned 27.9% a year against SPY's 14.6%**, with a comparable
  drawdown. The same hindsight caveat applies in full.
- **Trump's 147% volatility and -84.6% drawdown are one position.** DJT is 91% of that
  snapshot and did not trade until late 2021, so the curve sits flat near 1.0 and then
  inherits the ticker's swings almost directly. Both optimisers re-estimate weights from the
  lookback window and never take on that concentration, which is why mean-variance turns a
  0.05 Sharpe into 0.85.
- **Black-Litterman lands between its two inputs in every case**, which is what the model is
  built to do rather than a disappointment.

### View confidence interpolates between the two baselines

Because the equilibrium prior is implied from the disclosed weights, confidence sweeps the
posterior from one baseline to the other. Distances are the mean L2 norm between weight
vectors across rebalances (Buffett, real data).

| Confidence | Distance to disclosed | Distance to mean-variance | Sharpe | Turnover |
|---:|---:|---:|---:|---:|
| 0.01 | 0.064 | 0.387 | 0.75 | 5.9% |
| 0.20 | 0.348 | 0.242 | 0.72 | 24.7% |
| 0.40 | 0.396 | 0.210 | 0.71 | 27.4% |
| 0.65 | 0.413 | 0.169 | 0.68 | 29.4% |
| 0.80 | 0.416 | 0.139 | 0.64 | 30.7% |
| 0.95 | 0.415 | 0.094 | 0.58 | 32.6% |
| 0.999 | 0.414 | 0.072 | 0.56 | 33.6% |

Two things worth noting. Sharpe falls monotonically as confidence rises, so on this data
trusting the trailing sample means more is actively harmful, and turnover rises with it.
And the Black-Litterman strategy is best understood as a tunable blend of the other two
rather than an independent third model.

> **Caveat on the confidence axis.** `black_litterman_posterior` adds a fixed absolute
> regularisation term of `1e-6` to the view-uncertainty matrix. That matrix holds per-period
> variances, which across the three shipped universes span roughly `7e-8` to `6e-5`, so the
> constant is anywhere from negligible to 41x the quantity it stabilises (worst case: a
> BND-versus-VGIT relative view). The practical effect is that a
> configured confidence is not the realised one, and the gap depends on asset volatility: at
> a configured 0.65 the realised value is 0.14 for MUB and 0.64 for UNG. Relative views
> between similar assets are affected most. Treat the confidence column above as ordinal
> rather than as a calibrated probability until this is scaled to the matrix.

## Configuration

All backtest and model hyper-parameters live in `configs/case_studies.yaml`:

```yaml
backtest:
  lookback_periods: 6        # Historical periods used for estimation
  rebalance_frequency: ME    # Month-end rebalancing
  risk_aversion: 2.5         # λ in π = λΣw_mkt
  tau: 0.05                  # Prior uncertainty scalar
  view_confidence: 0.65      # BL analyst confidence (0, 1]
  use_sample_mean_views: true  # stack explicit views on the sample-mean views
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

## Expressing Views with a Pick Matrix

The Black-Litterman strategy accepts explicit analyst views per case study. Each view is
one row of the pick matrix `P` with its target return in `q`. Views live under the case
study in `configs/case_studies.yaml`:

```yaml
case_studies:
  buffett:
    person_label: Warren Buffett
    disclosure_aliases: ["buffett", "warren buffett", "berkshire hathaway"]
    views:
      - label: AAPL absolute
        assets: {AAPL: 1.0}            # absolute view: one ticker, coefficient 1
        annual_return: 0.08            # AAPL returns 8% per year
        confidence: 0.6                # optional; overrides backtest.view_confidence
      - label: CVX over OXY
        assets: {CVX: 1.0, OXY: -1.0}  # relative view: long side +1, short side -1
        annual_return: 0.02            # CVX beats OXY by 2% per year
```

Rules and behaviour:

- `assets` maps tickers to pick-matrix coefficients. An absolute view has one ticker with
  coefficient 1. A relative view has positive coefficients on the outperformers and negative
  coefficients on the underperformers; by convention each side sums to 1 in absolute value.
- `annual_return` is an annualised arithmetic return. It is divided by the number of return
  periods per year (252 for daily data) so it matches the scale of the covariance matrix.
- `confidence` is optional and per view, in `(0, 1]`. Views without it use
  `backtest.view_confidence`, as does the `view_confidence` override of `run_case_study`.
- By default (`use_sample_mean_views: true`) explicit views are stacked on top of the
  built-in sample-mean views, so the existing case studies are unchanged when no views are
  configured. Set `use_sample_mean_views: false` to optimise on explicit views alone. With
  that flag off and no views, the posterior equals the prior and the Black-Litterman
  weights track the disclosed portfolio.
- A view that references a ticker outside the case study's universe is ignored with a
  warning. A view naming a ticker without a *complete* lookback window of returns (a
  recently listed name) is not applied while that is true: the estimator drops any ticker
  with missing history from the window, so the ticker also carries zero Black-Litterman
  weight there. Both resume once the ticker has `lookback_periods` full periods of history,
  which is later than its first traded day.

Programmatic use:

```python
from portfolio_bl.models.views import View, build_view_matrices

views = [
    View({"AAPL": 1.0}, annual_return=0.08, confidence=0.6),
    View({"CVX": 1.0, "OXY": -1.0}, annual_return=0.02),
]
built = build_view_matrices(views, tickers, periods_per_year=252, default_confidence=0.65)
built.p_matrix, built.q_views, built.confidences  # P (k×n), q (k,), confidence per view (k,)
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
