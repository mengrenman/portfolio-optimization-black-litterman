`prices.csv` schema:
- `date` (YYYY-MM-DD)
- `ticker`
- `close`

## Current dataset

**90,298 rows of real adjusted daily close prices** sourced from Yahoo Finance via
[yfinance](https://github.com/ranaroussi/yfinance) (split- and dividend-adjusted).

| Field | Value |
|---|---|
| Source | Yahoo Finance (yfinance ≥ 0.2, `auto_adjust=True`) |
| Coverage | 2018-01-02 → 2025-12-30 |
| Tickers | 46 (all disclosed tickers across Buffett, Pelosi, Trump portfolios) |
| Frequency | Daily (trading days only) |

Tickers with a shorter history (post-IPO): CRWD (from 2019-06-12), DBX (from
2018-03-23), RBLX (from 2021-03-10), DJT (from 2021-09-30). The backtest engine
handles missing early history gracefully via the universe intersection and
lookback eligibility checks.

## Refreshing the data

```python
import yfinance as yf
import pandas as pd

TICKERS = [
    "AAPL", "AMZN", "AXP", "BAC", "BND", "CB", "CLNE", "CMCSA",
    "CRM", "CRWD", "CVX", "DBX", "DGRO", "DIS", "DJT", "DVA",
    "EMB", "GOOGL", "IBKR", "KHC", "KO", "LQD", "MA", "MCO",
    "MORN", "MSFT", "MUB", "NFLX", "NVDA", "OXY", "PANW", "PYPL",
    "QCOM", "RBLX", "SCHD", "SPY", "T", "UNG", "V", "VDE",
    "VGIT", "VIG", "VNQ", "VPL", "WBD", "XLE",
]

raw = yf.download(TICKERS, start="2018-01-01", auto_adjust=True, progress=False)["Close"]
long = (
    raw.reset_index()
    .melt(id_vars="Date", var_name="ticker", value_name="close")
    .rename(columns={"Date": "date"})
    .dropna(subset=["close"])
    .query("close > 0")
    .assign(date=lambda df: df["date"].dt.strftime("%Y-%m-%d"))
    .sort_values(["date", "ticker"])
    .reset_index(drop=True)
)
long.to_csv("data/raw/prices/prices.csv", index=False)
```
