`disclosures.csv` schema:
- `person`
- `as_of_date` (YYYY-MM-DD)
- `ticker`
- `value_usd`
- `source` (optional but strongly recommended)

## Current bundled dataset
The current `disclosures.csv` is populated from public filings:
- Warren Buffett: Berkshire Hathaway SEC Form 13F-HR (period end `2025-12-31`, filed `2026-02-17`).
- Nancy Pelosi: U.S. House financial disclosure (`10066169`, year `2024`).
- Donald Trump: OGE 278e annual report (`2025` filing year).

Coverage note:
- Buffett rows currently include a mapped subset of major 13F positions with clear ticker mapping in this starter dataset.
- Pelosi and Trump rows are range-based disclosures converted to numeric estimates.

## Normalization notes
- For exact dollar disclosures (for example SEC 13F line items), `value_usd` uses reported totals.
- For value ranges (House/OGE reports), `value_usd` uses midpoint estimates.
- For open-ended ranges (for example “Over $50M”), `value_usd` uses a lower-bound placeholder.

Example row:
`Warren Buffett,2025-12-31,AAPL,38268369638,SEC 13F-HR ...`
