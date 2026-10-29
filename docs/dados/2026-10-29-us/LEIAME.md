# US sleeves and PTAX for H2 (frozen 2026-10-29, cut at 16/10/2026)

- `spy_gld_daily.parquet` (ticker, date, close): SPY and GLD daily closes from the lake's `us_prices_listed.parquet`
  (Yahoo Finance via yfinance 1.7.0, `auto_adjust=True`: dividend-adjusted, so SPY is a total-return index; GLD pays
  none). SPY 1993-01-29 to 2026-10-16 (8,487 rows), GLD 2004-11-18 to 2026-10-16 (5,512 rows). Index ETFs that still
  trade: the lake's survivorship caveat for single stocks does not bite here.
- `ptax_daily.parquet` (date, ptax): BCB SGS 1, PTAX selling rate, R$ per US$, from the lake's `ptax` step
  (branch `feat/cda-ptax`), 1994-07-01 to 2026-10-16 (8,106 rows).
