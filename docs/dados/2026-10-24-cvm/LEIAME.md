# CVM fundamentals of 2026-10-24 (with gross profit)

Inputs of `scripts/avaliar_classicas.py` (Piotroski, Acquirer's Multiple and Magic Formula against the Conservadora,
DECISOES 130). Built by the lake's `fundamentals` command at commit `0b0bfc6` (24/10/2026 16:35), which adds the DRE
gross result (account 3.03) as `gross_profit`; every other account comes from the same code as
`docs/dados/2026-10-22-cvm/`.

- `fundamentals.parquet`: 417,773 account rows of 574 issuers (25,577 of them `gross_profit`), DFP 2010+ and ITR
  2011+, each value with the receipt date of its document at the CVM (`available_from`).
- `market_caps.parquet`: market value at the start of each month (2010-07 to 2026-10) with the split-adjusted share
  count (`shares_today_units`), after the lake's 1000x checks of session 39 (OIBR 2018-19 and PDGR 2015 fixed).

Against the 2026-10-20 snapshot, 176 account values differ: documents the CVM received again since then (the files
keep only the latest version of each document). Known limits are the lake's (README): about 70 filings a thousand
times off in some accounts, restated values known only from the re-filing date.
