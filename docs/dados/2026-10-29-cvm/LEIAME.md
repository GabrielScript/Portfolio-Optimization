# CVM fundamentals of 2026-10-29 (share-count slips fixed)

Built by the lake's `fundamentals` command at commit `7f365a3` (29/10/2026 09:59), which fixes the 1000x share-count
slips found in session 58 (DECISOES 170). The account code is the same as `docs/dados/2026-10-24-cvm/`.

- `fundamentals.parquet`: 417,930 account rows of 574 issuers: the 417,773 rows of 2026-10-24 unchanged, plus 157
  rows of documents the CVM received since then.
- `market_caps.parquet`: 54,472 issuer-months of market value (2010-07 to 2026-10). Against 2026-10-24: 220
  issuer-months of 43 issuers move more than 10x (86 of them in the liquidity universe), and PRML 2011-03/2012-05 and
  SMTO 2011-09/2012-08 (filings with zero shares, market value 0) are gone.

The three causes: a CVM code shared by two or three tickers (VVAR/BHIA, OIBR/BRTO/TEPR...) duplicated filings with
conflicting scales; an equity or assets value filed in thousands misled the price-to-book check (AMAR 12/2023, TELB
06/2014, SAUD 06/2021); runs of months a thousand times off with no exact x1000 between filings (RCSL, GFSA, OIBR
2023 and 2025-26, IFCM 06/2025). Known limits are the lake's (README): IPO months with no history before them, and
illiquid names whose scale flips every few months (BLUT 2013-14, TCNO 2019-20).

`scripts/sensibilidade_caps.py` reads only `market_caps.parquet` from here, paired with the 2026-10-24
`fundamentals.parquet`, so that the market value is the only input that changes.
In its output the `Files (sha256)` line carries both market-value files: `market_caps.parquet` is the
2026-10-24 folder's, hashed with its statements and not used; `market_caps.parquet (2026-10-29)` is this one.
