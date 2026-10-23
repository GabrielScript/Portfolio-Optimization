# Lake exports of 2026-10-21 (21:48), frozen on 2026-10-22

Input of the Conservadora's descriptive rerun (`scripts/rerun_conservadora.py`, DECISOES 106 and 115), copied from
`D:/b3-datalake/exports` as the nightly of 21/10/2026 published them (lake `main` = `3c96b53`, data through
21/10/2026). It is the first export with the CODBDI 58 quotes kept in gold and the RPMG3 2012 crash read as a crash
(DECISOES 103-104), and with the audit fixes of DECISOES 96-97.

- `universe_prices.parquet` and `manifest.json`: the daily total-return index (the rerun cuts it at 16/10/2026).
- `universe_monthly.parquet`: the point-in-time universe (rule D1); it gains 71 security-months (GOLL4, AZUL4, RPMG3,
  ENEV3, MMXM3, VAGV4) because the CODBDI 58 sessions now count in the liquidity rule.
- `sectors.parquet`: unchanged (same SHA256 as `docs/dados/2026-10-18/`).
- `monthly_total_returns.parquet` and `dividends.parquet`: the monthly total returns and the cash events.

The rerun checks the five files against `SHA256SUMS` and stops on a mismatch.
