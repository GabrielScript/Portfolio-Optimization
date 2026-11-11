# Monthly liquidity from the lake (11/11/2026)

Input of the volatility-target replay (`scripts/vol_target_motor_avaliar.py`, spec
`docs/superpowers/specs/2026-11-11-vol-target-motor-design.md`), cut by `scripts/exportar_liquidez.py` from the lake's
`data/b3lake/monthly_metrics.parquet` (sha256 `6cf3841c58b7565273a2285c3f342d81a848fbadd99f157b705a0fe70b4afcd0`).

- `liquidez.parquet`: one row per month and name, 2009-11 to 2026-10, for the names in the point-in-time universe
  (`docs/dados/2026-10-22/universe_monthly.parquet`) in some month since 2009-12, plus BOVA11 (370 names, 46,821 rows).
  Same columns and definitions as `docs/dados/2026-10-20-liquidez` (half the quoted spread, `vol_daily`,
  `median_dtv_brl`). Gaps stay empty (27 spread, 83 volatility); the replay uses the month's median.
- Against the 20/10 cut, on its 40,779 common rows: 225 rows of 2026-10 differ (that file stopped at 19/10) and 11 rows
  of earlier months differ (lake corrections since then). The rest is identical.

The script checks the file against `SHA256SUMS` and stops on a mismatch.
