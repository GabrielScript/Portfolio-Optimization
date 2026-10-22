# Lake macro series of 2026-10-21

Inputs of `scripts/avaliar_acoes_titulos.py` and `scripts/avaliar_filtros_macro.py`, copied from the lake's exports
(`D:/b3-datalake/exports/`, lake `main` = `3c96b53`).

- `cdi_daily.parquet`: daily CDI (BCB SGS 12) from 06/03/1986 to 19/10/2026, `date` and `cdi_pct` (% per day);
  written by the nightly of 20/10/2026 (22:27), the same file the app reads in `data/b3lake/`.
- `real_curve.parquet`: constant-maturity real yields (`date`, `tenor_years` 2/5/10, `real_rate`, `source`) from
  31/12/2004 to 21/08/2026, written by `b3lake.cli tesouro` on 2026-10-21 14:17. Built from the Tesouro Direto's
  yearly workbooks (cdn.tesouro.gov.br, public): the NTN-B Principal (zero-coupon) sell rates, interpolated linearly
  in years to maturity at each tenor, the coupon NTN-B only where no pair of Principals brackets the tenor (before
  07/2005, or past the longest Principal); no extrapolation. The CDN's 2026 workbooks stop at 21/08/2026.
- Point in time: the CDI of day d is in the level of d (the app's convention, `market_data.serie_rf_diaria`); the
  Tesouro's morning quote of day d is public on d.
