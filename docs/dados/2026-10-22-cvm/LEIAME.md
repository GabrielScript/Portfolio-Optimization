# CVM fundamentals of 2026-10-20 with the market values re-checked (2026-10-22)

A data correction for the macro filters (`scripts/avaliar_filtros_macro.py`, DECISOES 112 and 114). Only the market
values change; the formula, the configurations and N (38) do not.

- `fundamentals.parquet`: the same file as `docs/dados/2026-10-20-cvm/` (same SHA256).
- `market_caps.parquet`: `docs/dados/2026-10-20-cvm/market_caps.parquet` with its old `book_scale` undone
  (`market_cap / book_scale`, `shares_today_units / book_scale`) and the lake's `plausible_market_caps` of commit
  `3537a8f` applied again with that `fundamentals.parquet` (all sectors). Same rows, columns and types.

What changed (the old check skipped issuers without positive equity and banks): GOLL 2015-2025 (108 months, it was
~R$ 40 trillion and 90%+ of the universe's market value), RCSL, PRBC, AZEV, PDGR, LUPA, MTIG, IFCM, OIBR (2014-2015
filing), AZUL 2026 and a few small issuers: 466 issuer-months divided by 1000. Known debt, not fixed: OIBR
06/2018-05/2019 and PDGR 06-07/2015 are still a thousand times too big (no month of their filing crosses the check);
AZUL 2026 and OIBR are split-timing errors that the /1000 only shrinks.

The other inputs of the script keep their own folders and SHA256SUMS.
