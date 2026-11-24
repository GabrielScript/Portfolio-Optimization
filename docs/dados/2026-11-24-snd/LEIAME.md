# SND debenture data (2026-11-24): hashes only

The files live on `D:/b3-datalake/raw/snd/` and are never committed (DECISOES 110: public repository; ANBIMA data).
Downloaded on 2026-11-24 with the user's authorization (full SND history).

- `snd_trades.parquet`: secondary-market trades, 2004-01-02 .. 2026-11-23, 939,223 rows, active + matured papers.
  Missing sessions (SND internal server error to every query): 2020-12-16, 2021-04-09.
- `caracteristicas/caracteristicas.txt`: SND characteristics export, 10,091 debentures (CNPJ, indexer, terms).

Check with `sha256sum -c` from `D:/b3-datalake/raw/snd/`.
