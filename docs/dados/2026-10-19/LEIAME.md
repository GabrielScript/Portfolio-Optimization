# Exports do lago de 19/10/2026

Insumo do teste de robustez de 19/10/2026 (`scripts/avaliar_robustez.py`, PLAN §11, passo 3), copiado de
`data/b3lake/` (lago `b3-datalake` `7daf7bc`, manifest de 19/10/2026 16:51, dados até 16/10/2026).

`universe_prices.parquet` é o de 18/10 (`docs/dados/2026-10-18/`, sha256 b43dab68…) com uma coluna a mais, o IVVB11
diário desde 29/04/2014: sem ela os dois arquivos são idênticos (mesmas datas, mesmos valores, conferido com
`pandas.testing.assert_frame_equal`). `universe_monthly.parquet` e `sectors.parquet` não mudaram: o script lê os de
`docs/dados/2026-10-18/`, conferidos pelo `SHA256SUMS` de lá.

O script lê estes arquivos daqui (não de `data/b3lake/`, que a rotina do lago sobrescreve) e para se algum hash não
bater.
