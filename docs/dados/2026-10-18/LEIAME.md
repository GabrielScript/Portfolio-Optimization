# Exports do lago de 18/10/2026

Insumos das rodadas de 18/10/2026 (`docs/regras-no-motor-2026-10-18.txt`, Parte 1 = S1-P.b nos dados corrigidos) e do
1º registro do paper trading, copiados de `data/b3lake/` (lago `b3-datalake` `023f28e`, manifest de 18/10/2026 14:22,
dados até 16/10/2026). `SHA256SUMS` confere com os hashes impressos nessas saídas.

A saída original do S1-P.b (`docs/universo-dinamico-2026-10-18.txt`) leu um `universe_prices.parquet` anterior
(sha256 4a1fc288…), sobrescrito em 18/10 pela correção do custo de bonificação (lago `89803d5`); aquele arquivo não
existe mais e a saída só se confere pelo hash. Os preços diários do 1º registro do paper estão no git em
`lago/precos_diarios.parquet` do commit `260fd9a` (recorte em `tests/dados/precos_paper_v1.parquet`).

Para rodar um script de 18/10 sobre estes arquivos: copiar para `data/b3lake/` (ignorada pelo git).
