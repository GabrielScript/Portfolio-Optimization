# Exports do lago de 19/10/2026 (22h40)

Insumo do teste da Conservadora de Blitz e van Vliet (S2.1, `scripts/avaliar_conservadora.py`, PLAN §5), copiado de
`data/b3lake/` (lago `b3-datalake` `7daf7bc`, manifest de 19/10/2026 22:40, dados até 19/10/2026).

- `monthly_total_returns.parquet`: retorno total de cada mês (1994-07 em diante), todos os papéis.
- `dividends.parquet`: dinheiro pago por ação na data ex (dividendo, JCP líquido do IR na fonte, rendimento,
  restituição de capital) e o `yield` sobre o fechamento cru do dia ex.

Os preços diários, o universo ponto no tempo e os setores não mudaram desde os arquivos já guardados (mesmo sha256):
o script lê `universe_prices.parquet` de `docs/dados/2026-10-19/` e `universe_monthly.parquet` e `sectors.parquet` de
`docs/dados/2026-10-18/`, conferidos pelo `SHA256SUMS` de cada pasta. O script para se algum hash não bater.
