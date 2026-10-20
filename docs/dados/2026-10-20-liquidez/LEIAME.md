# Liquidez mensal do lago (20/10/2026)

Insumo da prova final com execução realista (S4, `scripts/avaliar_execucao.py`, PLAN §11), recortado de
`data/b3lake/monthly_metrics.parquet` (sha256 `8cf0f1df986d7a1209f633637f7f76639694670c7f591117d7370861f9385466`; lago
`b3-datalake`, manifest de 19/10/2026 22:40, o mesmo de `docs/dados/2026-10-20/`, dados até 19/10/2026).

- `liquidez.parquet`: uma linha por mês e papel, de 2011-11 a 2026-10, para os papéis que entram no universo ponto no
  tempo (`docs/dados/2026-10-18/universe_monthly.parquet`) em algum mês desde 2011-12, mais o BOVA11 (342 papéis).
  - `meio_spread`: metade do `quoted_spread` do lago (`sql/metrics.sql`: média do mês de (melhor oferta de venda -
    melhor oferta de compra) / ponto médio no fechamento do COTAHIST, nos pregões com as duas ofertas).
  - `vol_diaria`: `vol_daily` do lago (desvio-padrão amostral dos retornos totais diários do mês).
  - `volume_mediano`: `median_dtv_brl` do lago (volume financeiro diário mediano do mês, R$).
- O mês 2026-10 vai só até 19/10. Faltas do lago ficam vazias (24 de `meio_spread`, 67 de `vol_diaria`); o S4 usa a
  mediana do mês no lugar.

O script confere o arquivo pelo `SHA256SUMS` e para se o hash não bater.
