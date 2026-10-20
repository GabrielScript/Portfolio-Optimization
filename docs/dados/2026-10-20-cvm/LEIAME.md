# Fundamentos da CVM do lago de 20/10/2026 (03h01)

Insumo do teste da Fórmula Mágica de Greenblatt e do Trending Value de O'Shaughnessy (S2.2 e S2.3,
`scripts/avaliar_valor.py`, PLAN §5), copiado de `data/b3lake/` (lago `b3-datalake` `72f06c1`, comando
`uv run python -m b3lake.cli fundamentals`, gravado em 20/10/2026 03:01).

- `fundamentals.parquet`: contas fixas (`ST_CONTA_FIXA = S`) da DFP e do ITR desde 2010, consolidado primeiro, uma
  linha por emissor, conta e fim de período (`dt_refer`), com `available_from` = data de recebimento na CVM da versão
  do documento que os arquivos trazem (só a última). Saldos no fim do período; fluxos (receita, EBIT, lucro, caixa
  operacional, depreciação da DVA) em 12 meses (último acumulado no ano + ano anterior - acumulado do ano anterior até
  o mesmo trimestre), disponíveis na maior data de recebimento dos documentos usados.
- `market_caps.parquet`: valor de mercado de cada emissor na decisão de cada mês (1º dia): ações ordinárias e
  preferenciais conhecidas antes do mês, cada espécie pelo último fechamento cru do mês anterior (a espécie sem papel
  negociado leva o preço da outra). Contagem: composição do capital da DFP/ITR desde 2020, sem tesouraria; antes, o
  capital integralizado do FRE, datado pelo recebimento, com tesouraria; as duas levadas às unidades do fechamento
  pelos fatores de desdobramento do lago e conferidas pela contagem atual da B3 e pelo P/VP (fora do Financeiro).

Os outros insumos do script ficam nas pastas já guardadas, conferidos pelo `SHA256SUMS` de cada uma: retornos
mensais e proventos em `docs/dados/2026-10-20/`, preços diários em `docs/dados/2026-10-19/`, universo ponto no tempo e
setores em `docs/dados/2026-10-18/`. O script para se algum hash não bater.
