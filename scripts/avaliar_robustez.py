"""
Robustez fora da amostra (PLAN §11, passo 3): janelas móveis, subperíodos, P[> BOVA11] em log-retorno, MinTRL, DSR
contra o BOVA11 e a janela de estimação de 10 anos (R10). Descritivo: não muda o paper trading (paper-v1) nem o
critério de pronto. Rede só para o CDI.

Uso: .venv/Scripts/python -X utf8 scripts/avaliar_robustez.py > docs/robustez-2026-10-19.txt

Pré-registro (2026-10-19, escrito e commitado antes de rodar; depois desse commit o texto não muda):
- Dados: `docs/dados/2026-10-19/universe_prices.parquet` (o de 18/10 mais o IVVB11 diário) e, de
  `docs/dados/2026-10-18/`, `universe_monthly.parquet` e `sectors.parquet`. O script confere os três pelo
  `SHA256SUMS` de cada pasta e para se algum não bater.
- Motor e regras do `scripts/avaliar_regras_no_motor.py`, sem mudança: universo ponto no tempo, treino de 1.260
  pregões (R5), rebalanceio a cada 63, 10 bps por lado, Jorion, taxa livre de risco = CDI na data (Selic constante
  se o BCB falhar, com aviso), peso máximo 20%, até 10 ativos, teto de vol 40%, teto de 30% por setor no otimizador
  (setor do lago), `checar_carteira` recusando, perda diária de 5% no satélite congelando o pregão seguinte.
- Diferenças conhecidas para o satélite do paper (`scripts/paper_trading.satelite_do_dia`), escritas como limitação:
  o paper usa a lista líquida atual do app (com viés de sobrevivência), o setor do `assets.py` e a Selic constante
  como taxa livre de risco; aqui o universo é ponto no tempo, o setor é o do lago e a taxa é o CDI na data.
- Fases: o rebalanceio cai em pregões diferentes conforme o começo. O teste roda 3 fases, cortando 0, 21 e 42
  pregões do começo do arquivo de preços; a fase 0 é o arquivo inteiro. Tudo é reportado nas 3; nenhuma escolhe nada.
- Curvas (lista fechada; as 3 entram no DSR):
  1. "Kelly λ=2 com regras": o satélite com as regras 1 e 2, sem o stop (mede o otimizador);
  2. "Dinheiro real": BOVA11 70% + a curva 1 com o stop relativo 30%, de volta aos 70/30 a cada 63 pregões, com
     10 bps sobre o que muda e a volta congelada pela perda diária do satélite (`misturar_congelando`). O stop vale
     até o fim do horizonte: em cada janela, subperíodo ou período, ele é armado de novo no começo (o pico da razão
     satélite/BOVA11 começa ali), como para quem começou a investir ali;
  3. "Núcleo Mín. Vol 70/30 com regras": o padrão do estudo na tela, sem o stop (como no `avaliar_regras_no_motor`).
  Referências fora do N: BOVA11 e IVVB11 (desde 29/04/2014: janela ou subperíodo que começa antes fica sem ele).
- Período fora da amostra: da 1ª decisão de cada fase (~jan/2010) a 16/10/2026.
- Janelas móveis (`core.robustez.janelas_moveis`): 36 e 60 meses, uma começando em cada fim de mês. Por curva,
  comprimento e fase: nº de janelas, fração com CAGR acima do BOVA11 (riqueza), mediana e percentil 10 do excesso de
  CAGR, fração com P[> BOVA11] ≥ 0,95 em média aritmética e em log-retorno, pior MDD. Janelas sobrepostas não são
  amostras independentes: os números descrevem, não testam.
- Subperíodos fixados aqui: 2010–2014, 2015–2019 e 2020 a 16/10/2026 (curva recortada do último pregão antes do
  subperíodo ao último dentro dele): CAGR, excesso de CAGR sobre o BOVA11, P aritmético e em log, MDD.
- Período inteiro: Sharpe (excesso sobre o CDI), IR contra o BOVA11 (diferença diária, anualizada por √252),
  P[> BOVA11] aritmético e em log, DSR do IR (SR0 de `sharpe_referencia_deflacionado` com N = linhas de
  `docs/tentativas.csv` e V = variância dos IR diários das 3 curvas na mesma fase) e MinTRL
  (`anos_minimos_para_psr`: anos para o P aritmético chegar a 0,95 se o IR observado for o verdadeiro). O critério
  de pronto continua o do ROTEIRO §2 (PSR aritmético ≥ 0,95 contra o BOVA11); o P em log fica ao lado, não no lugar.
- R10 (DECISOES 61b): walk-forward com treino de 2.520 pregões; a R5 é a mesma estratégia estimada só nos últimos
  1.260 (`com_janela`), então as duas decidem nas mesmas datas e no mesmo universo (papéis com 10 anos de histórico
  na data: esta R5 não é a do paper nem a da parte de cima). Curva comparada: a 1 (o satélite sem stop; o P da
  diferença não muda com a fração fixa da mistura). Horizontes: os últimos 5 e 10 anos até 16/10/2026, curva
  recortada, nas 3 fases. A R10 passa se, nos dois horizontes e nas 3 fases: C1 P[R10 > R5] ≥ 0,95 (PSR aritmético
  da diferença diária); C2 MDD da R10 no máximo 5 p.p. pior que o da R5; C3 giro anual da R10 ≤ 2 × o da R5. Passar
  não muda o paper-v1: vira candidata a um pré-registro novo (paper-v2), registrada no DECISOES.
- Expectativa registrada: em janelas de 60 meses o satélite fica acima do BOVA11 em 40–65% delas, com percentil 10
  do excesso de CAGR abaixo de −8 p.p. ao ano e P em log ≥ 0,95 em menos de 15% delas; o "Dinheiro real" fica a
  poucos pontos do BOVA11 em quase todas; o sinal do excesso muda entre os subperíodos; MinTRL acima de 10 anos e DSR
  abaixo de 0,95 nas 3 curvas; R10 não passa (P[R10 > R5] entre 0,3 e 0,8).
"""
