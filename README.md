# 📊 Otimizador de Carteiras B3

> **Autor:** Gabriel Estrela Lopes

Dashboard em Python/Streamlit para otimizar carteiras de ações da B3 com foco no **investidor arrojado**.
Usa Markowitz/Kelly com estimadores robustos (Ledoit-Wolf + shrinkage estilo Jorion), alocação hierárquica
(HRP) e um **walk-forward fora da amostra, líquido de custos**, para que toda estratégia seja comparada nas
mesmas condições.

---

## 🚀 Funcionalidades

### 🎯 Objetivos de otimização
- **Núcleo Mín. Vol + satélite Kelly (padrão do estudo, só ações)** — (1 − f)·w_MínVol + f·w_Kelly, com o satélite f de
  0 a 50% (padrão 30%). Foi a mistura aprovada no teste pré-registrado sem viés de sobrevivência
  (`docs/universo-dinamico-2026-10-18.txt`): em 10 anos, Sharpe 0,35 × 0,36 do Kelly com drawdown de −46% × −63%.
- **Crescimento máximo (Kelly)** — `max μ'w − (λ/2)·w'Σw`, com aversão λ ajustável (padrão λ = 2). Opção: no mesmo
  teste, perdeu Sharpe para a Mínima variância em 5 anos.
- **Máximo Sharpe** — resolvido de forma exata (transformação de Charnes-Cooper, problema convexo).
- **Mínima variância.**
- **HRP** — Hierarchical Risk Parity (López de Prado); ignora o retorno esperado. Na tela, é projetado nos tetos
  antes de virar plano; nas comparações do walk-forward, fica puro.
- **Máximo retorno com teto de volatilidade** — mantido como legado: gera soluções de canto que concentram
  o erro de estimação.

### ⚙️ Restrições e parâmetros
- **Regras de risco fixas** (`core/risco.py`), conferidas na carteira de hoje e nos pesos efetivos do plano: sem
  alavancagem (pesos ≥ 0, soma ≤ 100%), até 20% por ação e por emissor, 30% por setor (papel sem setor também
  recusa) e 40% a.a. de volatilidade ex-ante. A tela só aperta: os sliders param nesses valores. Carteira fora das
  regras não vira plano. Stop relativo do satélite (razão satélite/BOVA11 −25% desde o pico) e pausa de um pregão
  depois de perda diária de 5% já existem como funções; valem quando houver posições registradas no paper trading
  (`core/registro.py`). A rotina diária que registra o satélite e aplica as duas regras (`scripts/paper_trading.py`,
  pré-registrada, com a decisão do item 3 do critério de pronto fixada antes) roda agendada de segunda a sexta às
  8h30. O 1º registro, no fechamento de 16/10/2026 (`paper/carteiras.csv`), começou o relógio de 6 a 12 meses.
- **As mesmas regras no motor do walk-forward com deslistadas** (`estrategia_otimizada_nas_regras`, `Congelamento`,
  recusa = mantém a carteira), testadas num pré-registro (`docs/regras-no-motor-2026-10-18.txt`): 0 recusas em 60
  rebalanceios; o stop relativo disparou nos dois horizontes (05/2025 e 12/2025); o padrão com regras teve Sharpe
  −0,06 (−0,13 sem) em 5 anos e 0,28 (0,35 sem) em 10. O teto de setor custa à Mín. Vol em 10 anos (0,19 × 0,28).
- **Robustez fora da amostra, 2010–2026** (`scripts/avaliar_robustez.py`, pré-registrado; saída em
  `docs/robustez-2026-10-19.txt`): janelas móveis de 3 e 5 anos, subperíodos, P[> BOVA11] também em log-retorno,
  DSR com as tentativas de `docs/tentativas.csv` e MinTRL, em 3 fases de rebalanceio. O satélite com regras teve IR
  de 0,32 a 0,37 contra o BOVA11 (P 0,90–0,93, MinTRL de 20 a 27 anos); o plano do dinheiro real, com o stop
  relativo, de 0,04 a 0,18 (o stop dispara em 2 a 3 anos na maioria das janelas de 5 anos); o Núcleo Mín. Vol 70/30
  chegou a P 0,95–0,96 no período inteiro, mas só por causa de 2010–2014 e com DSR abaixo de 0,95: não cumpre o
  critério de pronto. A janela de estimação de 10 anos não passou (P[R10 > R5] de 0,20 a 0,48). Detalhes no
  ROTEIRO §4.11.
- **S2.1, Conservadora de Blitz e van Vliet** (`scripts/avaliar_conservadora.py`, pré-registrado; saída em
  `docs/conservadora-2026-10-20.txt`): baixa vol + dividend yield + momentum 12−1, 15 nomes com pesos iguais. IC
  mensal de 0,081 (t 5,9) em 2008–2026 e 0,059 em 1998–2007; no walk-forward 2010–2026, 16,3–17,3% a.a. contra
  6,7–7,2% do BOVA11 e 13,0–14,3% do satélite Kelly, P[> BOVA11] 0,99 e DSR 0,80–0,88. Aprovada e candidata à prova
  final, mas ainda sem IR, com DSR abaixo de 0,95 e sem paper trading: não cumpre o critério de pronto. ROTEIRO §4.7.
- **S2.2 e S2.3, Fórmula Mágica de Greenblatt e Trending Value de O'Shaughnessy** (`scripts/avaliar_valor.py`,
  pré-registrado; saída em `docs/valor-2026-10-20.txt`), com os fundamentos da CVM ponto no tempo (data de entrega)
  do lago: IC mensal de 0,059 nas duas (t 5,4 e 5,5) em 2012–2026, as duas aprovadas como sinal; no walk-forward
  2012–2026, 10,2–12,1% e 11,1–13,5% a.a. contra 8,0–9,2% do BOVA11 e 16,7–18,5% da Conservadora nas mesmas datas,
  P[> BOVA11] 0,59–0,85: nenhuma candidata à prova final. ROTEIRO §4.7.
- **S3, Black-Litterman com views** (`scripts/avaliar_views.py`, pré-registrado; saída em `docs/views-2026-10-20.txt`):
  o Kelly atual com o μ trocado pelo posterior de um prior de mercado e das views da Conservadora e do valor, com a
  confiança pelo IC ponto no tempo. Retorno igual ao do Kelly (10,7–14,4% a.a., MDD −61 a −63%), P[> BOVA11]
  0,82–0,95: não candidata. ROTEIRO §4.8.
- **Alvo de vol com EGARCH** (`scripts/avaliar_vol_alvo.py`, pré-registrado; saída em `docs/vol-alvo-2026-10-20.txt`):
  a exposição de cada estratégia do S2 cai quando a vol prevista por um EGARCH passa da vol de 2 anos, e o resto
  fica no CDI. Corta 7 a 13 p.p. do MDD (Conservadora de −41% para −32/−33%), mas P[com alvo > original] de 0,14 a
  0,89: reprovado nas quatro, ficam as originais. ROTEIRO §4.13.
- **Prova final com execução realista (S4)** (`scripts/avaliar_execucao.py`, pré-registrada; saída em
  `docs/execucao-2026-10-20.txt`): a Conservadora refeita em reais com emolumentos da B3, meio spread e impacto pela
  raiz do tamanho da ordem, e com o IR de pessoa física (15%, isenção de vendas até R$ 20 mil no mês, preço médio,
  prejuízo acumulado), contra o BOVA11 que paga 15% do ganho na venda. Com R$ 10 mil, de 2012 a 2026: 16,1–17,8% a.a.
  líquidos de custo e de IR contra 7,2–8,4% do BOVA11, P[> BOVA11] da saída 0,981–0,995: **aprovada**, candidata ao
  paper-v2. Com R$ 1 milhão o custo sobe para 1,1–1,2% a.a. e o P cai para 0,94–0,98. Não cumpre ainda o critério de
  pronto (falta o paper trading). ROTEIRO §4.14.
- **Paper-v2 da Conservadora** (`scripts/paper_v2.py`, pré-registrado, tag `paper-v2`): registro em
  `paper-v2/carteiras.csv`, separado do paper-v1. A carteira do dia é a que o motor decidiria (conferido contra ele);
  rebalanceio a cada 63 pregões, com o `universe` do lago rodado no dia devido. 1º registro no fechamento de
  19/10/2026 (15 nomes, pesos iguais); leituras em 126 e 252 pregões com custo e IR de R$ 10 mil contra o BOVA11
  (`--leitura`). Tarefa "paper v2" do Windows, de segunda a sexta às 8h45 (`scripts/rodar_paper.py paper_v2`), conferida
  pelo vigia das 12h; o `universe` também roda toda noite na rotina do lago. ROTEIRO §4.10.
- **Núcleo BOVA11 + IVVB11** (`scripts/avaliar_nucleo.py`, pré-registrado; saída em `docs/nucleo-2026-10-20.txt`):
  metade BOVA11, metade IVVB11, de 2014 a 2026. 17,6–18,0% a.a. contra 11,9–12,0% do BOVA11, com vol de 15,7% contra
  23,8% e MDD de −34% contra −47%, mas P[> BOVA11] em log de 0,87–0,89 (pede 0,95) num regime de câmbio só: não
  candidato, o núcleo segue BOVA11. ROTEIRO §4.12.
- **HRP com a distância do paper** (`scripts/avaliar_hrp_distancia.py`, pré-registrado; saída em
  `docs/hrp-distancia-2026-10-20.txt`): López de Prado (2016) clusteriza a distância euclidiana entre as colunas da
  matriz de distâncias (d~), e o app clusteriza a própria d. Com d~, o HRP empata (P[d~ > d] 0,50–0,71, retorno total
  até 2 p.p. menor nas fases 0 e 21) e o HRP-Árvore reprova (P 0,03–0,39): o app fica com a d. A opção do paper está
  em `core/hrp_paper.py`, fora dos caminhos vigiados do paper.
- Teto de volatilidade anual (restrição cônica), peso máximo por ativo e número máximo de ativos
  (cardinalidade em duas etapas).
- Universo: lista líquida, filtro por setores ou todos os ativos de `assets.py`.
- Histórico de 3 a 10 anos, custo de transação em bps por lado e shrinkage de μ ligável/desligável.
- **Reamostragem de μ (bagging de Michaud)**, opcional e só para o Kelly: média de 100 carteiras calculadas
  em reamostras do histórico (bootstrap circular em blocos de 21 pregões), projetada nos limites. Espalha o
  peso entre mais ativos, mas vem **desligada**: na avaliação pré-registrada (`scripts/avaliar_reamostragem.py`,
  walk-forward de 5 e 10 anos), o Sharpe ficou abaixo do Kelly simples e o drawdown não melhorou. É lenta
  (~1 min a mais no walk-forward de 5 anos com a lista líquida).
- **Growth Score fora da tela.** O código continua em `core/growth.py`, mas o app não o usa: ele lê
  fundamentos e momentum de hoje (look-ahead se aplicado ao passado) e nunca foi testado no walk-forward.
  Assim, a carteira de hoje sai do mesmo método que o walk-forward avalia.

### 📈 Análises
- **Carteira de hoje:** cards de retorno/risco, fronteira eficiente (com o ponto 1/N), composição, tabela
  com setores, contribuição para o risco e matriz de correlação. **Teto por setor** (até 30%, regra fixa) na
  carteira, no plano de compra e no walk-forward; o HRP da tela é projetado nele.
- **Plano de compra, padrão = dinheiro real:** a regra até o critério de pronto, BOVA11 com as cotas inteiras mais
  perto de 70% do valor e, no satélite, a carteira registrada no paper trading (`paper/carteiras.csv`), sem
  reotimizar, antes da carteira do estudo (um estudo inválido não esconde o plano). O satélite fica em caixa quando
  algum papel registrado está sem cotação ou sem setor, quando passaria de 50% do valor, quando não cabem papéis
  bastantes com 2 unidades no teto, quando as quantidades inteiras furariam uma regra de risco ou quando o plano
  ficaria mais longe dos pesos registrados que o próprio caixa; depois do stop relativo do paper, tudo vai ao BOVA11
  (com o registro de 16/10/2026: R$ 300 e R$ 1.300 → 1 e 4 BOVA11 + caixa; R$ 3.000 → 10 BOVA11 + 8 papéis). Aviso
  do dinheiro real antes do plano.
- **Plano de compra do estudo** (escolha "Estudo"): cotação do dia (Yahoo, atraso de ~15 min; sem negócio, o último
  fechamento) e quantas ações comprar de cada uma da carteira otimizada, a partir de carteira zerada e de qualquer valor a partir de R$ 100. Quantidades
  inteiras com **mínimo de 2 ações por posição**; o custo estimado sai do valor antes da divisão; nenhuma
  posição passa do peso máximo. **Investe o valor todo:** antes de otimizar, tira as ações cujas 2 unidades
  passam do peso máximo × valor (com aviso); depois de comprar o piso de cada alvo, segue comprando a ação
  mais abaixo do alvo até nada caber no caixa e no teto (uma posição pode passar um pouco do alvo). Divide
  cada quantidade em lote padrão (múltiplos de 100) e fracionário (ticker com F) e mostra o peso efetivo, o
  valor investido e a sobra em caixa. A sobra não leva um setor além do teto, e o peso efetivo por setor
  aparece abaixo da tabela. Avisa quais ações ficaram de fora e por quê (2 ações passam do teto por ativo,
  passariam do teto do setor ou não cabem no caixa). Ação sem cotação sai do plano com aviso, e o valor dela vai para as outras só até os tetos (o resto fica em caixa). As
  cotações ficam guardadas por 5 min; o botão "Atualizar cotações" busca de novo só as da carteira na tela.
- **Walk-forward fora da amostra:** rebalanceamento trimestral com treino rolante na **mesma janela de
  estimação da carteira de hoje** e teste de N anos (sliders "Janela de estimação" e "Teste fora da amostra").
  Ele baixa janela + teste anos de preços, então ações com histórico mais curto ficam fora dele (a legenda
  lista quais) e continuam na carteira de hoje. As referências de mercado (BOVA11, SMAL11, DIVO11 e IVVB11) usam a mesma
  janela e as mesmas datas; a que não tem histórico para a janela aparece na legenda. Compara o objetivo
  escolhido (e a versão reamostrada, se ligada), 1/N, HRP e HRP-Árvore contra elas e o CDI; o alfa é contra o BOVA11. Mostra CAGR,
  volatilidade, Sharpe e Sortino (sobre o CDI vigente em cada data), Calmar, Omega, **PSR e DSR** (probabilidade
  de o Sharpe verdadeiro ser positivo e a mesma probabilidade descontada pelo número de estratégias comparadas;
  Bailey e López de Prado), VaR/CVaR, drawdown máximo, alfa por regressão com estatística t, pesos ao longo do
  tempo e métricas rolantes.

### 🏦 Dados
- **Preços diários:** lago de dados da B3 (projeto irmão `b3-datalake`), arquivo `lago/precos_diarios.parquet`:
  índice de retorno total por papel a partir do COTAHIST oficial e dos eventos da B3 e da CVM (cisões, bonificações,
  JCP líquido do IR), terminando no último fechamento; ações e units negociadas nas últimas duas semanas mais BOVA11,
  SMAL11, DIVO11 e IVVB11 (este desde 29/04/2014), desde 2005. Dia sem negócio fica sem preço (o Yahoo repetia o anterior), então ação pouco
  negociada cai na exigência de cobertura ≥ 95% da janela. A tela mostra até que pregão vão os preços.
- **Cotação do dia:** Yahoo Finance (`yfinance`), só para o plano de compra.
- **Banco Central (SGS):** CDI diário (série 12), Selic meta point-in-time (série 432) e PTAX (série 1, só para o painel macro),
  baixados em janelas de até 10 anos, com nova tentativa em falhas transitórias.
- **Lago de dados da B3 (projeto irmão `b3-datalake`):** exports em parquet gerados a partir do COTAHIST
  oficial (1986–hoje) e dos eventos da B3 e da CVM: retorno total ajustado desde 07/1994 de 3.948 papéis,
  **incluindo deslistados e em recuperação judicial** (troca de ticker, como ITAU4 → ITUB4, continua a mesma
  série), painel mensal de liquidez/volatilidade/microestrutura e índices próprios. Os preços diários do app
  já vêm dele. O lago também publica o **universo ponto no tempo**: em cada mês, as ações e units negociadas em
  ≥ 90% dos pregões dos 12 meses anteriores, com volume diário mediano ≥ R$ 1 mi corrigido pelo IPCA conhecido
  na data, um papel por emissor (417 papéis desde 2005, 186 deles hoje deslistados). O `core/backtest.py` já tem
  o walk-forward com esse universo (`walk_forward_dinamico`); a avaliação pré-registrada rodou em 2026-10-18
  (`docs/universo-dinamico-2026-10-18.txt`) e o app ainda usa a lista de hoje. Uma rotina diária no Windows (Agendador de Tarefas, seg–sex
  às 21h45 com novas tentativas até 23h45 e às 8h do dia seguinte) traz o pregão do dia para o lago, confere
  com portões automáticos (troca de ticker sem revisão, arquivo truncado, retorno ≥ 50% sem revisão, gold
  reproduzível) e só atualiza os exports quando todos passam; quando algo falha, avisa no Windows.
- **Setor por emissor, com as deslistadas** (`data/b3lake/sectors.parquet`, lido por `core.lago.ler_setores`):
  `SETOR_ATIV` do cadastro da CVM mapeado nos 10 setores do app; cobre 97% dos emissores do universo. É a taxonomia
  da CVM, mais grossa que a da B3 do `assets.py` (149 de 178 concordam): o motor dinâmico usa a do lago para vivas e
  mortas, e a tela continua com o `assets.py`.
- **Proventos por papel e data ex** (`data/b3lake/dividends.parquet`): dinheiro pago por ação (JCP líquido do IR),
  com o yield sobre o fechamento do dia ex, para os sinais de payout do S2.
- **Correção de 2026-10-18:** a B3 lista entre os proventos em dinheiro o custo atribuído das bonificações, que o
  lago contava de novo no dia do evento de ações (AXIA3 22/12/2025 +22% em vez de −3%; CYRE3 e RENT3 também). Os
  preços do app foram republicados sem isso.
- **Em desenvolvimento (só no `core/`, ainda sem tela):** painel macro mensal para cenários — Focus
  (expectativas do BCB), curva prefixada do Tesouro Direto e fatores de surpresa macro.

---

## 🛠️ Tecnologias

- **[Streamlit](https://streamlit.io/)** — interface.
- **[CVXPY](https://www.cvxpy.org/)** (solvers CLARABEL/SCS) — otimização convexa.
- **[scikit-learn](https://scikit-learn.org/)** — covariância de Ledoit-Wolf.
- **[SciPy](https://scipy.org/)** — clusterização hierárquica (HRP) e estatística.
- **[Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)** — dados.
- **[yfinance](https://pypi.org/project/yfinance/)** e **[API SGS/Olinda do BCB](https://dadosabertos.bcb.gov.br/)** — fontes.
- **[Plotly](https://plotly.com/)** — gráficos.

---

## 📂 Estrutura do Projeto

```
Portfolio-Optimization/
├── app.py                    # Orquestrador Streamlit (camada fina)
├── assets.py                 # Universo de ativos e setores da B3
├── visualizations.py         # Gráficos Plotly
├── core/                     # Lógica pura, sem Streamlit
│   ├── config.py             #   Objetivos, configurações, Selic → taxa aritmética
│   ├── lago.py               #   Preços diários do lago da B3 (lago/precos_diarios.parquet)
│   ├── market_data.py        #   Preços (lago), calendário B3, CDI, Selic, cotação do dia (Yahoo)
│   ├── ordens.py             #   Plano de compra: pesos + cotações → quantidades inteiras
│   ├── risco.py              #   Regras de risco fixas: aprova ou recusa a carteira (não muda pesos)
│   ├── registro.py           #   Registro do paper trading (CSV que só acrescenta linhas)
│   ├── paper.py              #   Paper trading: curva do registro, custos, decisão do dia, leituras de 6 e 12 meses
│   ├── bcb.py                #   Cliente SGS do Banco Central (janelas, cache, retry)
│   ├── estimation.py         #   Retornos, Ledoit-Wolf, Jorion, tilt growth
│   ├── optimization.py       #   Kelly, Max-Sharpe, Min-Vol, HRP, fronteira, projeção nos limites
│   ├── reamostragem.py       #   Carteira reamostrada (bagging, bootstrap em blocos)
│   ├── hierarchical.py       #   HRP e HRP-Árvore
│   ├── hrp_paper.py          #   HRP com a distância d~ do paper original (pesquisa, fora da tela)
│   ├── metrics.py            #   Sharpe, Sortino, Calmar, VaR/CVaR, alfa, drawdown, PSR/DSR
│   ├── robustez.py           #   MinTRL, P[> referência] em log-retorno, janelas móveis
│   ├── backtest.py           #   Walk-forward com custos e CDI point-in-time (universo ponto no tempo, regras no motor)
│   ├── poder.py              #   Poder dos portões do S2 por simulação (IC plantado no painel mensal)
│   ├── sinais.py             #   Sinais do S2: Conservadora (vol 36m, DY 12m, momentum 12−1) e a carteira dela
│   ├── valor.py              #   Sinais do S2 com fundamentos da CVM: Fórmula Mágica e Trending Value
│   ├── vol_alvo.py           #   Alvo de vol com EGARCH sobre uma curva fora da amostra (pesquisa)
│   ├── views.py              #   Black-Litterman com views dos sinais do S2 (S3)
│   ├── impostos.py           #   IR de pessoa física em ações: apuração mensal, isenção e prejuízo (S4)
│   ├── execucao.py           #   Curva refeita em reais com custo por liquidez, IR e valor de saída (S4)
│   ├── growth.py             #   Growth Score (Yahoo/CVM), fora da tela até ser validado
│   ├── focus.py              #   Expectativas Focus (em desenvolvimento)
│   ├── tesouro.py            #   Curva prefixada do Tesouro (em desenvolvimento)
│   └── fatores_macro.py      #   Painel e fatores macro (em desenvolvimento)
├── ui/
│   ├── servicos.py           # Cache do Streamlit com chaves explícitas
│   └── secoes.py             # Seções e textos da interface
├── scripts/
│   ├── conferir_painel_macro.py  # Conferência ao vivo do painel macro (usa rede)
│   ├── avaliar_reamostragem.py   # Avaliação pré-registrada da reamostragem (usa rede, ~30 min)
│   ├── avaliar_janelas.py        # Teste pré-registrado das janelas de estimação (usa rede)
│   ├── avaliar_universo_dinamico.py  # Teste pré-registrado com o universo ponto no tempo do lago (rodado)
│   ├── avaliar_regras_no_motor.py    # Regras de risco no motor, pré-registrado (rodado)
│   ├── poder_dos_portoes.py          # Poder dos portões do S2, pré-registrado (rodado)
│   ├── avaliar_robustez.py           # Robustez 2010–2026 e janela de 10 anos, pré-registrado (rodado)
│   ├── avaliar_conservadora.py       # S2.1, Conservadora de Blitz e van Vliet, pré-registrado (rodado)
│   ├── avaliar_nucleo.py             # Núcleo BOVA11 + IVVB11, pré-registrado (rodado)
│   ├── avaliar_valor.py              # S2.2 e S2.3, Fórmula Mágica e Trending Value, pré-registrado (rodado)
│   ├── avaliar_vol_alvo.py           # Alvo de vol com EGARCH, pré-registrado (rodado)
│   ├── avaliar_views.py              # S3, Black-Litterman com views, pré-registrado (rodado)
│   ├── avaliar_execucao.py           # Prova final com execução realista (S4), pré-registrada (rodada)
│   ├── avaliar_hrp_distancia.py      # HRP com a distância d~ do paper, pré-registrado (rodado)
│   ├── paper_v2.py                   # Paper-v2 da Conservadora: rotina e leitura, pré-registrado
│   └── paper_trading.py              # Rotina diária do paper trading (S5), pré-registrada (agendada seg-sex 8h30)
├── docs/
│   ├── ROTEIRO.md                # Roteiro: critério de pronto, fases, riscos e decisões pendentes
│   ├── janelas-tiro1-*.txt       # Saídas do tiro 1 do teste das janelas
│   ├── universo-dinamico-2026-10-18.txt  # Saída do S1-P.b (sem viés de sobrevivência)
│   ├── regras-no-motor-2026-10-18.txt    # Saída das regras de risco no motor
│   ├── poder-dos-portoes-2026-10-18.txt  # Saída do poder dos portões do S2
│   ├── robustez-2026-10-19.txt   # Saída do teste de robustez
│   ├── conservadora-2026-10-20.txt  # Saída do S2.1 (Conservadora)
│   ├── nucleo-2026-10-20.txt     # Saída do teste do núcleo BOVA11 + IVVB11
│   ├── valor-2026-10-20.txt      # Saída do S2.2 e do S2.3 (Fórmula Mágica e Trending Value)
│   ├── vol-alvo-2026-10-20.txt   # Saída do teste do alvo de vol com EGARCH
│   ├── views-2026-10-20.txt      # Saída do S3 (Black-Litterman com views)
│   ├── execucao-2026-10-20.txt   # Saída da prova final com custo por liquidez e IR (S4)
│   ├── hrp-distancia-2026-10-20.txt  # Saída do teste do HRP com a distância d~ do paper
│   ├── tentativas.csv            # Configurações já avaliadas fora da amostra (o N do DSR)
│   └── dados/                    # Exports do lago fixados pelos pré-registros (com SHA256SUMS)
├── lago/
│   └── precos_diarios.parquet    # Preços diários de retorno total publicados pelo b3-datalake
├── tests/                    # Testes com pytest, sem rede (o app inteiro roda via AppTest com dados sintéticos)
└── requirements.txt
```

---

## 🚀 Como Executar

### Pré-requisitos
- Python **3.11** ou superior (NumPy 2 e CVXPY 1.9).

### Passo a passo

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/GabrielScript/Portfolio-Optimization.git
   cd Portfolio-Optimization
   ```

2. **Crie e ative um ambiente virtual** (recomendado — evita conflito com um NumPy 1.x global):
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute a aplicação:**
   ```bash
   streamlit run app.py
   ```
   O dashboard abre em `http://localhost:8501`. A primeira execução baixa preços e fundamentos e pode
   levar um ou dois minutos.

5. **Rode os testes** (opcional; precisa do `pytest`):
   ```bash
   pip install pytest
   python -m pytest tests -q
   ```

---

## ⚠️ Limitações

- **Viés de sobrevivência.** O walk-forward do app usa a lista atual de tickers; empresas que quebraram ou
  fecharam capital não aparecem, o que favorece as estratégias contra o BOVA11 (o índice não tem o viés). O universo ponto no tempo, com as
  deslistadas, já existe (lago + `walk_forward_dinamico`). Na avaliação pré-registrada sem o viés
  (`docs/universo-dinamico-2026-10-18.txt`), o Kelly λ=2 teve Sharpe −0,31 em 5 anos e 0,36 em 10 (MDD −63%),
  e o núcleo Mín. Vol 70% + satélite Kelly 30% teve 0,35 em 10 anos com MDD −46%: ele virou o padrão da tela.
  Refeito com os preços corrigidos (bonificação) no pré-registro das regras, as duas decisões não mudaram.
- **Setor das deslistadas é o de hoje.** O `SETOR_ATIV` da CVM não tem data de vigência: empresa que mudou de
  ramo fica com o setor atual na história toda; 10 holdings sem setor principal ficam sem setor.
- **Migração de classe sem continuidade.** Quem migrou de PN para ON no Novo Mercado (PCAR4 → PCAR3 em
  2020, SUZB5 → SUZB3) começa no lago sem a história da classe antiga e fica fora de janelas longas.
- Os preços diários vêm do lago desde 2026-10 (antes, Yahoo: errava cisões e usava JCP bruto; registrou
  +112% num dia em que o retorno total real foi +14,1%, cisão do Assaí). Medições publicadas antes disso
  usaram os dados do Yahoo.
- **Sem vantagem comprovada sobre o BOVA11.** A última medição fora da amostra deu empate estatístico; use o
  BOVA11 como núcleo e o otimizador, no máximo, como satélite pequeno. Os horizontes de 5 e 10 anos terminam no
  mesmo dia (são uma história só: ganho forte em 2016–2021, perda em 2021–2026); com tracking error de 16–19%,
  provar o alfa contra o BOVA11 com PSR ≥ 0,95 pediria ~14–30 anos de histórico. A coluna P[> BOVA11] usa a média
  aritmética dos retornos diários, não a riqueza final, e favorece as curvas mais voláteis.
- **O plano do estudo só compra ações.** No modo "Estudo", com valores pequenos (R$ 300) o plano para em "Aumente
  o valor a investir" porque cabem menos de 5 papéis, e o filtro por preço reotimiza num universo menor. O padrão da
  tela é o plano do dinheiro real (BOVA11 + satélite registrado), que não reotimiza.
- **Paper trading próprio, sem plataforma de corretora.** A rotina usa os fechamentos oficiais do lago, o mesmo código
  do walk-forward e um registro versionado que só cresce (auditável; uma conta demo pode ser zerada). Não mede a
  execução: spread, slippage, ordens parciais e a liquidez do fracionário ficam fora (custo fixo de 10 bps por lado).
- **Sem alavancagem.** A regra fixa é 1,0×: com o empréstimo a CDI + spread perto do retorno histórico do BOVA11, a
  alavancagem custaria mais do que rende enquanto o alfa não estiver provado.
- **S&P 500 só como referência.** O IVVB11 (S&P 500 em reais, sem hedge) está no lago desde 2014: 20,1% a.a. ×
  12,4% do BOVA11 em 10 anos e 9,2% × 15,2% em 5, correlação mensal −0,18. O núcleo metade BOVA11, metade IVVB11
  não passou no teste pré-registrado (ROTEIRO §4.12).
- **Imposto de renda não é modelado** (15% sobre o ganho; isenção de vendas até R$ 20 mil/mês). O Yahoo
  também usa o JCP bruto; o lago da B3 já desconta o IR retido na fonte (15% até 2025 e 17,5% desde
  01/01/2026, LC 224/2025).
- **Janelas curtas.** Com 3–5 anos de histórico, alfas e diferenças de Sharpe raramente são
  estatisticamente significativos; o app mostra a estatística t para lembrar disso. A análise de poder
  (`docs/poder-dos-portoes-2026-10-18.txt`) mediu isso para os sinais do S2: o portão "IC médio mensal ≥ 0,02 e
  t ≥ 2" aprova 87% dos sinais com IC 0,03 e 1% dos sinais sem valor nos 225 meses de 2008 a 2026; em 5 anos, com IC
  que varia de mês a mês, nenhum portão passa de 47%. Por isso os sinais do S2 são julgados na janela inteira.

O que falta e em que ordem: [`docs/ROTEIRO.md`](docs/ROTEIRO.md).

---

## 📝 Isenção de Responsabilidade (Disclaimer)

Esta é uma ferramenta de **apoio à decisão**, sem vantagem comprovada sobre o BOVA11 até o fim da validação em paper trading.

> ⚠️ **Não representa recomendação de investimento.** As decisões financeiras devem ser tomadas com base em análise própria ou orientação profissional. Retornos passados não garantem retornos futuros.

---

© 2026 Gabriel Estrela Lopes
