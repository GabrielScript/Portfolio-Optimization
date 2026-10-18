# 📊 Otimizador de Carteiras B3

> **Autor:** Gabriel Estrela Lopes

Dashboard em Python/Streamlit para otimizar carteiras de ações da B3 com foco no **investidor arrojado**.
Usa Markowitz/Kelly com estimadores robustos (Ledoit-Wolf + shrinkage estilo Jorion), alocação hierárquica
(HRP) e um **walk-forward fora da amostra, líquido de custos**, para que toda estratégia seja comparada nas
mesmas condições.

---

## 🚀 Funcionalidades

### 🎯 Objetivos de otimização
- **Núcleo Mín. Vol + satélite Kelly (padrão, recomendado)** — (1 − f)·w_MínVol + f·w_Kelly, com o satélite f de
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
  (`core/registro.py`).
- **As mesmas regras no motor do walk-forward com deslistadas** (`estrategia_otimizada_nas_regras`, `Congelamento`,
  recusa = mantém a carteira), testadas num pré-registro (`docs/regras-no-motor-2026-10-18.txt`): 0 recusas em 60
  rebalanceios; o stop relativo disparou nos dois horizontes (05/2025 e 12/2025); o padrão com regras teve Sharpe
  −0,06 (−0,13 sem) em 5 anos e 0,28 (0,35 sem) em 10. O teto de setor custa à Mín. Vol em 10 anos (0,19 × 0,28).
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
- **Plano de compra:** cotação do dia (Yahoo, atraso de ~15 min; sem negócio, o último fechamento) e quantas
  ações comprar de cada uma, a partir de carteira zerada e de qualquer valor a partir de R$ 100. Quantidades
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
  lista quais) e continuam na carteira de hoje. As referências de mercado (BOVA11, SMAL11 e DIVO11) usam a mesma
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
  SMAL11 e DIVO11, desde 2005. Dia sem negócio fica sem preço (o Yahoo repetia o anterior), então ação pouco
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
│   ├── bcb.py                #   Cliente SGS do Banco Central (janelas, cache, retry)
│   ├── estimation.py         #   Retornos, Ledoit-Wolf, Jorion, tilt growth
│   ├── optimization.py       #   Kelly, Max-Sharpe, Min-Vol, HRP, fronteira, projeção nos limites
│   ├── reamostragem.py       #   Carteira reamostrada (bagging, bootstrap em blocos)
│   ├── hierarchical.py       #   HRP e HRP-Árvore
│   ├── metrics.py            #   Sharpe, Sortino, Calmar, VaR/CVaR, alfa, drawdown, PSR/DSR
│   ├── backtest.py           #   Walk-forward com custos e CDI point-in-time (universo ponto no tempo, regras no motor)
│   ├── poder.py              #   Poder dos portões do S2 por simulação (IC plantado no painel mensal)
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
│   └── poder_dos_portoes.py          # Poder dos portões do S2, pré-registrado (ainda não rodado)
├── docs/
│   ├── ROTEIRO.md                # Roteiro: critério de pronto, fases, riscos e decisões pendentes
│   ├── janelas-tiro1-*.txt       # Saídas do tiro 1 do teste das janelas
│   ├── universo-dinamico-2026-10-18.txt  # Saída do S1-P.b (sem viés de sobrevivência)
│   └── regras-no-motor-2026-10-18.txt    # Saída das regras de risco no motor
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
  fecharam capital não aparecem, o que favorece todas as estratégias. O universo ponto no tempo, com as
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
  BOVA11 como núcleo e o otimizador, no máximo, como satélite pequeno.
- **Imposto de renda não é modelado** (15% sobre o ganho; isenção de vendas até R$ 20 mil/mês). O Yahoo
  também usa o JCP bruto; o lago da B3 já desconta o IR retido na fonte (15% até 2025 e 17,5% desde
  01/01/2026, LC 224/2025).
- **Janelas curtas.** Com 3–5 anos de histórico, alfas e diferenças de Sharpe raramente são
  estatisticamente significativos; o app mostra a estatística t para lembrar disso.

O que falta e em que ordem: [`docs/ROTEIRO.md`](docs/ROTEIRO.md).

---

## 📝 Isenção de Responsabilidade (Disclaimer)

Esta é uma ferramenta de **apoio à decisão**, sem vantagem comprovada sobre o BOVA11 até o fim da validação em paper trading.

> ⚠️ **Não representa recomendação de investimento.** As decisões financeiras devem ser tomadas com base em análise própria ou orientação profissional. Retornos passados não garantem retornos futuros.

---

© 2026 Gabriel Estrela Lopes
