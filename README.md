# 📊 Otimizador de Carteiras B3

> **Autor:** Gabriel Estrela Lopes

Dashboard em Python/Streamlit para otimizar carteiras de ações da B3 com foco no **investidor arrojado**.
Usa Markowitz/Kelly com estimadores robustos (Ledoit-Wolf + shrinkage estilo Jorion), alocação hierárquica
(HRP) e um **walk-forward fora da amostra, líquido de custos**, para que toda estratégia seja comparada nas
mesmas condições.

---

## 🚀 Funcionalidades

### 🎯 Objetivos de otimização
- **Crescimento máximo (Kelly)** — `max μ'w − (λ/2)·w'Σw`, com aversão λ ajustável (padrão λ = 2). Recomendado.
- **Máximo Sharpe** — resolvido de forma exata (transformação de Charnes-Cooper, problema convexo).
- **Mínima variância.**
- **HRP** — Hierarchical Risk Parity (López de Prado); ignora o retorno esperado.
- **Máximo retorno com teto de volatilidade** — mantido como legado: gera soluções de canto que concentram
  o erro de estimação.

### ⚙️ Restrições e parâmetros
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
  com setores, contribuição para o risco e matriz de correlação. **Teto por setor** (padrão 30%; 100% = sem
  limite) na carteira, no plano de compra e no walk-forward; o HRP ignora.
- **Plano de compra:** cotação do dia (Yahoo, atraso de ~15 min; sem negócio, o último fechamento) e quantas
  ações comprar de cada uma, a partir de carteira zerada e de qualquer valor a partir de R$ 100. Quantidades
  inteiras com **mínimo de 2 ações por posição**; o custo estimado sai do valor antes da divisão; nenhuma
  posição passa do peso máximo. **Investe o valor todo:** antes de otimizar, tira as ações cujas 2 unidades
  passam do peso máximo × valor (com aviso); depois de comprar o piso de cada alvo, segue comprando a ação
  mais abaixo do alvo até nada caber no caixa e no teto (uma posição pode passar um pouco do alvo). Divide
  cada quantidade em lote padrão (múltiplos de 100) e fracionário (ticker com F) e mostra o peso efetivo, o
  valor investido e a sobra em caixa. A sobra não leva um setor além do teto, e o peso efetivo por setor
  aparece abaixo da tabela. Avisa quais ações ficaram de fora e por quê (2 ações passam do teto por ativo,
  passariam do teto do setor ou não cabem no caixa). Ação sem cotação sai do plano com aviso, e o valor dela vai para as outras. As
  cotações ficam guardadas por 5 min; o botão "Atualizar cotações" busca de novo só as da carteira na tela.
- **Walk-forward fora da amostra:** rebalanceamento trimestral com treino rolante na **mesma janela de
  estimação da carteira de hoje** e teste de N anos (sliders "Janela de estimação" e "Teste fora da amostra").
  Ele baixa janela + teste anos de preços, então ações com histórico mais curto ficam fora dele (a legenda
  lista quais) e continuam na carteira de hoje. O BOVA11 usa a mesma janela e as mesmas datas. Compara o objetivo
  escolhido (e a versão reamostrada, se ligada), 1/N, HRP e HRP-Árvore contra o BOVA11 e o CDI. Mostra CAGR,
  volatilidade, Sharpe e Sortino (sobre o CDI vigente em cada data), Calmar, Omega, **PSR e DSR** (probabilidade
  de o Sharpe verdadeiro ser positivo e a mesma probabilidade descontada pelo número de estratégias comparadas;
  Bailey e López de Prado), VaR/CVaR, drawdown máximo, alfa por regressão com estatística t, pesos ao longo do
  tempo e métricas rolantes.

### 🏦 Dados
- **Preços:** Yahoo Finance (`yfinance`), alinhados ao calendário da B3, com exigência de cobertura ≥ 95%.
- **Banco Central (SGS):** CDI diário (série 12), Selic meta point-in-time (série 432) e PTAX (série 1, só para o painel macro),
  baixados em janelas de até 10 anos, com nova tentativa em falhas transitórias.
- **Lago de dados da B3 (projeto irmão `b3-datalake`, ainda não ligado ao app):** exports em parquet gerados
  a partir do COTAHIST oficial (1986–hoje) e dos eventos da B3 e da CVM: retorno total ajustado
  desde 07/1994 de 3.948 papéis, **incluindo deslistados e em recuperação judicial** (troca de ticker,
  como ITAU4 → ITUB4, continua a mesma série), painel mensal de liquidez/volatilidade/microestrutura e
  índices próprios. Vai substituir o Yahoo (preços e proventos) e remover o viés de sobrevivência quando
  for integrado; hoje o app ainda usa o Yahoo. Uma rotina diária no Windows (Agendador de Tarefas, seg–sex
  às 21h45 com novas tentativas até 23h45 e às 8h do dia seguinte) traz o pregão do dia para o lago, confere
  com portões automáticos (troca de ticker sem revisão, arquivo truncado, retorno ≥ 50% sem revisão, gold
  reproduzível) e só atualiza os exports quando todos passam; quando algo falha, avisa no Windows.
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
│   ├── market_data.py        #   Preços, calendário B3, CDI, Selic, cotação do dia
│   ├── ordens.py             #   Plano de compra: pesos + cotações → quantidades inteiras
│   ├── bcb.py                #   Cliente SGS do Banco Central (janelas, cache, retry)
│   ├── estimation.py         #   Retornos, Ledoit-Wolf, Jorion, tilt growth
│   ├── optimization.py       #   Kelly, Max-Sharpe, Min-Vol, HRP, fronteira, projeção nos limites
│   ├── reamostragem.py       #   Carteira reamostrada (bagging, bootstrap em blocos)
│   ├── hierarchical.py       #   HRP e HRP-Árvore
│   ├── metrics.py            #   Sharpe, Sortino, Calmar, VaR/CVaR, alfa, drawdown, PSR/DSR
│   ├── backtest.py           #   Walk-forward com custos e CDI point-in-time
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
│   └── avaliar_janelas.py        # Teste pré-registrado das janelas de estimação (usa rede)
├── docs/
│   ├── ROTEIRO.md                # Roteiro: critério de pronto, fases, riscos e decisões pendentes
│   └── janelas-tiro1-*.txt       # Saídas do tiro 1 do teste das janelas
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

- **Viés de sobrevivência.** O universo é a lista atual de tickers; empresas que quebraram ou fecharam
  capital não aparecem, o que favorece todas as estratégias. Os dados para corrigir isso já existem no
  lago de dados da B3 (`b3-datalake`), mas ainda não alimentam o app.
- **Preços e proventos do Yahoo têm erros conhecidos** (cisões, JCP bruto): comparado ao COTAHIST, o Yahoo
  registrou +112% num dia em que o retorno total real foi +14,1% (cisão do Assaí). Resultados do walk-forward
  herdam esses erros até a integração do lago.
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
