# 📊 Otimizador de Carteiras B3

> **TCC - Ciência de Dados para Negócios - UFPB**  
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
- **Growth Score** (fundamentos Yahoo, opcionalmente receita de 10 anos via CVM): filtro por percentil e
  tilt no μ, aplicados **só à carteira de hoje** (ver Limitações).
- Cripto (BTC/ETH convertidos para R$ pela PTAX) com teto agregado — ainda presente no app, mas o projeto
  passará a ser só ações.

### 📈 Análises
- **Carteira de hoje:** cards de retorno/risco, fronteira eficiente (com o ponto 1/N), composição, tabela
  com setores, contribuição para o risco e matriz de correlação.
- **Walk-forward fora da amostra:** rebalanceamento trimestral com 2 anos de treino; compara o objetivo
  escolhido, 1/N, HRP e HRP-Árvore contra o BOVA11 e o CDI. Mostra CAGR, volatilidade, Sharpe e Sortino
  (sobre o CDI vigente em cada data), Calmar, Omega, VaR/CVaR, drawdown máximo, alfa por regressão com estatística t,
  pesos ao longo do tempo e métricas rolantes.

### 🏦 Dados
- **Preços:** Yahoo Finance (`yfinance`), alinhados ao calendário da B3, com exigência de cobertura ≥ 95%.
- **Banco Central (SGS):** CDI diário (série 12), Selic meta point-in-time (série 432) e PTAX (série 1),
  baixados em janelas de até 10 anos, com nova tentativa em falhas transitórias.
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
├── analise_top_portfolio.py  # Script de análise (com aviso de data snooping)
├── core/                     # Lógica pura, sem Streamlit
│   ├── config.py             #   Objetivos, configurações, Selic → taxa aritmética
│   ├── market_data.py        #   Preços, calendário B3, CDI, Selic, câmbio
│   ├── bcb.py                #   Cliente SGS do Banco Central (janelas, cache, retry)
│   ├── estimation.py         #   Retornos, Ledoit-Wolf, Jorion, tilt growth
│   ├── optimization.py       #   Kelly, Max-Sharpe, Min-Vol, HRP, fronteira
│   ├── hierarchical.py       #   HRP e HRP-Árvore
│   ├── metrics.py            #   Sharpe, Sortino, Calmar, VaR/CVaR, alfa, drawdown
│   ├── backtest.py           #   Walk-forward com custos e CDI point-in-time
│   ├── growth.py             #   Growth Score (Yahoo/CVM)
│   ├── focus.py              #   Expectativas Focus (em desenvolvimento)
│   ├── tesouro.py            #   Curva prefixada do Tesouro (em desenvolvimento)
│   └── fatores_macro.py      #   Painel e fatores macro (em desenvolvimento)
├── ui/
│   ├── servicos.py           # Cache do Streamlit com chaves explícitas
│   └── secoes.py             # Seções e textos da interface
├── scripts/
│   └── conferir_painel_macro.py  # Conferência ao vivo do painel macro (usa rede)
├── tests/                    # Testes com pytest, sem rede
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

- **Filtro e tilt growth não entram no backtest.** Usam fundamentos e momentum de hoje; aplicá-los ao
  passado seria look-ahead. O walk-forward mede só a alocação sobre o universo pré-filtro.
- **Viés de sobrevivência.** O universo é a lista atual de tickers; empresas que quebraram ou fecharam
  capital não aparecem, o que favorece todas as estratégias.
- **Imposto de renda não é modelado** (15% sobre o ganho; isenção de vendas até R$ 20 mil/mês).
- **Janelas curtas.** Com 3–5 anos de histórico, alfas e diferenças de Sharpe raramente são
  estatisticamente significativos; o app mostra a estatística t para lembrar disso.
- **Cripto** usa o fechamento do Yahoo (00h UTC), que não coincide com o da B3.

---

## 📝 Isenção de Responsabilidade (Disclaimer)

Esta ferramenta foi desenvolvida para fins **acadêmicos e educacionais** como parte do Trabalho de Conclusão de Curso (TCC). 

> ⚠️ **Não representa recomendação de investimento.** As decisões financeiras devem ser tomadas com base em análise própria ou orientação profissional. Retornos passados não garantem retornos futuros.

---

© 2026 Gabriel Estrela Lopes
