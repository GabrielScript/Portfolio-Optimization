# 📊 Otimizador de Carteiras B3

> **TCC - Ciência de Dados para Negócios - UFPB**  
> **Autor:** Gabriel Estrela Lopes

Este projeto é um dashboard interativo desenvolvido em Python com Streamlit para otimização de carteiras de investimentos no mercado brasileiro (B3), utilizando a **Teoria Moderna do Portfólio (Markowitz)**.

---

## 🚀 Funcionalidades

O sistema oferece uma interface amigável para investidores simularem e otimizarem seus portfólios com base em diferentes perfis de risco.

### 🎯 Perfis de Risco
- **Conservador**: Prioriza segurança (Minimiza Volatilidade).
- **Moderado**: Busca equilíbrio (Maximiza Sharpe).
- **Agressivo**: Foca em rentabilidade (Maximiza Retorno).

### ⚙️ Personalização
- **Orçamento**: Defina o valor inicial do investimento.
- **Filtros**: Selecione ativos por setores específicos.
- **Restrições**: Configure número máximo de ativos e limites de exposição.
- **Parâmetros**: Ajuste período de análise, Taxa Selic e pesos máximos.

### 📊 Análises e Visualizações
- **Fronteira Eficiente**: Gráfico interativo risco x retorno.
- **Composição da Carteira**: Gráficos de pizza e barras da alocação sugerida.
- **Backtesting Walk-Forward**: Simulação histórica do desempenho da carteira.
- **Métricas de Risco**: VaR (Value at Risk), CVaR, Drawdown Máximo, Sharpe e Sortino.
- **Matriz de Correlação**: Análise de diversificação entre ativos.

---

## 🛠️ Tecnologias Utilizadas

- **[Streamlit](https://streamlit.io/)**: Framework para construção do dashboard.
- **[Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)**: Manipulação e análise de dados.
- **[yfinance](https://pypi.org/project/yfinance/)**: Coleta de dados históricos da B3.
- **[Plotly](https://plotly.com/)**: Gráficos interativos.
- **[SciPy](https://scipy.org/)**: Algoritmos de otimização matemática.

---

## 📂 Estrutura do Projeto

```
Codigo_Projeto_TCC/
├── app.py                # Aplicação principal (Streamlit)
├── assets.py             # Definição de ativos e setores da B3
├── backtesting.py        # Lógica de simulação e métricas de risco
├── data_loader.py        # Coleta e processamento de dados (Yahoo Finance)
├── optimizer.py          # Algoritmos de otimização (Markowitz)
├── risk_profiles.py      # Configuração dos perfis de investidor
├── visualizations.py     # Funções geradoras de gráficos
└── requirements.txt      # Dependências do projeto
```

---

## 🚀 Como Executar

### Pré-requisitos
- Python 3.8 ou superior instalado.

### Passo a Passo

1. **Clone ou baixe o repositório** para sua máquina local.

2. **Crie um ambiente virtual (opcional, mas recomendado):**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute a aplicação:**
   ```bash
   streamlit run app.py
   ```

5. O dashboard abrirá automaticamente no seu navegador padrão (geralmente em `http://localhost:8501`).

---

## 📝 Isenção de Responsabilidade (Disclaimer)

Esta ferramenta foi desenvolvida para fins **acadêmicos e educacionais** como parte do Trabalho de Conclusão de Curso (TCC). 

> ⚠️ **Não representa recomendação de investimento.** As decisões financeiras devem ser tomadas com base em análise própria ou orientação profissional. Retornos passados não garantem retornos futuros.

---

© 2024 Gabriel Estrela Lopes
