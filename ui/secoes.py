"""
secoes.py - Blocos de renderização do dashboard.

Segurança (code-reviewer / XSS): o único ponto com ``unsafe_allow_html`` é
``card``. Todo texto dinâmico passa por ``html.escape``.

Fronteira de aconselhamento (prediction-market-risk-review): a saída é
informativa. Retornos esperados aparecem como estimativa de modelo com faixa
de incerteza, nunca como valor prometido em R$.
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st

from core.backtest import ResultadoBacktest
from core.metrics import ResumoDesempenho, comparar_benchmark, resumo_desempenho
from core.optimization import ResultadoOtimizacao

CSS = """
<style>
.stApp { background: linear-gradient(135deg, #0E1117 0%, #1a1f2e 100%); }
.metric-card { background: linear-gradient(145deg, #1E2130, #2D3250); border-radius: 16px;
  padding: 20px; border: 1px solid rgba(102,126,234,.2); text-align: center; }
.metric-value { font-size: 1.9rem; font-weight: 700; color: #8FA2FF; }
.metric-value.pos { color: #00D4AA; } .metric-value.neg { color: #FF5252; }
.metric-label { font-size: .8rem; color: #999; text-transform: uppercase; letter-spacing: 1px; margin-top: 6px; }
.metric-sub { font-size: .75rem; color: #777; margin-top: 4px; }
.main-title { font-size: 2.4rem; font-weight: 800; text-align: center; color: #8FA2FF; margin-bottom: .2rem; }
.subtitle { text-align: center; color: #888; font-size: 1rem; margin-bottom: 1.5rem; }
</style>
"""


def card(valor: str, rotulo: str, sub: str = "", tom: str = "") -> None:
    classe = {"pos": "metric-value pos", "neg": "metric-value neg"}.get(tom, "metric-value")
    sub_html = f'<div class="metric-sub">{html.escape(sub)}</div>' if sub else ""
    st.markdown(
        f'<div class="metric-card"><div class="{classe}">{html.escape(valor)}</div>'
        f'<div class="metric-label">{html.escape(rotulo)}</div>{sub_html}</div>',
        unsafe_allow_html=True,
    )


def _fmt(valor: float, pct: bool = True, casas: int = 2) -> str:
    if valor is None or not np.isfinite(valor):
        return "∞" if valor == float("inf") else "—"
    return f"{valor * 100:.{casas}f}%" if pct else f"{valor:.{casas}f}"


def cards_otimizacao(resultado: ResultadoOtimizacao, orcamento: float, intensidade_shrink: float) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card(_fmt(resultado.retorno_esperado), "Retorno esperado (modelo, a.a.)",
             f"shrinkage Jorion {intensidade_shrink:.0%}")
    with c2:
        card(_fmt(resultado.volatilidade), "Volatilidade (a.a.)")
    with c3:
        card(_fmt(resultado.sharpe, pct=False, casas=2), "Sharpe ex-ante",
             tom="pos" if resultado.sharpe > 0 else "neg")
    with c4:
        baixo = orcamento * (resultado.retorno_esperado - resultado.volatilidade)
        alto = orcamento * (resultado.retorno_esperado + resultado.volatilidade)
        card(f"R$ {baixo:,.0f} a {alto:,.0f}", "Faixa de ±1σ em 12 meses",
             "estimativa de modelo; não é promessa")


def tabela_carteira(pesos: pd.Series, orcamento: float, info: dict,
                    growth: pd.DataFrame | None) -> None:
    linhas = []
    for ticker, peso in pesos[pesos > 0].sort_values(ascending=False).items():
        dados = info.get(ticker, {})
        linha = {"Ticker": ticker.replace(".SA", ""), "Nome": dados.get("nome", "—"),
                 "Setor": dados.get("setor", "—"), "Peso": f"{peso:.2%}",
                 "Valor (R$)": f"{peso * orcamento:,.2f}"}
        if growth is not None and ticker in growth.index:
            linha["Growth Score"] = f"{growth.loc[ticker, 'score']:.0f}"
            linha["Queima de caixa"] = "sim" if growth.loc[ticker, "queima_caixa"] else "—"
        linhas.append(linha)
    st.dataframe(pd.DataFrame(linhas), width="stretch", hide_index=True)


def _coluna_resumo(r: ResumoDesempenho, bt: ResultadoBacktest) -> list[str]:
    return [
        _fmt(r.retorno_total), _fmt(r.cagr), _fmt(r.volatilidade),
        _fmt(r.sharpe, False), _fmt(r.sortino, False), _fmt(r.calmar, False), _fmt(r.omega, False),
        _fmt(r.max_drawdown), _fmt(r.var_95_diario), _fmt(r.cvar_95_diario),
        f"{bt.turnover_anual_medio:.1f}x", f"R$ {bt.custos_totais:,.0f}", str(len(bt.falhas)),
    ]


ROTULOS_RESUMO = ["Retorno total", "CAGR", "Volatilidade", "Sharpe (CDI)", "Sortino (CDI)",
                  "Calmar", "Omega", "Max drawdown", "VaR 95% diário (C-F)", "CVaR 95% diário",
                  "Giro anual", "Custos pagos", "Falhas do otimizador"]


def tabela_comparativa(resultados: dict[str, ResultadoBacktest], rf_diaria: pd.Series,
                       benchmark: ResultadoBacktest | None) -> dict[str, ResumoDesempenho]:
    todos = dict(resultados)
    if benchmark is not None:
        todos[benchmark.nome] = benchmark
    resumos = {nome: resumo_desempenho(bt.valores, rf_diaria) for nome, bt in todos.items()}
    tabela = {"Métrica": ROTULOS_RESUMO}
    tabela.update({nome: _coluna_resumo(resumos[nome], bt) for nome, bt in todos.items()})
    st.dataframe(pd.DataFrame(tabela), width="stretch", hide_index=True)
    return resumos


def bloco_alfa(otimizada: ResultadoBacktest, benchmark: ResultadoBacktest, rf_diaria: pd.Series) -> None:
    try:
        comp = comparar_benchmark(otimizada.valores, benchmark.valores, rf_diaria)
    except ValueError as erro:
        st.info(f"Comparação com {benchmark.nome} indisponível: {erro}")
        return
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card(_fmt(comp.alfa_anual), f"Alfa de Jensen vs {benchmark.nome}", f"t = {comp.t_stat_alfa:.2f}",
             tom="pos" if comp.alfa_anual > 0 else "neg")
    with c2:
        card(_fmt(comp.beta, False), "Beta")
    with c3:
        card(_fmt(comp.tracking_error), "Tracking error")
    with c4:
        card(_fmt(comp.information_ratio, False), "Information ratio")
    if abs(comp.t_stat_alfa) < 2:
        st.caption("|t| < 2: o alfa não é estatisticamente distinguível de zero neste histórico.")


LIMITACOES = """
**O que este backtest mede e o que não mede**

- **Walk-forward fora da amostra.** Cada rebalanceamento trimestral usa só os 2 anos anteriores.
  A taxa livre de risco é o CDI vigente na data. Todas as estratégias usam as mesmas datas e custos.
- **O filtro e o tilt growth NÃO entram no backtest.** Eles usam fundamentos e momentum de HOJE. Filtrar
  o universo com eles e testar no mesmo período seleciona os vencedores do período (look-ahead).
  A curva OOS mede só a **alocação** sobre o universo pré-filtro.
- **Viés de sobrevivência.** O universo é a lista atual de tickers. Empresas que quebraram ou
  fecharam capital não aparecem, o que favorece todas as estratégias.
- **Custos.** O custo em bps é cobrado sobre o giro (compra e venda). **Imposto de renda não é modelado**
  (15% sobre ganho líquido; isenção para vendas de ações até R$ 20 mil/mês; cripto tem regras próprias).
- **Cripto.** BTC/ETH em BRL no calendário da B3: o fim de semana entra no retorno de segunda.
  O fechamento do Yahoo (00h UTC) não coincide com o da B3, e as correlações diárias ficam subestimadas.
- **Ferramenta educacional.** Não é recomendação de investimento. Retorno passado não garante retorno futuro.
"""
