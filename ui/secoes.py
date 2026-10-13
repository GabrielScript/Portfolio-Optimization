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
from core.metrics import (MomentosSharpe, ResumoDesempenho, comparar_benchmark, momentos_sharpe,
                          resumo_desempenho, retornos_de_valores, sharpe_probabilistico,
                          sharpe_referencia_deflacionado)
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


def tabela_estabilidade(pesos: pd.Series, frequencia: pd.Series) -> None:
    """Peso final contra a fração das reamostras em que o ativo teve peso > 1%: frequência baixa = ruído."""
    tabela = pd.DataFrame({"Peso": pesos, "% das reamostras em que aparece": frequencia})
    tabela = tabela[(tabela > 0).any(axis=1)].sort_values("% das reamostras em que aparece", ascending=False)
    tabela.index = tabela.index.str.replace(".SA", "", regex=False).rename("Ticker")
    st.dataframe(tabela, width="stretch", column_config={
        "Peso": st.column_config.NumberColumn(format="percent"),
        "% das reamostras em que aparece": st.column_config.NumberColumn(format="percent"),
    })


def _coluna_resumo(r: ResumoDesempenho, bt: ResultadoBacktest, psr: float, dsr: float) -> list[str]:
    return [
        _fmt(r.retorno_total), _fmt(r.cagr), _fmt(r.volatilidade),
        _fmt(r.sharpe, False), _fmt(r.sortino, False), _fmt(r.calmar, False), _fmt(r.omega, False),
        _fmt(psr, casas=1), _fmt(dsr, casas=1),
        _fmt(r.max_drawdown), _fmt(r.var_95_diario), _fmt(r.cvar_95_diario),
        f"{bt.turnover_anual_medio:.1f}x", f"R$ {bt.custos_totais:,.0f}", str(len(bt.falhas)),
    ]


ROTULOS_RESUMO = ["Retorno total", "CAGR", "Volatilidade", "Sharpe (CDI)", "Sortino (CDI)",
                  "Calmar", "Omega", "PSR (Sharpe > 0)", "DSR (deflacionado)",
                  "Max drawdown", "VaR 95% diário (C-F)", "CVaR 95% diário",
                  "Giro anual", "Custos pagos", "Falhas do otimizador"]

LEGENDA_PSR_DSR = (
    "**PSR**: probabilidade de o Sharpe verdadeiro sobre o CDI ser positivo, dado o tamanho da amostra, "
    "a assimetria e a curtose (Bailey e López de Prado, 2012). **DSR**: o mesmo, descontando a escolha da "
    "melhor entre as {n} estratégias da tabela (2014). Ajustes manuais de parâmetros não entram na conta, "
    "então o DSR é otimista. Abaixo de 95%, a vantagem pode ser sorte."
)


def _momentos_do_excesso(bt: ResultadoBacktest, rf_diaria: pd.Series) -> MomentosSharpe:
    retornos = retornos_de_valores(bt.valores)
    return momentos_sharpe(retornos - rf_diaria.reindex(retornos.index).fillna(0.0))


def _psr_e_dsr(resultados: dict[str, ResultadoBacktest], benchmark: ResultadoBacktest | None,
               rf_diaria: pd.Series) -> dict[str, tuple[float, float]]:
    """
    PSR de todas; DSR só das estratégias (o benchmark não foi uma escolha entre tentativas).
    Sharpe NaN (curva constante) fica fora de N e da variância, senão anularia o DSR das outras.
    """
    momentos = {nome: _momentos_do_excesso(bt, rf_diaria) for nome, bt in resultados.items()}
    sharpes = [m.sharpe for m in momentos.values() if np.isfinite(m.sharpe)]
    variancia = float(np.var(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
    referencia = sharpe_referencia_deflacionado(variancia, len(sharpes))
    saida = {nome: (sharpe_probabilistico(m), sharpe_probabilistico(m, referencia))
             for nome, m in momentos.items()}
    if benchmark is not None:
        saida[benchmark.nome] = (sharpe_probabilistico(_momentos_do_excesso(benchmark, rf_diaria)),
                                 float("nan"))
    return saida


def tabela_comparativa(resultados: dict[str, ResultadoBacktest], rf_diaria: pd.Series,
                       benchmark: ResultadoBacktest | None) -> dict[str, ResumoDesempenho]:
    todos = dict(resultados)
    if benchmark is not None:
        todos[benchmark.nome] = benchmark
    resumos = {nome: resumo_desempenho(bt.valores, rf_diaria) for nome, bt in todos.items()}
    psr_dsr = _psr_e_dsr(resultados, benchmark, rf_diaria)
    tabela = {"Métrica": ROTULOS_RESUMO}
    tabela.update({nome: _coluna_resumo(resumos[nome], bt, *psr_dsr[nome]) for nome, bt in todos.items()})
    st.dataframe(pd.DataFrame(tabela), width="stretch", hide_index=True)
    st.caption(LEGENDA_PSR_DSR.format(n=len(resultados)))
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
  (15% sobre ganho líquido; isenção para vendas de ações até R$ 20 mil/mês).
- **Ferramenta educacional.** Não é recomendação de investimento. Retorno passado não garante retorno futuro.
"""
