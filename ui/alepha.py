"""
alepha.py - The past results block of the Aleph Quant page: the Composite and the Conservadora rerun without
survivorship bias up to the last close (``scripts/curvas_ao_vivo.py``), next to BOVA11, the Selic and the B3's two
bitcoin ETFs. The buy plan from the held positions left the page on 2026-11-04 (the user's UI list).

Advice boundary (prediction-market-risk-review): backtest numbers appear as measured past; nothing is promised in R$.
UI strings are Portuguese, like the rest of the app.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.periodos import (EXCESSO_DO_JURO, juro_por_ano, medidas_por_ano, medidas_por_periodo, resumo_do_periodo,
                           rotulo_do_ano)
from visualizations import grafico_curvas_capital
from ui.filtros import filtrar
from ui.formato import com_traco

TITULO_SEM_VIES = "## Backtest sem viés de sobrevivência: Composite, Conservadora e BOVA11"
SESSOES_DO_SEMAFORO = 21           # the DeepSeek pack reads the CVM filings of the last 21 sessions

BRACOS = ("Conservadora", "Composto", "BOVA11")              # the live rerun's arms (scripts/curvas_ao_vivo.py)
SELIC = "Selic"
CURVAS_DA_TELA = (*BRACOS, "BITH11", "QBTC11")
EXCESSO_BOVA11 = "Excesso sobre o BOVA11"
MEDIDAS_POR_ANO = ("Retorno", EXCESSO_BOVA11, EXCESSO_DO_JURO, "Vol", "Queda máx.", "Pior mês", "Sharpe", "Sortino",
                   "Calmar", "Beta (BOVA11)")
SEM_PORCENTO = {"Sharpe", "Sortino", "Calmar", "Beta (BOVA11)", "t do alfa", "Information ratio"}
PRIMEIRO_ANO = 2010


def selic_diaria(parte: pd.DataFrame, juros: pd.DataFrame | None) -> pd.Series:
    """The Selic of each session of ``parte`` (decimal per day), from the published rates (``juros``). Without the
    file, the curves' own CDI stands in (CDI and Selic differ by about 0.1 point a year)."""
    if juros is not None and SELIC in juros:
        return juros[SELIC].dropna().reindex(parte.index, method="ffill").fillna(0.0)
    return parte["CDI"].fillna(0.0)


def _tabela_por_ano(curvas: pd.DataFrame, selic: pd.Series, medida: str) -> pd.DataFrame:
    """Years since ``PRIMEIRO_ANO`` x columns of ``curvas``: ``medida`` of ``core.periodos.medidas_por_ano``, or the
    year's return minus BOVA11's."""
    medidas = medidas_por_ano(curvas, selic)
    if medida == EXCESSO_BOVA11:
        retornos = medidas["Retorno"].unstack()
        tabela = retornos.sub(retornos["BOVA11"], axis=0)
    else:
        tabela = medidas[medida].unstack()
    tabela = tabela.reindex(columns=list(curvas.columns)).astype(float)
    return tabela.loc[tabela.index >= PRIMEIRO_ANO]


def resultados_da_conservadora(quadro: pd.DataFrame | None, juros: pd.DataFrame | None = None) -> None:
    """The whole period, the last 5 and 3 years, each calendar year since 2010 in the measure picked, and an aggressive
    investor's risk summary. One rebalance phase (the first): the user found the phase picker meaningless for an
    investor (2026-11-04). ``juros``: the CDI and Selic of each published day (``scripts/juros_diarios.py``); the Selic
    is the risk-free rate and a fourth line of the chart (the user, 2026-11-04: the Selic only, no CDI)."""
    st.markdown(TITULO_SEM_VIES)
    if quadro is None:
        st.caption("Sem as curvas do backtest.")
        return
    parte = quadro[quadro["fase"] == quadro["fase"].min()].set_index("date")
    selic = selic_diaria(parte, juros)
    na_selic = (1.0 + selic).cumprod().rename(SELIC)
    st.plotly_chart(grafico_curvas_capital({**{nome: parte[nome].dropna() for nome in BRACOS}, SELIC: na_selic},
                                           destaque="Composto",
                                           titulo="Curva de capital (base 100), antes de custos e IR"),
                    width="stretch")
    com_selic = parte[list(BRACOS)].assign(**{SELIC: na_selic})
    medidas = medidas_por_periodo(com_selic)
    nomes = [*BRACOS, SELIC]
    tabela = pd.DataFrame({
        "Período": medidas.index,
        "De": [f"{data:%d/%m/%Y}" for data in medidas[("Início", "")]],
        "Até": [f"{data:%d/%m/%Y}" for data in medidas[("Fim", "")]],
        **{f"{medida} {nome}": medidas[(nome, coluna)].astype(float).to_numpy()
           for medida, coluna in (("Retorno a.a.", "CAGR"), ("Vol", "Vol anual"), ("Queda máx.", "MDD"))
           for nome in nomes}})
    st.dataframe(filtrar(tabela, "periodos"), width="stretch", hide_index=True,
                 column_config={coluna: st.column_config.NumberColumn(format="percent")
                                for coluna in tabela.columns[3:]},
                 alt="Conservadora, Composto, BOVA11 e Selic no período inteiro e nos últimos 5 e 3 anos")

    curvas = parte[list(CURVAS_DA_TELA)]
    medida = st.selectbox("Medida de cada ano", MEDIDAS_POR_ANO, key="medida_por_ano")
    anos = _tabela_por_ano(curvas, selic, medida)
    taxa = ({SELIC: juro_por_ano(juros[SELIC].dropna(), parte.index).reindex(anos.index).to_numpy()}
            if juros is not None and SELIC in juros and medida == "Retorno" else {})
    por_ano = pd.DataFrame({"Ano": [rotulo_do_ano(ano, parte.index) for ano in anos.index],
                            **{nome: anos[nome].to_numpy() for nome in BRACOS}, **taxa,
                            **{nome: anos[nome].to_numpy() for nome in CURVAS_DA_TELA[len(BRACOS):]}})
    filtrado = filtrar(por_ano, "por_ano")
    # "—" for the years before a fund listed (the bitcoin ETFs), not Streamlit's "None" (the user, 2026-11-04)
    tabela_do_ano, config_do_ano = com_traco(filtrado, por_ano.columns[1:], porcento=medida not in SEM_PORCENTO)
    st.dataframe(tabela_do_ano, width="stretch", hide_index=True, column_config=config_do_ano,
                 alt=f"{medida} de cada ano civil desde {PRIMEIRO_ANO}: {', '.join(por_ano.columns[1:])}")

    inicio_cripto = curvas[["BITH11", "QBTC11"]].dropna().index[0]
    mesma_janela = st.toggle(f"Mesma janela para todos (desde {inicio_cripto:%d/%m/%Y}, quando os dois ETFs de "
                             "bitcoin já negociavam)", key="mesma_janela")
    resumo = resumo_do_periodo(curvas.loc[inicio_cripto:] if mesma_janela else curvas, selic)
    resumo = resumo.assign(Desde=[f"{data:%d/%m/%Y}" for data in resumo["Desde"]]).rename_axis(
        "Estratégia").reset_index()
    numericas = [coluna for coluna in resumo.columns if coluna not in ("Estratégia", "Desde", "Anos acima do BOVA11")]
    filtrado = filtrar(resumo.astype({coluna: float for coluna in numericas}), "resumo")
    config: dict = {}
    for porcento in (True, False):                              # BOVA11 against itself has no alpha: "—"
        filtrado, parte_da_config = com_traco(
            filtrado, [c for c in numericas if (c in SEM_PORCENTO) != porcento], porcento=porcento)
        config.update(parte_da_config)
    st.dataframe(filtrado, width="stretch", hide_index=True, column_config=config,
                 alt="Retorno, risco e alfa de cada estratégia no período")
    inteiro = medidas.iloc[0]

    def ao_ano(nome: str) -> str:
        return f"{inteiro[(nome, 'CAGR')]:.1%}".replace(".", ",")
    st.caption(
        f"Simulação no passado sem viés de sobrevivência (as empresas que saíram da bolsa continuam na conta), antes de "
        f"custos e IR, até {parte.index[-1]:%d/%m/%Y}. Ao ano no período inteiro: Composite {ao_ano('Composto')}, "
        f"Conservadora {ao_ano('Conservadora')}, BOVA11 {ao_ano('BOVA11')} e Selic {ao_ano(SELIC)}. Sharpe e Sortino "
        "usam a Selic. "
        "BITH11 e QBTC11 (bitcoin) são só comparação. Passado não garante futuro.")
