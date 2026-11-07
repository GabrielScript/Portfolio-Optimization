"""
comparacao.py - The strategies comparison of the Aleph Quant page (the user's item 15, 2026-11-05, DECISOES 257): every
portfolio method of the project and the references in one precomputed run (``scripts/comparacao_estrategias.py``),
same engine, dates, universe and costs, with every risk measure of ``core.periodos.resumo_do_periodo``, the turnover
and the names held, and a year table. The Selic is a curve too, with a dash where a premium over itself is meaningless.

Advice boundary: measured past, before taxes; nothing is promised. UI strings are Portuguese, like the rest of the app.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.periodos import medidas_por_ano, resumo_do_periodo, rotulo_do_ano
from ui.alepha import EXCESSO_BOVA11, MEDIDAS_POR_ANO, SELIC, SEM_PORCENTO, SEM_SENTIDO_NA_SELIC, selic_diaria
from ui.filtros import filtrar
from ui import formato
from visualizations import grafico_curvas_capital

TITULO = "## Comparação das estratégias"
REFERENCIA = "BOVA11 (Ibovespa)"
# the chart's curves, fixed: the "Curvas no gráfico" picker left on 2026-11-06 (the user); the Conservadora off the
# main line (DECISOES 261)
NO_GRAFICO = ("Composite", REFERENCIA, SELIC)
GIRO, NOMES = "Giro anual", "Nomes médios"
PRIMEIRO_ANO = 2010
LEGENDA = (
    "Todas as estratégias rodam no mesmo motor, nas mesmas datas, com o mesmo universo (todas as ações elegíveis da "
    "B3 em cada data, inclusive as que depois saíram da bolsa), troca a cada 3 meses e 0,1% de custo por lado, antes "
    "de Imposto de Renda. Média-variância, Kelly e reamostrada: no máximo 20% por ação e até 10 ações. Paridade de risco, HRP e 1/N "
    "usam todas as elegíveis. Composite top-5: as 5 ações de maior nota da Composite, uma por setor, 20% cada. "
    "A quântica escolhe 15 ações por recozimento simulado (computador clássico) e pondera por "
    "máximo Sharpe. Tesouro IPCA+: média dos títulos IPCA+ com mais de um ano até o vencimento (Tesouro Transparente). "
    "Sharpe e Sortino sobre a Selic; alfa, beta e Information Ratio contra o BOVA11. Passado não garante futuro.")


def _curvas_com_selic(curvas: pd.DataFrame, juros: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.Series]:
    selic = selic_diaria(curvas, juros)
    nomes = [coluna for coluna in curvas.columns if coluna != "CDI"]
    return curvas[nomes].assign(**{SELIC: 10_000 * (1.0 + selic).cumprod()}), selic


def _resumo(curvas: pd.DataFrame, selic: pd.Series, giro: pd.DataFrame) -> pd.DataFrame:
    resumo = resumo_do_periodo(curvas, selic, REFERENCIA)
    resumo.loc[SELIC, [coluna for coluna in SEM_SENTIDO_NA_SELIC if coluna in resumo.columns]] = float("nan")
    resumo[GIRO] = giro["giro_anual"].reindex(resumo.index)
    resumo[NOMES] = giro["nomes_medios"].reindex(resumo.index)
    return resumo.assign(Desde=[f"{data:%d/%m/%Y}" for data in resumo["Desde"]]).rename_axis(
        "Estratégia").reset_index()


def secao(dados: tuple[pd.DataFrame, pd.DataFrame] | None, juros: pd.DataFrame | None) -> None:
    st.markdown(TITULO)
    if dados is None:
        st.caption("Sem a comparação pré-calculada (scripts/comparacao_estrategias.py).")
        return
    brutas, giro = dados
    curvas, selic = _curvas_com_selic(brutas, juros)
    nomes = list(curvas.columns)
    no_grafico = [nome for nome in NO_GRAFICO if nome in nomes]
    if no_grafico:
        st.plotly_chart(grafico_curvas_capital({nome: curvas[nome].dropna() for nome in no_grafico},
                                               destaque="Composite",
                                               titulo="Curva de capital (base 100), depois de custos e antes do "
                                                      "Imposto de Renda"),
                        width="stretch")
    resumo = _resumo(curvas, selic, giro)
    numericas = [c for c in resumo.columns if c not in ("Estratégia", "Desde", "Anos acima do BOVA11")]
    filtrado = filtrar(resumo.astype({coluna: float for coluna in numericas}), "comparacao")
    sem_porcento = SEM_PORCENTO | {GIRO, NOMES}
    filtrado, config = formato.br(filtrado, {c: formato.decimal if c in sem_porcento else formato.percentual
                                             for c in numericas})
    st.dataframe(filtrado, width="stretch", hide_index=True, column_config=config,
                 alt="Retorno, risco, alfa, giro e número de ações de cada estratégia e referência")

    medida = st.selectbox("Medida de cada ano", MEDIDAS_POR_ANO, key="medida_comparacao")
    anos = _tabela_por_ano(curvas, selic, medida)
    por_ano = pd.DataFrame({"Ano": [rotulo_do_ano(ano, curvas.index) for ano in anos.index],
                            **{nome: anos[nome].to_numpy() for nome in nomes}})
    escrever = formato.decimal if medida in SEM_PORCENTO else formato.percentual
    tabela, config_do_ano = formato.br(filtrar(por_ano, "comparacao_por_ano"),
                                       dict.fromkeys(por_ano.columns[1:], escrever))
    st.dataframe(tabela, width="stretch", hide_index=True,
                 column_config={**config_do_ano, "Ano": formato.COLUNA_DO_ANO},
                 alt=f"{medida} de cada ano civil desde {PRIMEIRO_ANO} para cada estratégia e referência")
    st.caption(LEGENDA)


def _tabela_por_ano(curvas: pd.DataFrame, selic: pd.Series, medida: str) -> pd.DataFrame:
    """Years since ``PRIMEIRO_ANO`` x curves: ``medida`` of ``medidas_por_ano`` against ``REFERENCIA``, or the year's
    return minus the reference's; the Selic's meaningless cells blank."""
    medidas = medidas_por_ano(curvas, selic, REFERENCIA)
    if medida == EXCESSO_BOVA11:
        retornos = medidas["Retorno"].unstack()
        tabela = retornos.sub(retornos[REFERENCIA], axis=0)
    else:
        tabela = medidas[medida].unstack()
    tabela = tabela.reindex(columns=list(curvas.columns)).astype(float)
    if medida in SEM_SENTIDO_NA_SELIC:
        tabela[SELIC] = float("nan")
    return tabela.loc[tabela.index >= PRIMEIRO_ANO]
