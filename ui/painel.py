"""
painel.py - Screen block "Mercado agora": the election timing, the news tone of the lake's forward log, the CVM
material filings, the Focus survey and a table of every stock (risk from the prices, valuation from the CVM
statements). The user asked for it on 2026-10-23 (DECISOES 124).

Advice boundary (prediction-market-risk-review): information only. Nothing here changes the Alepha plan or the
risk rules; the news tone has not been tested as a signal (its pre-registered test reads in ~October 2027). UI strings
are Portuguese, like the rest of the app.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from core.indicadores import risco_por_acao
from core.macro import resumo_focus
from core.sentimento import (P_DO_ESTUDO, RAZAO_DE_VOL_DO_ESTUDO, RODADAS_DO_ESTUDO, ROTULOS_DOS_TOPICOS,
                             TOPICOS_DO_PAINEL, TURNOS_DE_2026, janela_eleitoral, ultimo_fato)

AVISO = ("Informação, não recomendação: nada aqui muda o plano do Alepha (a Conservadora registrada) nem as "
         "regras de risco. O tom das notícias e os fatos relevantes ainda não foram testados como sinal: o teste "
         "pré-registrado lê em ~outubro de 2027 (docs/preregistro-sentimento-2026-10-23.md).")
PERCENTUAIS = ("Retorno 12m", "Vol anual", "Queda máx. (3 anos)", "ROE", "Margem líquida", "DY 12m")


def secao_mercado(hoje: date, diario: pd.DataFrame | None, fatos: pd.DataFrame | None,
                  fundamentos: pd.DataFrame | None, manchetes: pd.DataFrame | None,
                  focus_anual: dict[str, pd.DataFrame] | None, indicadores_bcb: dict | None, selic,
                  precos: pd.DataFrame, info: dict, referencia: str) -> None:
    st.markdown("## Mercado agora: eleição, notícias, macro e ações")
    st.caption(AVISO)
    _timing(hoje)
    _noticias(diario, manchetes)
    _macro(hoje, focus_anual, indicadores_bcb, selic)
    _acoes(precos, info, fundamentos, fatos, referencia)


def _virgula(numero: float, casas: int) -> str:
    return f"{numero:.{casas}f}".replace(".", ",")


def _timing(hoje: date) -> None:
    janela = janela_eleitoral(hoje)
    if janela is None:
        return
    turno = "1º turno" if janela.turno == TURNOS_DE_2026[0] else "2º turno"
    quando = (f"faltam {janela.dias_ate} dia(s)" if janela.dias_ate > 0 else
              "é hoje" if janela.dias_ate == 0 else f"foi há {-janela.dias_ate} dia(s)")
    agora = " Estamos dentro dessa janela." if janela.dentro else ""
    st.info(f":material/how_to_vote: Eleição: {turno} em {janela.turno:%d/%m/%Y} ({quando}). No estudo pré-registrado "
            f"({RODADAS_DO_ESTUDO} votações de 2006 a 2022, docs/eleicao-2026-10-23.txt), a volatilidade do Ibovespa "
            f"nos pregões 0 a +4 depois da votação ({janela.janela_inicio:%d/%m} a {janela.janela_fim:%d/%m} desta "
            f"vez) foi {_virgula(RAZAO_DE_VOL_DO_ESTUDO, 2)} vez a de antes (p = {_virgula(P_DO_ESTUDO, 4)}), em todas "
            f"as votações; a direção não foi testada.{agora} O plano do Alepha não muda por isso: quem comprar "
            "ou vender nesses dias pode encontrar spread maior e preços mais nervosos.")


def _noticias(diario: pd.DataFrame | None, manchetes: pd.DataFrame | None) -> None:
    st.markdown("### Notícias e sentimento (política, eleição e economia)")
    if diario is None or diario.empty:
        st.caption("Sem o painel de notícias (lago/sentimento_diario.parquet).")
        return
    dias = diario.set_index(pd.to_datetime(diario["dia"]))
    ultimo = dias.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Itens no dia", f"{int(ultimo['itens'])}", f"{dias.index[-1]:%d/%m/%Y}", delta_color="off",
              delta_arrow="off", border=True)
    c2.metric("Relevantes para a bolsa", f"{int(ultimo['relevantes'])}", border=True)
    c3.metric("Tom do mercado", "—" if pd.isna(ultimo["tom"]) else f"{ultimo['tom']:+.2f}",
              help="Média de P(positivo) − P(negativo) do Jev nos itens relevantes: −1 a +1.", border=True)
    c4.metric("Sobre a eleição", "—" if pd.isna(ultimo["eleicao"]) else f"{ultimo['eleicao']:.0%}",
              help="Fração dos itens relevantes que falam da eleição de 2026.", border=True)
    if len(dias) > 1:
        st.line_chart(dias[["tom", "eleicao"]].rename(columns={"tom": "Tom", "eleicao": "Eleição (fração)"}),
                      y_label="Tom / fração")
    temas = dias[list(TOPICOS_DO_PAINEL)].tail(7).mean().rename(ROTULOS_DOS_TOPICOS)
    st.bar_chart(temas.rename("Peso médio nos últimos 7 dias"), horizontal=True, x_label="Peso médio do tema")
    if manchetes is not None and not manchetes.empty:
        st.dataframe(manchetes, width="stretch", hide_index=True, height=320,
                     column_config={"quando": st.column_config.DatetimeColumn("Quando", format="DD/MM HH:mm"),
                                    "fonte": "Fonte", "titulo": "Manchete", "tema": "Tema",
                                    "tom": st.column_config.NumberColumn("Tom", format="%+.2f"),
                                    "link": st.column_config.LinkColumn("Link", display_text="abrir")},
                     alt="Manchetes relevantes dos últimos 3 dias, com tema e tom do Jev")
    else:
        st.caption("As manchetes ficam só no computador do lago (o texto é dos veículos); na versão online vão só "
                   "os números.")
    st.caption(f"Log desde 23/10/2026: 9 veículos por RSS a cada 2 h, buscas e redes (Reddit, Threads) duas vezes por "
               f"dia; notas do Jev (TypeSafe) com perguntas fixas. Dados até {dias.index[-1]:%d/%m/%Y}.")


def _macro(hoje: date, focus_anual: dict[str, pd.DataFrame] | None, indicadores_bcb: dict | None, selic) -> None:
    st.markdown("### Macroeconomia (Banco Central, ao vivo)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Selic meta", f"{selic.valor:.2%}", selic.fonte, delta_color="off", delta_arrow="off", border=True)
    for coluna, (nome, (quando, valor)) in zip((c2, c3), (indicadores_bcb or {}).items()):
        coluna.metric(nome, f"{valor:,.2f}", f"{quando:%d/%m/%Y}", delta_color="off", delta_arrow="off", border=True)
    if focus_anual is None:
        st.caption("Focus indisponível agora (API do BCB).")
        return
    resumo = resumo_focus(focus_anual, hoje.year)
    st.dataframe(resumo, width="stretch",
                 column_config={"Pesquisa": st.column_config.DateColumn(format="DD/MM/YYYY")},
                 alt="Mediana do Focus para este ano e o próximo e a mudança em 4 semanas")
    st.caption("Focus: mediana das projeções do mercado (Selic e PIB em % a.a., IPCA em %, câmbio em R$/US$ no fim do "
               "ano). Os filtros macro foram testados no BOVA11 e não passaram (DECISOES 112): aqui é contexto.")


def _acoes(precos: pd.DataFrame, info: dict, fundamentos: pd.DataFrame | None, fatos: pd.DataFrame | None,
           referencia: str) -> None:
    st.markdown("### Todas as ações: preço, risco, balanço e fatos relevantes")
    com_dados = precos.loc[:, precos.notna().any()]
    if referencia not in com_dados.columns:
        st.caption("Sem os preços do BOVA11 para medir o beta.")
        return
    risco = risco_por_acao(com_dados, referencia).drop(index=referencia).rename(
        columns={"Queda máx.": "Queda máx. (3 anos)"})
    ultimos = com_dados.ffill().iloc[-1]
    datas = com_dados.apply(lambda serie: serie.last_valid_index())
    tabela = pd.DataFrame({"Ticker": [ticker.removesuffix(".SA") for ticker in risco.index],
                           "Empresa": [info.get(ticker, {}).get("nome", "") for ticker in risco.index],
                           "Setor": [info.get(ticker, {}).get("setor", "") for ticker in risco.index],
                           "Preço (R$)": ultimos.reindex(risco.index).to_numpy(),
                           "Fechamento de": datas.reindex(risco.index).to_numpy()}).join(
        risco.reset_index(drop=True))
    if fundamentos is not None:
        tabela = tabela.join(fundamentos.drop(columns=["Balanço de"]), on="Ticker")
    if fatos is not None and not fatos.empty:
        tabela["Último fato relevante"] = tabela["Ticker"].map(ultimo_fato(fatos)).fillna("")
    st.dataframe(tabela, width="stretch", hide_index=True, height=420,
                 column_config={"Preço (R$)": st.column_config.NumberColumn(format="R$ %.2f"),
                                "Fechamento de": st.column_config.DateColumn(format="DD/MM/YYYY"),
                                "Beta": st.column_config.NumberColumn(format="%.2f"),
                                "P/L": st.column_config.NumberColumn(format="%.1f"),
                                "P/VP": st.column_config.NumberColumn(format="%.2f"),
                                "Dív. líq./EBITDA": st.column_config.NumberColumn(format="%.2f"),
                                **{coluna: st.column_config.NumberColumn(format="percent") for coluna in PERCENTUAIS}},
                 alt="Todas as ações da lista: preço, risco, múltiplos do balanço e o último fato relevante")
    st.caption("Preço: fechamento de retorno total do lago (COTAHIST e eventos da B3/CVM), um pregão de atraso no "
               "máximo; risco nos últimos 252 pregões (beta contra o BOVA11) e queda máxima em 3 anos. Balanço: último "
               "ITR/DFP da CVM recebido (fluxos em 12 meses); bancos e seguradoras ficam sem múltiplos (outro plano de "
               "contas). "
               "Alguns valores de mercado logo depois de um IPO estão mil vezes pequenos (limitação do lago).")
    if fatos is not None and not fatos.empty:
        with st.expander(f"Fatos relevantes e comunicados ao mercado dos últimos 90 dias ({len(fatos)})"):
            st.dataframe(fatos, width="stretch", hide_index=True, height=360,
                         column_config={"ticker": "Ticker", "empresa": "Empresa", "categoria": "Categoria",
                                        "assunto": "Assunto", "data": st.column_config.DateColumn(
                                            "Entregue em", format="DD/MM/YYYY"),
                                        "link": st.column_config.LinkColumn("Documento", display_text="CVM")},
                         alt="Fatos relevantes e comunicados ao mercado da CVM, mais novos primeiro")
