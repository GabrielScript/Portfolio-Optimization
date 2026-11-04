"""
painel.py - Screen block "Mercado agora": the election timing, the news tone of the lake's forward log, the CVM
material filings, the Focus survey and a table of every stock (risk from the prices, valuation from the CVM
statements). The user asked for it on 2026-10-23 (DECISOES 124).

Advice boundary (prediction-market-risk-review): information only. Nothing here changes the Aleph Quant plan or the
risk rules; the news tone has not been tested as a signal (its pre-registered test reads in ~October 2027). UI strings
are Portuguese, like the rest of the app.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import streamlit as st

from core.indicadores import tabela_por_acao
from core.macro import resumo_focus
from core.sentimento import ROTULOS_DOS_TOPICOS, TOPICOS_DO_PAINEL, ultimo_fato
from ui.filtros import filtrar

AVISO = "Informação, não recomendação: nada aqui muda o plano nem as regras de risco."
AJUDA_DO_TOM = ("Média do tom das notícias relevantes do dia, de −1 a +1: acima de 0, mais notícias boas que ruins para "
                "a bolsa; abaixo de 0, mais ruins que boas.")
DOLAR_DO_FOCUS = "Câmbio"          # the Focus line the panel keeps (the adviser's pack still reads all of them)
PERCENTUAIS = ("Retorno 12m", "Vol anual", "Queda máx. (3 anos)", "ROE", "Margem líquida", "DY 12m")


def secao_mercado(hoje: date, diario: pd.DataFrame | None, fatos: pd.DataFrame | None,
                  fundamentos: pd.DataFrame | None, manchetes: pd.DataFrame | None, publicadas_em: datetime | None,
                  focus_anual: dict[str, pd.DataFrame] | None, indicadores_bcb: dict | None,
                  precos: pd.DataFrame, info: dict, referencia: str) -> None:
    st.markdown("## Mercado agora: notícias, macro e ações")
    st.caption(AVISO)
    _noticias(diario, manchetes)
    _macro(hoje, focus_anual, indicadores_bcb)
    _acoes(precos, info, fundamentos, fatos, referencia)


def _virgula(numero: float, casas: int) -> str:
    return f"{numero:.{casas}f}".replace(".", ",")


def _noticias(diario: pd.DataFrame | None, manchetes: pd.DataFrame | None) -> None:
    st.markdown("### Notícias e sentimento")
    if diario is None or diario.empty:
        st.caption("Sem o painel de notícias (lago/sentimento_diario.parquet).")
        return
    dias = diario.set_index(pd.to_datetime(diario["dia"]))
    ultimo = dias.iloc[-1]
    c1, c2 = st.columns(2)
    c1.metric("Notícias relevantes para a bolsa", f"{int(ultimo['relevantes'])}", border=True)
    c2.metric("Tom do mercado", "—" if pd.isna(ultimo["tom"]) else f"{ultimo['tom']:+.2f}", help=AJUDA_DO_TOM,
              border=True)
    if len(dias) > 1:
        st.line_chart(dias["tom"].rename("Tom do mercado"), y_label="Tom (−1 a +1)")
    temas = dias[list(TOPICOS_DO_PAINEL)].tail(7).mean().rename(ROTULOS_DOS_TOPICOS)
    st.bar_chart(temas.rename("Peso médio nos últimos 7 dias"), horizontal=True, x_label="Peso médio do tema")
    if manchetes is not None and not manchetes.empty:
        st.dataframe(filtrar(manchetes, "manchetes"), width="stretch", hide_index=True, height=320,
                     column_config={"quando": st.column_config.DatetimeColumn("Quando", format="DD/MM HH:mm"),
                                    "fonte": "Fonte", "titulo": "Manchete", "tema": "Tema",
                                    "tom": st.column_config.NumberColumn("Tom", format="%+.2f"),
                                    "link": st.column_config.LinkColumn("Link", display_text="abrir")},
                     alt="Manchetes relevantes dos últimos 3 dias, com tema e tom do Jev")
    else:
        st.caption("Manchetes indisponíveis agora.")


def _macro(hoje: date, focus_anual: dict[str, pd.DataFrame] | None, indicadores_bcb: dict | None) -> None:
    """The dollar alone (the user's UI list of 2026-11-03): the last PTAX and the Focus median for this year and the
    next, with no source codes or dates."""
    st.markdown("### Dólar (Banco Central, ao vivo)")
    ptax = next((valor for nome, (_, valor) in (indicadores_bcb or {}).items() if nome.startswith("Dólar")), None)
    colunas = st.columns(3)
    colunas[0].metric("Dólar PTAX hoje", "—" if ptax is None else f"R$ {_virgula(ptax, 2)}", border=True)
    if focus_anual is None or DOLAR_DO_FOCUS not in focus_anual:
        st.caption("Projeção do Focus indisponível agora (API do BCB).")
        return
    linha = resumo_focus({DOLAR_DO_FOCUS: focus_anual[DOLAR_DO_FOCUS]}, hoje.year).iloc[0]
    for coluna, ano in zip(colunas[1:], (hoje.year, hoje.year + 1)):
        valor = linha[str(ano)]
        coluna.metric(f"Projeção Focus para o fim de {ano}", "—" if pd.isna(valor) else f"R$ {_virgula(valor, 2)}",
                      help="Mediana das projeções dos economistas que o Banco Central consulta toda semana.",
                      border=True)


def _acoes(precos: pd.DataFrame, info: dict, fundamentos: pd.DataFrame | None, fatos: pd.DataFrame | None,
           referencia: str) -> None:
    st.markdown("### Todas as ações, BDRs e ETFs: preço, risco, balanço e fatos relevantes")
    com_dados = precos.loc[:, precos.notna().any()]
    if referencia not in com_dados.columns:
        st.caption("Sem os preços do BOVA11 para medir o beta.")
        return
    tabela = tabela_por_acao(com_dados, info, None if fundamentos is None else fundamentos.drop(columns=["Balanço de"]),
                             referencia).rename_axis("Ticker").reset_index()
    tabela.insert(1, "Tipo", [info.get(f"{ticker}.SA", {}).get("tipo", "") for ticker in tabela["Ticker"]])
    if fatos is not None and not fatos.empty:
        tabela["Último fato relevante"] = tabela["Ticker"].map(ultimo_fato(fatos)).fillna("")
    tabela = filtrar(tabela, "acoes", so_para_filtro=["Tipo"])     # the type is a filter, not a column (2026-11-03)
    st.caption(f"{len(tabela)} papéis na tabela.")
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
    st.caption("BDRs: retorno só de preço (o lago não tem os dividendos dos BDRs), em reais com o dólar dentro, sem "
               "balanço da CVM; ganho em BDR paga 15% de IR sem a isenção de R$ 20 mil por mês. ETFs reinvestem. "
               "Entram todos os papéis negociados nos últimos 15 dias, sem piso de liquidez: veja o \"Fechamento "
               "de\" e desconfie do risco de quem quase não negocia. Nome e setor dos BDRs e ETFs vêm do cadastro da "
               "B3; saltos de preço deles não passam pela revisão do lago.")
    if fatos is not None and not fatos.empty:
        with st.expander(f"Fatos relevantes e comunicados ao mercado dos últimos 90 dias ({len(fatos)})"):
            st.dataframe(filtrar(fatos, "fatos"), width="stretch", hide_index=True, height=360,
                         column_config={"ticker": "Ticker", "empresa": "Empresa", "categoria": "Categoria",
                                        "assunto": "Assunto", "data": st.column_config.DateColumn(
                                            "Entregue em", format="DD/MM/YYYY"),
                                        "link": st.column_config.LinkColumn("Documento", display_text="CVM")},
                         alt="Fatos relevantes e comunicados ao mercado da CVM, mais novos primeiro")
