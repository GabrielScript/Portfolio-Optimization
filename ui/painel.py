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

from core import dolar_bcb, macro_bcb
from core.indicadores import tabela_por_acao
from core.macro import resumo_focus
from core.sentimento import ultimo_fato
from ui.filtros import filtrar

AVISO = "Informação, não recomendação: nada aqui muda o plano nem as regras de risco."
AJUDA_DO_TOM = ("Média do tom das notícias relevantes do dia, de −1 a +1: acima de 0, mais notícias boas que ruins para "
                "a bolsa; abaixo de 0, mais ruins que boas.")
DOLAR_DO_FOCUS = "Câmbio"          # the Focus line the panel keeps (the adviser's pack still reads all of them)
PERCENTUAIS = ("Retorno 12m", "Vol anual", "Queda máx. (3 anos)", "ROE", "Margem líquida", "DY 12m", "CAGR",
               "CVaR 95% diário", "Alfa a.a.")
# the measures the user asked for on 2026-11-04 (core.indicadores.desempenho_por_acao), with a "?" each
AJUDA_DAS_MEDIDAS = {
    "CAGR": "Retorno composto ao ano nos 3 anos da tabela (ou desde a estreia).",
    "Sharpe": "Retorno acima da Selic por unidade de risco (volatilidade), ao ano. Acima de 0,5 é bom para ação.",
    "Sortino": "Como o Sharpe, mas só conta a volatilidade das quedas.",
    "Calmar": "Retorno ao ano dividido pela maior queda do período: quanto rende para cada ponto de tombo.",
    "CVaR 95% diário": "Perda média nos 5% piores dias: o tamanho típico de um dia muito ruim.",
    "Alfa a.a.": "Retorno ao ano que sobra depois de descontar o que o BOVA11 (o mercado) explica.",
    "t do alfa": "Confiança no alfa: acima de 2 (ou abaixo de −2), dificilmente é só sorte.",
    "Information ratio": "Retorno acima do BOVA11 por unidade de distância dele, ao ano.",
}
COLUNAS_COM_DUAS_CASAS = ("Beta", "P/VP", "Dív. líq./EBITDA", "Sharpe", "Sortino", "Calmar", "t do alfa",
                          "Information ratio")
# the material facts' tone (Jev's labels) in Portuguese, and what each score means (the user, 2026-11-04)
TONS_DO_FATO = {"good": "bom", "bad": "ruim", "neutral": "neutro", "unclear": "incerto"}
SEM_NOTA = "sem nota"
AJUDA_DO_TOM_DO_FATO = ("Leitura do documento pelo modelo Jev: bom, ruim, neutro ou incerto para a ação. \"Sem nota\": "
                        "ainda não lido.")
AJUDA_P_RUIM = "Chance, segundo o Jev, de o fato ser ruim para a ação."
AJUDA_P_BOM = "Chance, segundo o Jev, de o fato ser bom para a ação."
AJUDA_P_FORTE = ("Chance, segundo o Jev, de o impacto ser grande ou extraordinário (e não pequeno ou rotineiro). Um "
                 "fato com tom ruim e esta chance acima de 50% acende o alerta vermelho da ação para o assessor.")


def secao_mercado(hoje: date, diario: pd.DataFrame | None, fatos: pd.DataFrame | None,
                  fundamentos: pd.DataFrame | None, manchetes: pd.DataFrame | None, publicadas_em: datetime | None,
                  focus_anual: dict[str, pd.DataFrame] | None, indicadores_bcb: dict | None,
                  precos: pd.DataFrame, info: dict, referencia: str, desempenho: pd.DataFrame | None = None,
                  ata: macro_bcb.Ata | None = None, selic: float | None = None,
                  dolar: dolar_bcb.DolarDoDia | None = None) -> None:
    st.markdown("## Mercado agora: notícias, macro e ações")
    st.caption(AVISO)
    _noticias(diario, manchetes)
    _macro(hoje, focus_anual, indicadores_bcb, ata, selic, dolar)
    _acoes(precos, info, fundamentos, fatos, referencia, desempenho)


def _virgula(numero: float, casas: int) -> str:
    return f"{numero:.{casas}f}".replace(".", ",")


# The headline is its own link (the user, 2026-11-04): Streamlit's LinkColumn takes its text only from the URL, so the
# title rides after a fragment marker the article's site ignores, and a capture group shows it.
MARCA_DA_MANCHETE = "#manchete="
TEXTO_DA_MANCHETE = MARCA_DA_MANCHETE + "(.*)$"


def manchetes_com_link(manchetes: pd.DataFrame) -> pd.DataFrame:
    """The headlines with ``titulo`` as "link#manchete=title" and no ``link`` column; a headline without a link keeps
    its bare title."""
    if "link" not in manchetes.columns:
        return manchetes
    com_link = manchetes["link"].fillna("").astype(str) != ""
    titulos = manchetes["link"].astype(str) + MARCA_DA_MANCHETE + manchetes["titulo"].astype(str)
    return manchetes.assign(titulo=titulos.where(com_link, manchetes["titulo"])).drop(columns="link")


def _noticias(diario: pd.DataFrame | None, manchetes: pd.DataFrame | None) -> None:
    st.markdown("### Notícias e sentimento")
    if diario is None or diario.empty:
        st.caption("Painel de notícias indisponível agora.")
        return
    dias = diario.set_index(pd.to_datetime(diario["dia"]))
    ultimo = dias.iloc[-1]
    c1, c2 = st.columns(2)
    c1.metric("Notícias relevantes para a bolsa", f"{int(ultimo['relevantes'])}", border=True)
    c2.metric("Tom do mercado", "—" if pd.isna(ultimo["tom"]) else f"{ultimo['tom']:+.2f}", help=AJUDA_DO_TOM,
              border=True)
    if len(dias) > 1:
        st.line_chart(dias["tom"].rename("Tom do mercado"), y_label="Tom (−1 a +1)")
    if manchetes is not None and not manchetes.empty:
        st.dataframe(filtrar(manchetes_com_link(manchetes), "manchetes"), width="stretch", hide_index=True, height=320,
                     column_config={"quando": st.column_config.DatetimeColumn("Quando", format="DD/MM HH:mm"),
                                    "fonte": "Fonte", "tema": "Tema",
                                    "titulo": st.column_config.LinkColumn("Manchete", display_text=TEXTO_DA_MANCHETE,
                                                                          alignment="left"),
                                    "tom": st.column_config.NumberColumn("Tom", format="%+.2f")},
                     alt="Manchetes relevantes dos últimos 3 dias, com tema e tom do Jev")
    else:
        st.caption("Manchetes indisponíveis agora.")


IPCA_12M = "IPCA 12 meses (%)"         # the key of ``servicos.carregar_indicadores_bcb``
AJUDA_DA_SELIC = ("Taxa básica de juros da economia, definida pelo Copom a cada 45 dias. É o rendimento de referência "
                  "da renda fixa e a taxa sem risco usada no Sharpe e no Sortino deste app.")
AJUDA_DO_IPCA = "Inflação oficial do país (IBGE) acumulada nos últimos 12 meses."
AJUDA_DO_JURO_REAL = ("Quanto a Selic rende acima da inflação: (1 + Selic) / (1 + IPCA 12 meses) − 1. Juro real alto "
                      "deixa a renda fixa atraente e pesa no preço das ações.")
AJUDA_DO_DOLAR = ("PTAX de venda do Banco Central: média das cotações do mercado medida em boletins ao longo do dia "
                  "(abertura por volta das 10h, intermediários às 11h e 12h, fechamento às 13h). Antes do primeiro "
                  "boletim e em feriados, vale o último publicado.")


def _juros_e_inflacao(selic: float | None, indicadores: dict) -> None:
    """Today's Selic target, the IPCA over 12 months and the real rate between them (the user, 2026-11-05)."""
    ipca = indicadores.get(IPCA_12M, (None, None))[1]
    colunas = st.columns(3)
    colunas[0].metric("Selic hoje (meta do Copom)", "—" if selic is None else f"{_virgula(100 * selic, 2)}% a.a.",
                      help=AJUDA_DA_SELIC, border=True)
    colunas[1].metric("IPCA em 12 meses", "—" if ipca is None else f"{_virgula(ipca, 2)}%", help=AJUDA_DO_IPCA,
                      border=True)
    real = None if selic is None or ipca is None else (1 + selic) / (1 + ipca / 100) - 1
    colunas[2].metric("Juro real (Selic acima do IPCA)", "—" if real is None else f"{_virgula(100 * real, 2)}% a.a.",
                      help=AJUDA_DO_JURO_REAL, border=True)


def _macro(hoje: date, focus_anual: dict[str, pd.DataFrame] | None, indicadores_bcb: dict | None,
           ata: macro_bcb.Ata | None = None, selic: float | None = None,
           dolar: dolar_bcb.DolarDoDia | None = None) -> None:
    """The Banco Central block: today's Selic, the IPCA and the real rate (the user, 2026-11-05); activity, direct
    investment, credit, its cost and defaults and the last Copom minutes, each with a "?" (the user, 2026-11-04); then
    the day's last PTAX bulletin (``core.dolar_bcb``: the SGS close was a day behind until 13h) and the Focus median for
    the dollar this year and the next (the user's UI list of 2026-11-03)."""
    st.markdown("### Banco Central (ao vivo)")
    _juros_e_inflacao(selic, indicadores_bcb or {})
    _economia(indicadores_bcb or {}, ata)
    st.markdown("#### Dólar")
    colunas = st.columns(3)
    if dolar is None:
        colunas[0].metric("Dólar hoje (PTAX do Banco Central)", "—", help=AJUDA_DO_DOLAR, border=True)
    else:
        colunas[0].metric("Dólar hoje (PTAX do Banco Central)", f"R$ {_virgula(dolar.valor, 2)}",
                          f"{dolar.boletim}, {dolar.quando:%d/%m às %H:%M}", delta_color="off", delta_arrow="off",
                          help=AJUDA_DO_DOLAR, border=True)
    if focus_anual is None or DOLAR_DO_FOCUS not in focus_anual:
        st.caption("Projeção do Focus indisponível agora (API do BCB).")
        return
    linha = resumo_focus({DOLAR_DO_FOCUS: focus_anual[DOLAR_DO_FOCUS]}, hoje.year).iloc[0]
    for coluna, ano in zip(colunas[1:], (hoje.year, hoje.year + 1)):
        valor = linha[str(ano)]
        coluna.metric(f"Projeção Focus para o fim de {ano}", "—" if pd.isna(valor) else f"R$ {_virgula(valor, 2)}",
                      help="Relatório Focus: mediana das projeções de cerca de 100 bancos, gestoras e consultorias "
                           "que o Banco Central consulta toda semana (dólar, inflação, PIB e Selic; aqui, o dólar).",
                      border=True)


FORMATOS_DA_ECONOMIA = {
    macro_bcb.IBC_BR: lambda v: f"{_virgula(v, 1)}%", macro_bcb.IDP: lambda v: f"US$ {_virgula(v, 1)} bi",
    macro_bcb.CONCESSOES: lambda v: f"R$ {_virgula(v, 1)} bi", macro_bcb.CONCESSOES_12M: lambda v: f"{_virgula(v, 1)}%",
    macro_bcb.JUROS_DO_CREDITO: lambda v: f"{_virgula(v, 2)}%", macro_bcb.INADIMPLENCIA: lambda v: f"{_virgula(v, 2)}%"}


def _economia(indicadores: dict, ata: macro_bcb.Ata | None) -> None:
    """One metric per ``core.macro_bcb`` indicator present, in a fixed order, and the Copom minutes' link."""
    presentes = [nome for nome in FORMATOS_DA_ECONOMIA if nome in indicadores]
    if not presentes:
        st.caption("Indicadores do Banco Central indisponíveis agora (API do BCB).")
    else:
        colunas = st.columns(3)
        for posicao, nome in enumerate(presentes):
            quando, valor = indicadores[nome]
            colunas[posicao % 3].metric(nome, "—" if pd.isna(valor) else FORMATOS_DA_ECONOMIA[nome](valor),
                                        f"dado de {quando:%m/%Y}", delta_color="off", delta_arrow="off",
                                        help=macro_bcb.EXPLICACOES[nome], border=True)
    if ata is not None:
        st.markdown(f"**{macro_bcb.ATA_DO_COPOM}:** [{ata.titulo}]({ata.link}), publicada em {ata.data:%d/%m/%Y}",
                    help=macro_bcb.EXPLICACOES[macro_bcb.ATA_DO_COPOM])


def _acoes(precos: pd.DataFrame, info: dict, fundamentos: pd.DataFrame | None, fatos: pd.DataFrame | None,
           referencia: str, desempenho: pd.DataFrame | None = None) -> None:
    st.markdown("### Todas as ações, BDRs e ETFs: preço, risco, balanço e fatos relevantes")
    com_dados = precos.loc[:, precos.notna().any()]
    if referencia not in com_dados.columns:
        st.caption("Sem os preços do BOVA11 para medir o beta.")
        return
    tabela = tabela_por_acao(com_dados, info, None if fundamentos is None else fundamentos.drop(columns=["Balanço de"]),
                             referencia, desempenho).rename_axis("Ticker").reset_index()
    tabela.insert(1, "Tipo", [info.get(f"{ticker}.SA", {}).get("tipo", "") for ticker in tabela["Ticker"]])
    if fatos is not None and not fatos.empty:
        tabela["Último fato relevante"] = tabela["Ticker"].map(ultimo_fato(fatos)).fillna("")
    tabela = filtrar(tabela, "acoes", so_para_filtro=["Tipo"])     # the type is a filter, not a column (2026-11-03)
    st.caption(f"{len(tabela)} papéis na tabela.")
    st.dataframe(tabela, width="stretch", hide_index=True, height=420,
                 column_config={"Preço (R$)": st.column_config.NumberColumn(format="R$ %.2f"),
                                "Fechamento de": st.column_config.DateColumn(format="DD/MM/YYYY"),
                                "P/L": st.column_config.NumberColumn(format="%.1f"),
                                **{coluna: st.column_config.NumberColumn(format="%.2f", help=AJUDA_DAS_MEDIDAS.get(coluna))
                                   for coluna in COLUNAS_COM_DUAS_CASAS},
                                **{coluna: st.column_config.NumberColumn(format="percent", help=AJUDA_DAS_MEDIDAS.get(coluna))
                                   for coluna in PERCENTUAIS}},
                 alt="Todas as ações da lista: preço, risco, múltiplos do balanço e o último fato relevante")
    st.caption("Preços com dividendos reinvestidos (B3 e CVM), até o último pregão. Retorno, volatilidade e beta: 12 "
               "meses; CAGR, Sharpe, Sortino, Calmar, CVaR e alfa: 3 anos, contra a Selic e o BOVA11. Balanço: último "
               "ITR/DFP da CVM; bancos e seguradoras ficam sem múltiplos.")
    st.caption("BDRs: retorno só de preço (sem os dividendos), em reais; ganho paga 15% de IR, sem a isenção de R$ 20 "
               "mil por mês. Papéis com pouco negócio: confira o \"Fechamento de\".")
    if fatos is not None and not fatos.empty:
        with st.expander(f"Fatos relevantes e comunicados ao mercado dos últimos 90 dias ({len(fatos)})"):
            st.caption("A CVM publica estes dados uma vez por semana (aos domingos): os fatos dos últimos dias entram "
                       "na publicação seguinte. Tom e chances: leitura de cada documento pelo modelo Jev.")
            st.dataframe(filtrar(fatos_na_tela(fatos), "fatos"), width="stretch", hide_index=True, height=360,
                         column_config={"ticker": "Ticker", "empresa": "Empresa", "categoria": "Categoria",
                                        "assunto": "Assunto", "data": st.column_config.DateColumn(
                                            "Entregue em", format="DD/MM/YYYY"),
                                        "link": st.column_config.LinkColumn("Documento", display_text="CVM (link)"),
                                        "tom_do_fato": st.column_config.TextColumn("Tom do fato",
                                                                                   help=AJUDA_DO_TOM_DO_FATO),
                                        "p_ruim": st.column_config.NumberColumn("Chance de ruim", format="percent",
                                                                                help=AJUDA_P_RUIM),
                                        "p_bom": st.column_config.NumberColumn("Chance de bom", format="percent",
                                                                               help=AJUDA_P_BOM),
                                        "p_forte": st.column_config.NumberColumn("Chance de impacto forte",
                                                                                 format="percent", help=AJUDA_P_FORTE)},
                         alt="Fatos relevantes e comunicados ao mercado da CVM, mais novos primeiro")


def fatos_na_tela(fatos: pd.DataFrame) -> pd.DataFrame:
    """The filings with Jev's tone in Portuguese ("sem nota" for a filing not scored yet)."""
    if "tom_do_fato" not in fatos.columns:
        return fatos
    return fatos.assign(tom_do_fato=fatos["tom_do_fato"].map(TONS_DO_FATO).fillna(SEM_NOTA))
